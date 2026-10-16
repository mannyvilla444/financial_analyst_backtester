// static/app.js

// ── Tab switching ──────────────────────────────────────────────────────────────
const tabHooks = {};   // tab name → function run when the tab is shown
function showTab(name) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + name));
  if (tabHooks[name]) tabHooks[name]();
}
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => showTab(btn.dataset.tab));
});

// ── Shareable link for an open saved report (/?report=<id>) ───────────────────
function setReportParam(id) {
  const url = new URL(location.href);
  if (id) url.searchParams.set("report", id); else url.searchParams.delete("report");
  if (url.href !== location.href) history.replaceState(null, "", url.pathname + url.search + url.hash);
}
function reportLink(id) {
  return `${location.origin}${location.pathname}?report=${encodeURIComponent(id)}`;
}
function fmtRunTime(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return "";
  return d.toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
}

// ── Utility: escape text before putting it into innerHTML ─────────────────────
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// ── Utility: parse a JSON response, tolerating HTML error pages ───────────────
async function readJson(res) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch (_) {
    // e.g. a 413 "Request Entity Too Large" or a proxy timeout page
    return { error: `Server returned ${res.status} ${res.statusText || ""}`.trim() };
  }
}

// ── Utility: render HTML into an iframe ───────────────────────────────────────
// The iframe fills the viewport-height .report-area and scrolls internally.
// (Auto-sizing the frame to its scrollHeight can loop forever with responsive
// Chart.js canvases, so it is intentionally not done.)
function renderInFrame(frameId, htmlString) {
  document.getElementById(frameId).srcdoc = htmlString;
}

// ── Utility: download HTML as a file ──────────────────────────────────────────
function downloadHtml(htmlString, filename) {
  const blob = new Blob([htmlString], { type: "text/html" });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 1000);
}

// ── Utility: show/hide loading + error ────────────────────────────────────────
function setLoading(loadingId, errorId, on, text) {
  const l = document.getElementById(loadingId);
  const e = document.getElementById(errorId);
  l.hidden = !on;
  if (text) {
    const t = l.querySelector("#loading-text");
    if (t) t.textContent = text;
  }
  if (on) e.hidden = true;
}
function showError(errorId, msg) {
  const e = document.getElementById(errorId);
  e.textContent = "⚠ " + msg;
  e.hidden = false;
}

async function postJson(url, body) {
  const res  = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await readJson(res);
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

// ══════════════════════════════════════════════════════════════════════════════
//  SAVE TO LIBRARY — the prompt shown after a report is generated
// ══════════════════════════════════════════════════════════════════════════════
const REPORT_TICKER_RE = /^[A-Z0-9][A-Z0-9.\-]{0,9}$/;
const libState = { available: null, deleteEnabled: false, stale: true };

async function loadLibraryStatus() {
  try {
    const res = await fetch("/api/reports/status");
    const d = await readJson(res);
    libState.available = !!d.available;
    libState.deleteEnabled = !!d.delete_enabled;
  } catch (_) {
    libState.available = false;
  }
  return libState;
}

// One prompt per tab. rows: { token, save, label, ticker, state, name, error }
function makeSavePrompt(panelId) {
  const el = document.getElementById(panelId);
  let rows = [];
  let shown = false;

  function nameFor(r) {
    const t = (r.ticker || "").trim().toUpperCase();
    return REPORT_TICKER_RE.test(t) ? `${t}_${r.save.type_part}_${r.save.run_date}` : null;
  }
  const open = () => rows.filter(r => r.state === "pending" || r.state === "error");

  function render() {
    if (!rows.length) { el.hidden = true; el.innerHTML = ""; shown = false; return; }
    const pend = open();
    el.hidden = false;
    el.innerHTML = "";
    if (!pend.length) {
      // Everything decided → a one-line summary that can be dismissed
      const saved = rows.filter(r => r.state === "saved");
      const line = document.createElement("div");
      line.className = "save-done";
      if (saved.length) {
        line.innerHTML = `<span class="ok">✓ Saved as ${saved.map(r => `<b>${esc(r.name)}</b>`).join(", ")}</span>`;
        const lib = Object.assign(document.createElement("button"), { type: "button", className: "link-btn", textContent: "Open Saved Reports" });
        lib.onclick = () => openLibrary({});
        line.appendChild(lib);
      } else {
        line.innerHTML = `<span class="muted">Not saved to the library.</span>`;
      }
      const x = Object.assign(document.createElement("button"), { type: "button", className: "save-x", textContent: "×", title: "Dismiss" });
      x.setAttribute("aria-label", "Dismiss");
      x.onclick = () => { rows = []; render(); };
      line.appendChild(x);
      el.appendChild(line);
      return;
    }
    const head = document.createElement("div");
    head.className = "save-head";
    head.innerHTML = `<strong id="${panelId}-title">Save ${pend.length > 1 ? `these ${pend.length} reports` : "this report"} to the library?</strong>
      <span class="save-help">Check the ticker — it names the report. Saved reports are shared: anyone who can open this app can see them.</span>`;
    el.setAttribute("aria-labelledby", `${panelId}-title`);
    el.appendChild(head);

    rows.forEach((r, i) => {
      if (r.state === "skipped") return;
      const row = document.createElement("div");
      row.className = "save-row " + r.state;
      if (r.state === "saved") {
        row.innerHTML = `<span class="save-label">${esc(r.label)}</span><span class="ok">✓ Saved as <b>${esc(r.name)}</b></span>`;
        el.appendChild(row);
        return;
      }
      const id = `${panelId}-t${i}`;
      const lab = Object.assign(document.createElement("label"), { htmlFor: id, className: "save-label", textContent: r.label });
      const inp = Object.assign(document.createElement("input"), {
        type: "text", id, value: r.ticker, maxLength: 10, autocomplete: "off", spellcheck: false
      });
      inp.setAttribute("autocapitalize", "characters");
      inp.setAttribute("aria-describedby", `${id}-name`);
      const nm = document.createElement("code");
      nm.className = "save-name";
      nm.id = `${id}-name`;
      const upd = () => {
        r.ticker = inp.value;
        const n = nameFor(r);
        nm.textContent = n || "Enter a ticker: 1–10 letters, digits, . or -";
        nm.classList.toggle("bad", !n);
        inp.setAttribute("aria-invalid", String(!n));
        const btn = el.querySelector(".save-go");
        if (btn) btn.disabled = !open().some(nameFor);
      };
      inp.addEventListener("input", upd);
      inp.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); saveAll(); } });
      row.append(lab, inp, nm);
      if (pend.length > 1) {
        const skip = Object.assign(document.createElement("button"), { type: "button", className: "link-btn save-skip", textContent: "Skip" });
        skip.onclick = () => { r.state = "skipped"; render(); };
        row.appendChild(skip);
      }
      if (r.error) {
        const er = Object.assign(document.createElement("div"), { className: "save-err", textContent: "⚠ " + r.error });
        row.appendChild(er);
      }
      el.appendChild(row);
      setTimeout(upd);
    });

    const foot = document.createElement("div");
    foot.className = "save-foot";
    const go = Object.assign(document.createElement("button"), {
      type: "button", className: "btn-primary save-go", textContent: pend.length > 1 ? `Save all (${pend.length})` : "Save"
    });
    go.onclick = saveAll;
    const no = Object.assign(document.createElement("button"), { type: "button", className: "btn-ghost", textContent: "Don't save" });
    no.onclick = dismiss;
    foot.append(go, no);
    el.appendChild(foot);
  }

  function dismiss() {
    open().forEach(r => { r.state = "skipped"; });
    render();
  }

  async function saveAll() {
    const todo = open().filter(r => nameFor(r));
    if (!todo.length || el.dataset.busy) return;
    el.dataset.busy = "1";
    const btn = el.querySelector(".save-go");
    if (btn) { btn.disabled = true; btn.textContent = "Saving…"; }
    for (const r of todo) {
      try {
        const res = await fetch("/api/reports", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ report_token: r.token, ticker: r.ticker.trim().toUpperCase() })
        });
        const d = await readJson(res);
        if (!res.ok) throw new Error(d.error || "Save failed");
        Object.assign(r, { state: "saved", name: d.name, error: null });
        libState.stale = true;
      } catch (err) {
        Object.assign(r, { state: "error", error: err.message });
      }
    }
    delete el.dataset.busy;
    render();
  }

  el.addEventListener("keydown", e => { if (e.key === "Escape" && open().length) { e.preventDefault(); dismiss(); } });

  return {
    // data = a report response: { report_token, report_save: {...} }
    offer(data, label) {
      const save = data && data.report_save;
      if (!save) return;
      if (!save.available) {
        if (!rows.length) { el.hidden = false; el.innerHTML = `<div class="save-done"><span class="muted">Report library is unavailable on this server.</span></div>`; }
        return;
      }
      if (!data.report_token) return;
      rows.push({ token: data.report_token, save, label, ticker: save.suggested_ticker || "", state: "pending", name: null, error: null });
      const first = !shown;
      shown = true;
      render();
      if (first) {
        const ae = document.activeElement;
        if (!ae || ae === document.body || ae.tagName === "BUTTON") {
          const inp = el.querySelector("input");
          if (inp) { inp.focus({ preventScroll: true }); inp.select(); }
        }
      }
    },
    clear() { rows = []; render(); },
  };
}

const TYPE_LABELS = { financial: "Financial analysis", estimates: "Estimates", backtest: "Backtest" };

function savedBannerHtml(meta) {
  const p = meta.params || {};
  const extra = meta.type === "backtest" && p.timeframe
    ? ` · ${esc(p.timeframe)} MACD refs · stop: ${esc(p.stop_label || p.stop || "none")}` : "";
  return `<span class="sb-icon" aria-hidden="true">📁</span>
    <span class="sb-text">Saved report · <b>${esc(meta.name)}</b> · run ${esc(fmtRunTime(meta.created_at))}${extra}</span>
    <button type="button" class="link-btn sb-copy">Copy link</button>`;
}
function showSavedBanner(bannerId, meta) {
  const b = document.getElementById(bannerId);
  if (!meta) { b.hidden = true; b.innerHTML = ""; return; }
  b.innerHTML = savedBannerHtml(meta);
  b.hidden = false;
  const btn = b.querySelector(".sb-copy");
  btn.onclick = () => copyText(reportLink(meta.id), btn);
}
async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    if (btn) { const t = btn.textContent; btn.textContent = "Link copied ✓"; setTimeout(() => { btn.textContent = t; }, 1600); }
  } catch (_) {
    // Older browsers / blocked clipboard API: copy via a temporary text field
    const ta = Object.assign(document.createElement("textarea"), { value: text });
    ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch (_) {}
    ta.remove();
    if (btn) { const t = btn.textContent; btn.textContent = ok ? "Link copied ✓" : text; setTimeout(() => { btn.textContent = t; }, ok ? 1600 : 8000); }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  ANALYSIS TAB
// ══════════════════════════════════════════════════════════════════════════════
const anSave = makeSavePrompt("an-save");

// State: map of key → { html, filename, summary?, label? }
// Company reports are keyed by ticker; the comparison lives under COMPARE_KEY.
const COMPARE_KEY     = "__compare__";
const analysisResults = new Map();
let   activeResult    = null;

function showResultUI(show) {
  document.getElementById("analysis-input").hidden = show;
  document.getElementById("result-tabs").hidden    = !show;
  document.getElementById("report-area").hidden    = !show;
  if (!show && tgCtx === tgContexts.analysis) toggleTelegram(false);
}

function renderResultTabs() {
  const bar = document.getElementById("result-tab-bar");
  bar.innerHTML = "";
  // Comparison tab first, then companies in the order they were added
  const keys = [...analysisResults.keys()].sort((a, b) => (b === COMPARE_KEY) - (a === COMPARE_KEY));
  keys.forEach(key => {
    const v   = analysisResults.get(key);
    const btn = document.createElement("button");
    btn.className = "result-tab" + (key === COMPARE_KEY ? " compare" : "") + (key === activeResult ? " active" : "");
    btn.textContent = v.label || key;
    btn.onclick = () => { activeResult = key; renderResultTabs(); renderActiveResult(); };
    bar.appendChild(btn);
  });
  updateTelegramScope();
}

function renderActiveResult() {
  if (!activeResult || !analysisResults.has(activeResult)) return;
  const { html, saved } = analysisResults.get(activeResult);
  renderInFrame("report-frame", html);
  showSavedBanner("an-banner", saved || null);
  setReportParam(saved ? saved.id : null);
}

// offer = the server response (carries report_token / report_save); offerLabel names it in the save prompt
function addResult(ticker, html, filename, summary, { activate = true, offer = null, offerLabel = null } = {}) {
  analysisResults.set(ticker, { html, filename, summary });
  if (activate || !activeResult) activeResult = ticker;
  renderResultTabs();
  if (activate) renderActiveResult();
  showResultUI(true);
  if (offer) anSave.offer(offer, offerLabel || `${ticker} · financial analysis`);
}

// A saved financial report opens as one more result tab (it isn't offered for saving again)
function openSavedFinancial(meta, html) {
  const key = "saved:" + meta.id;
  analysisResults.set(key, {
    html, filename: meta.name + ".html", saved: meta,
    label: `📁 ${meta.ticker} ${meta.name.slice(meta.name.indexOf(meta.run_date))}`, caption: `📁 ${meta.name}`
  });
  activeResult = key;
  renderResultTabs();
  renderActiveResult();
  showResultUI(true);
  showTab("analysis");
}

// ── Peer comparison (built after a multi-file upload) ────────────────────────
function companySummaries() {
  return [...analysisResults.entries()]
    .filter(([k, v]) => k !== COMPARE_KEY && v.summary)
    .map(([, v]) => v.summary);
}

async function buildComparison() {
  const companies = companySummaries();
  if (companies.length < 2) return;
  setLoading("analysis-loading", "analysis-error", true, `Building comparison of ${companies.length} companies…`);
  try {
    const data = await postJson("/api/analyze/compare", { companies });
    analysisResults.set(COMPARE_KEY, {
      html: data.html, filename: data.filename, label: `📊 Compare (${companies.length})`,
      caption: `📊 Peer comparison — ${companies.map(c => c.ticker).join(", ")}`
    });
    activeResult = COMPARE_KEY;
    renderResultTabs();
    renderActiveResult();
  } catch (err) {
    showError("analysis-error", "Comparison failed: " + err.message);
  } finally {
    setLoading("analysis-loading", "analysis-error", false);
  }
}

// Download button
document.getElementById("download-btn").addEventListener("click", () => {
  if (!activeResult) return;
  const { html, filename } = analysisResults.get(activeResult);
  downloadHtml(html, filename);
});

// New analysis button
document.getElementById("new-analysis-btn").addEventListener("click", () => {
  analysisResults.clear();
  activeResult = null;
  anSave.clear();
  showSavedBanner("an-banner", null);
  setReportParam(null);
  document.getElementById("report-frame").srcdoc = "";
  showResultUI(false);
  document.getElementById("analysis-error").hidden = true;
  document.getElementById("batch-list").hidden = true;
});

// ── Send to Telegram ─────────────────────────────────────────────────────────
const tgPanel  = document.getElementById("tg-panel");
const tgToggle = document.getElementById("tg-toggle");
// The one Telegram panel is shared by the Analysis, Estimates and Backtest tabs:
// it moves next to whichever tab's button opened it, and asks that tab what to send.
const tgContexts = {};
let   tgCtx = null;
function registerTelegram(name, ctx) { tgContexts[name] = ctx; }
const tgLink   = document.getElementById("tg-link");
const tgSend   = document.getElementById("tg-send");
const tgStatus = document.getElementById("tg-status");
let   tgState  = { configured: null, bot_username: null, links: [] };

function tgSetStatus(msg, kind) {
  tgStatus.textContent = msg || "";
  tgStatus.className = "tg-status" + (kind ? " " + kind : "");
}

function toggleTelegram(open, name) {
  const ctx = name ? tgContexts[name] : tgCtx;
  if (!ctx) return;
  const switching = tgCtx && tgCtx !== ctx;
  const show = open ?? (switching || tgPanel.hidden);
  Object.values(tgContexts).forEach(c => {
    c.toggle.setAttribute("aria-expanded", "false");
    c.toggle.classList.remove("active");
  });
  if (show) {
    tgCtx = ctx;
    ctx.anchor().after(tgPanel);
    ctx.toggle.setAttribute("aria-expanded", "true");
    ctx.toggle.classList.add("active");
    updateTelegramScope();
  }
  tgPanel.hidden = !show;
  if (show) { loadTelegramStatus(); tgLink.focus(); }
}
tgToggle.addEventListener("click", () => toggleTelegram(undefined, "analysis"));

function renderSavedLinks() {
  const dl = document.getElementById("tg-link-list");
  const box = document.getElementById("tg-saved");
  dl.innerHTML = "";
  box.innerHTML = "";
  (tgState.links || []).forEach(({ link, title }) => {
    const opt = document.createElement("option");
    opt.value = link;
    if (title) opt.label = title;
    dl.appendChild(opt);

    const chip = document.createElement("span");
    chip.className = "tg-chip";
    const use = document.createElement("button");
    use.className = "tg-chip-use";
    use.textContent = title && title !== link ? `${title}` : link.replace(/^https:\/\//, "");
    use.title = `Use ${link}`;
    use.onclick = () => { tgLink.value = link; tgLink.focus(); };
    const del = document.createElement("button");
    del.className = "tg-chip-del";
    del.textContent = "×";
    del.title = `Forget ${link}`;
    del.setAttribute("aria-label", `Forget ${link}`);
    del.onclick = async () => {
      try {
        const res  = await fetch("/api/telegram/links", {
          method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ link })
        });
        const data = await readJson(res);
        tgState.links = data.links || [];
        renderSavedLinks();
      } catch (_) {}
    };
    chip.append(use, del);
    box.appendChild(chip);
  });
  if ((tgState.links || []).length) {
    const lbl = document.createElement("span");
    lbl.className = "tg-saved-label";
    lbl.textContent = "Recent:";
    box.prepend(lbl);
  }
}

async function loadTelegramStatus() {
  try {
    const res  = await fetch("/api/telegram/status");
    tgState = await readJson(res);
    renderSavedLinks();
    if (!tgLink.value && tgState.links && tgState.links.length) tgLink.value = tgState.links[0].link;
    if (!tgState.configured) {
      tgSetStatus("Telegram isn't set up yet: add TELEGRAM_BOT_TOKEN (the same token your Telegram bot uses) to the .env file in the finbot-web folder, then restart the app.", "warn");
    } else if (tgState.error) {
      tgSetStatus(tgState.error, "warn");
    } else if (!tgStatus.classList.contains("ok")) {
      tgSetStatus(tgState.bot_username
        ? `Posts as @${tgState.bot_username}. Channel: make the bot an admin with “Post messages”. Your private chat: enter your chat ID (send /chatid to the bot).`
        : "Channel: make the bot an admin with “Post messages”. Your private chat: enter your chat ID.");
    }
  } catch (_) {
    tgSetStatus("Couldn't check Telegram setup.", "warn");
  }
}

function updateTelegramScope() {
  const wrap = document.getElementById("tg-all-wrap");
  const n = analysisResults.size;
  wrap.hidden = n < 2 || !tgCtx || tgCtx !== tgContexts.analysis;
  document.getElementById("tg-all-label").textContent = `Send all ${n} open reports`;
}

function analysisReportsToSend() {
  const all = document.getElementById("tg-all").checked && analysisResults.size > 1;
  const keys = all ? [...analysisResults.keys()] : [activeResult];
  return keys.filter(k => analysisResults.has(k)).map(k => {
    const r = analysisResults.get(k);
    const caption = r.caption || `📊 ${k} — Financial Analysis`
      + (r.summary && r.summary.price ? ` · $${r.summary.price.toFixed(2)}` : "");
    return { html: r.html, filename: r.filename, caption };
  });
}

async function sendToTelegram() {
  const link = tgLink.value.trim();
  if (!link) { tgSetStatus("Enter your channel's public link, e.g. https://t.me/yourchannel", "warn"); tgLink.focus(); return; }
  if (!tgCtx || tgSend.disabled) return;
  const reports = tgCtx.reports();
  if (!reports.length) { tgSetStatus("Open a report first.", "warn"); return; }
  tgSend.disabled = true;
  tgSetStatus(`Sending ${reports.length > 1 ? reports.length + " reports" : "report"}…`);
  try {
    const res  = await fetch("/api/telegram/send", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ link, reports })
    });
    const data = await readJson(res);
    if (!res.ok || data.error) {
      const partial = data.sent && data.sent.length ? ` (sent ${data.sent.length} before the error)` : "";
      throw new Error((data.error || "Send failed") + partial);
    }
    tgState.links = data.links || tgState.links;
    renderSavedLinks();
    tgSetStatus(`✓ Sent ${data.sent.length > 1 ? data.sent.length + " reports" : data.sent[0]} to ${data.chat_title}.`, "ok");
  } catch (err) {
    tgSetStatus("⚠ " + err.message, "err");
  } finally {
    tgSend.disabled = false;
  }
}
async function testTelegram() {
  const btn = document.getElementById("tg-test");
  if (btn.disabled) return;
  btn.disabled = true;
  tgSetStatus("Checking with Telegram…");
  try {
    const res  = await fetch("/api/telegram/diagnose", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ link: tgLink.value.trim() })
    });
    const d = await readJson(res);
    tgStatus.className = "tg-status diag " + (d.ok ? "ok-box" : "err-box");
    tgStatus.innerHTML = "";
    const head = document.createElement("div");
    head.className = "diag-verdict";
    head.textContent = (d.ok ? "✓ " : "⚠ ") + (d.verdict || d.error || "Test failed");
    const ul = document.createElement("ul");
    (d.checks || []).forEach(c => {
      const li = document.createElement("li");
      li.className = c.ok === false ? "bad" : c.ok ? "good" : "";
      li.textContent = `${c.ok === false ? "✗" : c.ok ? "✓" : "•"} ${c.label}${c.detail ? " — " + c.detail : ""}`;
      ul.appendChild(li);
    });
    const meta = document.createElement("div");
    meta.className = "diag-meta";
    meta.textContent = `App folder: ${d.app_folder || "?"} · token from: ${d.token_source || "not found"}`
      + (d.chat_id_source ? ` · chat ID from: ${d.chat_id_source}` : "");
    tgStatus.append(head, ul, meta);
  } catch (err) {
    tgSetStatus("⚠ Couldn't run the test: " + err.message, "err");
  } finally {
    btn.disabled = false;
  }
}
document.getElementById("tg-test").addEventListener("click", testTelegram);
tgSend.addEventListener("click", sendToTelegram);
registerTelegram("analysis", {
  toggle: tgToggle, anchor: () => document.getElementById("result-tabs"), reports: analysisReportsToSend
});
tgLink.addEventListener("keydown", e => { if (e.key === "Enter") sendToTelegram(); });

// ── Ticker input ──────────────────────────────────────────────────────────────
async function runTickerAnalysis() {
  const input  = document.getElementById("ticker-input");
  const btn    = document.getElementById("ticker-btn");
  const ticker = input.value.trim().toUpperCase();
  if (!ticker || btn.disabled) return;
  btn.disabled = true;
  setLoading("analysis-loading", "analysis-error", true, `Fetching ${ticker}…`);
  try {
    const data = await postJson("/api/analyze/ticker", { ticker });
    addResult(data.ticker, data.html, data.filename, data.summary,
              { offer: data, offerLabel: `${data.ticker} · financial analysis (live data)` });
    input.value = "";
  } catch (err) {
    showError("analysis-error", err.message);
  } finally {
    setLoading("analysis-loading", "analysis-error", false);
    btn.disabled = false;
  }
}

document.getElementById("ticker-btn").addEventListener("click", runTickerAnalysis);
document.getElementById("ticker-input").addEventListener("keydown", e => {
  if (e.key === "Enter") runTickerAnalysis();
});

// ── CSV upload (single + batch) ───────────────────────────────────────────────
async function analyzeCSV(file) {
  const form = new FormData();
  form.append("file", file);
  const res  = await fetch("/api/analyze/csv", { method: "POST", body: form });
  const data = await readJson(res);
  if (!res.ok) throw new Error(data.error || "CSV analysis failed");
  return data;
}

async function handleFiles(fileList) {
  const files = Array.from(fileList);
  if (files.length === 0) return;
  document.getElementById("analysis-error").hidden = true;

  if (files.length === 1) {
    // Single file — simple loading state
    document.getElementById("batch-list").hidden = true;
    setLoading("analysis-loading", "analysis-error", true, `Analyzing ${files[0].name}…`);
    try {
      const data = await analyzeCSV(files[0]);
      addResult(data.ticker, data.html, data.filename, data.summary, { offer: data, offerLabel: files[0].name });
    } catch (err) {
      showError("analysis-error", err.message);
    } finally {
      setLoading("analysis-loading", "analysis-error", false);
    }
    return;
  }

  // Batch — show progress list (files are processed one at a time)
  const list = document.getElementById("batch-list");
  list.hidden = false;
  list.innerHTML = "";

  const items = files.map(file => {
    const li = document.createElement("li");
    li.className = "batch-item";
    li.innerHTML = `<span class="batch-status">⏳</span> <span class="batch-name">${esc(file.name)}</span> <span class="batch-msg"></span>`;
    list.appendChild(li);
    return li;
  });

  let failures = 0, successes = 0;
  for (let i = 0; i < files.length; i++) {
    const li = items[i];
    li.querySelector(".batch-status").textContent = "🔄";
    try {
      const data = await analyzeCSV(files[i]);
      // Show the first report straight away; later ones open as tabs
      addResult(data.ticker, data.html, data.filename, data.summary,
                { activate: successes === 0, offer: data, offerLabel: files[i].name });
      successes++;
      li.querySelector(".batch-status").textContent = "✅";
      li.classList.add("ok");
    } catch (err) {
      failures++;
      li.querySelector(".batch-status").textContent = "❌";
      li.querySelector(".batch-msg").textContent = err.message;
      li.classList.add("failed");
      li.title = err.message;
    }
  }
  if (failures === files.length) {
    showError("analysis-error", `All ${files.length} files failed — see the list above.`);
  }
  // Two or more companies → build the side-by-side comparison
  if (successes >= 1 && companySummaries().length >= 2) await buildComparison();
}

// File input
const csvInput = document.getElementById("csv-input");
csvInput.addEventListener("change", e => {
  handleFiles(e.target.files);
  csvInput.value = "";   // allow re-selecting the same file later
});

// Drag and drop
const dz = document.getElementById("dropzone");
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("drag-over"); });
dz.addEventListener("dragleave", ()  => dz.classList.remove("drag-over"));
dz.addEventListener("drop", e => {
  e.preventDefault();
  dz.classList.remove("drag-over");
  const csvFiles = Array.from(e.dataTransfer.files).filter(f => f.name.toLowerCase().endsWith(".csv"));
  if (csvFiles.length === 0) {
    showError("analysis-error", "Only .csv files can be analyzed.");
    return;
  }
  handleFiles(csvFiles);
});
// Keyboard access to the dropzone
dz.tabIndex = 0;
dz.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); csvInput.click(); }
});

// ══════════════════════════════════════════════════════════════════════════════
//  ESTIMATES + BACKTEST TABS (same shape: ticker in → report in iframe)
// ══════════════════════════════════════════════════════════════════════════════
function wireReportTab({ tab, inputId, btnId, loadingId, errorId, areaId, toolbarId, downloadId,
                         frameId, url, extraBody, savePanelId, bannerId, tgToggleId, caption, offerLabel }) {
  const input   = document.getElementById(inputId);
  const btn     = document.getElementById(btnId);
  const toolbar = document.getElementById(toolbarId);
  const prompt  = makeSavePrompt(savePanelId);
  let last = null;   // { html, filename, caption, saved? }

  function show(html) {
    renderInFrame(frameId, html);
    document.getElementById(areaId).hidden = false;
    toolbar.hidden = false;
  }

  async function run() {
    const ticker = input.value.trim().toUpperCase();
    if (!ticker || btn.disabled) return;
    btn.disabled = true;
    document.getElementById(loadingId).hidden = false;
    document.getElementById(errorId).hidden   = true;
    document.getElementById(areaId).hidden    = true;
    toolbar.hidden = true;
    if (tgCtx === tgContexts[tab]) toggleTelegram(false);
    prompt.clear();
    showSavedBanner(bannerId, null);
    try {
      const body = { ticker, ...(extraBody ? extraBody() : {}) };
      const data = await postJson(url, body);
      last = { html: data.html, filename: data.filename || `${data.ticker}_report.html`, caption: caption(data) };
      setReportParam(null);
      show(data.html);
      prompt.offer(data, offerLabel(data));
    } catch (err) {
      showError(errorId, err.message);
    } finally {
      document.getElementById(loadingId).hidden = true;
      btn.disabled = false;
    }
  }

  btn.addEventListener("click", run);
  input.addEventListener("keydown", e => { if (e.key === "Enter") run(); });
  document.getElementById(downloadId).addEventListener("click", () => {
    if (last) downloadHtml(last.html, last.filename);
  });
  const tgBtn = document.getElementById(tgToggleId);
  registerTelegram(tab, {
    toggle: tgBtn, anchor: () => toolbar,
    reports: () => last ? [{ html: last.html, filename: last.filename, caption: last.caption }] : []
  });
  tgBtn.addEventListener("click", () => toggleTelegram(undefined, tab));

  return {
    // Reopen a saved report here (never offered for saving again)
    openSaved(meta, html) {
      prompt.clear();
      document.getElementById(errorId).hidden = true;
      if (tgCtx === tgContexts[tab]) toggleTelegram(false);
      last = { html, filename: meta.name + ".html", caption: `📁 ${meta.name}`, saved: meta };
      show(html);
      showSavedBanner(bannerId, meta);
      setReportParam(meta.id);
      showTab(tab);
    }
  };
}

// "atr3" → "3× ATR", from the Stop menu's own labels
function stopText(key) {
  const opt = [...document.getElementById("bt-stop").options].find(o => o.value === key);
  return opt ? opt.text.replace(/^Stop:\s*/, "") : (key || "none");
}

const estimatesTab = wireReportTab({
  tab: "estimates",
  inputId: "est-ticker", btnId: "est-btn", loadingId: "est-loading",
  errorId: "est-error", areaId: "est-area", toolbarId: "est-toolbar", downloadId: "est-download",
  frameId: "est-frame", url: "/api/analyze/estimates",
  savePanelId: "est-save", bannerId: "est-banner", tgToggleId: "est-tg",
  caption: d => `📈 ${d.ticker} — Estimates`,
  offerLabel: d => `${d.ticker} · estimates`
});

const backtestTab = wireReportTab({
  tab: "backtest",
  inputId: "bt-ticker", btnId: "bt-btn", loadingId: "bt-loading",
  errorId: "bt-error", areaId: "bt-area", toolbarId: "bt-toolbar", downloadId: "bt-download",
  frameId: "bt-frame", url: "/api/backtest",
  savePanelId: "bt-save", bannerId: "bt-banner", tgToggleId: "bt-tg",
  extraBody: () => ({ timeframe: document.getElementById("bt-timeframe").value,
                      stop: document.getElementById("bt-stop").value }),
  caption: d => `🧪 ${d.ticker} — Backtest (${d.timeframe} MACD refs · stop: ${stopText(d.stop)})`,
  offerLabel: d => `${d.ticker} · backtest (${d.timeframe} · stop: ${stopText(d.stop)})`
});

// ══════════════════════════════════════════════════════════════════════════════
//  SIGNALS TAB
// ══════════════════════════════════════════════════════════════════════════════
let lastSignalTs = 0;

async function loadSignals() {
  const status = document.getElementById("feed-status");
  try {
    const res  = await fetch(`/api/signals?since=${lastSignalTs}`);
    const data = await readJson(res);
    if (!res.ok) throw new Error(data.error || "Request failed");
    renderWatchedList(data.subscribed || [], data.watchers || {});
    const newSignals = data.signals || [];
    if (newSignals.length) {
      renderSignals(newSignals);
      lastSignalTs = Math.max(lastSignalTs, ...newSignals.map(s => Number(s.timestamp) || 0));
    }
    status.textContent = "· updated " + new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    status.classList.remove("offline");
  } catch (_) {
    status.textContent = "· offline";
    status.classList.add("offline");
  }
}

function renderWatchedList(tickers, watchers) {
  const ul = document.getElementById("watched-list");
  ul.innerHTML = "";
  if (!tickers.length) {
    const li = document.createElement("li");
    li.className = "watched-empty";
    li.textContent = "Not watching any tickers.";
    ul.appendChild(li);
    return;
  }
  tickers.forEach(ticker => {
    const li   = document.createElement("li");
    const name = document.createElement("span");
    const btn  = document.createElement("button");
    const w    = watchers[ticker];
    name.className = "watched-name";
    name.textContent = ticker;
    if (w) {
      const meta = document.createElement("small");
      const strat = w.strategy === "crossover" ? "Crossover + MA50" : "Histogram";
      meta.textContent = `${strat} · ${w.params_label}`;
      meta.title = `Backtest (weekly, $10k/trade): ${w.backtest_pnl >= 0 ? "+" : "−"}$${Math.abs(w.backtest_pnl).toLocaleString()} · ${w.win_rate}% win · ${w.trades} trades`;
      name.appendChild(meta);
    }
    btn.textContent = "×";
    btn.className = "unsub-btn";
    btn.title = `Stop monitoring ${ticker}`;
    btn.setAttribute("aria-label", `Stop monitoring ${ticker}`);
    btn.onclick = () => unsubscribe(ticker);
    li.append(name, btn);
    ul.appendChild(li);
  });
}

// `signals` arrives newest-first; prepend oldest-first so the newest ends on top.
function renderSignals(signals) {
  const log = document.getElementById("signal-log");
  document.querySelector(".signal-empty")?.remove();
  [...signals].reverse().forEach(s => {
    const li    = document.createElement("li");
    const isBuy = s.type === "BUY";
    li.className = "signal-item " + (isBuy ? "buy" : "sell");
    const d     = new Date((Number(s.timestamp) || 0) * 1000);
    const price = Number.isFinite(Number(s.price)) && s.price !== null ? "$" + Number(s.price).toFixed(2) : "—";
    li.innerHTML =
      `<span class="sig-type">${esc(s.type)}</span> <strong>${esc(s.ticker)}</strong> @ ${esc(price)} ` +
      `<span class="sig-time">${esc(d.toLocaleString())}</span><br><small>${esc(s.message || "")}</small>`;
    log.prepend(li);
  });
}

async function subscribe() {
  const input  = document.getElementById("sig-ticker");
  const btn    = document.getElementById("sig-btn");
  const ticker = input.value.trim().toUpperCase();
  if (!ticker || btn.disabled) return;
  btn.disabled = true;
  document.getElementById("sig-error").hidden = true;
  document.getElementById("sig-loading-text").textContent = `Backtesting ${ticker} to pick a strategy…`;
  document.getElementById("sig-loading").hidden = false;
  try {
    await postJson("/api/signals/subscribe", { ticker });
    if (input.value.trim().toUpperCase() === ticker) input.value = "";
  } catch (err) {
    showError("sig-error", err.message);
  } finally {
    document.getElementById("sig-loading").hidden = true;
    btn.disabled = false;
  }
  loadSignals();
}

async function unsubscribe(ticker) {
  try {
    await fetch("/api/signals/subscribe", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker })
    });
  } catch (_) {}
  loadSignals();
}

document.getElementById("sig-btn").addEventListener("click", subscribe);
document.getElementById("sig-ticker").addEventListener("keydown", e => {
  if (e.key === "Enter") subscribe();
});

// Poll signals every 60 seconds
loadSignals();
setInterval(loadSignals, 60_000);

// ══════════════════════════════════════════════════════════════════════════════
//  SAVED REPORTS TAB (shared library)
// ══════════════════════════════════════════════════════════════════════════════
const ADMIN_KEY = "finbot_reports_admin";
const lib = {
  list: document.getElementById("lib-list"), more: document.getElementById("lib-more"),
  type: document.getElementById("lib-type"), ticker: document.getElementById("lib-ticker"),
  q: document.getElementById("lib-q"), count: document.getElementById("lib-count"),
  empty: document.getElementById("lib-empty"), loading: document.getElementById("lib-loading"),
  offset: 0, total: 0, seq: 0, pageSize: 25,
};
function adminToken() {
  try { return sessionStorage.getItem(ADMIN_KEY) || ""; } catch (_) { return ""; }
}

function libRow(m) {
  const li = document.createElement("li");
  li.className = "lib-item";
  const p = m.params || {};
  const extra = m.type === "backtest" && p.timeframe ? ` · ${esc(p.timeframe)} · stop: ${esc(p.stop_label || p.stop || "none")}` : "";
  const src = m.source && m.source.kind === "csv" ? " · from CSV" : "";
  const open = document.createElement("button");
  open.type = "button";
  open.className = "lib-open";
  open.innerHTML = `<span class="lib-name">${esc(m.name)}</span>
    <span class="lib-meta"><span class="badge badge-${esc(m.type)}">${esc(TYPE_LABELS[m.type] || m.type)}</span>
    <span>${esc(m.ticker)} · run ${esc(fmtRunTime(m.created_at))}${extra}${src}</span></span>`;
  open.onclick = () => openSavedReport(m.id);

  const acts = document.createElement("div");
  acts.className = "lib-actions";
  const dl = Object.assign(document.createElement("a"), {
    className: "btn-ghost sm", href: `/api/reports/${encodeURIComponent(m.id)}/download`, textContent: "⬇", title: "Download"
  });
  dl.setAttribute("aria-label", `Download ${m.name}`);
  const cp = Object.assign(document.createElement("button"), { type: "button", className: "btn-ghost sm", textContent: "🔗", title: "Copy link" });
  cp.setAttribute("aria-label", `Copy link to ${m.name}`);
  cp.onclick = () => copyText(reportLink(m.id), null).then(() => { cp.textContent = "✓"; setTimeout(() => { cp.textContent = "🔗"; }, 1500); });
  acts.append(dl, cp);
  if (libState.deleteEnabled && adminToken()) {
    const del = Object.assign(document.createElement("button"), { type: "button", className: "btn-ghost sm danger", textContent: "Delete" });
    del.setAttribute("aria-label", `Delete ${m.name}`);
    let armed = null;
    del.onclick = async () => {
      if (!armed) {                                   // two-step: click again within 4 s to confirm
        del.textContent = "Confirm delete";
        del.classList.add("armed");
        armed = setTimeout(() => { armed = null; del.textContent = "Delete"; del.classList.remove("armed"); }, 4000);
        return;
      }
      clearTimeout(armed);
      del.disabled = true;
      try {
        const res = await fetch(`/api/reports/${encodeURIComponent(m.id)}`, { method: "DELETE", headers: { "X-Admin-Token": adminToken() } });
        const d = await readJson(res);
        if (!res.ok) throw new Error(d.error || "Delete failed");
        li.remove();
        lib.total = Math.max(0, lib.total - 1);
        lib.offset = Math.max(0, lib.offset - 1);
        renderLibCount();
      } catch (err) {
        showError("lib-error", err.message);
        del.disabled = false; del.textContent = "Delete"; del.classList.remove("armed"); armed = null;
      }
    };
    acts.appendChild(del);
  }
  li.append(open, acts);
  return li;
}

function renderLibCount() {
  lib.count.textContent = lib.total ? `(${lib.total})` : "";
  lib.more.hidden = lib.offset >= lib.total;
  const filtered = lib.type.value || lib.ticker.value.trim() || lib.q.value.trim();
  lib.empty.hidden = lib.total > 0;
  lib.empty.textContent = filtered
    ? "No saved reports match these filters."
    : "No saved reports yet. Run a Financial Analysis, Estimates or Backtest report and save it — it will show up here.";
}

async function loadLibrary({ append = false } = {}) {
  if (libState.available === null) await loadLibraryStatus();
  document.getElementById("lib-unavailable").hidden = libState.available !== false;
  document.getElementById("lib-admin").hidden = !libState.deleteEnabled || !libState.available;
  if (!libState.available) { lib.list.innerHTML = ""; lib.empty.hidden = true; lib.more.hidden = true; return; }
  const seq = ++lib.seq;
  if (!append) lib.offset = 0;
  const qs = new URLSearchParams({ limit: lib.pageSize, offset: lib.offset });
  if (lib.type.value) qs.set("type", lib.type.value);
  if (lib.ticker.value.trim()) qs.set("ticker", lib.ticker.value.trim().toUpperCase());
  if (lib.q.value.trim()) qs.set("q", lib.q.value.trim());
  lib.loading.hidden = false;
  document.getElementById("lib-error").hidden = true;
  try {
    const res = await fetch("/api/reports?" + qs);
    const d = await readJson(res);
    if (seq !== lib.seq) return;                       // a newer search superseded this one
    if (!res.ok) throw new Error(d.error || "Couldn't load saved reports");
    if (!append) lib.list.innerHTML = "";
    d.reports.forEach(m => lib.list.appendChild(libRow(m)));
    lib.offset += d.reports.length;
    lib.total = d.total;
    libState.stale = false;
    renderLibCount();
  } catch (err) {
    if (seq === lib.seq) showError("lib-error", err.message);
  } finally {
    if (seq === lib.seq) lib.loading.hidden = true;
  }
}

let libTimer = null;
function reloadLibrarySoon() { clearTimeout(libTimer); libTimer = setTimeout(() => loadLibrary(), 250); }
lib.type.addEventListener("change", () => loadLibrary());
lib.ticker.addEventListener("input", reloadLibrarySoon);
lib.q.addEventListener("input", reloadLibrarySoon);
lib.more.addEventListener("click", () => loadLibrary({ append: true }));
document.getElementById("lib-refresh").addEventListener("click", () => loadLibrary());
tabHooks.library = () => loadLibrary();

// "Past reports" buttons → library pre-filtered to that tab's type (+ typed ticker)
function openLibrary({ type = "", ticker = "" }) {
  lib.type.value = type;
  lib.ticker.value = ticker;
  lib.q.value = "";
  showTab("library");
}
document.querySelectorAll(".past-btn").forEach(b => {
  b.addEventListener("click", () => {
    const inp = document.getElementById(b.dataset.tickerInput);
    openLibrary({ type: b.dataset.past, ticker: inp ? inp.value.trim().toUpperCase() : "" });
  });
});

async function openSavedReport(id) {
  document.getElementById("lib-error").hidden = true;
  try {
    const res = await fetch(`/api/reports/${encodeURIComponent(id)}`);
    const d = await readJson(res);
    if (!res.ok) throw new Error(d.error || "Couldn't open that report");
    const { html, ...meta } = d;
    if (meta.type === "financial") openSavedFinancial(meta, html);
    else if (meta.type === "estimates") estimatesTab.openSaved(meta, html);
    else if (meta.type === "backtest") backtestTab.openSaved(meta, html);
    return true;
  } catch (err) {
    showTab("library");
    showError("lib-error", err.message);
    return false;
  }
}

// Admin unlock (token kept in sessionStorage only; the server checks it on every delete)
document.getElementById("lib-admin-btn").addEventListener("click", async () => {
  const inp = document.getElementById("lib-admin-token");
  const st = document.getElementById("lib-admin-status");
  const tok = inp.value.trim();
  if (!tok) { st.textContent = "Enter the admin token."; return; }
  try {
    const res = await fetch("/api/reports/admin", { method: "POST", headers: { "X-Admin-Token": tok } });
    const d = await readJson(res);
    if (!res.ok) throw new Error(d.error || "Not accepted");
    try { sessionStorage.setItem(ADMIN_KEY, tok); } catch (_) {}
    inp.value = "";
    st.textContent = "✓ Delete buttons unlocked for this browser tab.";
    loadLibrary();
  } catch (err) {
    st.textContent = "⚠ " + err.message;
  }
});

// ── Startup: library status, then a ?report=<id> deep link ───────────────────
(async () => {
  await loadLibraryStatus();
  const id = new URLSearchParams(location.search).get("report");
  if (id) await openSavedReport(id);
})();
