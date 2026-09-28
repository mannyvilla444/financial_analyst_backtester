// static/app.js

// ── Tab switching ──────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

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
//  ANALYSIS TAB
// ══════════════════════════════════════════════════════════════════════════════

// State: map of key → { html, filename, summary?, label? }
// Company reports are keyed by ticker; the comparison lives under COMPARE_KEY.
const COMPARE_KEY     = "__compare__";
const analysisResults = new Map();
let   activeResult    = null;

function showResultUI(show) {
  document.getElementById("analysis-input").hidden = show;
  document.getElementById("result-tabs").hidden    = !show;
  document.getElementById("report-area").hidden    = !show;
  if (!show) toggleTelegram(false);
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
  const { html } = analysisResults.get(activeResult);
  renderInFrame("report-frame", html);
}

function addResult(ticker, html, filename, summary, { activate = true } = {}) {
  analysisResults.set(ticker, { html, filename, summary });
  if (activate || !activeResult) activeResult = ticker;
  renderResultTabs();
  if (activate) renderActiveResult();
  showResultUI(true);
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
  document.getElementById("report-frame").srcdoc = "";
  showResultUI(false);
  document.getElementById("analysis-error").hidden = true;
  document.getElementById("batch-list").hidden = true;
});

// ── Send to Telegram ─────────────────────────────────────────────────────────
const tgPanel  = document.getElementById("tg-panel");
const tgToggle = document.getElementById("tg-toggle");
const tgLink   = document.getElementById("tg-link");
const tgSend   = document.getElementById("tg-send");
const tgStatus = document.getElementById("tg-status");
let   tgState  = { configured: null, bot_username: null, links: [] };

function tgSetStatus(msg, kind) {
  tgStatus.textContent = msg || "";
  tgStatus.className = "tg-status" + (kind ? " " + kind : "");
}

function toggleTelegram(open) {
  const show = open ?? tgPanel.hidden;
  tgPanel.hidden = !show;
  tgToggle.setAttribute("aria-expanded", String(show));
  tgToggle.classList.toggle("active", show);
  if (show) { loadTelegramStatus(); tgLink.focus(); }
}
tgToggle.addEventListener("click", () => toggleTelegram());

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
  wrap.hidden = n < 2;
  document.getElementById("tg-all-label").textContent = `Send all ${n} open reports`;
}

function reportsToSend() {
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
  if (!activeResult || tgSend.disabled) return;
  const reports = reportsToSend();
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
    addResult(data.ticker, data.html, data.filename, data.summary);
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
      addResult(data.ticker, data.html, data.filename, data.summary);
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
      addResult(data.ticker, data.html, data.filename, data.summary, { activate: successes === 0 });
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
function wireReportTab({ inputId, btnId, loadingId, errorId, areaId, toolbarId, downloadId,
                         frameId, url, extraBody }) {
  const input   = document.getElementById(inputId);
  const btn     = document.getElementById(btnId);
  const toolbar = document.getElementById(toolbarId);
  let last = null;   // { html, filename }

  async function run() {
    const ticker = input.value.trim().toUpperCase();
    if (!ticker || btn.disabled) return;
    btn.disabled = true;
    document.getElementById(loadingId).hidden = false;
    document.getElementById(errorId).hidden   = true;
    document.getElementById(areaId).hidden    = true;
    toolbar.hidden = true;
    try {
      const data = await postJson(url, { ticker, ...(extraBody ? extraBody() : {}) });
      last = { html: data.html, filename: data.filename || `${data.ticker}_report.html` };
      renderInFrame(frameId, data.html);
      document.getElementById(areaId).hidden = false;
      toolbar.hidden = false;
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
}

wireReportTab({
  inputId: "est-ticker", btnId: "est-btn", loadingId: "est-loading",
  errorId: "est-error", areaId: "est-area", toolbarId: "est-toolbar", downloadId: "est-download",
  frameId: "est-frame", url: "/api/analyze/estimates"
});

wireReportTab({
  inputId: "bt-ticker", btnId: "bt-btn", loadingId: "bt-loading",
  errorId: "bt-error", areaId: "bt-area", toolbarId: "bt-toolbar", downloadId: "bt-download",
  frameId: "bt-frame", url: "/api/backtest",
  extraBody: () => ({ timeframe: document.getElementById("bt-timeframe").value })
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
