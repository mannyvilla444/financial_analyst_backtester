"""
compare_report.py

Peer-comparison HTML report built from the per-company summaries that
pipeline.run_analysis() returns (one per uploaded CSV / typed ticker).

Compares: current market cap, revenue TTM, revenue growth YoY, EBITDA margin,
FCF margin, net debt / EBITDA, the three current valuation multiples
(Mkt Cap / Revenue, / EBITDA, / FCF — the same ones as the single-company
report) and an estimated market share.

MARKET SHARE
────────────
There's no industry-size data in a TradingView export, so by default share is
each company's slice of the *combined TTM revenue of the companies compared*
(a peer-group share — only meaningful when the uploads are from the same
space). The report has a "total market size" box: enter the market's annual
revenue in $B and shares are recomputed against it, with the remainder shown
as "Rest of market".
"""

from datetime import datetime

from web_reports import _page, _header, _e, _json

FIELDS = (
    "ticker", "price", "price_date", "financials_date", "mkt_cap", "revenue_ttm",
    "revenue_growth", "gross_margin", "ebitda_margin", "fcf_margin", "fcf_ttm",
    "net_debt_ebitda", "pe", "ebitda_ttm", "mult_revenue", "mult_ebitda", "mult_fcf",
    "median_mult_revenue", "median_mult_ebitda", "median_mult_fcf",
)


def _clean(company: dict) -> dict:
    out = {}
    for k in FIELDS:
        v = company.get(k)
        if k in ("ticker", "price_date", "financials_date"):
            out[k] = str(v)[:20] if v is not None else ""
        else:
            try:
                f = float(v)
                out[k] = f if f == f and abs(f) != float("inf") else None
            except (TypeError, ValueError):
                out[k] = None
    return out


def generate_comparison_html(companies: list[dict], market_size_b: float | None = None) -> str:
    """companies: list of summary dicts from pipeline.build_summary()."""
    rows = [_clean(c) for c in companies if c and c.get("ticker")]
    # one row per ticker (last one wins, matching the UI's result tabs)
    rows = list({r["ticker"]: r for r in rows}.values())
    if len(rows) < 2:
        raise ValueError("Need at least two companies to compare.")

    tickers = " · ".join(r["ticker"] for r in rows)
    price_dates = sorted({r["price_date"] for r in rows if r["price_date"]})
    price_note = (f"Prices as of {datetime.strptime(price_dates[-1], '%Y-%m-%d'):%b %d, %Y}"
                  if price_dates else "")
    subtitle = (f"{len(rows)} companies &nbsp;·&nbsp; {_e(tickers)}"
                + (f" &nbsp;·&nbsp; {price_note}" if price_note else "")
                + f" &nbsp;·&nbsp; Generated {datetime.now():%B %d, %Y}")

    data = {"companies": rows, "marketSizeB": market_size_b}
    body = (_EXTRA_CSS
            + f"<script type='application/json' id='cmp-data'>{_json(data)}</script>"
            + _BODY + _SCRIPT)
    return _page("Peer Comparison", _header("Peer Comparison", subtitle), body,
                 "⚠️ Figures from the uploaded TradingView exports, with current prices from the latest "
                 "close. Market share is estimated from TTM revenue — by default as a share of the "
                 "compared group's combined revenue, not the whole industry. Not financial advice.")


_EXTRA_CSS = """
<style>
  :root { --bar-pos: #2a78d6; --bar-neg: #e34948; --bar-rest: #b9bcc4; --best-bg: rgba(42,120,214,0.10); }
  @media (prefers-color-scheme: dark) {
    :root { --bar-pos: #3987e5; --bar-neg: #e66767; --bar-rest: #4a4f5c; --best-bg: rgba(57,135,229,0.16); }
  }
  .leaders { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
  @media (min-width: 720px) { .leaders { grid-template-columns: repeat(4, 1fr); } }
  .leader-tk { font-size: 20px; font-weight: 700; }
  .cmp-table th { cursor: pointer; user-select: none; }
  .cmp-table th .arrow { opacity: .5; font-size: 10px; margin-left: 3px; }
  .cmp-table td { vertical-align: top; }
  .cmp-table td.best { background: var(--best-bg) !important; font-weight: 700; }
  .cmp-table td.best::after { content: " ★"; font-size: 11px; }
  .cmp-table .sub { display: block; font-size: 11px; color: var(--muted); font-weight: 400; }
  .cmp-table td.tk { font-weight: 700; }
  .grid2 { display: grid; grid-template-columns: 1fr; gap: 12px; }
  @media (min-width: 760px) { .grid2 { grid-template-columns: 1fr 1fr; } }
  .grid2 .card { margin-bottom: 0; }
  .chart-box { position: relative; width: 100%; }
  .chart-box canvas { max-height: none; }
  .nm-note { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .tam { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 4px 0 14px; font-size: 13px; }
  .tam input {
    width: 140px; height: 34px; padding: 0 10px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--bg); color: var(--text); font: inherit;
  }
  .tam button {
    height: 34px; padding: 0 12px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--card); color: var(--text); font: inherit; cursor: pointer;
  }
  .tam button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  .share-note { font-size: 12px; color: var(--muted); margin-top: 8px; }
  .share-note.warn { color: var(--red); }
</style>
"""

_BODY = """
<div class='section-heading'>Leaders</div>
<div class='leaders' id='leaders'></div>

<div class='section-heading'>Estimated Market Share</div>
<div class='card'>
  <div class='card-title'>Share of revenue (TTM)</div>
  <div class='card-desc' id='share-desc'></div>
  <div class='tam'>
    <label for='tam-input'>Total market size (annual revenue, $B):</label>
    <input id='tam-input' type='number' min='0' step='any' inputmode='decimal' placeholder='optional'>
    <button class='primary' id='tam-apply'>Apply</button>
    <button id='tam-reset'>Use peer group</button>
  </div>
  <div class='chart-box' id='share-box'><canvas id='chartShare' role='img' aria-label='Estimated market share'></canvas></div>
  <div class='share-note' id='share-note'></div>
</div>

<div class='section-heading'>Side-by-side</div>
<div class='card'>
  <div class='card-desc'>Click a column to sort. ★ = best in the group for that column
    (higher growth, margins and share; lower leverage and multiples).
    Multiples use today's market cap; “n/m” = not meaningful (TTM EBITDA or FCF ≤ 0).
    Small print under a multiple is that company's own historical median.</div>
  <div class='table-wrap'><table class='data-table cmp-table' id='cmp-table'></table></div>
</div>

<div class='section-heading'>Growth vs Valuation</div>
<div class='card'>
  <div class='card-title'>Revenue growth vs Mkt Cap / Revenue</div>
  <div class='card-desc'>Up-and-left = paying more for less growth; down-and-right = more growth for a lower multiple.</div>
  <div class='chart-box' style='height:340px'><canvas id='chartScatter' role='img' aria-label='Revenue growth versus revenue multiple'></canvas></div>
</div>

<div class='section-heading'>Metric by Metric</div>
<div class='grid2' id='metric-grid'></div>
"""

_SCRIPT = r"""
<script>
(function () {
  var D = JSON.parse(document.getElementById('cmp-data').textContent);
  var C = D.companies;
  var T = window.__fbTheme || { grid: 'rgba(0,0,0,0.06)', tick: '#555' };
  var css = getComputedStyle(document.documentElement);
  var POS = css.getPropertyValue('--bar-pos').trim() || '#2a78d6';
  var NEG = css.getPropertyValue('--bar-neg').trim() || '#e34948';
  var REST = css.getPropertyValue('--bar-rest').trim() || '#b9bcc4';
  var hasChart = typeof Chart !== 'undefined';

  // ── formatting ────────────────────────────────────────────────────────────
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function big(v) {
    if (!isNum(v)) return 'N/A';
    var a = Math.abs(v), s = v < 0 ? '−' : '';
    if (a >= 1e12) return s + '$' + (a / 1e12).toFixed(2) + 'T';
    if (a >= 1e9)  return s + '$' + (a / 1e9).toFixed(2) + 'B';
    if (a >= 1e6)  return s + '$' + (a / 1e6).toFixed(1) + 'M';
    return s + '$' + a.toLocaleString();
  }
  function pct(v)  { return isNum(v) ? v.toFixed(1) + '%' : 'N/A'; }
  function spct(v) { return isNum(v) ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) + '%' : 'N/A'; }
  function mult(v) { return isNum(v) ? v.toFixed(1) + 'x' : 'n/m'; }
  function lev(v)  { return isNum(v) ? v.toFixed(2) + 'x' : 'N/A'; }
  function esc(s)  { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }

  // ── market share ──────────────────────────────────────────────────────────
  var marketSizeB = isNum(D.marketSizeB) && D.marketSizeB > 0 ? D.marketSizeB : null;
  function computeShares() {
    var total = 0;
    C.forEach(function (c) { if (isNum(c.revenue_ttm) && c.revenue_ttm > 0) total += c.revenue_ttm; });
    var base = marketSizeB ? marketSizeB * 1e9 : total;
    C.forEach(function (c) {
      c.share = (isNum(c.revenue_ttm) && c.revenue_ttm > 0 && base > 0) ? c.revenue_ttm / base * 100 : null;
    });
    return { total: total, base: base };
  }

  // ── columns ───────────────────────────────────────────────────────────────
  var COLS = [
    { key: 'mkt_cap',         label: 'Mkt Cap',            fmt: big },
    { key: 'revenue_ttm',     label: 'Revenue TTM',        fmt: big },
    { key: 'revenue_growth',  label: 'Rev Growth YoY',     fmt: spct, best: 'high', chart: true, unit: '%' },
    { key: 'ebitda_margin',   label: 'EBITDA Margin',      fmt: pct,  best: 'high', chart: true, unit: '%' },
    { key: 'fcf_margin',      label: 'FCF Margin',         fmt: pct,  best: 'high', chart: true, unit: '%' },
    { key: 'net_debt_ebitda', label: 'Net Debt / EBITDA',  fmt: lev,  best: 'low',  chart: true, unit: 'x',
      note: 'Below 0 = net cash', noSignColor: true },
    { key: 'mult_revenue',    label: 'Mkt Cap / Revenue',  fmt: mult, best: 'low',  chart: true, unit: 'x', median: 'median_mult_revenue' },
    { key: 'mult_ebitda',     label: 'Mkt Cap / EBITDA',   fmt: mult, best: 'low',  chart: true, unit: 'x', median: 'median_mult_ebitda',
      nm: 'negative or zero TTM EBITDA' },
    { key: 'mult_fcf',        label: 'Mkt Cap / FCF',      fmt: mult, best: 'low',  chart: true, unit: 'x', median: 'median_mult_fcf',
      nm: 'negative or zero TTM FCF' },
    { key: 'share',           label: 'Est. Share',         fmt: pct,  best: 'high' }
  ];

  // Tickers holding the best value in a column (all of them on a tie; none if
  // every company has the same value, since then nobody is "best").
  function bestTickers(col) {
    var vals = C.filter(function (c) { return isNum(c[col.key]); }).map(function (c) { return c[col.key]; });
    if (vals.length < 2 || !col.best) return [];
    var top = col.best === 'high' ? Math.max.apply(null, vals) : Math.min.apply(null, vals);
    var same = function (v) { return Math.abs(v - top) <= 1e-9 * Math.max(1, Math.abs(top)); };
    if (vals.every(same)) return [];
    return C.filter(function (c) { return isNum(c[col.key]) && same(c[col.key]); }).map(function (c) { return c.ticker; });
  }

  // ── leaders row ───────────────────────────────────────────────────────────
  function leader(label, key, dir, fmt) {
    var vals = C.filter(function (c) { return isNum(c[key]); });
    if (!vals.length) return '';
    vals.sort(function (a, b) { return dir === 'high' ? b[key] - a[key] : a[key] - b[key]; });
    var w = vals[0];
    var tied = vals.filter(function (c) { return c[key] === w[key]; }).map(function (c) { return c.ticker; });
    return "<div class='metric-card'><div class='metric-label'>" + esc(label) + "</div>" +
           "<div class='leader-tk'>" + esc(tied.length > 1 ? tied.join(' / ') : w.ticker) + "</div>" +
           "<div class='metric-sub'>" + esc(fmt(w[key])) + "</div></div>";
  }
  document.getElementById('leaders').innerHTML =
    leader('Largest (Mkt Cap)', 'mkt_cap', 'high', big) +
    leader('Fastest growth', 'revenue_growth', 'high', spct) +
    leader('Best FCF margin', 'fcf_margin', 'high', pct) +
    leader('Cheapest on revenue', 'mult_revenue', 'low', function (v) { return mult(v) + ' Mkt Cap / Revenue'; });

  // ── table ─────────────────────────────────────────────────────────────────
  var sortKey = 'mkt_cap', sortDir = -1;
  function renderTable() {
    var bests = {};
    COLS.forEach(function (col) { bests[col.key] = bestTickers(col); });
    var rows = C.slice().sort(function (a, b) {
      if (sortKey === 'ticker') return sortDir * a.ticker.localeCompare(b.ticker);
      var x = a[sortKey], y = b[sortKey];
      if (!isNum(x) && !isNum(y)) return 0;
      if (!isNum(x)) return 1;
      if (!isNum(y)) return -1;
      return sortDir * (x - y);
    });
    function th(key, label) {
      var arrow = key === sortKey ? (sortDir > 0 ? '▲' : '▼') : '↕';
      return "<th data-key='" + key + "'>" + esc(label) + "<span class='arrow'>" + arrow + "</span></th>";
    }
    var h = '<thead><tr>' + th('ticker', 'Company') + th('price', 'Price');
    COLS.forEach(function (col) { h += th(col.key, col.label); });
    h += '</tr></thead><tbody>';
    rows.forEach(function (c) {
      h += "<tr><td class='tk'>" + esc(c.ticker) + "<span class='sub'>fin. " + esc(c.financials_date) + "</span></td>" +
           "<td>" + (isNum(c.price) ? '$' + c.price.toFixed(2) : 'N/A') + "</td>";
      COLS.forEach(function (col) {
        var cls = bests[col.key].indexOf(c.ticker) >= 0 ? " class='best' title='Best in group'" : '';
        var sub = col.median && isNum(c[col.median]) ? "<span class='sub'>med " + mult(c[col.median]) + "</span>" : '';
        h += '<td' + cls + '>' + esc(col.fmt(c[col.key])) + sub + '</td>';
      });
      h += '</tr>';
    });
    h += '</tbody>';
    var t = document.getElementById('cmp-table');
    t.innerHTML = h;
    t.querySelectorAll('th').forEach(function (el) {
      el.addEventListener('click', function () {
        var k = el.getAttribute('data-key');
        if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = k === 'ticker' ? 1 : -1; }
        renderTable();
      });
    });
  }

  // ── charts ────────────────────────────────────────────────────────────────
  function hBar(canvas, labels, values, colors, unitFmt) {
    return new Chart(canvas, {
      type: 'bar',
      data: { labels: labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4,
              borderSkipped: false, barPercentage: 0.72, categoryPercentage: 0.9, maxBarThickness: 22 }] },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false },
                   tooltip: { callbacks: { label: function (ctx) { return unitFmt(ctx.parsed.x); } } } },
        scales: {
          x: { grid: { color: T.grid }, border: { display: false },
               ticks: { color: T.tick, font: { size: 11 }, callback: function (v) { return unitFmt(v, true); } } },
          y: { grid: { display: false }, border: { display: false }, ticks: { color: T.tick, font: { size: 12, weight: '600' } } }
        }
      }
    });
  }
  function boxHeight(n) { return Math.max(120, 34 + n * 28) + 'px'; }

  var shareChart = null;
  function renderShare() {
    var s = computeShares();
    var rows = C.filter(function (c) { return isNum(c.share); })
                .sort(function (a, b) { return b.share - a.share; });
    var labels = rows.map(function (c) { return c.ticker; });
    var vals = rows.map(function (c) { return +c.share.toFixed(2); });
    var colors = rows.map(function () { return POS; });
    var desc = document.getElementById('share-desc'), note = document.getElementById('share-note');
    note.className = 'share-note';
    if (marketSizeB) {
      var rest = 100 - vals.reduce(function (a, b) { return a + b; }, 0);
      if (rest > 0) { labels.push('Rest of market'); vals.push(+rest.toFixed(2)); colors.push(REST); }
      desc.textContent = 'Each company\'s TTM revenue as a share of a $' + marketSizeB.toLocaleString() +
                         'B total market.';
      if (s.total > s.base) {
        note.className = 'share-note warn';
        note.textContent = '⚠ The market size is smaller than these companies\' combined revenue (' +
                           big(s.total) + '), so shares add up to more than 100%.';
      } else {
        note.textContent = 'Together these companies hold ' + (s.total / s.base * 100).toFixed(1) + '% of the market.';
      }
    } else {
      desc.textContent = 'Each company\'s TTM revenue as a share of the ' + rows.length +
                         ' companies\' combined revenue (' + big(s.total) + '). Enter a total market size to ' +
                         'estimate share of the whole space instead.';
      note.textContent = 'Peer-group share only — it assumes these companies make up the space.';
    }
    var box = document.getElementById('share-box');
    box.style.height = boxHeight(labels.length);
    if (!hasChart) return;
    if (shareChart) shareChart.destroy();
    shareChart = hBar(document.getElementById('chartShare'), labels, vals, colors,
                      function (v) { return v.toFixed(1) + '%'; });
  }

  function renderMetricCharts() {
    var grid = document.getElementById('metric-grid');
    COLS.filter(function (c) { return c.chart; }).forEach(function (col, i) {
      var rows = C.filter(function (c) { return isNum(c[col.key]); })
                  .sort(function (a, b) { return col.best === 'low' ? a[col.key] - b[col.key] : b[col.key] - a[col.key]; });
      var missing = C.filter(function (c) { return !isNum(c[col.key]); }).map(function (c) { return c.ticker; });
      var card = document.createElement('div');
      card.className = 'card';
      var dirText = col.best === 'low' ? (col.unit === 'x' && col.key.indexOf('mult') === 0 ? 'Lower = cheaper' : 'Lower is better')
                                       : 'Higher is better';
      card.innerHTML = "<div class='card-title'>" + esc(col.label) + "</div>" +
        "<div class='card-desc'>" + dirText + (col.note ? ' · ' + esc(col.note) : '') + " · best at top</div>" +
        "<div class='chart-box' style='height:" + boxHeight(rows.length) + "'><canvas id='cm" + i + "' role='img' aria-label='" + esc(col.label) + "'></canvas></div>" +
        (missing.length ? "<div class='nm-note'>" + (col.nm ? 'n/m (' + esc(col.nm) + '): ' : 'No data: ') + esc(missing.join(', ')) + "</div>" : '');
      grid.appendChild(card);
      if (!hasChart || !rows.length) return;
      var fmt = col.unit === '%' ? function (v, axis) { return (axis ? v : v.toFixed(1)) + '%'; }
                                 : function (v, axis) { return (axis ? v : v.toFixed(col.key === 'net_debt_ebitda' ? 2 : 1)) + 'x'; };
      hBar(document.getElementById('cm' + i), rows.map(function (c) { return c.ticker; }),
           rows.map(function (c) { return +c[col.key].toFixed(3); }),
           // red only where a negative value is bad (growth, margins) — for
           // leverage a negative number means net cash, which is good
           rows.map(function (c) { return (!col.noSignColor && c[col.key] < 0) ? NEG : POS; }), fmt);
    });
  }

  function renderScatter() {
    if (!hasChart) return;
    var pts = C.filter(function (c) { return isNum(c.revenue_growth) && isNum(c.mult_revenue); })
               .map(function (c) { return { x: +c.revenue_growth.toFixed(2), y: +c.mult_revenue.toFixed(2), t: c.ticker }; });
    var labelPlugin = {
      id: 'tickerLabels',
      afterDatasetsDraw: function (chart) {
        var ctx = chart.ctx, meta = chart.getDatasetMeta(0);
        ctx.save();
        ctx.font = '600 12px -apple-system, Segoe UI, system-ui, sans-serif';
        ctx.fillStyle = T.tick; ctx.textBaseline = 'middle';
        // Place each ticker to the right of its dot; if it would overlap a label
        // already drawn, try below / above / left until it fits.
        var placed = [], h = 14;
        function hits(r) {
          return placed.some(function (p) {
            return r.x < p.x + p.w && p.x < r.x + r.w && r.y < p.y + p.h && p.y < r.y + r.h;
          });
        }
        meta.data.forEach(function (el, i) {
          var w = ctx.measureText(pts[i].t).width, best = null;
          var tries = [[9, 0], [9, h], [9, -h], [-w - 9, 0], [9, 2 * h], [9, -2 * h], [-w - 9, h], [-w - 9, -h]];
          for (var k = 0; k < tries.length; k++) {
            var r = { x: el.x + tries[k][0], y: el.y + tries[k][1] - h / 2, w: w, h: h };
            if (!hits(r)) { best = r; break; }
          }
          best = best || { x: el.x + 9, y: el.y - h / 2, w: w, h: h };
          placed.push(best);
          ctx.fillText(pts[i].t, best.x, best.y + h / 2);
        });
        ctx.restore();
      }
    };
    new Chart(document.getElementById('chartScatter'), {
      type: 'scatter',
      data: { datasets: [{ data: pts, backgroundColor: POS, borderColor: getComputedStyle(document.body).backgroundColor,
                           borderWidth: 2, pointRadius: 6, pointHoverRadius: 8 }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        layout: { padding: { right: 40 } },
        plugins: { legend: { display: false },
                   tooltip: { callbacks: { label: function (ctx) {
                     var p = pts[ctx.dataIndex];
                     return p.t + ': growth ' + spct(p.x) + ', ' + p.y.toFixed(1) + 'x revenue'; } } } },
        scales: {
          x: { title: { display: true, text: 'Revenue growth YoY', color: T.tick },
               grid: { color: T.grid }, ticks: { color: T.tick, callback: function (v) { return v + '%'; } } },
          y: { title: { display: true, text: 'Mkt Cap / Revenue', color: T.tick }, beginAtZero: true,
               grid: { color: T.grid }, ticks: { color: T.tick, callback: function (v) { return v + 'x'; } } }
        }
      },
      plugins: [labelPlugin]
    });
  }

  // ── market size controls ──────────────────────────────────────────────────
  var input = document.getElementById('tam-input');
  if (marketSizeB) input.value = marketSizeB;
  function applyTam() {
    var v = parseFloat(input.value);
    marketSizeB = isFinite(v) && v > 0 ? v : null;
    renderShare(); renderTable();
  }
  document.getElementById('tam-apply').addEventListener('click', applyTam);
  input.addEventListener('keydown', function (e) { if (e.key === 'Enter') applyTam(); });
  document.getElementById('tam-reset').addEventListener('click', function () {
    input.value = ''; marketSizeB = null; renderShare(); renderTable();
  });

  renderShare();
  renderTable();
  renderScatter();
  renderMetricCharts();
})();
</script>
"""
