"""
lab_report.py — HTML report for the Backtest tab (regime_lab.LabResult).
Same look as the other reports (web_reports page shell).
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from rsi2.ab import STOP_PRESETS, default_time_stop
from web_reports import _page, _header, _e, _json, _table

ORDER = ["histogram", "crossover", "A2", "A3", "A4", "B2", "B3", "B4"]
# Validated reference palette. Colour follows the RSI length (same in A and B —
# the chart/label says which strategy); the two MACD references get their own slots.
RSI_COLORS = {2: ("#2a78d6", "#3987e5"), 3: ("#eb6834", "#d95926"), 4: ("#1baf7a", "#199e70")}
COLORS = {"histogram": ("#4a3aa7", "#9085e9"), "crossover": ("#eda100", "#c98500")}
for _L, _c in RSI_COLORS.items():
    COLORS[f"A{_L}"] = COLORS[f"B{_L}"] = _c
SHORT = {"histogram": "Histogram", "crossover": "Crossover",
         **{f"A{L}": f"A·RSI{L}" for L in RSI_COLORS}, **{f"B{L}": f"B·RSI{L}" for L in RSI_COLORS}}
REASONS = {
    "stop": "ATR stop", "signal": "Sell signal",
    "rsi_decline": "RSI decline-watch", "macd_ma5": "MACD/MA5",
    "atr_stop": "ATR stop", "time_stop": "Time stop",
}


def _n(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pct(v, signed=True, d=1):
    f = _n(v)
    if f is None:
        return "–"
    return f"{f:+.{d}f}%" if signed else f"{f:.{d}f}%"


def _num(v, d=2):
    if v == "inf":
        return "∞"
    f = _n(v)
    return "–" if f is None else f"{f:.{d}f}"


def _money(v):
    f = _n(v)
    return "–" if f is None else f"{'−' if f < 0 else ''}${abs(f):,.0f}"


def _tone(v):
    f = _n(v)
    return "" if f is None else ("positive" if f > 0 else "negative" if f < 0 else "")


_EXTRA_CSS_T = """
<style>
  :root { __LIGHT__
          --bench:#8a8f98; --best-bg: rgba(42,120,214,0.10); }
  @media (prefers-color-scheme: dark) {
    :root { __DARK__
            --bench:#9aa0aa; --best-bg: rgba(57,135,229,0.16); }
  }
  body { max-width: 1040px; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; vertical-align:0; flex:none; }
  .fam { display:inline-block; white-space:nowrap; font-size:10px; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
         color:var(--muted); border:1px solid var(--border); border-radius:999px; padding:0 7px; margin-left:6px; }
  .score td, .score th { text-align:right; padding: 8px 5px; font-size: 12.5px; }
  .score th { font-size: 10px; white-space: normal; line-height: 1.25; vertical-align: bottom; }
  .score td:first-child, .score th:first-child { text-align:left; white-space:normal; min-width:150px; }
  .score td.best { background: var(--best-bg) !important; font-weight:700; }
  .score td.best::after { content:" ★"; font-size:10px; }
  .score tr.bench td { color: var(--muted); font-style: italic; }
  .score .sub { display:block; font-size:11px; color:var(--muted); font-weight:400; }
  .score tr.na td { color: var(--muted); }
  .grid3 { display:grid; grid-template-columns:1fr; gap:12px; }
  @media (min-width: 760px) { .grid3 { grid-template-columns: repeat(2, 1fr); } }
  .kv2 { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 4px 16px; font-size:13px; margin-top:8px; }
  .kv2 div { display:flex; justify-content:space-between; gap:8px; border-bottom:1px dashed var(--border); padding:2px 0; }
  .kv2 span:first-child { color: var(--muted); }
  .kv2 span:last-child { font-weight:600; font-variant-numeric: tabular-nums; }
  .card h4 { margin: 0; font-size: 15px; display:flex; align-items:center; }
  .handoff { display:grid; grid-template-columns: 1fr; gap:12px; }
  @media (min-width: 760px) { .handoff { grid-template-columns: 1fr 1fr; } }
  .verdict { font-size: 14px; margin-top: 10px; padding: 10px 12px; border-radius: 8px; background: rgba(46,134,171,.08); }
  .notice { font-size: 13px; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border); background: var(--card); margin: 10px 0; }
  details.card > summary { cursor:pointer; font-weight:600; font-size:15px; }
  details.card[open] > summary { margin-bottom: 12px; }
  .legend-row { display:flex; flex-wrap:wrap; gap:6px 16px; font-size:12px; color:var(--muted); margin-top:8px; }
  .legend-row span { display:inline-flex; align-items:center; }
  .legend-row i { display:inline-block; width:18px; height:3px; border-radius:2px; margin-right:6px; }
  .legend-row i.dash { background: repeating-linear-gradient(90deg, var(--bench) 0 5px, transparent 5px 8px) !important; }
  .range-btns { display:flex; gap:4px; margin: 0 0 8px; }
  .range-btns button { font: inherit; font-size: 12px; padding: 3px 10px; border-radius: 6px; cursor: pointer;
    border: 1px solid var(--border); background: var(--card); color: var(--text); }
  .range-btns button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  .eq-pair { display:grid; grid-template-columns:1fr; gap:12px; }
  @media (min-width: 860px) { .eq-pair { grid-template-columns: 1fr 1fr; } .eq-pair .card { margin-bottom:0; } }
  .ab td:first-child, .ab th:first-child { min-width: 52px !important; width: 52px; white-space: nowrap; text-align: center; }
  .ab td, .ab th { padding: 8px 5px; }
  .ab td:nth-child(2), .ab th:nth-child(2) { text-align: center; }
  .insights { margin: 12px 0 0 18px; padding: 0; font-size: 13px; line-height: 1.55; }
  .insights li { margin: 3px 0; }
  .dl-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  .dl { font: inherit; font-size: 13px; padding: 6px 12px; border-radius: 8px; cursor: pointer;
        border: 1px solid var(--border); background: var(--card); color: var(--text); }
  .dl:hover { border-color: var(--accent); }
  .stop-badge { font-size: 13px; margin: 0 0 8px; padding: 8px 12px; border-radius: 8px; background: var(--best-bg); }
  .stop-badge span { color: var(--muted); }
  .sg td, .sg th { text-align:right; padding: 8px 6px; font-size: 12.5px; font-variant-numeric: tabular-nums; }
  .sg th { font-size: 10px; white-space: normal; line-height: 1.25; vertical-align: bottom; }
  .sg td:first-child, .sg th:first-child { text-align:left; white-space:nowrap; }
  .sg th.inuse { box-shadow: inset 0 -3px 0 var(--accent); }
  .sg td.inuse { box-shadow: inset 2px 0 0 var(--accent), inset -2px 0 0 var(--accent); }
  .sg td.best { background: var(--best-bg) !important; font-weight:700; }
  .sg .star { font-size:10px; }
  .sg .d { display:block; font-size:10.5px; color: var(--muted); font-weight:400; }
  .yr td, .yr th { text-align:right; }
  .yr td:first-child, .yr th:first-child { text-align:left; }
</style>
"""


_EXTRA_CSS = (_EXTRA_CSS_T
              .replace("__LIGHT__", " ".join(f"--s-{k}:{v[0]};" for k, v in COLORS.items()))
              .replace("__DARK__", " ".join(f"--s-{k}:{v[1]};" for k, v in COLORS.items())))


def _dot(key):
    return f"<span class='dot' style='background:var(--s-{key})'></span>"


def render(lab) -> str:
    strategies = [lab.strategies[k] for k in ORDER]
    avail = [s for s in strategies if s.available]
    ranked = lab.ranked()
    winner = ranked[0] if ranked else None
    bench = lab.benchmark
    parts = [_EXTRA_CSS]

    # ── notes ────────────────────────────────────────────────────────────────
    for n in lab.notes:
        parts.append(f"<div class='notice'>ℹ️ {_e(n)}</div>")
    for key, msg in lab.pending.items():
        parts.append(f"<div class='notice'>⚡ <b>{_e(lab.strategies[key].name)}</b>: {_e(msg)}.</div>")

    # ── winner banner ────────────────────────────────────────────────────────
    if winner:
        w = winner.summary
        by_sharpe = max(avail, key=lambda s: _n(s.summary.get("sharpe")) or -1e9)
        vs = w["total_return_pct"] - bench["total_return_pct"]
        extra = ""
        if by_sharpe.key != winner.key:
            extra = (f"<br>Best risk-adjusted: <b>{_e(by_sharpe.name)}</b> (Sharpe "
                     f"{_num(by_sharpe.summary.get('sharpe'))} vs {_num(w.get('sharpe'))}).")
        parts.append(f"""
<div class='winner'>
  <div style='font-size:28px'>🏆</div>
  <div>
    <div class='big'>{_e(winner.name)} returned {_pct(w['total_return_pct'])}</div>
    <div class='metric-sub' style='margin:0'>{_pct(vs)} vs buy &amp; hold ({_pct(bench['total_return_pct'])}) ·
      Sharpe {_num(w.get('sharpe'))} · max drawdown {_pct(w['max_drawdown_pct'])} · in the market
      {_pct(w['exposure_pct'], False, 0)} of the time.{extra}</div>
  </div>
</div>""")

    # ── scoreboard ───────────────────────────────────────────────────────────
    cols = [
        ("Total return", "total_return_pct", lambda v: _pct(v), "high"),
        ("CAGR", "cagr_pct", lambda v: _pct(v), "high"),
        ("Max DD", "max_drawdown_pct", lambda v: _pct(v), "high"),
        ("Sharpe", "sharpe", lambda v: _num(v), "high"),
        ("Trades", "trades", lambda v: "–" if v is None else f"{int(v)}", None),
        ("Win %", "win_rate_pct", lambda v: _pct(v, False, 0), "high"),
        ("Avg trade", "avg_net_ret_pct", lambda v: _pct(v, True, 2), "high"),
        ("Profit factor", "profit_factor", lambda v: _num(v), "high"),
        ("Exposure", "exposure_pct", lambda v: _pct(v, False, 0), None),
        ("Return ÷ exposure", "return_per_exposure_pct", lambda v: _pct(v, True, 0), "high"),
        ("t-stat", "t_stat", lambda v: _num(v), "high"),
    ]
    best = {}
    for _, key, _, d in cols:
        if not d:
            continue
        vals = [(_n(s.summary.get(key)) if s.summary.get(key) != "inf" else 1e18, s.key) for s in avail]
        vals = [v for v in vals if v[0] is not None]
        if len(vals) >= 2 and len({round(v[0], 9) for v in vals}) > 1:
            top = max(v[0] for v in vals)
            best[key] = {k for v, k in vals if abs(v - top) <= 1e-9 * max(1.0, abs(top))}
    head = "".join(f"<th>{_e(c[0])}</th>" for c in cols)
    body = ""
    for s in strategies:
        if not s.available:
            body += (f"<tr class='na'><td>{_dot(s.key)}{_e(s.name)}<span class='fam'>{_e(s.family)}</span>"
                     f"<span class='sub'>{_e(s.error)}</span></td>" + "<td>–</td>" * len(cols) + "</tr>")
            continue
        cells = ""
        for _, key, fmt, _ in cols:
            v = s.summary.get(key)
            cls = []
            if s.key in best.get(key, ()):
                cls.append("best")
            if key in ("total_return_pct", "cagr_pct", "avg_net_ret_pct"):
                cls.append(_tone(v))
            cells += f"<td class='{' '.join(c for c in cls if c)}'>{fmt(v)}</td>"
        open_note = ""
        if s.open_trades:
            ot = s.open_trades[0]
            open_note = (f" · <b>in a trade</b> since {pd.Timestamp(ot['entry_date']):%b %d} "
                         f"({_pct(ot['unrealized_pct'])})")
        sub = f"<span class='sub'>{open_note.lstrip(' ·')}</span>" if open_note else ""
        body += (f"<tr><td>{_dot(s.key)}{_e(s.name)}<span class='fam'>{_e(s.family)}</span>{sub}</td>{cells}</tr>")
    b = bench
    body += ("<tr class='bench'><td><span class='dot' style='background:var(--bench)'></span>Buy &amp; hold"
             "<span class='sub'>benchmark · always invested</span></td>"
             f"<td>{_pct(b['total_return_pct'])}</td><td>{_pct(b['cagr_pct'])}</td><td>{_pct(b['max_drawdown_pct'])}</td>"
             f"<td>{_num(b.get('sharpe'))}</td><td>–</td><td>–</td><td>–</td><td>–</td><td>100%</td>"
             f"<td>{_pct(b['return_per_exposure_pct'], True, 0)}</td><td>–</td></tr>")
    parts.append(f"""
<div class='section-heading'>Scoreboard</div>
<div class='card'>
  <div class='card-desc'>Same window ({lab.window_start:%b %d, %Y} – {lab.window_end:%b %d, %Y}), same costs
    ({lab.cost_bps:g} bps per side), 100% of equity per trade, marked to market
    daily. ★ = best in column. <b>Return ÷ exposure</b> = total return per unit of time in the market (buy &amp; hold
    is always in). <b>t-stat</b> below ~2 means the average trade isn't distinguishable from noise.</div>
  <div class='table-wrap'><table class='data-table score'><thead><tr><th>Strategy</th>{head}</tr></thead>
  <tbody>{body}</tbody></table></div>
</div>""")

    # ── A vs B × RSI length (the requested comparison table) ────────────────
    stop = getattr(lab, "stop", "none") or "none"
    stop_label = STOP_PRESETS[stop].label
    full_comp = lab.comparison.copy()
    hide = ["Stop-loss", "Worst trade %"] + (["Stop exits %"] if stop == "none" else [])
    comp = full_comp.drop(columns=[c for c in hide if c in full_comp.columns])
    def cfmt(col, v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if col in ("RSI length", "Total trades", "Open at end"):
            return f"{int(v)}"
        if col == "Strategy":
            return str(v)
        if col == "Profit factor":
            return "∞" if v == float("inf") else f"{v:.2f}"
        if col in ("Avg holding (bars)",):
            return f"{v:.1f}"
        if col in ("Win rate %", "MACD/MA5 exits %", "RSI decline exits %", "Stop exits %"):
            return f"{v:.0f}%"
        return f"{v:+.2f}%" if col in ("Avg win %", "Avg loss %") else f"{v:+.1f}%"
    comp_best = {}
    for col, d in (("Total return %", "high"), ("Win rate %", "high"), ("Profit factor", "high"),
                   ("Max drawdown %", "high")):
        vals = comp[col].dropna()
        if len(vals) > 1 and vals.nunique() > 1:
            comp_best[col] = set(vals[vals == vals.max()].index)
    head = "".join(f"<th>{_e(c)}</th>" for c in comp.columns)
    body = ""
    for idx, row in comp.iterrows():
        key = f"{row['Strategy']}{int(row['RSI length'])}"
        cells = ""
        for col in comp.columns:
            v = row[col]
            cls = "best" if idx in comp_best.get(col, ()) else ""
            if col in ("Total return %", "Avg win %", "Avg loss %"):
                cls += " " + _tone(v)
            txt = cfmt(col, v)
            if col == "RSI length":
                txt = f"{_dot(key)}{txt}"
            cells += f"<td class='{cls.strip()}'>{txt}</td>"
        body += f"<tr>{cells}</tr>"

    def line(strat):
        c = comp[comp.Strategy == strat].set_index("RSI length")
        bits = []
        for L, r in c.iterrows():
            bits.append(f"RSI({int(L)}) {int(r['Total trades'])} trades, {cfmt('Win rate %', r['Win rate %'])} win, "
                        f"{cfmt('Total return %', r['Total return %'])}, holds {cfmt('Avg holding (bars)', r['Avg holding (bars)'])} bars")
        best = c["Total return %"].idxmax() if c["Total return %"].notna().any() else None
        return "; ".join(bits) + (f". Best: <b>RSI({int(best)})</b>." if best is not None else ".")
    insights = [f"<li><b>Strategy A</b> — {line('A')}</li>", f"<li><b>Strategy B</b> — {line('B')}</li>"]
    ab = comp.pivot(index="RSI length", columns="Strategy", values="Total return %")
    diffs = [f"RSI({int(L)}) {r['B'] - r['A']:+.1f} pts" for L, r in ab.iterrows() if pd.notna(r.get("A")) and pd.notna(r.get("B"))]
    if diffs:
        insights.append(f"<li><b>B minus A</b> (total return): {', '.join(diffs)}.</li>")
    tr = comp[comp.Strategy == "A"].set_index("RSI length")["Total trades"]
    if len(tr) >= 2 and tr.iloc[0] > tr.iloc[-1]:
        insights.append(f"<li>Longer RSI settings reach &lt;10 less often: {int(tr.iloc[0])} → {int(tr.iloc[-1])} "
                        f"Strategy A trades from RSI({int(tr.index[0])}) to RSI({int(tr.index[-1])}).</li>")
    if (comp["Open at end"] > 0).any():
        open_runs = ", ".join(f"{r['Strategy']}·RSI{int(r['RSI length'])}" for _, r in comp[comp['Open at end'] > 0].iterrows())
        why = (" With no stop-loss, a trade whose RSI never crosses over 80 stays open." if stop == "none" else "")
        insights.append(f"<li>Still in a trade at the end (not counted in the trade stats, included in total return): "
                        f"{open_runs}.{why}</li>")

    csv_data = {
        "comparison": {"name": f"{lab.ticker}_AB_comparison{'' if stop == 'none' else '_' + stop}.csv",
                       "cols": list(full_comp.columns),
                       "rows": [[None if (isinstance(v, float) and not math.isfinite(v)) else
                                 (v.item() if hasattr(v, "item") else v) for v in r] for r in full_comp.itertuples(index=False)]},
        "trades": {"name": f"{lab.ticker}_AB_trades{'' if stop == 'none' else '_' + stop}.csv", "cols": list(lab.trades.columns),
                   "rows": [[None if (isinstance(v, float) and not math.isfinite(v)) else
                             (v.item() if hasattr(v, "item") else v) for v in r] for r in lab.trades.itertuples(index=False)]},
    }
    grid = getattr(lab, "stop_grid", None)
    if grid is not None and len(grid):
        csv_data["stops"] = {"name": f"{lab.ticker}_AB_stop_comparison.csv", "cols": list(grid.columns),
                             "rows": [[None if (isinstance(v, float) and not math.isfinite(v)) else
                                       (v.item() if hasattr(v, "item") else v) for v in r]
                                      for r in grid.itertuples(index=False)]}
    parts.append(f"""
<div class='section-heading'>Strategy A vs B × RSI Length</div>
<div class='card'>
  <div class='stop-badge'>Stop-loss: <b>{_e(stop_label)}</b>{_e(_stop_desc(stop))}</div>
  <div class='card-desc'>6 runs on daily bars, same window and costs. Returns are net of {lab.cost_bps:g} bps per side;
    total return and drawdown use daily mark-to-market (100% of equity per trade). Trade statistics count closed
    trades; an open trade shows under “Open at end”. ★ = best of the six.</div>
  <div class='table-wrap'><table class='data-table score ab'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
  <ul class='insights'>{''.join(insights)}</ul>
  <div class='dl-row'>
    <button class='dl' data-csv='comparison'>⬇ comparison.csv</button>
    <button class='dl' data-csv='trades'>⬇ trade log (all 6 runs).csv</button>
  </div>
</div>
<script type='application/json' id='ab-csv'>{_json(csv_data)}</script>
<script>
(function() {{
  var D = JSON.parse(document.getElementById('ab-csv').textContent);
  function cell(v) {{
    if (v === null || v === undefined) return '';
    var s = typeof v === 'number' ? String(Math.round(v * 1e4) / 1e4) : String(v);
    return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }}
  document.addEventListener('click', function(e) {{
    var b = e.target.closest ? e.target.closest('button.dl') : null;
    if (!b) return;
    var t = D[b.getAttribute('data-csv')];
    if (!t) return;
    var csv = [t.cols.map(cell).join(',')].concat(t.rows.map(function(r) {{ return r.map(cell).join(','); }})).join('\\n');
    var url = URL.createObjectURL(new Blob([csv], {{ type: 'text/csv' }}));
    var a = document.createElement('a'); a.href = url; a.download = t.name;
    document.body.appendChild(a); a.click(); setTimeout(function() {{ URL.revokeObjectURL(url); a.remove(); }}, 500);
  }});
}})();
</script>""")

    if grid is not None and len(grid):
        parts.append(_stop_section(lab, grid, stop))

    # ── equity curves: A and B, coloured by RSI length ──────────────────────
    idx = lab.benchmark_equity.index
    step = max(1, len(idx) // 700)
    take = list(range(0, len(idx), step))
    if take[-1] != len(idx) - 1:
        take.append(len(idx) - 1)
    charts = {}
    for strat in ("A", "B"):
        ser = []
        for L in (2, 3, 4):
            st = lab.strategies.get(f"{strat}{L}")
            if st and st.available:
                eq = st.equity.reindex(idx).ffill()
                ser.append({"key": f"{strat}{L}", "name": f"RSI({L})", "data": [round(float(eq.iloc[i]), 2) for i in take]})
        charts[strat] = ser
    eq_data = {"labels": [idx[i].strftime("%Y-%m-%d") for i in take], "charts": charts,
               "bench": [round(float(lab.benchmark_equity.iloc[i]), 2) for i in take]}
    legend = "".join(f"<span><i style='background:var(--s-A{L})'></i>RSI({L})</span>" for L in (2, 3, 4))
    legend += "<span><i class='dash'></i>Buy &amp; hold</span>"
    parts.append(f"""
<div class='section-heading'>Equity Curves — $10,000 each, net of costs</div>
<div class='card-desc'>Log scale, so equal vertical distances are equal percentage moves; both charts share one scale.</div>
<div class='eq-pair'>
  <div class='card'><div class='card-title'>Strategy A · Pure RSI mean reversion</div>
    <div style='position:relative;height:300px'><canvas id='chartEqA' role='img' aria-label='Strategy A equity curves'></canvas></div>
    <div class='legend-row'>{legend}</div></div>
  <div class='card'><div class='card-title'>Strategy B · RSI + MACD filter</div>
    <div style='position:relative;height:300px'><canvas id='chartEqB' role='img' aria-label='Strategy B equity curves'></canvas></div>
    <div class='legend-row'>{legend}</div></div>
</div>
<script>
(function() {{
  var d = {_json(eq_data)}, T = window.__fbTheme, css = getComputedStyle(document.documentElement);
  var lo = Infinity, hi = -Infinity;
  ['A', 'B'].forEach(function(k) {{ d.charts[k].forEach(function(s) {{ s.data.forEach(function(v) {{ lo = Math.min(lo, v); hi = Math.max(hi, v); }}); }}); }});
  d.bench.forEach(function(v) {{ lo = Math.min(lo, v); hi = Math.max(hi, v); }});
  ['A', 'B'].forEach(function(k) {{
    var ds = d.charts[k].map(function(s) {{
      var c = css.getPropertyValue('--s-' + s.key).trim();
      return {{ label: s.name, data: s.data, borderColor: c, backgroundColor: c, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4 }};
    }});
    ds.push({{ label: 'Buy & hold', data: d.bench, borderColor: css.getPropertyValue('--bench').trim(), borderWidth: 1.5,
              borderDash: [5, 4], pointRadius: 0, pointHoverRadius: 3 }});
    new Chart(document.getElementById('chartEq' + k), {{
      type: 'line', data: {{ labels: d.labels, datasets: ds }},
      options: {{ responsive: true, maintainAspectRatio: false, animation: false,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{ legend: {{ display: false }},
          tooltip: {{ itemSort: function(a, b) {{ return b.parsed.y - a.parsed.y; }},
            callbacks: {{ label: function(c) {{ return c.dataset.label + ': $' + Math.round(c.parsed.y).toLocaleString(); }} }} }} }},
        scales: {{
          x: {{ grid: {{ color: T.grid }}, ticks: {{ color: T.tick, maxTicksLimit: 6, maxRotation: 0,
                 callback: function(v) {{ return this.getLabelForValue(v).slice(0, 7); }} }} }},
          y: {{ type: 'logarithmic', min: lo * 0.92, max: hi * 1.08,
                grid: {{ color: T.grid }}, ticks: {{ color: T.tick, maxTicksLimit: 8,
                  callback: function(v) {{ var m = v / Math.pow(10, Math.floor(Math.log10(v)));
                    return [1, 2, 5].some(function(k) {{ return Math.abs(m - k) < 1e-6; }})
                      ? '$' + (v >= 1000 ? (v / 1000).toLocaleString() + 'k' : v) : ''; }} }} }}
        }} }}
    }});
  }});
}})();
</script>""")

    # ── Strategy B: which exit path did trades take? ────────────────────────
    cards = ""
    for L in (2, 3, 4):
        st = lab.strategies.get(f"B{L}")
        if not st or not st.available:
            continue
        h = st.extra.get("paths", {})
        if not h.get("closed"):
            verdict = "No closed trades yet."
        else:
            ma_n, rs_n = h.get("ma5_n", 0), h.get("rsi_n", 0)
            if ma_n == 0:
                verdict = "The MACD BUY state never came before RSI crossed over 80, so every trade used A's decline watch."
            elif rs_n == 0:
                verdict = (f"Every trade that reached RSI 80 took the MACD/MA5 exit — a MACD BUY state had always appeared before RSI "
                           f"crossed over 80 — averaging {_pct(h.get('ma5_avg'), True, 2)} over "
                           f"{_num(h.get('ma5_bars'), 1)} bars.")
            else:
                better = "better" if (h.get("ma5_avg") or 0) > (h.get("rsi_avg") or 0) else "worse"
                verdict = (f"MACD/MA5 exits averaged {_pct(h.get('ma5_avg'), True, 2)} vs {_pct(h.get('rsi_avg'), True, 2)} "
                           f"for RSI decline-watch exits ({better}), holding {_num(h.get('ma5_bars'), 1)} vs "
                           f"{_num(h.get('rsi_bars'), 1)} bars.")
                if h.get("p_value") is not None:
                    verdict += (f" Welch p = {h['p_value']:.2f}"
                                + (" — not statistically distinguishable." if h["p_value"] > 0.05 else "."))
        cards += f"""
<div class='card'>
  <h4>{_dot(f'B{L}')}B · RSI({L})</h4>
  <div class='kv2'>
    <div><span>Closed trades</span><span>{h.get('closed', 0)}</span></div>
    <div><span>MACD/MA5 exits</span><span>{h.get('ma5_n', 0)} ({_pct(h.get('ma5_pct'), False, 0)})</span></div>
    <div><span>RSI decline-watch exits</span><span>{h.get('rsi_n', 0)} ({_pct(h.get('rsi_pct'), False, 0)})</span></div>
    {f"<div><span>Stop exits</span><span>{h.get('stop_n', 0)} ({_pct(h.get('stop_pct'), False, 0)}) · avg {_pct(h.get('stop_avg'), True, 2)}</span></div>" if stop != "none" else ""}
    <div><span>Avg return MA5 / RSI</span><span>{_pct(h.get('ma5_avg'), True, 2)} / {_pct(h.get('rsi_avg'), True, 2)}</span></div>
    <div><span>Win rate MA5 / RSI</span><span>{_pct(h.get('ma5_win'), False, 0)} / {_pct(h.get('rsi_win'), False, 0)}</span></div>
    <div><span>Avg bars MA5 / RSI</span><span>{_num(h.get('ma5_bars'), 1)} / {_num(h.get('rsi_bars'), 1)}</span></div>
  </div>
  <div class='verdict'>{verdict}</div>
</div>"""
    parts.append(f"""
<div class='section-heading'>Strategy B — Exit Paths</div>
<div class='card-desc'>Each Strategy B trade decides its exit when RSI crosses over 80: if a MACD-histogram BUY state
  (first bar of a fresh 2-rising-bars streak, daily MACD 12/26/9) happened since entry, it exits on the first close below
  MA5; otherwise it uses Strategy A's RSI decline watch.{"" if stop == "none" else " Trades closed by the " + _e(stop_label.lower()) + " before that point are counted under “Stop exits”."}</div>
<div class='handoff'>{cards}</div>""")

    # ── per-strategy cards ───────────────────────────────────────────────────
    cards = ""
    for s in strategies:
        if not s.available:
            continue
        m = s.summary
        mix = " · ".join(f"{REASONS.get(x['exit_reason'], x['exit_reason'])} {x['count']} "
                         f"({_pct(x['avg_net_ret_pct'], True, 2)})" for x in m.get("exit_mix", []))
        cards += f"""
<div class='card'>
  <h4>{_dot(s.key)}{_e(s.name)}<span class='fam'>{_e(s.family)}</span></h4>
  <div class='card-desc' style='margin:6px 0 0'>{_e(s.rules)}</div>
  <div class='metric-sub'>{_e(s.params)}</div>
  <div class='kv2'>
    <div><span>Total return</span><span class='{_tone(m['total_return_pct'])}'>{_pct(m['total_return_pct'])}</span></div>
    <div><span>Fixed $10k/trade PnL</span><span class='{_tone(m.get('fixed_pnl'))}'>{_money(m.get('fixed_pnl'))}</span></div>
    <div><span>CAGR</span><span>{_pct(m['cagr_pct'])}</span></div>
    <div><span>Max drawdown</span><span>{_pct(m['max_drawdown_pct'])} ({m['max_drawdown_days']}d)</span></div>
    <div><span>Sharpe / Sortino</span><span>{_num(m.get('sharpe'))} / {_num(m.get('sortino'))}</span></div>
    <div><span>Trades / year</span><span>{_num(m.get('trades_per_year'), 1)}</span></div>
    <div><span>Avg win / loss</span><span>{_pct(m.get('avg_win_pct'), True, 2)} / {_pct(m.get('avg_loss_pct'), True, 2)}</span></div>
    <div><span>Avg days held</span><span>{_num(m.get('avg_bars_held'), 1)}</span></div>
    <div><span>Max losing streak</span><span>{m.get('max_consecutive_losses', '–')}</span></div>
    <div><span>Open at end</span><span>{m['open_trades']}</span></div>
  </div>
  {f"<div class='metric-sub' style='margin-top:8px'>Exits: {mix}</div>" if mix else ""}
</div>"""
    parts.append(f"<div class='section-heading'>Strategy Details</div><div class='grid3'>{cards}</div>")

    # ── price & trades for the winner ────────────────────────────────────────
    if winner:
        d = lab.daily.iloc[lab.start_bar:]
        st = lab.start_bar
        n = len(d)
        buys, sells = [None] * n, [None] * n
        for f in winner.fills:
            if f.entry_bar >= st:
                buys[f.entry_bar - st] = round(f.entry_price, 2)
            if f.exit_bar is not None:
                sells[f.exit_bar - st] = round(f.exit_price, 2)
        pc = {"labels": [x.strftime("%Y-%m-%d") for x in d.index], "close": [round(float(v), 2) for v in d["Close"]],
              "buys": buys, "sells": sells}
        parts.append(f"""
<div class='section-heading'>Price &amp; Trades — {_e(winner.name)}</div>
<div class='card'>
  <div class='card-desc'>Daily close · ▲ entry · ▼ exit, drawn at the fill price{" (next bar's open)" if winner.key[0] in "AB" else ""}.</div>
  <div class='range-btns' id='pxRange'><button data-m='0' class='on'>All</button><button data-m='12'>1Y</button>
    <button data-m='6'>6M</button><button data-m='3'>3M</button></div>
  <div style='position:relative;height:360px'><canvas id='chartPx' role='img' aria-label='Price with trades'></canvas></div>
</div>
<script>
(function() {{
  var d = {_json(pc)}, T = window.__fbTheme, css = getComputedStyle(document.documentElement);
  var ink = T.dark ? '#cbd5e0' : '#4a5568';
  var ds = [
    {{ label: 'Close', data: d.close, borderColor: ink, borderWidth: 1.2, pointRadius: 0, order: 4 }},
    {{ label: 'Entry', data: d.buys, showLine: false, pointStyle: 'triangle', pointRadius: 6, pointHoverRadius: 8,
       backgroundColor: '#1f8a4c', borderColor: '#1f8a4c', order: 1 }},
    {{ label: 'Exit', data: d.sells, showLine: false, pointStyle: 'triangle', rotation: 180, pointRadius: 6,
       pointHoverRadius: 8, backgroundColor: '#d03b3b', borderColor: '#d03b3b', order: 1 }}
  ];
  var zoomed = false;
  var px = new Chart(document.getElementById('chartPx'), {{
    type: 'line', data: {{ labels: d.labels, datasets: ds }},
    options: {{ responsive: true, maintainAspectRatio: false, animation: false, spanGaps: false,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ color: T.tick, usePointStyle: true, boxWidth: 10,
                    filter: function(it, data) {{ return data.datasets[it.datasetIndex].data.some(function(v) {{ return v != null; }}); }} }} }},
        tooltip: {{ filter: function(i) {{ return i.raw != null; }},
          callbacks: {{ label: function(c) {{ return c.dataset.label + ': $' + Number(c.parsed.y).toFixed(2); }} }} }} }},
      scales: {{
        x: {{ grid: {{ color: T.grid }}, ticks: {{ color: T.tick, maxTicksLimit: 10, maxRotation: 0,
               callback: function(v) {{ var s = this.getLabelForValue(v); return zoomed ? s.slice(5) : s.slice(0, 7); }} }} }},
        y: {{ grid: {{ color: T.grid }}, ticks: {{ color: T.tick, includeBounds: false,
               callback: function(v) {{ return '$' + (Math.abs(v) >= 100 ? Math.round(v) : Number(v.toFixed(2))); }} }} }}
      }} }}
  }});
  // Range buttons: show the last N months (y-axis rescales to the visible prices)
  var btns = document.querySelectorAll('#pxRange button');
  btns.forEach(function(b) {{
    b.addEventListener('click', function() {{
      btns.forEach(function(x) {{ x.classList.toggle('on', x === b); }});
      var m = +b.getAttribute('data-m'), n = d.labels.length, lo = 0;
      if (m) {{
        var last = new Date(d.labels[n - 1]); last.setMonth(last.getMonth() - m);
        var cut = last.toISOString().slice(0, 10);
        while (lo < n - 1 && d.labels[lo] < cut) lo++;
      }}
      px.options.scales.x.min = d.labels[lo];
      px.options.scales.x.max = d.labels[n - 1];
      var vis = d.close.slice(lo);
      var ymin = Math.min.apply(null, vis), ymax = Math.max.apply(null, vis);
      var step = Math.pow(10, Math.floor(Math.log10(Math.max(ymax - ymin, 1e-6)))) / 2;
      zoomed = m > 0;
      px.options.scales.y.min = m ? Math.floor(ymin / step) * step : undefined;
      px.options.scales.y.max = m ? Math.ceil(ymax / step) * step : undefined;
      px.update('none');
    }});
  }});
}})();
</script>""")

    # ── returns by year ──────────────────────────────────────────────────────
    from rsi2.metrics import yearly_returns
    yr = {s.key: yearly_returns(s.equity) for s in avail}
    yb = yearly_returns(lab.benchmark_equity)
    years = list(yb.index)
    head = "".join(f"<th title='{_e(s.name)}'>{_dot(s.key)}{_e(SHORT[s.key])}</th>" for s in avail) + "<th>Buy &amp; hold</th>"
    rows = ""
    for y in years:
        cells = "".join(f"<td class='{_tone(yr[s.key].get(y))}'>{_pct(yr[s.key].get(y))}</td>" for s in avail)
        rows += f"<tr><td>{y}{' (partial)' if y in (years[0], years[-1]) else ''}</td>{cells}<td>{_pct(yb.get(y))}</td></tr>"
    parts.append(f"""
<div class='section-heading'>Returns by Year</div>
<div class='card'><div class='table-wrap'><table class='data-table yr'><thead><tr><th>Year</th>{head}</tr></thead>
<tbody>{rows}</tbody></table></div></div>""")

    # ── parameter sweeps & trade logs (collapsed) ────────────────────────────
    for kind, label in (("histogram", "MACD Histogram"), ("crossover", "MACD Crossover + MA50")):
        parts.append(f"""
<details class='card'><summary>Parameter sweep — {label} ({lab.tf_label.lower()} candles)</summary>
  <div class='card-desc'>All 12 combinations over the same window, net of costs. ★ = the set used above and in the
    combined strategy (best in-sample, min. {'10' if kind == 'histogram' else '5'} trades) — treat it as optimistic.</div>
  {_table(pd.DataFrame(lab.sweeps[kind]), numeric_cols=("PnL ($10k/trade, net)", "Avg trade"))}
</details>""")
    ab_log = lab.trades.copy()
    if len(ab_log):
        show = ab_log[["Strategy", "RSI length", "Entry date", "Entry price", "Exit date", "Exit price",
                       "Exit type", "Trade return %", "Bars held", "Status"]].copy()
        show["Entry price"] = show["Entry price"].map(lambda v: f"${v:.2f}")
        show["Exit price"] = show["Exit price"].map(lambda v: "" if pd.isna(v) else f"${v:.2f}")
        show["Trade return %"] = show["Trade return %"].map(lambda v: f"{v:+.2f}%")
        show = show.rename(columns={"Trade return %": "Return (net)"})
    parts.append(f"""
<details class='card'><summary>Trade log — Strategy A &amp; B, all RSI lengths ({len(ab_log)} trades)</summary>
  <div class='card-desc'>Also downloadable as CSV from the comparison table above.</div>
  {_table(show if len(ab_log) else pd.DataFrame(), 'No trades.', numeric_cols=("Return (net)",))}
</details>""")
    for s in strategies:
        if not s.available or s.family != "Momentum":
            continue
        led = s.ledger.copy() if s.ledger is not None else pd.DataFrame()
        rows = []
        for _, t in led.iterrows():
            row = {"Entry": pd.Timestamp(t.entry_date).strftime("%Y-%m-%d"), "Exit": pd.Timestamp(t.exit_date).strftime("%Y-%m-%d"),
                   "Entry $": f"${t.entry_price:.2f}", "Exit $": f"${t.exit_price:.2f}", "Days": int(t.bars_held),
                   "Net": f"{t.net_ret_pct:+.2f}%", "Reason": REASONS.get(t.exit_reason, t.exit_reason),
                   "MAE / MFE": f"{t.mae_pct:+.1f}% / {t.mfe_pct:+.1f}%"}
            if "extended" in led.columns:
                row["Hand-off"] = t.handoff_date if bool(t.extended) else ""
            rows.append(row)
        for ot in s.open_trades:
            row = {"Entry": pd.Timestamp(ot["entry_date"]).strftime("%Y-%m-%d"), "Exit": "open", "Entry $": f"${ot['entry_price']:.2f}",
                   "Exit $": f"${ot['last_close']:.2f} (last)", "Days": int(ot["bars_held"]), "Net": f"{ot['unrealized_pct']:+.2f}%",
                   "Reason": "still open", "MAE / MFE": ""}
            if "extended" in led.columns or "extended" in ot:
                row["Hand-off"] = ot.get("handoff_date") or ""
            rows.append(row)
        parts.append(f"""
<details class='card'><summary>{_dot(s.key)}Trade log — {_e(s.name)} ({len(led)} closed{', 1 open' if s.open_trades else ''})</summary>
  {_table(pd.DataFrame(rows), 'No trades in the window.', numeric_cols=("Net",))}
</details>""")

    subtitle = (f"Strategy Lab &nbsp;·&nbsp; RSI strategies on daily bars, MACD references on {lab.tf_label.lower()} candles "
                f"&nbsp;·&nbsp; {lab.window_start:%b %Y} – {lab.window_end:%b %d, %Y} "
                f"&nbsp;·&nbsp; Generated {datetime.now():%B %d, %Y}")
    return _page(f"{lab.ticker} — Strategy Lab", _header(lab.ticker, subtitle), "".join(parts),
                 "⚠️ Hypothetical results on split/dividend-adjusted Yahoo Finance prices, net of 4 bps per side "
                 "(commission + slippage), no taxes. RSI strategies fill at the next bar's open. MACD momentum parameters are "
                 "the best of 12 tested on this same data, so their results are optimistic. Backtesting a currently-listed "
                 "ticker is survivorship-biased. "
                 "Past performance ≠ future results. Not financial advice.")


# ── Stop-loss comparison ──────────────────────────────────────────────────────

def _stop_desc(stop: str) -> str:
    rule = STOP_PRESETS[stop]
    bits = []
    if rule.time_stop:
        bits.append(f"sell at the next open if RSI hasn't crossed over 80 by bar "
                    f"{default_time_stop(2)} / {default_time_stop(3)} / {default_time_stop(4)} (RSI 2 / 3 / 4)")
    if rule.atr_mult:
        bits.append(f"sell at the next open after a close below entry − {rule.atr_mult:g}× ATR(14)")
    return (" — " + "; ".join(bits) + ".") if bits else " — trades exit only on the strategy's own sell rule."


_STOP_METRICS = [
    # column, button label, better ("high" or None), format
    ("Total return %", "Total return", "high", "pct1"),
    ("Max drawdown %", "Max drawdown", "high", "pct1"),
    ("Return ÷ |max DD|", "Return ÷ drawdown", "high", "x"),
    ("Worst trade %", "Worst trade", "high", "pct1"),
    ("Win rate %", "Win rate", "high", "pct0"),
    ("Profit factor", "Profit factor", "high", "num2"),
    ("Avg holding (bars)", "Avg hold (bars)", None, "num1"),
    ("Stop exits %", "Stop exits", None, "pct0u"),
]


def _ending_ratio(r_stop, r_none):
    a, b = _n(r_stop), _n(r_none)
    return None if a is None or b is None or b <= -100 else (1 + a / 100) / (1 + b / 100) - 1


def _stop_section(lab, grid: pd.DataFrame, stop: str) -> str:
    g = grid.copy()
    g["Return ÷ |max DD|"] = [(_n(r) / abs(d)) if _n(r) is not None and _n(d) not in (None, 0) else None
                             for r, d in zip(g["Total return %"], g["Max drawdown %"])]
    labels = [STOP_PRESETS[k].label for k in STOP_PRESETS]
    none_label = STOP_PRESETS["none"].label
    runs = []
    for L in sorted(g["RSI length"].unique()):
        for strat in ("A", "B"):
            if ((g["RSI length"] == L) & (g["Strategy"] == strat)).any():
                runs.append((strat, int(L)))
    vals = {}
    for col, *_ in _STOP_METRICS:
        vals[col] = {}
        for strat, L in runs:
            row = []
            for lab_ in labels:
                v = g.loc[(g["RSI length"] == L) & (g["Strategy"] == strat) & (g["Stop-loss"] == lab_), col]
                f = v.iloc[0] if len(v) else None
                if col == "Profit factor" and f == float("inf"):
                    row.append("inf")
                else:
                    row.append(_n(f))
            vals[col][f"{strat}{L}"] = row
    data = {"stops": labels, "inuse": labels.index(STOP_PRESETS[stop].label),
            "runs": [{"key": f"{s}{L}", "label": f"{s} · RSI({L})"} for s, L in runs],
            "metrics": [{"col": c, "label": lb, "better": b, "fmt": f} for c, lb, b, f in _STOP_METRICS],
            "vals": vals}

    # ── plain-English takeaways ──
    ins = []
    n_runs = len(runs)
    best_ret = {}
    for strat, L in runs:
        rv = vals["Total return %"][f"{strat}{L}"]
        ok = [(v, -i) for i, v in enumerate(rv) if v is not None]
        if ok:
            best_ret[(strat, L)] = labels[-max(ok)[1]]          # ties go to the earlier column (No stop first)
    nostop_best = sum(1 for v in best_ret.values() if v == none_label)
    if best_ret:
        if nostop_best == n_runs:
            ins.append(f"<li><b>Total return:</b> no stop was best (or tied) in all {n_runs} runs.</li>")
        else:
            winners = ", ".join(f"{s}·RSI{L} → {_e(v)}" for (s, L), v in best_ret.items() if v != none_label)
            ins.append(f"<li><b>Total return:</b> no stop was best (or tied) in {nostop_best} of {n_runs} runs; "
                       f"a stop did better in {winners}.</li>")
    # ending equity vs no stop, averaged (geometric) across runs
    eff = []
    for k, lab_ in enumerate(labels):
        if lab_ == none_label:
            continue
        ratios = [_ending_ratio(vals["Total return %"][f"{s}{L}"][k], vals["Total return %"][f"{s}{L}"][0]) for s, L in runs]
        ratios = [r for r in ratios if r is not None]
        if ratios:
            gm = float(np.exp(np.mean(np.log1p(ratios)))) - 1
            eff.append(f"{_e(lab_)} {gm * 100:+.0f}%")
    if eff:
        ins.append(f"<li><b>Ending equity vs no stop</b> (average of the {n_runs} runs): {', '.join(eff)}.</li>")
    # drawdown
    dd_better, biggest = 0, None
    for s, L in runs:
        row = vals["Max drawdown %"][f"{s}{L}"]
        base = row[0]
        if base is None:
            continue
        cands = [(v, i) for i, v in enumerate(row[1:], 1) if v is not None]
        if not cands:
            continue
        v, i = max(cands)
        if v > base + 0.05:
            dd_better += 1
            if biggest is None or v - base > biggest[0]:
                biggest = (v - base, s, L, base, v, labels[i])
    txt = f"<li><b>Max drawdown:</b> at least one stop made it shallower in {dd_better} of {n_runs} runs"
    if biggest:
        _, s_, L_, b_, v_, lb_ = biggest
        txt += f" (biggest: {s_}·RSI{L_} {b_:.1f}% → {v_:.1f}% with {_e(lb_)})"
    ins.append(txt + ".</li>")
    # worst trade
    worse = better = 0
    for s, L in runs:
        row = vals["Worst trade %"][f"{s}{L}"]
        if row[0] is None:
            continue
        for v in row[1:]:
            if v is None:
                continue
            if v > row[0] + 0.05:
                better += 1
            elif v < row[0] - 0.05:
                worse += 1
    if better or worse:
        note = (" A stop sells near the low; without one, some of those trades recovered." if worse else "")
        ins.append(f"<li><b>Worst trade:</b> across all stop/run pairs, a stop improved the worst trade {better} "
                   f"time(s) and made it worse {worse} time(s).{note}</li>")
    # risk-adjusted
    rd = {}
    for s, L in runs:
        row = vals["Return ÷ |max DD|"][f"{s}{L}"]
        ok = [(v, -i) for i, v in enumerate(row) if v is not None]
        if ok:
            k = labels[-max(ok)[1]]
            rd[k] = rd.get(k, 0) + 1
    if rd:
        top = sorted(rd.items(), key=lambda kv: -kv[1])
        ins.append("<li><b>Best return ÷ drawdown:</b> " + ", ".join(f"{_e(k)} in {v} run(s)" for k, v in top) + ".</li>")

    on = " class='on'"
    btns = "".join(f"<button data-i='{i}'{on if i == 0 else ''}>{_e(m[1])}</button>"
                   for i, m in enumerate(_STOP_METRICS))
    return f"""
<div class='section-heading'>Stop-Loss Comparison</div>
<div class='card'>
  <div class='card-desc'>Strategy A and B × RSI 2/3/4 re-run with each stop — same entries, window and {lab.cost_bps:g} bps
    costs. <b>Time stop</b>: if RSI hasn't crossed over 80 by bar {default_time_stop(2)} / {default_time_stop(3)} /
    {default_time_stop(4)} of the trade (RSI 2 / 3 / 4), sell at the next open. <b>ATR stop</b>: a daily close below
    entry − k × ATR(14) sells at the next open, so a gap can fill below the stop. The underlined column is the stop used in
    the rest of this report (pick another in the Stop menu above the report). ★ = best in each row (a stop that ties
    No stop never fired).</div>
  <div class='range-btns' id='sgBtns' style='flex-wrap:wrap'>{btns}</div>
  <div class='table-wrap'><table class='data-table sg' id='sgTable'></table></div>
  <ul class='insights'>{''.join(ins)}</ul>
  <div class='dl-row'><button class='dl' data-csv='stops'>⬇ stop_comparison.csv (all 36 runs)</button></div>
</div>
<script type='application/json' id='sg-data'>{_json(data)}</script>
<script>
(function() {{
  var D = JSON.parse(document.getElementById('sg-data').textContent);
  var tbl = document.getElementById('sgTable'), btns = document.querySelectorAll('#sgBtns button');
  function fmt(v, f) {{
    if (v === 'inf') return '∞';
    if (v === null || v === undefined) return '–';
    if (f === 'pct1') return (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(1) + '%';
    if (f === 'pct0' || f === 'pct0u') return v.toFixed(0) + '%';
    if (f === 'num2') return v.toFixed(2);
    if (f === 'x') return v.toFixed(1) + '×';
    return v.toFixed(1);
  }}
  function num(v) {{ return v === 'inf' ? Infinity : v; }}
  function draw(mi) {{
    var m = D.metrics[mi];
    var h = '<thead><tr><th>Run</th>' + D.stops.map(function(s, i) {{
      return '<th' + (i === D.inuse ? ' class="inuse" title="Used in the rest of this report"' : '') + '>' + s + '</th>'; }}).join('') + '</tr></thead><tbody>';
    D.runs.forEach(function(r) {{
      var row = D.vals[m.col][r.key], best = null;
      if (m.better) {{
        var ok = row.filter(function(v) {{ return v !== null; }}).map(num);
        if (ok.length > 1 && Math.max.apply(null, ok) !== Math.min.apply(null, ok)) best = Math.max.apply(null, ok);
      }}
      h += '<tr><td><span class="dot" style="background:var(--s-' + r.key + ')"></span>' + r.label + '</td>';
      row.forEach(function(v, i) {{
        var cls = [];
        if (best !== null && v !== null && num(v) === best) cls.push('best');
        if (i === D.inuse) cls.push('inuse');
        if (m.fmt === 'pct1' && v !== null) cls.push(v > 0 ? 'positive' : v < 0 ? 'negative' : '');
        var sub = '';
        if (m.col === 'Total return %' && i > 0 && v !== null && row[0] !== null) {{
          var d = v - row[0]; sub = '<span class="d">' + (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(0) + ' pts</span>';
        }}
        var star = cls.indexOf('best') >= 0 ? '<span class="star"> ★</span>' : '';
        h += '<td class="' + cls.join(' ').trim() + '">' + fmt(v, m.fmt) + star + sub + '</td>';
      }});
      h += '</tr>';
    }});
    tbl.innerHTML = h + '</tbody>';
  }}
  btns.forEach(function(b) {{
    b.addEventListener('click', function() {{
      btns.forEach(function(x) {{ x.classList.remove('on'); }}); b.classList.add('on'); draw(+b.getAttribute('data-i'));
    }});
  }});
  draw(0);
}})();
</script>"""
