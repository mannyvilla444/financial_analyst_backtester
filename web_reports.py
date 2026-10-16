"""
web_reports.py

Self-contained HTML reports for the web app's Estimates and Backtest tabs.
The Telegram bot sends these as PDFs (estimates_report.py / backtest_report.py);
the web app renders them inline instead, styled to match html_report.py.

Public entry points (called by app.py):
    render_estimates(ticker)            -> str   (never raises; "no data" page on failure)
    render_backtest(ticker, timeframe)  -> str   (raises RuntimeError on bad ticker / no data)
"""

import html
import json
import math
from datetime import datetime

import pandas as pd

CHART_JS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"

# Stop-loss presets the Backtest tab offers (keys of rsi2.ab.STOP_PRESETS).
from rsi2.ab import STOP_PRESETS  # noqa: E402

TIMEFRAMES = {
    # web value : (macd_backtest interval, label)
    "weekly": ("1wk", "Weekly"),
    "daily":  ("1d",  "Daily"),
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _num(v):
    """float or None (NaN/inf/None/non-numeric → None)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _money(v, decimals=2):
    f = _num(v)
    return "N/A" if f is None else f"${f:,.{decimals}f}"


def _signed_money(v):
    f = _num(v)
    if f is None:
        return "N/A"
    return f"{'+' if f >= 0 else '−'}${abs(f):,.0f}"


def _billions(v):
    f = _num(v)
    return "N/A" if f is None else f"${f / 1e9:,.2f}B"


def _pct(v, signed=True):
    f = _num(v)
    if f is None:
        return "N/A"
    return f"{f:+.1f}%" if signed else f"{f:.1f}%"


def _mult(v):
    f = _num(v)
    return "N/A" if f is None else f"{f:.1f}x"


def _tone(v):
    f = _num(v)
    if f is None:
        return "neutral"
    return "positive" if f >= 0 else "negative"


def _upside(target, current):
    t, c = _num(target), _num(current)
    if t is None or not c:
        return None
    return (t / c - 1) * 100


def _e(s):
    return html.escape(str(s if s is not None else ""))


def _json(data) -> str:
    """JSON for embedding in a <script> block: NaN → null, no '</script>' breakouts."""
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float):
            return None if math.isnan(o) or math.isinf(o) else o
        return o
    return json.dumps(clean(data), allow_nan=False).replace("</", "<\\/")


def _table(df: pd.DataFrame, empty_msg="No rows.", numeric_cols=()) -> str:
    if df is None or df.empty:
        return f"<p class='no-data'>{_e(empty_msg)}</p>"
    head = "".join(f"<th>{_e(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v   = r[c]
            txt = _e(v)
            cls = ""
            if isinstance(v, str) and c in numeric_cols:
                if v.startswith(("+", "$+")):
                    cls = "positive"
                elif v.startswith(("-", "$-")):
                    cls = "negative"
            cells.append(f"<td class='{cls}'>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<div class='table-wrap'><table class='data-table'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


# ── Shared page shell (matches html_report.py) ────────────────────────────────

_CSS = """
  :root {
    --bg: #f5f7fa; --card: #ffffff; --border: #e2e8f0;
    --text: #1a202c; --muted: #718096; --accent: #2E86AB;
    --green: #27AE60; --red: #E74C3C; --orange: #E67E22; --amber: #B7791F;
    --radius: 12px; --shadow: 0 2px 8px rgba(0,0,0,0.07);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1117; --card: #1a1d27; --border: #2d3148;
      --text: #e2e8f0; --muted: #a0aec0; --amber: #F6C453;
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 15px; line-height: 1.5;
    padding: 16px; max-width: 900px; margin: 0 auto;
  }
  .header {
    background: linear-gradient(135deg, #1a365d 0%, #2E86AB 100%);
    border-radius: var(--radius); padding: 24px 20px 20px;
    margin-bottom: 20px; color: white;
  }
  .header-ticker { font-size: 32px; font-weight: 700; letter-spacing: -0.5px; }
  .header-name { font-size: 15px; opacity: 0.92; margin-top: 2px; }
  .header-sub { font-size: 13px; opacity: 0.8; margin-top: 4px; }
  .section-heading {
    font-size: 13px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--muted);
    margin: 28px 0 12px; padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }
  .metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
  @media (min-width: 520px) { .metric-grid { grid-template-columns: repeat(3, 1fr); } .metric-grid.four { grid-template-columns: repeat(2, 1fr); } }
  @media (min-width: 720px) { .metric-grid.four { grid-template-columns: repeat(4, 1fr); } }
  .metric-card, .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 14px 12px; box-shadow: var(--shadow);
  }
  .card { padding: 18px 16px; margin-bottom: 14px; }
  .metric-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--muted); margin-bottom: 6px;
  }
  .metric-value { font-size: 20px; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
  .metric-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .positive { color: var(--green) !important; }
  .negative { color: var(--red) !important; }
  .neutral  { color: var(--muted) !important; }
  .card-title { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
  .card-desc { font-size: 12px; color: var(--muted); margin-bottom: 14px; }
  canvas { max-height: 340px; }
  .no-data { color: var(--muted); font-size: 13px; padding: 8px 0; }
  .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .data-table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
  .data-table th {
    background: #1a365d; color: white; padding: 8px 10px; text-align: right;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
    white-space: nowrap;
  }
  .data-table th:first-child, .data-table td:first-child { text-align: left; }
  .data-table td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }
  .data-table tr:last-child td { border-bottom: none; }
  .data-table tr:nth-child(even) td { background: rgba(0,0,0,0.02); }
  @media (prefers-color-scheme: dark) { .data-table tr:nth-child(even) td { background: rgba(255,255,255,0.02); } }
  .data-table tr.hl td { background: rgba(46,134,171,0.10) !important; font-weight: 600; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600; border: 1px solid currentColor;
  }
  .badge.bull { color: var(--green); } .badge.bear { color: var(--red); } .badge.flat { color: var(--amber); }
  .winner {
    display: flex; gap: 12px; align-items: center;
    background: var(--card); border: 1px solid var(--border); border-left: 4px solid var(--green);
    border-radius: var(--radius); padding: 14px 16px; box-shadow: var(--shadow);
  }
  .winner .big { font-size: 18px; font-weight: 700; }
  .two-col { display: grid; grid-template-columns: 1fr; gap: 12px; }
  @media (min-width: 700px) { .two-col { grid-template-columns: 1fr 1fr; } }
  .strategy.win { border-color: var(--green); box-shadow: 0 0 0 1px var(--green), var(--shadow); }
  .kv { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; font-size: 13px; font-variant-numeric: tabular-nums; }
  .kv dt { color: var(--muted); }
  .kv dd { text-align: right; font-weight: 600; }
  .rules { font-size: 12px; color: var(--muted); margin-top: 10px; }
  .rec-bar { display: flex; height: 14px; border-radius: 7px; overflow: hidden; margin: 10px 0 6px; background: var(--border); }
  .rec-bar span { display: block; height: 100%; }
  .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: var(--muted); }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
  .empty-state { text-align: center; padding: 40px 16px; color: var(--muted); }
  .empty-state .big { font-size: 18px; font-weight: 600; color: var(--text); margin-bottom: 6px; }
  .disclaimer {
    font-size: 11px; color: var(--muted); text-align: center;
    padding: 20px 16px 8px; border-top: 1px solid var(--border); margin-top: 28px;
  }
"""

# Shared Chart.js theming, defined once per page.
_CHART_BOOT = """
<script>
window.__fbTheme = (function () {
  var dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  return { dark: dark, grid: dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)', tick: dark ? '#aaa' : '#555' };
})();
</script>"""


def _page(title: str, header_html: str, body_html: str, disclaimer: str, charts=True) -> str:
    chart_tags = f'<script src="{CHART_JS}"></script>{_CHART_BOOT}' if charts else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_e(title)}</title>
{chart_tags}
<style>{_CSS}</style>
</head>
<body>
{header_html}
{body_html}
<div class="disclaimer">{disclaimer}</div>
</body>
</html>"""


def _header(ticker, subtitle, name=None):
    name_html = f"<div class='header-name'>{_e(name)}</div>" if name and name != ticker else ""
    return (f"<div class='header'><div class='header-ticker'>{_e(ticker)}</div>{name_html}"
            f"<div class='header-sub'>{subtitle}</div></div>")


def _metric(label, value, sub="", tone=""):
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return (f"<div class='metric-card'><div class='metric-label'>{_e(label)}</div>"
            f"<div class='metric-value {tone}'>{_e(value)}</div>{sub_html}</div>")


def _empty_page(ticker, kind, reason):
    body = (f"<div class='card empty-state'><div class='big'>No {kind} data available for {_e(ticker)}</div>"
            f"<div>{_e(reason)}</div></div>")
    return _page(f"{ticker} — {kind.title()}",
                 _header(ticker, f"{kind.title()} &nbsp;·&nbsp; Generated {datetime.now():%B %d, %Y}"),
                 body, "Data from Yahoo Finance via yfinance. Not financial advice.", charts=False)


# ══════════════════════════════════════════════════════════════════════════════
#  ESTIMATES
# ══════════════════════════════════════════════════════════════════════════════

def render_estimates(ticker: str) -> str:
    """
    Run estimates_analysis for `ticker` and return a self-contained HTML page.
    Never raises: if yfinance has nothing (bad ticker, ETF, no coverage) the
    page says "No estimates data available" instead.
    """
    ticker = ticker.upper().strip()
    try:
        from estimates_analysis import run_estimates_analysis
        result = run_estimates_analysis(ticker)
    except Exception as e:
        return _empty_page(ticker, "estimates", str(e) or "yfinance returned no data.")

    try:
        return _estimates_html(result)
    except Exception as e:     # a malformed field shouldn't take the tab down
        return _empty_page(ticker, "estimates", f"Could not build the report: {e}")


def _estimates_html(r) -> str:
    price = _num(r.current_price)
    at    = r.analyst_targets
    periods = [p for p in (r.periods or [])
               if any(_num(getattr(p, a)) is not None
                      for a in ("rev_mean", "eps_mean", "rev_low", "eps_low"))]
    has_targets = at is not None and any(_num(getattr(at, a)) is not None
                                         for a in ("low", "mean", "median", "high"))

    if price is None and not periods and not has_targets:
        return _empty_page(r.ticker, "estimates",
                           "Yahoo Finance returned no price, estimates or analyst targets for this ticker.")

    parts = []

    # ── Snapshot ──────────────────────────────────────────────────────────────
    parts.append("<div class='section-heading'>Snapshot</div><div class='metric-grid'>")
    parts.append(_metric("Current Price", _money(price), f"Mkt Cap: {_billions(r.market_cap)}"))
    parts.append(_metric("Revenue TTM", _billions(r.ttm_revenue),
                         f"Rev multiple: {_mult(r.current_rev_multiple)}"))
    eps_val = _num(r.ttm_eps)
    parts.append(_metric("EPS TTM", "N/A" if eps_val is None else f"${eps_val:.2f}",
                         f"TTM P/E: {_mult(r.ttm_pe)}"))
    parts.append("</div>")

    # ── Implied price range chart ─────────────────────────────────────────────
    cats, ranges, means, colors = [], [], [], []
    for p in periods:
        for label, lo, mid, hi, col in (
            ("Revenue", p.rev_price_low, p.rev_price_mean, p.rev_price_high, "#2E86AB"),
            ("EPS",     p.eps_price_low, p.eps_price_mean, p.eps_price_high, "#27AE60"),
        ):
            lo, mid, hi = _num(lo), _num(mid), _num(hi)
            if lo is None and hi is None and mid is None:
                continue
            lo = lo if lo is not None else mid
            hi = hi if hi is not None else mid
            cats.append(f"{p.period} · {label}")
            ranges.append([round(min(lo, hi), 2), round(max(lo, hi), 2)])
            means.append(round(mid, 2) if mid is not None else None)
            colors.append(col)
    if has_targets:
        lo, mid, hi = _num(at.low), _num(at.mean), _num(at.high)
        if lo is not None or hi is not None:
            lo = lo if lo is not None else mid
            hi = hi if hi is not None else mid
            cats.append("Analyst targets")
            ranges.append([round(min(lo, hi), 2), round(max(lo, hi), 2)])
            means.append(round(mid, 2) if mid is not None else None)
            colors.append("#E67E22")

    if cats:
        chart = {"labels": cats, "ranges": ranges, "means": means, "colors": colors, "price": price}
        parts.append(f"""
<div class='section-heading'>Implied Price Ranges</div>
<div class='card'>
  <div class='card-title'>Where the estimates put the share price</div>
  <div class='card-desc'>Bars span the low → high estimate; the dot is the mean.
    Revenue bars hold today's Mkt Cap / Revenue multiple; EPS bars hold today's TTM P/E.
    Dashed line = current price.</div>
  <canvas id='chartImplied' role='img' aria-label='Implied price ranges'></canvas>
</div>
<script>
(function() {{
  var d = {_json(chart)}, T = window.__fbTheme;
  var n = d.labels.length;
  var ds = [
    {{ type: 'bar', label: 'Low → High', data: d.ranges, backgroundColor: d.colors.map(function(c){{return c + '55';}}),
       borderColor: d.colors, borderWidth: 1.5, borderRadius: 4, borderSkipped: false, barPercentage: 0.55, order: 2 }},
    {{ type: 'line', label: 'Mean', data: d.means, showLine: false, pointRadius: 5, pointHoverRadius: 7,
       pointBackgroundColor: d.colors, pointBorderColor: '#fff', pointBorderWidth: 1.5, order: 1 }}
  ];
  if (d.price) ds.push({{ type: 'line', label: 'Current $' + d.price.toFixed(2), data: Array(n).fill(d.price),
       borderColor: '#E74C3C', borderDash: [6,4], borderWidth: 1.5, pointRadius: 0, order: 0 }});
  new Chart(document.getElementById('chartImplied'), {{
    data: {{ labels: d.labels, datasets: ds }},
    options: {{
      indexAxis: 'x', responsive: true,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: function(c) {{
          var v = c.raw;
          if (Array.isArray(v)) return 'Range: $' + v[0].toFixed(2) + ' → $' + v[1].toFixed(2);
          return v == null ? null : c.dataset.label + ': $' + Number(v).toFixed(2);
        }} }} }}
      }},
      scales: {{
        x: {{ grid: {{ color: T.grid }}, ticks: {{ color: T.tick, font: {{ size: 11 }} }} }},
        y: {{ grid: {{ color: T.grid }}, ticks: {{ color: T.tick, callback: function(v) {{ return '$' + v; }} }} }}
      }}
    }}
  }});
}})();
</script>""")

    # ── Per-period detail ─────────────────────────────────────────────────────
    for p in periods:
        sentiment = p.pe_sentiment or ""
        badge_cls = {"Bullish": "bull", "Bearish": "bear", "Neutral": "flat"}.get(sentiment, "flat")
        badge = (f"<span class='badge {badge_cls}'>{_e(sentiment)} vs TTM P/E</span>"
                 if sentiment else "")

        def row(label, rev, rev_px, eps, eps_px, fwd_pe, hl=False):
            up_r = _upside(rev_px, price)
            up_e = _upside(eps_px, price)
            return (f"<tr class='{'hl' if hl else ''}'><td>{label}</td>"
                    f"<td>{_billions(rev)}</td><td>{_money(rev_px)}</td>"
                    f"<td class='{_tone(up_r)}'>{_pct(up_r)}</td>"
                    f"<td>{'N/A' if _num(eps) is None else f'${_num(eps):.2f}'}</td>"
                    f"<td>{_mult(fwd_pe)}</td><td>{_money(eps_px)}</td>"
                    f"<td class='{_tone(up_e)}'>{_pct(up_e)}</td></tr>")

        parts.append(f"""
<div class='section-heading'>{_e(p.period)} ({_e(p.period_date)})</div>
<div class='card'>
  <div style='display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px'>
    <div class='card-title'>Consensus estimates</div>{badge}
  </div>
  <div class='table-wrap'><table class='data-table'>
    <thead>
      <tr><th></th><th>Revenue</th><th>Implied $</th><th>Upside</th>
          <th>EPS</th><th>Fwd P/E</th><th>Implied $</th><th>Upside</th></tr>
    </thead>
    <tbody>
      {row('Low',  p.rev_low,  p.rev_price_low,  p.eps_low,  p.eps_price_low,  p.fwd_pe_high)}
      {row('Mean', p.rev_mean, p.rev_price_mean, p.eps_mean, p.eps_price_mean, p.fwd_pe_mean, hl=True)}
      {row('High', p.rev_high, p.rev_price_high, p.eps_high, p.eps_price_high, p.fwd_pe_low)}
    </tbody>
  </table></div>
  <div class='rules'>
    Revenue growth: <b class='{_tone(p.rev_growth_pct)}'>{_pct(p.rev_growth_pct)}</b> ({int(p.rev_num_analysts or 0)} analysts)
    &nbsp;·&nbsp; EPS growth: <b class='{_tone(p.eps_growth_pct)}'>{_pct(p.eps_growth_pct)}</b> ({int(p.eps_num_analysts or 0)} analysts)
    &nbsp;·&nbsp; TTM P/E: <b>{_mult(r.ttm_pe)}</b>
  </div>
</div>""")

    if not periods:
        parts.append("<div class='section-heading'>Forward Estimates</div>"
                     "<div class='card'><p class='no-data'>No revenue or EPS estimates "
                     "available for this ticker.</p></div>")

    # ── Analyst targets ───────────────────────────────────────────────────────
    parts.append("<div class='section-heading'>Analyst Price Targets</div>")
    if has_targets:
        n = int(at.num_analysts or 0)
        parts.append("<div class='metric-grid four'>")
        for label, val, up in (("Low", at.low, at.upside_low), ("Mean", at.mean, at.upside_mean),
                               ("Median", at.median, at.upside_median), ("High", at.high, at.upside_high)):
            parts.append(_metric(f"{label} target", _money(val),
                                 f"<span class='{_tone(up)}'>{_pct(up)}</span> vs current"))
        total = at.strong_buy + at.buy + at.hold + at.sell + at.strong_sell
        parts.append("</div>")
        if total:
            segs = [("Strong Buy", at.strong_buy, "#1E8449"), ("Buy", at.buy, "#27AE60"),
                    ("Hold", at.hold, "#B7791F"), ("Sell", at.sell, "#E67E22"),
                    ("Strong Sell", at.strong_sell, "#C0392B")]
            bar = "".join(f"<span style='width:{c / total * 100:.2f}%;background:{col}' title='{l}: {c}'></span>"
                          for l, c, col in segs if c)
            legend = "".join(f"<span><i style='background:{col}'></i>{l} {c}</span>" for l, c, col in segs)
            parts.append(f"""
<div class='card' style='margin-top:10px'>
  <div class='card-title'>Consensus: {_e(at.consensus)}</div>
  <div class='card-desc'>{total} ratings this month{f' · {n} analysts with price targets' if n else ''}</div>
  <div class='rec-bar'>{bar}</div>
  <div class='legend'>{legend}</div>
</div>""")
        elif n:
            parts.append(f"<p class='no-data'>{n} analysts with price targets; no rating breakdown available.</p>")
    else:
        parts.append("<div class='card'><p class='no-data'>No analyst price targets available.</p></div>")

    subtitle = f"Forward Estimates &nbsp;·&nbsp; Generated {datetime.now():%B %d, %Y}"
    return _page(f"{r.ticker} — Estimates", _header(r.ticker, subtitle, r.company_name),
                 "".join(parts),
                 "⚠️ Analyst consensus from Yahoo Finance. Implied prices assume today's multiples hold — "
                 "they are scenarios, not forecasts. Not financial advice.",
                 charts=bool(cats))


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def render_backtest(ticker: str, timeframe: str = "weekly", stop: str = "none") -> str:
    """
    Strategy Lab: RSI mean-reversion Strategy A and B (MACD filter) for RSI
    2/3/4, plus MACD Histogram and MACD Crossover + MA50 as references, compared
    on the same data, window and costs (see regime_lab.py and rsi2/ab.py).
    stop: a rsi2.ab.STOP_PRESETS key used for the A/B runs; the report also
    compares every stop preset. Raises RuntimeError on bad ticker.
    """
    import lab_report
    import regime_lab
    return lab_report.render(regime_lab.run_lab(ticker, timeframe, stop=stop))
