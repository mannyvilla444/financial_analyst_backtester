"""
html_report.py

Generates a self-contained, mobile-friendly HTML financial analysis report.
Replaces the PDF report for TradingView CSV uploads.

Charts included:
  - Quarterly Revenue / EBITDA / EPS / FCF grouped by Q1-Q4
  - EBITDA Margin % history (line)
  - Revenue / EBITDA / FCF Multiple history (line) with current marker
  - Price Predictions table with Current multiple row
"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fv(val, fmt=".2f") -> str:
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        return format(float(val), fmt)
    except Exception:
        return "N/A"


def _fb(val) -> str:
    try:
        v = float(val)
        if math.isnan(v): return "N/A"
        return f"${v / 1e9:.2f}B"
    except Exception:
        return "N/A"


def _fpct(val) -> str:
    try:
        v = float(val)
        if math.isnan(v): return "N/A"
        return f"{v:.1f}%"
    except Exception:
        return "N/A"


def _fx(val) -> str:
    try:
        v = float(val)
        if math.isnan(v): return "N/A"
        return f"{v:.2f}x"
    except Exception:
        return "N/A"


def _spct(val) -> str:
    try:
        v = float(val)
        if math.isnan(v): return "N/A"
        return f"{v:+.1f}%"
    except Exception:
        return "N/A"


def _color_pct(val) -> str:
    try:
        v = float(val)
        if math.isnan(v): return "neutral"
        return "positive" if v >= 0 else "negative"
    except Exception:
        return "neutral"


def _quarter_num(ts) -> int:
    try:
        return (pd.Timestamp(ts).month - 1) // 3 + 1
    except Exception:
        return 0


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return None if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return None


# ── Chart data builders ───────────────────────────────────────────────────────

YEAR_COLORS = [
    "#2E86AB", "#27AE60", "#E67E22", "#8E44AD",
    "#1ABC9C", "#E74C3C", "#F39C12", "#2980B9",
]


def _build_quarterly_grouped(df: pd.DataFrame, col: str, label: str,
                              divisor: float = 1e9, unit: str = "B") -> dict:
    """Grouped bar: x=Q1..Q4, one dataset per calendar year."""
    sub = df[df[col].notna() & (df[col] != 0)].copy()
    if sub.empty:
        return {}

    sub["_year"] = sub["time"].apply(lambda t: pd.Timestamp(t).year)
    sub["_qnum"] = sub["time"].apply(_quarter_num)
    sub = sub[sub["_qnum"].between(1, 4)]

    # Monthly CSVs can have multiple rows per quarter — keep the last per (year, quarter)
    sub = sub.sort_values("time").groupby(["_year", "_qnum"], as_index=False).last()

    years = sorted(sub["_year"].unique())
    datasets = []
    for idx, yr in enumerate(years):
        yr_data = sub[sub["_year"] == yr].set_index("_qnum")[col]
        vals = []
        for q in range(1, 5):
            if q in yr_data.index:
                raw = float(yr_data[q])
                vals.append(round(raw / divisor, 3) if divisor != 1 else round(raw, 3))
            else:
                vals.append(None)
        datasets.append({
            "label": str(yr),
            "data": vals,
            "backgroundColor": YEAR_COLORS[idx % len(YEAR_COLORS)],
            "borderColor": YEAR_COLORS[idx % len(YEAR_COLORS)],
            "borderWidth": 1,
            "borderRadius": 4,
        })

    return {
        "type": "bar",
        "labels": ["Q1", "Q2", "Q3", "Q4"],
        "datasets": datasets,
        "title": label,
        "yLabel": f"${unit}",
    }


def _build_multiple_line(df: pd.DataFrame, col: str, title: str,
                         color: str, current_val: float | None = None) -> dict:
    """
    Historical multiple line chart.
    Annotates current value with a dashed horizontal line.
    """
    if col not in df.columns:
        return {}

    # Filter to rows with valid, finite, positive multiple
    sub = df[df[col].notna()].copy()
    sub = sub[sub[col].apply(lambda v: _safe_float(v) is not None and _safe_float(v) > 0)]
    if sub.empty:
        return {}

    labels = [pd.Timestamp(t).strftime("%b '%y") for t in sub["time"]]
    values = [round(float(v), 2) for v in sub[col]]

    datasets = [{
        "label": title,
        "data": values,
        "borderColor": color,
        "backgroundColor": f"{color}18",
        "fill": True,
        "tension": 0.3,
        "pointRadius": 3,
        "pointHoverRadius": 6,
        "borderWidth": 2,
    }]

    # Add current value as a flat annotation dataset if available
    if current_val is not None:
        datasets.append({
            "label": f"Current ({current_val:.2f}x)",
            "data": [current_val] * len(values),
            "borderColor": "#E74C3C",
            "backgroundColor": "transparent",
            "fill": False,
            "tension": 0,
            "pointRadius": 0,
            "borderWidth": 1.5,
            "borderDash": [6, 4],
        })

    return {
        "type": "line",
        "labels": labels,
        "datasets": datasets,
        "title": title,
        "yLabel": "x",
    }


def _build_ebitda_margin_line(df: pd.DataFrame) -> dict:
    """EBITDA Margin % historical line."""
    col = "EBITDA_Margin_TTM"
    if col not in df.columns:
        return {}
    sub = df[df[col].notna()].copy()
    if sub.empty:
        return {}
    labels = [pd.Timestamp(t).strftime("%b '%y") for t in sub["time"]]
    values = [round(float(v), 2) for v in sub[col]]
    return {
        "type": "line",
        "labels": labels,
        "datasets": [{
            "label": "EBITDA Margin % (TTM)",
            "data": values,
            "borderColor": "#2E86AB",
            "backgroundColor": "rgba(46, 134, 171, 0.1)",
            "fill": True,
            "tension": 0.3,
            "pointRadius": 3,
            "pointHoverRadius": 6,
            "borderWidth": 2,
        }],
        "title": "EBITDA Margin % Over Time",
        "yLabel": "%",
    }


# ── JavaScript chart renderer ─────────────────────────────────────────────────

def _js_chart(data: dict, canvas_id: str) -> str:
    if not data:
        return "<p class='no-data'>No data available for this chart.</p>"
    d_json = json.dumps(data, allow_nan=False)
    return f"""
<canvas id="{canvas_id}" role="img" aria-label="{data.get('title','Chart')}"></canvas>
<script>
(function() {{
  var raw = {d_json};
  var ctx = document.getElementById('{canvas_id}').getContext('2d');
  var isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  var gridClr = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)';
  var tickClr = isDark ? '#aaa' : '#555';
  var dss = raw.datasets.map(function(ds) {{
    if (ds.borderDash) ds.borderDash = ds.borderDash;
    return ds;
  }});
  new Chart(ctx, {{
    type: raw.type,
    data: {{ labels: raw.labels, datasets: dss }},
    options: {{
      responsive: true,
      maintainAspectRatio: true,
      plugins: {{
        legend: {{
          position: 'bottom',
          labels: {{ color: tickClr, font: {{ size: 12 }}, boxWidth: 14, padding: 10 }}
        }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              var v = ctx.parsed.y;
              if (v === null || v === undefined) return null;
              var u = raw.yLabel || '';
              if (u === '%') return ctx.dataset.label + ': ' + v.toFixed(1) + '%';
              if (u === 'x') return ctx.dataset.label + ': ' + v.toFixed(2) + 'x';
              if (u === '$B') return ctx.dataset.label + ': $' + v.toFixed(2) + 'B';
              return ctx.dataset.label + ': ' + u + v.toFixed(2);
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          grid: {{ color: gridClr }},
          ticks: {{ color: tickClr, font: {{ size: 11 }}, maxRotation: 45 }}
        }},
        y: {{
          grid: {{ color: gridClr }},
          ticks: {{
            color: tickClr,
            font: {{ size: 11 }},
            callback: function(v) {{
              var u = raw.yLabel || '';
              if (u === '%') return v + '%';
              if (u === 'x') return v + 'x';
              if (u === '$B') return '$' + v + 'B';
              return u + v;
            }}
          }}
        }}
      }}
    }}
  }});
}})();
</script>"""


# ── HTML builder ──────────────────────────────────────────────────────────────

def generate_html(ticker: str,
                  df_ttm: pd.DataFrame,
                  metrics: dict,
                  predictions: pd.DataFrame | None) -> str:
    """
    Generate a self-contained mobile-friendly HTML report.
    Returns the HTML as a string.
    """
    gen_date = datetime.now().strftime("%B %d, %Y")

    # ── Metric card values ────────────────────────────────────────────────────
    rev_ttm       = _fb(metrics.get("Revenue_TTM"))
    rev_growth    = _spct(metrics.get("Revenue_Growth_YoY"))
    rev_g_cls     = _color_pct(metrics.get("Revenue_Growth_YoY"))
    gp_margin     = _fpct(metrics.get("Gross_Profit_Margin"))
    ebitda_margin = _fpct(metrics.get("EBITDA_Margin"))
    fcf_margin    = _fpct(metrics.get("FCF_Margin"))
    fcf_ttm_val   = _fb(metrics.get("FCF_TTM"))
    nd_ebitda     = _fx(metrics.get("Net_Debt_to_EBITDA"))
    curr_price    = f"${_fv(metrics.get('Current_Price'))}"
    mkt_cap       = _fb(metrics.get("Current_Mkt_Cap"))
    eps_ttm       = _fv(metrics.get("EPS_TTM"))
    pe_ratio      = _fx(metrics.get("PE_Ratio"))
    rev_surprise  = _spct(metrics.get("Revenue_Surprise_Pct"))
    rev_s_cls     = _color_pct(metrics.get("Revenue_Surprise_Pct"))

    price_date_html = ""
    if metrics.get("Price_Date") is not None:
        try:
            price_date_html = f" &nbsp;·&nbsp; {pd.Timestamp(metrics['Price_Date']).strftime('%b %d, %Y')}"
        except Exception:
            pass

    ticker_date = metrics.get("ticker_date", "")
    try:
        ticker_date = pd.Timestamp(ticker_date).strftime("%b %d, %Y")
    except Exception:
        ticker_date = str(ticker_date)[:10]

    # ── Current multiples (latest non-null value from df_ttm) ─────────────────
    def _current_multiple(col):
        if col not in df_ttm.columns:
            return None
        # The latest value IS the current multiple. (Previously this skipped
        # back to the most recent *positive* value, so a company whose TTM
        # EBITDA/FCF had turned negative showed an old multiple as "current".)
        sub = df_ttm[df_ttm[col].notna()]
        if sub.empty:
            return None
        v = _safe_float(sub.iloc[-1][col])
        return v if v is not None and v > 0 else None

    curr_rev_mult   = _current_multiple("Market_Cap_to_Revenue")
    curr_ebi_mult   = _current_multiple("Market_Cap_to_EBITDA")
    curr_fcf_mult   = _current_multiple("Market_Cap_to_FCF")

    # ── Price predictions table with "Current" row ────────────────────────────
    # Build the "Current" row from live multiples and current price
    curr_price_raw = _safe_float(metrics.get("Current_Price"))

    def _current_pred_row():
        def cell(mult_val, ttm_col):
            if mult_val is None:
                return "N/A", "N/A", _fx(None)
            # Current multiple IS the scenario — upside is 0% by definition
            return f"${curr_price_raw:.2f}" if curr_price_raw else "N/A", "—", _fx(mult_val)

        rev_p, rev_u, rev_m   = cell(curr_rev_mult, "Revenue_TTM")
        ebi_p, ebi_u, ebi_m   = cell(curr_ebi_mult, "EBITDA_TTM")
        fcf_p, fcf_u, fcf_m   = cell(curr_fcf_mult, "FCF_TTM")

        return f"""
            <tr class="current-row">
              <td class="scenario">&#9733; Current</td>
              <td class="price">{rev_p}</td>
              <td class="neutral">{rev_u}</td>
              <td class="mult current-mult">{rev_m}</td>
              <td class="price">{ebi_p}</td>
              <td class="neutral">{ebi_u}</td>
              <td class="mult current-mult">{ebi_m}</td>
              <td class="price">{fcf_p}</td>
              <td class="neutral">{fcf_u}</td>
              <td class="mult current-mult">{fcf_m}</td>
            </tr>"""

    pred_rows_html = _current_pred_row()

    if predictions is not None and not predictions.empty:
        for _, row in predictions.iterrows():
            scenario = row.get("Scenario", "")

            rev_p  = row.get("Price (Rev)", "N/A")
            rev_u  = row.get("Upside (Rev)", "N/A")
            rev_m  = row.get("Rev Multiple", "N/A")
            rev_ucls = "positive" if (isinstance(rev_u, str) and rev_u.startswith("+")) else \
                       "negative" if (isinstance(rev_u, str) and rev_u.startswith("-")) else "neutral"

            ebi_p  = row.get("Price (EBITDA)", "N/A")
            ebi_u  = row.get("Upside (EBITDA)", "N/A")
            ebi_m  = row.get("EBITDA Multiple", "N/A")
            ebi_ucls = "positive" if (isinstance(ebi_u, str) and ebi_u.startswith("+")) else \
                       "negative" if (isinstance(ebi_u, str) and ebi_u.startswith("-")) else "neutral"

            fcf_p  = row.get("Price (FCF)", "N/A")
            fcf_u  = row.get("Upside (FCF)", "N/A")
            fcf_m  = row.get("FCF Multiple", "N/A")
            fcf_ucls = "positive" if (isinstance(fcf_u, str) and fcf_u.startswith("+")) else \
                       "negative" if (isinstance(fcf_u, str) and fcf_u.startswith("-")) else "neutral"

            pred_rows_html += f"""
            <tr>
              <td class="scenario">{scenario}</td>
              <td class="price">{rev_p}</td>
              <td class="{rev_ucls}">{rev_u}</td>
              <td class="mult">{rev_m}</td>
              <td class="price">{ebi_p}</td>
              <td class="{ebi_ucls}">{ebi_u}</td>
              <td class="mult">{ebi_m}</td>
              <td class="price">{fcf_p}</td>
              <td class="{fcf_ucls}">{fcf_u}</td>
              <td class="mult">{fcf_m}</td>
            </tr>"""

    # ── Chart data ─────────────────────────────────────────────────────────────
    rev_chart    = _build_quarterly_grouped(df_ttm, "Quarterly Revenue",
                                            "Quarterly Revenue", 1e9, "B")
    ebitda_chart = _build_quarterly_grouped(df_ttm, "EBITDA",
                                            "Quarterly EBITDA", 1e9, "B")
    eps_chart    = _build_quarterly_grouped(df_ttm, "EPS estimates",
                                            "Quarterly EPS", 1, "$")
    fcf_chart    = _build_quarterly_grouped(df_ttm, "Free cash flow",
                                            "Quarterly Free Cash Flow", 1e9, "B")
    margin_chart = _build_ebitda_margin_line(df_ttm)

    rev_mult_chart = _build_multiple_line(df_ttm, "Market_Cap_to_Revenue",
                                          "Mkt Cap / Revenue Multiple",
                                          "#2E86AB", curr_rev_mult)
    ebi_mult_chart = _build_multiple_line(df_ttm, "Market_Cap_to_EBITDA",
                                          "Mkt Cap / EBITDA Multiple",
                                          "#27AE60", curr_ebi_mult)
    fcf_mult_chart = _build_multiple_line(df_ttm, "Market_Cap_to_FCF",
                                          "Mkt Cap / FCF Multiple",
                                          "#E67E22", curr_fcf_mult)

    rev_chart_html     = _js_chart(rev_chart,     "chartRevenue")
    ebitda_chart_html  = _js_chart(ebitda_chart,  "chartEbitda")
    eps_chart_html     = _js_chart(eps_chart,      "chartEps")
    fcf_chart_html     = _js_chart(fcf_chart,      "chartFcf")
    margin_chart_html  = _js_chart(margin_chart,   "chartMargin")
    rev_mult_html      = _js_chart(rev_mult_chart, "chartRevMult")
    ebi_mult_html      = _js_chart(ebi_mult_chart, "chartEbiMult")
    fcf_mult_html      = _js_chart(fcf_mult_chart, "chartFcfMult")

    # ── Current multiple summary cards ────────────────────────────────────────
    curr_rev_card  = _fx(curr_rev_mult)
    curr_ebi_card  = _fx(curr_ebi_mult)
    curr_fcf_card  = _fx(curr_fcf_mult)

    # ── Full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker} — Financial Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #f5f7fa; --card: #ffffff; --border: #e2e8f0;
    --text: #1a202c; --muted: #718096; --accent: #2E86AB;
    --green: #27AE60; --red: #E74C3C; --orange: #E67E22;
    --radius: 12px; --shadow: 0 2px 8px rgba(0,0,0,0.07);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0f1117; --card: #1a1d27; --border: #2d3148;
      --text: #e2e8f0; --muted: #a0aec0;
    }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 15px; line-height: 1.5;
    padding: 16px; max-width: 900px; margin: 0 auto;
  }}
  .header {{
    background: linear-gradient(135deg, #1a365d 0%, #2E86AB 100%);
    border-radius: var(--radius); padding: 24px 20px 20px;
    margin-bottom: 20px; color: white;
  }}
  .header-ticker {{ font-size: 32px; font-weight: 700; letter-spacing: -0.5px; }}
  .header-sub {{ font-size: 13px; opacity: 0.8; margin-top: 4px; }}
  .section-heading {{
    font-size: 13px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--muted);
    margin: 28px 0 12px; padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }}
  .metric-grid {{
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 4px;
  }}
  @media (min-width: 520px) {{ .metric-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
  @media (min-width: 720px) {{ .metric-grid {{ grid-template-columns: repeat(4, 1fr); }} }}
  .metric-card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 14px 12px; box-shadow: var(--shadow);
  }}
  .metric-label {{
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--muted); margin-bottom: 6px;
  }}
  .metric-value {{ font-size: 20px; font-weight: 700; color: var(--text); line-height: 1.1; }}
  .metric-sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .positive {{ color: var(--green) !important; }}
  .negative {{ color: var(--red) !important; }}
  .neutral  {{ color: var(--muted) !important; }}
  .chart-card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px 16px 16px;
    box-shadow: var(--shadow); margin-bottom: 14px;
  }}
  .chart-title {{ font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: 4px; }}
  .chart-desc {{ font-size: 12px; color: var(--muted); margin-bottom: 16px; }}
  canvas {{ max-height: 320px; }}
  .no-data {{ color: var(--muted); font-size: 13px; padding: 12px 0; }}
  .table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
  .pred-table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
    min-width: 520px; background: var(--card);
    border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow);
  }}
  .pred-table th {{
    background: #1a365d; color: white; padding: 10px;
    text-align: center; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap;
  }}
  .pred-table th.group-header {{ background: #2E86AB; font-size: 12px; }}
  .pred-table td {{
    padding: 9px 10px; text-align: center;
    border-bottom: 1px solid var(--border); white-space: nowrap;
  }}
  .pred-table tr:last-child td {{ border-bottom: none; }}
  .pred-table tr:nth-child(even) td {{ background: rgba(0,0,0,0.02); }}
  @media (prefers-color-scheme: dark) {{
    .pred-table tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
  }}
  .pred-table tr.current-row td {{
    background: rgba(46, 134, 171, 0.08) !important;
    font-weight: 600;
  }}
  .pred-table td.scenario {{ font-weight: 600; text-align: left; color: var(--accent); }}
  .pred-table td.price {{ font-weight: 600; }}
  .pred-table td.mult {{ color: var(--muted); font-size: 12px; }}
  .pred-table td.current-mult {{ color: var(--accent) !important; font-weight: 700; font-size: 13px; }}
  .disclaimer {{
    font-size: 11px; color: var(--muted); text-align: center;
    padding: 20px 16px 8px; border-top: 1px solid var(--border); margin-top: 28px;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-ticker">{ticker}</div>
  <div class="header-sub">Financial Analysis &nbsp;·&nbsp; Financials as of {ticker_date} &nbsp;·&nbsp; Generated {gen_date}</div>
</div>

<!-- Snapshot -->
<div class="section-heading">Snapshot</div>
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Current Price</div>
    <div class="metric-value">{curr_price}</div>
    <div class="metric-sub">Mkt Cap: {mkt_cap}{price_date_html}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Revenue TTM</div>
    <div class="metric-value">{rev_ttm}</div>
    <div class="metric-sub">YoY Growth: <span class="{rev_g_cls}">{rev_growth}</span></div>
  </div>
  <div class="metric-card">
    <div class="metric-label">EPS TTM</div>
    <div class="metric-value">${eps_ttm}</div>
    <div class="metric-sub">P/E Ratio: {pe_ratio}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Rev vs Estimates</div>
    <div class="metric-value {rev_s_cls}">{rev_surprise}</div>
    <div class="metric-sub">Surprise %</div>
  </div>
</div>

<!-- Profitability -->
<div class="section-heading">Profitability (TTM)</div>
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Gross Margin</div>
    <div class="metric-value">{gp_margin}</div>
    <div class="metric-sub">Capital to create product</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">EBITDA Margin</div>
    <div class="metric-value">{ebitda_margin}</div>
    <div class="metric-sub">Before debt &amp; capex</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">FCF Margin</div>
    <div class="metric-value">{fcf_margin}</div>
    <div class="metric-sub">After all capex</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">FCF TTM</div>
    <div class="metric-value">{fcf_ttm_val}</div>
    <div class="metric-sub">Free cash flow</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Net Debt / EBITDA</div>
    <div class="metric-value">{nd_ebitda}</div>
    <div class="metric-sub">Leverage ratio</div>
  </div>
</div>

<!-- Current Multiples -->
<div class="section-heading">Current Valuation Multiples</div>
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Mkt Cap / Revenue</div>
    <div class="metric-value" style="color:var(--accent)">{curr_rev_card}</div>
    <div class="metric-sub">Revenue multiple</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Mkt Cap / EBITDA</div>
    <div class="metric-value" style="color:var(--green)">{curr_ebi_card}</div>
    <div class="metric-sub">EBITDA multiple</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Mkt Cap / FCF</div>
    <div class="metric-value" style="color:var(--orange)">{curr_fcf_card}</div>
    <div class="metric-sub">FCF multiple</div>
  </div>
</div>

<!-- Revenue Chart -->
<div class="section-heading">Quarterly Revenue</div>
<div class="chart-card">
  <div class="chart-title">Revenue — Q1 / Q2 / Q3 / Q4 Grouped by Year</div>
  <div class="chart-desc">Same quarter across years — shows seasonality and YoY growth within each quarter.</div>
  {rev_chart_html}
</div>

<!-- EBITDA Chart -->
<div class="section-heading">Quarterly EBITDA</div>
<div class="chart-card">
  <div class="chart-title">EBITDA — Q1 / Q2 / Q3 / Q4 Grouped by Year</div>
  <div class="chart-desc">Earnings before interest, taxes, depreciation &amp; amortization per quarter.</div>
  {ebitda_chart_html}
</div>

<!-- EPS Chart -->
<div class="section-heading">Quarterly EPS</div>
<div class="chart-card">
  <div class="chart-title">EPS — Q1 / Q2 / Q3 / Q4 Grouped by Year</div>
  <div class="chart-desc">Earnings per share per quarter — reveals seasonal patterns and year-over-year earnings growth.</div>
  {eps_chart_html}
</div>

<!-- FCF Chart -->
<div class="section-heading">Quarterly Free Cash Flow</div>
<div class="chart-card">
  <div class="chart-title">FCF — Q1 / Q2 / Q3 / Q4 Grouped by Year</div>
  <div class="chart-desc">Free cash flow per quarter after all capital expenditures — the truest measure of cash generation.</div>
  {fcf_chart_html}
</div>

<!-- EBITDA Margin History -->
<div class="section-heading">EBITDA Margin % History</div>
<div class="chart-card">
  <div class="chart-title">EBITDA Margin % (TTM) Over Time</div>
  <div class="chart-desc">Rolling TTM margin — expanding trend signals improving operational leverage.</div>
  {margin_chart_html}
</div>

<!-- Multiple History Charts -->
<div class="section-heading">Historical Valuation Multiples</div>
<div class="chart-card">
  <div class="chart-title">Revenue Multiple History (Mkt Cap / Revenue TTM)</div>
  <div class="chart-desc">Red dashed line = current multiple. Where the current multiple sits vs history shows whether the stock is cheap or expensive on revenue.</div>
  {rev_mult_html}
</div>
<div class="chart-card">
  <div class="chart-title">EBITDA Multiple History (Mkt Cap / EBITDA TTM)</div>
  <div class="chart-desc">Red dashed line = current multiple. Tells you how the market is pricing the company's operating profit today vs historically.</div>
  {ebi_mult_html}
</div>
<div class="chart-card">
  <div class="chart-title">FCF Multiple History (Mkt Cap / FCF TTM)</div>
  <div class="chart-desc">Red dashed line = current multiple. FCF multiple is the most reliable valuation anchor — shows whether the market is paying a premium or discount for cash flow.</div>
  {fcf_mult_html}
</div>

<!-- Price Predictions -->
<div class="section-heading">Price Predictions by Historical Multiple</div>
<div class="table-wrap">
  <table class="pred-table">
    <colgroup>
      <col style="width:72px"><col><col><col><col><col><col><col><col><col>
    </colgroup>
    <thead>
      <tr>
        <th rowspan="2">Scenario</th>
        <th colspan="3" class="group-header">Revenue Multiple</th>
        <th colspan="3" class="group-header">EBITDA Multiple</th>
        <th colspan="3" class="group-header">FCF Multiple</th>
      </tr>
      <tr>
        <th>Price</th><th>Upside</th><th>Multiple</th>
        <th>Price</th><th>Upside</th><th>Multiple</th>
        <th>Price</th><th>Upside</th><th>Multiple</th>
      </tr>
    </thead>
    <tbody>
      {pred_rows_html}
    </tbody>
  </table>
</div>

<div class="disclaimer">
  ⚠️ Data from TradingView CSV export. Not financial advice.
  Predictions use historical valuation ranges — not forecasts.
  ★ Current row shows today's actual multiples.
</div>

</body>
</html>"""

    return html
