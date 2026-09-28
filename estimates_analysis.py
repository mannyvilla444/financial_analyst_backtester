"""
estimates_analysis.py

Fetches forward estimates from yfinance and computes:

1. REVENUE-BASED PRICE TARGETS
   For each revenue estimate period (current year, next year):
   - If revenue hits consensus estimate AND the Mkt Cap/Revenue multiple
     stays the same → implied price
   - Range: low analyst estimate → high analyst estimate

2. EPS-BASED FORWARD P/E ANALYSIS
   - Forward P/E at current price for each estimate period
   - Compare to TTM P/E (trailing) → bullish or bearish signal
   - Implied price if forward P/E stays the same as today's TTM P/E
   - Implied price range using low/high EPS estimates

3. ANALYST PRICE TARGETS
   - Low, mean, median, high targets
   - Upside/downside vs current price
   - Number of analysts, buy/hold/sell breakdown
"""

import warnings
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_estimates_data(ticker: str) -> dict:
    """
    Pull all estimates and valuation data from yfinance.
    Returns a rich dict of raw data.
    Raises RuntimeError on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    t = yf.Ticker(ticker)

    # Fetch each field individually so one missing attribute doesn't
    # block everything else — each field falls back to None gracefully.
    def _get(attr):
        try: return getattr(t, attr)
        except Exception: return None

    def _has_data(v):
        """Return True if v is a non-empty DataFrame/dict/Series, False otherwise.
        Never uses bare bool(v) which raises on DataFrames."""
        if v is None:
            return False
        import pandas as pd
        if isinstance(v, (pd.DataFrame, pd.Series)):
            return not v.empty
        if isinstance(v, dict):
            return len(v) > 0
        return v is not None

    def _first(*attrs):
        """Return the first attribute that has data, never using bare bool()."""
        for attr in attrs:
            v = _get(attr)
            if _has_data(v):
                return v
        return None

    try:
        info = t.info
    except Exception as e:
        raise RuntimeError(f"yfinance fetch failed for '{ticker}': {e}")

    fast          = _get("fast_info")
    income_annual = _get("income_stmt")
    income_q      = _get("quarterly_income_stmt")
    analyst_price = _get("analyst_price_targets")
    # recommendations_summary is the correct attribute in current yfinance
    recommendations = _first("recommendations_summary", "recommendations")
    eps_trend     = _get("eps_trend")
    revenue_est   = _get("revenue_estimate")
    # earnings_estimate is the correct attribute (not eps_estimates)
    eps_est       = _first("earnings_estimate", "eps_estimates")

    if info is None or not info:
        raise RuntimeError(f"No data found for '{ticker}'. Check the ticker symbol.")

    return {
        "info":           info,
        "fast":           fast,
        "income_annual":  income_annual,
        "income_q":       income_q,
        "analyst_price":  analyst_price,
        "recommendations": recommendations,
        "eps_trend":      eps_trend,
        "revenue_est":    revenue_est,
        "eps_est":        eps_est,
    }


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PeriodEstimate:
    """Estimates for a single forward period (e.g. current year, next year)."""
    period:           str        # "Current Year", "Next Year"
    period_date:      str        # e.g. "Dec 2025"

    # Revenue
    rev_low:          Optional[float] = None
    rev_mean:         Optional[float] = None
    rev_high:         Optional[float] = None
    rev_num_analysts: int = 0
    rev_growth_pct:   Optional[float] = None   # YoY growth vs TTM

    # Revenue-implied prices (multiple held constant)
    rev_price_low:    Optional[float] = None
    rev_price_mean:   Optional[float] = None
    rev_price_high:   Optional[float] = None

    # EPS
    eps_low:          Optional[float] = None
    eps_mean:         Optional[float] = None
    eps_high:         Optional[float] = None
    eps_num_analysts: int = 0
    eps_growth_pct:   Optional[float] = None   # YoY growth vs TTM EPS

    # Forward P/E at consensus EPS
    fwd_pe_low:       Optional[float] = None
    fwd_pe_mean:      Optional[float] = None
    fwd_pe_high:      Optional[float] = None

    # EPS-implied prices (TTM P/E held constant)
    eps_price_low:    Optional[float] = None
    eps_price_mean:   Optional[float] = None
    eps_price_high:   Optional[float] = None

    # Sentiment
    pe_sentiment:     str = ""    # "Bullish", "Bearish", "Neutral"
    pe_compression:   Optional[float] = None   # fwd_pe_mean vs ttm_pe


@dataclass
class AnalystTargets:
    low:          Optional[float] = None
    mean:         Optional[float] = None
    median:       Optional[float] = None
    high:         Optional[float] = None
    num_analysts: int = 0
    upside_low:   Optional[float] = None
    upside_mean:  Optional[float] = None
    upside_median:Optional[float] = None
    upside_high:  Optional[float] = None
    strong_buy:   int = 0
    buy:          int = 0
    hold:         int = 0
    sell:         int = 0
    strong_sell:  int = 0

    @property
    def bullish_count(self): return self.strong_buy + self.buy
    @property
    def bearish_count(self): return self.sell + self.strong_sell
    @property
    def consensus(self):
        total = self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell
        if total == 0: return "N/A"
        score = (self.strong_buy*5 + self.buy*4 + self.hold*3 +
                 self.sell*2 + self.strong_sell*1) / total
        if score >= 4.0: return "Strong Buy"
        if score >= 3.5: return "Buy"
        if score >= 2.5: return "Hold"
        if score >= 1.5: return "Sell"
        return "Strong Sell"


@dataclass
class EstimatesResult:
    ticker:          str
    company_name:    str
    current_price:   float
    market_cap:      float
    shares_out:      float
    ttm_revenue:     Optional[float]
    ttm_eps:         Optional[float]
    ttm_pe:          Optional[float]
    current_rev_multiple: Optional[float]   # Mkt Cap / TTM Revenue
    currency:        str = "USD"
    periods:         list = field(default_factory=list)   # list[PeriodEstimate]
    analyst_targets: Optional[AnalystTargets] = None


# ── Analysis engine ───────────────────────────────────────────────────────────

def _safe(d, *keys, default=None):
    """Safe dict/attr access."""
    for k in keys:
        if d is None: return default
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            d = getattr(d, k, default)
        if d is None: return default
    return d


def _find_row(df, keyword):
    if df is None or df.empty: return None
    for idx in df.index:
        if keyword.lower() in str(idx).lower():
            return df.loc[idx]
    return None


def run_estimates_analysis(ticker: str) -> EstimatesResult:
    """
    Full pipeline: fetch → compute → return EstimatesResult.
    """
    data   = fetch_estimates_data(ticker)
    info   = data["info"]
    fast   = data["fast"]

    # ── Current market data ───────────────────────────────────────────────────
    current_price = (
        _safe(fast, "last_price") or
        info.get("currentPrice") or
        info.get("regularMarketPrice", 0.0)
    )
    market_cap = (
        info.get("marketCap") or
        (_safe(fast, "shares") or 0) * current_price
    )
    shares_out = (
        info.get("sharesOutstanding") or
        _safe(fast, "shares") or
        (market_cap / current_price if current_price else 0)
    )
    company_name = info.get("longName") or info.get("shortName") or ticker

    # ── TTM financials ────────────────────────────────────────────────────────
    income_q = data["income_q"]
    ttm_revenue = None
    ttm_eps     = None

    if income_q is not None and not income_q.empty:
        rev_row = _find_row(income_q, "Total Revenue")
        eps_row = _find_row(income_q, "Basic EPS")
        if rev_row is not None:
            cols = sorted(rev_row.index, reverse=True)[:4]
            vals = [float(rev_row[c]) for c in cols if pd.notna(rev_row[c])]
            if vals: ttm_revenue = sum(vals)
        if eps_row is not None:
            cols = sorted(eps_row.index, reverse=True)[:4]
            vals = [float(eps_row[c]) for c in cols if pd.notna(eps_row[c])]
            if vals: ttm_eps = sum(vals)

    # Fallback to info fields
    if ttm_revenue is None:
        ttm_revenue = info.get("totalRevenue")
    if ttm_eps is None:
        ttm_eps = info.get("trailingEps")

    ttm_pe = (current_price / ttm_eps) if (ttm_eps and ttm_eps > 0 and current_price) else None
    current_rev_multiple = (market_cap / ttm_revenue) if (ttm_revenue and ttm_revenue > 0) else None

    # ── Revenue estimates ─────────────────────────────────────────────────────
    rev_est = data["revenue_est"]
    rev_by_period = {}

    if rev_est is not None and not rev_est.empty:
        for period_key in rev_est.index:
            row = rev_est.loc[period_key]
            avg_val = (
                _safe(row, "avg") or
                _safe(row, "mean") or
                _safe(row, "Avg Estimate")
            )
            low_val  = _safe(row, "low")  or _safe(row, "Low Estimate")
            high_val = _safe(row, "high") or _safe(row, "High Estimate")
            n_analysts = int(
                _safe(row, "numberOfAnalysts", default=0) or
                _safe(row, "No. of Analysts", default=0) or 0
            )
            rev_by_period[str(period_key)] = {
                "low":          low_val,
                "avg":          avg_val,
                "high":         high_val,
                "num_analysts": n_analysts,
                "growth":       _safe(row, "growth") or _safe(row, "Growth"),
            }

    # ── EPS estimates ─────────────────────────────────────────────────────────
    # earnings_estimate (current yfinance) uses columns:
    #   avg, low, high, numberOfAnalysts, growth, yearAgoEps
    eps_est = data["eps_est"]
    eps_by_period = {}

    if eps_est is not None and not eps_est.empty:
        # Print columns for debugging
        cols = eps_est.columns.tolist() if hasattr(eps_est, "columns") else []
        for period_key in eps_est.index:
            row = eps_est.loc[period_key]
            # Handle both old (eps_estimates) and new (earnings_estimate) column names
            avg_val = (
                _safe(row, "avg") or
                _safe(row, "mean") or
                _safe(row, "consensus") or
                _safe(row, "Avg Estimate")
            )
            low_val = _safe(row, "low") or _safe(row, "Low Estimate")
            high_val = _safe(row, "high") or _safe(row, "High Estimate")
            n_analysts = int(
                _safe(row, "numberOfAnalysts", default=0) or
                _safe(row, "No. of Analysts", default=0) or 0
            )
            eps_by_period[str(period_key)] = {
                "low":          low_val,
                "avg":          avg_val,
                "high":         high_val,
                "num_analysts": n_analysts,
                "growth":       _safe(row, "growth") or _safe(row, "Growth"),
                "year_ago":     _safe(row, "yearAgoEps") or _safe(row, "Year Ago EPS"),
            }

    # ── Build period estimates ────────────────────────────────────────────────
    # yfinance period keys: '0q'=current qtr, '+1q'=next qtr,
    #                       '0y'=current year, '+1y'=next year
    YEAR_PERIODS = {
        "0y":  "Current Year",
        "+1y": "Next Year",
    }

    periods = []
    for key, label in YEAR_PERIODS.items():
        rv = rev_by_period.get(key, {})
        ep = eps_by_period.get(key, {})

        # Get the calendar year from info if possible
        fiscal_year_end = info.get("fiscalYearEnd", "")
        try:
            yr_offset = 0 if key == "0y" else 1
            year_num  = datetime.now().year + yr_offset
            period_date = f"{year_num}"
        except Exception:
            period_date = label

        pe = PeriodEstimate(period=label, period_date=period_date)

        # Revenue fields
        for attr, src_key in [("rev_low","low"),("rev_mean","avg"),("rev_high","high")]:
            v = rv.get(src_key)
            if v is not None and pd.notna(v):
                setattr(pe, attr, float(v))
        pe.rev_num_analysts = rv.get("num_analysts", 0)
        if rv.get("growth") is not None and pd.notna(rv["growth"]):
            pe.rev_growth_pct = float(rv["growth"]) * 100

        # Revenue-implied prices (same Mkt Cap/Rev multiple)
        if current_rev_multiple and shares_out and shares_out > 0:
            for attr, rev_attr in [
                ("rev_price_low",  "rev_low"),
                ("rev_price_mean", "rev_mean"),
                ("rev_price_high", "rev_high"),
            ]:
                rev_val = getattr(pe, rev_attr)
                if rev_val is not None:
                    implied_mkt_cap = current_rev_multiple * rev_val
                    setattr(pe, attr, implied_mkt_cap / shares_out)

        # EPS fields
        for attr, src_key in [("eps_low","low"),("eps_mean","avg"),("eps_high","high")]:
            v = ep.get(src_key)
            if v is not None and pd.notna(v):
                setattr(pe, attr, float(v))
        pe.eps_num_analysts = ep.get("num_analysts", 0)
        if ep.get("growth") is not None and pd.notna(ep["growth"]):
            pe.eps_growth_pct = float(ep["growth"]) * 100

        # Forward P/E at current price
        if current_price:
            for attr, eps_attr in [
                ("fwd_pe_low",  "eps_high"),  # low P/E = high EPS
                ("fwd_pe_mean", "eps_mean"),
                ("fwd_pe_high", "eps_low"),   # high P/E = low EPS
            ]:
                eps_val = getattr(pe, eps_attr)
                if eps_val and eps_val > 0:
                    setattr(pe, attr, current_price / eps_val)

        # EPS-implied prices (TTM P/E held constant)
        if ttm_pe and ttm_pe > 0:
            for attr, eps_attr in [
                ("eps_price_low",  "eps_low"),
                ("eps_price_mean", "eps_mean"),
                ("eps_price_high", "eps_high"),
            ]:
                eps_val = getattr(pe, eps_attr)
                if eps_val is not None and eps_val > 0:
                    setattr(pe, attr, ttm_pe * eps_val)

        # P/E sentiment vs TTM
        if pe.fwd_pe_mean and ttm_pe:
            pe.pe_compression = pe.fwd_pe_mean - ttm_pe
            if pe.fwd_pe_mean < ttm_pe * 0.90:
                pe.pe_sentiment = "Bullish"     # forward P/E cheaper
            elif pe.fwd_pe_mean > ttm_pe * 1.10:
                pe.pe_sentiment = "Bearish"     # forward P/E more expensive
            else:
                pe.pe_sentiment = "Neutral"

        periods.append(pe)

    # ── Analyst price targets ─────────────────────────────────────────────────
    # analyst_price_targets returns a dict in current yfinance:
    #   {"low": x, "mean": x, "median": x, "high": x, "currentPrice": x}
    analyst_targets = AnalystTargets()
    apt = data["analyst_price"]

    def _safe_float(v):
        """Convert to float safely, returning None if not possible."""
        try:
            if v is None: return None
            if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series): return None
            f = float(v)
            return None if np.isnan(f) else f
        except Exception:
            return None

    if apt is not None:
        try:
            if isinstance(apt, dict):
                analyst_targets.low    = _safe_float(apt.get("low"))
                analyst_targets.mean   = _safe_float(apt.get("mean"))
                analyst_targets.median = _safe_float(apt.get("median"))
                analyst_targets.high   = _safe_float(apt.get("high"))
            elif isinstance(apt, pd.DataFrame) and not apt.empty:
                # Some versions return a single-row DataFrame
                row = apt.iloc[-1]
                for col_name in ["low", "Low"]:
                    if col_name in row.index:
                        analyst_targets.low = _safe_float(row[col_name]); break
                for col_name in ["mean", "Mean"]:
                    if col_name in row.index:
                        analyst_targets.mean = _safe_float(row[col_name]); break
                for col_name in ["median", "Median"]:
                    if col_name in row.index:
                        analyst_targets.median = _safe_float(row[col_name]); break
                for col_name in ["high", "High"]:
                    if col_name in row.index:
                        analyst_targets.high = _safe_float(row[col_name]); break
        except Exception:
            pass

    # Fallback to info fields for anything still missing
    if analyst_targets.low    is None: analyst_targets.low    = _safe_float(info.get("targetLowPrice"))
    if analyst_targets.mean   is None: analyst_targets.mean   = _safe_float(info.get("targetMeanPrice"))
    if analyst_targets.median is None: analyst_targets.median = _safe_float(info.get("targetMedianPrice"))
    if analyst_targets.high   is None: analyst_targets.high   = _safe_float(info.get("targetHighPrice"))
    analyst_targets.num_analysts = int(info.get("numberOfAnalystOpinions", 0) or 0)

    # Upside calcs
    if current_price:
        for attr, tgt_attr in [
            ("upside_low",    "low"),
            ("upside_mean",   "mean"),
            ("upside_median", "median"),
            ("upside_high",   "high"),
        ]:
            tgt = getattr(analyst_targets, tgt_attr)
            if tgt is not None and not isinstance(tgt, (pd.DataFrame, pd.Series)):
                try:
                    setattr(analyst_targets, attr, (float(tgt) / current_price - 1) * 100)
                except Exception:
                    pass

    # Recommendation counts
    recs = data["recommendations"]
    if recs is not None and not recs.empty:
        try:
            # recommendations_summary has columns: strongBuy, buy, hold, sell, strongSell
            # recommendations has columns: period, strongBuy, buy, hold, sell, strongSell
            # Both are ordered newest-first (period 0m, -1m, -2m, -3m), so the
            # current month is row 0, not the last row.
            if "period" in recs.columns and (recs["period"] == "0m").any():
                latest = recs[recs["period"] == "0m"].iloc[0]
            else:
                latest = recs.iloc[0]
            def _rc(row, *keys):
                for k in keys:
                    try:
                        v = row[k] if k in row.index else row.get(k, 0)
                        if v is not None and pd.notna(v): return int(v)
                    except Exception: pass
                return 0
            analyst_targets.strong_buy  = _rc(latest, "strongBuy",  "Strong Buy")
            analyst_targets.buy         = _rc(latest, "buy",        "Buy")
            analyst_targets.hold        = _rc(latest, "hold",       "Hold")
            analyst_targets.sell        = _rc(latest, "sell",       "Sell")
            analyst_targets.strong_sell = _rc(latest, "strongSell", "Strong Sell")
        except Exception:
            pass

    # (Removed an old fallback that overwrote `buy` with the total analyst count
    #  whenever strong_buy was 0 — it inflated the Buy count and the consensus.
    #  With no breakdown available the counts stay 0 and consensus shows "N/A".)

    return EstimatesResult(
        ticker               = ticker.upper(),
        company_name         = company_name,
        current_price        = current_price,
        market_cap           = market_cap,
        shares_out           = shares_out,
        ttm_revenue          = ttm_revenue,
        ttm_eps              = ttm_eps,
        ttm_pe               = ttm_pe,
        current_rev_multiple = current_rev_multiple,
        periods              = periods,
        analyst_targets      = analyst_targets,
    )


# ── Telegram message formatter ────────────────────────────────────────────────

def _fmt_p(v):
    return f"${v:.2f}" if v and pd.notna(v) else "N/A"

def _fmt_pct(v):
    if v is None or pd.isna(v): return "N/A"
    return f"{v:+.1f}%"

def _fmt_b(v):
    if v is None or pd.isna(v): return "N/A"
    return f"${v/1e9:.2f}B"

def _fmt_x(v):
    if v is None or pd.isna(v): return "N/A"
    return f"{v:.1f}x"


def build_telegram_estimates(result: EstimatesResult) -> str:
    r  = result
    at = r.analyst_targets
    lines = [
        f"🔭 *{r.ticker} — Forward Estimates Analysis*",
        f"_{r.company_name}_",
        f"Current Price: *{_fmt_p(r.current_price)}*  |  "
        f"Mkt Cap: *{_fmt_b(r.market_cap)}*",
        f"TTM Revenue: *{_fmt_b(r.ttm_revenue)}*  |  "
        f"TTM P/E: *{_fmt_x(r.ttm_pe)}*  |  "
        f"Rev Multiple: *{_fmt_x(r.current_rev_multiple)}*",
        "",
    ]

    for p in r.periods:
        sentiment_emoji = {"Bullish":"🟢","Bearish":"🔴","Neutral":"🟡"}.get(p.pe_sentiment,"⚪")
        lines += [
            f"━━━━━━━━━━━━━━━━━━━━",
            f"📅 *{p.period} ({p.period_date})*",
            "",
            f"💰 *Revenue Estimates*",
            f"  Low:  {_fmt_b(p.rev_low)}   Mean: {_fmt_b(p.rev_mean)}   High: {_fmt_b(p.rev_high)}",
            f"  YoY Growth: {_fmt_pct(p.rev_growth_pct)}  ({p.rev_num_analysts} analysts)",
            f"  Implied Price (same Rev multiple):",
            f"  Low {_fmt_p(p.rev_price_low)} → Mean {_fmt_p(p.rev_price_mean)} → High {_fmt_p(p.rev_price_high)}",
            "",
            f"📊 *EPS Estimates*",
            f"  Low: {_fmt_p(p.eps_low)}   Mean: {_fmt_p(p.eps_mean)}   High: {_fmt_p(p.eps_high)}",
            f"  YoY Growth: {_fmt_pct(p.eps_growth_pct)}  ({p.eps_num_analysts} analysts)",
            f"  Forward P/E (at current price):",
            f"  Low {_fmt_x(p.fwd_pe_low)} / Mean {_fmt_x(p.fwd_pe_mean)} / High {_fmt_x(p.fwd_pe_high)}",
            f"  vs TTM P/E {_fmt_x(r.ttm_pe)}  →  {sentiment_emoji} {p.pe_sentiment}",
            f"  Implied Price (same TTM P/E):",
            f"  Low {_fmt_p(p.eps_price_low)} → Mean {_fmt_p(p.eps_price_mean)} → High {_fmt_p(p.eps_price_high)}",
            "",
        ]

    if at:
        lines += [
            f"━━━━━━━━━━━━━━━━━━━━",
            f"🎯 *Analyst Price Targets*  ({at.num_analysts} analysts)",
            f"  Low:    {_fmt_p(at.low)}  ({_fmt_pct(at.upside_low)})",
            f"  Mean:   {_fmt_p(at.mean)}  ({_fmt_pct(at.upside_mean)})",
            f"  Median: {_fmt_p(at.median)}  ({_fmt_pct(at.upside_median)})",
            f"  High:   {_fmt_p(at.high)}  ({_fmt_pct(at.upside_high)})",
            f"",
            f"  Consensus: *{at.consensus}*",
            f"  🟢 Buy/Strong Buy: {at.bullish_count}  "
            f"⚪ Hold: {at.hold}  "
            f"🔴 Sell/Strong Sell: {at.bearish_count}",
        ]

    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "⚠️ _Estimates are analyst consensus. Not financial advice._",
    ]
    return "\n".join(lines)
