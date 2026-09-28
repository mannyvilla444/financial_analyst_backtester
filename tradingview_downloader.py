"""
tradingview_downloader.py

Fetches quarterly financial data via yfinance and assembles it into a CSV
that exactly replicates the structure of the original TradingView export.

CRITICAL STRUCTURE (reverse-engineered from original CSV)
─────────────────────────────────────────────────────────
The original TradingView CSV has 300 WEEKLY rows covering ~6 years.
Quarterly financial data is embedded like this:

  • "Quarterly Revenue" and "Mkt cap" are populated on EVERY weekly row
    (the quarter's value forward-fills into all weeks of that quarter)

  • "EBITDA", "Gross Profit (B)", "Free cash flow",
    "Net debt to EBITDA ratio", "EPS estimates", "Revenue estimates"
    appear ONLY on the single week that coincides with the earnings
    report date (quarter end), and are NaN on all other rows.

  • Price columns (open, high, low, close, Volume) are the actual
    weekly OHLCV values for every row.

All financial values are RAW DOLLARS (not billions), despite column names.

DATA SOURCES
────────────
  yfinance — quarterly financials + weekly OHLCV price history
  tradingview-scraper — forward estimates (optional, best-effort)
"""

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


US_EXCHANGES = {
    "AMD": "NASDAQ", "NVDA": "NASDAQ", "AAPL": "NASDAQ", "MSFT": "NASDAQ",
    "GOOGL": "NASDAQ", "GOOG": "NASDAQ", "META": "NASDAQ", "TSLA": "NASDAQ",
    "AMZN": "NASDAQ", "NFLX": "NASDAQ", "INTC": "NASDAQ", "QCOM": "NASDAQ",
    "AVGO": "NASDAQ", "MU":   "NASDAQ", "AMAT": "NASDAQ", "LRCX": "NASDAQ",
    "ARM":  "NASDAQ", "SMCI": "NASDAQ", "MRVL": "NASDAQ", "KLAC": "NASDAQ",
    "JPM":  "NYSE",   "BAC":  "NYSE",   "GS":   "NYSE",   "WMT":  "NYSE",
    "XOM":  "NYSE",   "CVX":  "NYSE",   "BA":   "NYSE",   "JNJ":  "NYSE",
    "V":    "NYSE",   "MA":   "NYSE",   "UNH":  "NYSE",   "HD":   "NYSE",
    "ORCL": "NYSE",   "CRM":  "NYSE",   "IBM":  "NYSE",
}


def _tv_symbol(ticker: str) -> str:
    exchange = US_EXCHANGES.get(ticker.upper(), "NASDAQ")
    return f"{exchange}:{ticker.upper()}"


def _find_row(df: pd.DataFrame, keyword: str):
    """Return first row whose index label contains keyword (case-insensitive)."""
    if df is None or df.empty:
        return None
    for idx in df.index:
        if keyword.lower() in str(idx).lower():
            return df.loc[idx]
    return None


def _val(series, col) -> float:
    if series is None:
        return np.nan
    v = series.get(col, np.nan)
    return float(v) if pd.notna(v) else np.nan


def _fetch_yfinance(ticker: str) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    logger.info(f"[yfinance] Fetching {ticker}...")
    stock = yf.Ticker(ticker)

    try:
        income_q   = stock.quarterly_income_stmt
        cashflow_q = stock.quarterly_cash_flow
        balance_q  = stock.quarterly_balance_sheet
        fast_info  = stock.fast_info
        hist       = stock.history(period="6y", interval="1wk")
    except Exception as e:
        raise RuntimeError(f"yfinance fetch failed for '{ticker}': {e}")

    if income_q is None or income_q.empty:
        raise RuntimeError(
            f"No quarterly data found for '{ticker}'. "
            "Check the ticker (e.g. AMD, AAPL, NVDA, TSLA)."
        )

    return {
        "income_q":   income_q,
        "cashflow_q": cashflow_q,
        "balance_q":  balance_q,
        "fast_info":  fast_info,
        "hist":       hist,
    }


def _fetch_tv_estimates(tv_symbol: str) -> dict:
    result = {"eps_estimate": None, "revenue_estimate": None}
    try:
        from tradingview_scraper.symbols.overview import Overview
        fin = Overview().get_financials(symbol=tv_symbol)
        if fin and fin.get("status") == "success":
            d = fin.get("data", {})
            result["eps_estimate"]     = d.get("earnings_per_share_forecast_next_fq")
            result["revenue_estimate"] = d.get("revenue_forecast_next_fq")
    except Exception as e:
        logger.warning(f"[TV scraper] Non-fatal: {e}")
    return result


def _build_dataframe(ticker: str, data: dict, estimates: dict) -> pd.DataFrame:
    income_q   = data["income_q"]
    cashflow_q = data["cashflow_q"]
    balance_q  = data["balance_q"]
    fast_info  = data["fast_info"]
    hist       = data["hist"]

    # ── Shares outstanding for market cap ────────────────────────────────────
    try:
        shares_out = getattr(fast_info, "shares", None) or getattr(fast_info, "shares_outstanding", None)
        shares_out = float(shares_out) if shares_out else np.nan
    except Exception:
        shares_out = np.nan

    # ── Financial rows (yfinance returns raw dollars) ────────────────────────
    rev_row      = _find_row(income_q,   "Total Revenue")
    gp_row       = _find_row(income_q,   "Gross Profit")
    ebitda_row   = _find_row(income_q,   "EBITDA")
    fcf_row      = _find_row(cashflow_q, "Free Cash Flow")
    net_debt_row = _find_row(balance_q,  "Net Debt")
    eps_row      = _find_row(income_q,   "Basic EPS")

    # ── Normalise hist index (remove timezone) ───────────────────────────────
    hist = hist.copy()
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)

    # ── Build a quarterly lookup dict keyed by quarter-end Timestamp ─────────
    # yfinance columns are Timestamps; sort oldest → newest
    quarters = sorted(income_q.columns, key=lambda t: pd.Timestamp(t))

    def ts(q):
        t = pd.Timestamp(q)
        return t.tz_localize(None) if t.tz else t

    quarterly_data = {}
    for q in quarters:
        q_ts = ts(q)
        quarterly_data[q_ts] = {
            "revenue":      _val(rev_row,      q),
            "gross_profit": _val(gp_row,       q),
            "ebitda":       _val(ebitda_row,   q),
            "fcf":          _val(fcf_row,      q),
            "net_debt":     _val(net_debt_row, q),
            "eps":          _val(eps_row,      q),
        }

    # ── Build weekly rows matching TradingView's structure ───────────────────
    #
    # For each weekly bar:
    #   - close/open/high/low/volume  → actual weekly OHLCV
    #   - Mkt cap                     → shares × weekly close  (every row)
    #   - Quarterly Revenue           → current quarter's value (every row, forward-filled)
    #   - EBITDA, GP, FCF, ND/EBITDA, EPS est, Rev est
    #                                 → only on the earnings week, NaN elsewhere
    #
    # "Current quarter" for a weekly bar = the most recent quarter that
    # has ended on or before that week's date.

    q_dates = sorted(quarterly_data.keys())   # ascending

    rows = []
    for week_ts, bar in hist.iterrows():
        week_ts = pd.Timestamp(week_ts)

        close  = float(bar.get("Close",  np.nan))
        open_  = float(bar.get("Open",   np.nan))
        high   = float(bar.get("High",   np.nan))
        low    = float(bar.get("Low",    np.nan))
        volume = float(bar.get("Volume", np.nan))

        mkt_cap = (shares_out * close) if (pd.notna(shares_out) and pd.notna(close)) else np.nan

        # Find the most recent quarter that has ended by this week
        current_q = None
        for qd in q_dates:
            if qd <= week_ts:
                current_q = qd

        if current_q is None:
            # Before any earnings data — still include the price row
            rows.append({
                "time":                     f"{week_ts.month}/{week_ts.day}/{week_ts.year}",
                "open":                     round(open_, 4),
                "high":                     round(high, 4),
                "low":                      round(low, 4),
                "close":                    round(close, 4),
                "Volume":                   volume,
                "Mkt cap":                  mkt_cap,
                "Quarterly Revenue":        np.nan,
                "Gross Profit (B)":         np.nan,
                "EBITDA":                   np.nan,
                "Free cash flow":           np.nan,
                "Net debt to EBITDA ratio": np.nan,
                "EPS estimates":            np.nan,
                "Revenue estimates":        np.nan,
            })
            continue

        qdata   = quarterly_data[current_q]
        revenue = qdata["revenue"]

        # Earnings week = the weekly bar that contains the quarter-end date
        # (within a 7-day window of the quarter end)
        is_earnings_week = abs((week_ts - current_q).days) <= 7

        ebitda   = qdata["ebitda"]
        net_debt = qdata["net_debt"]
        nd_ratio = (
            (net_debt / ebitda)
            if (pd.notna(net_debt) and pd.notna(ebitda) and ebitda != 0)
            else np.nan
        )

        # Format date the same way TradingView does: M/D/YYYY
        try:
            date_str = f"{week_ts.month}/{week_ts.day}/{week_ts.year}"
        except Exception:
            date_str = str(week_ts)

        rows.append({
            "time":                     date_str,
            "open":                     round(open_, 4)  if pd.notna(open_)  else np.nan,
            "high":                     round(high, 4)   if pd.notna(high)   else np.nan,
            "low":                      round(low, 4)    if pd.notna(low)    else np.nan,
            "close":                    round(close, 4)  if pd.notna(close)  else np.nan,
            "Volume":                   volume,
            "Mkt cap":                  mkt_cap,
            # Revenue forward-fills into every week of the quarter
            "Quarterly Revenue":        revenue,
            # All other metrics: only on the earnings week
            "Gross Profit (B)":         qdata["gross_profit"] if is_earnings_week else np.nan,
            "EBITDA":                   qdata["ebitda"]       if is_earnings_week else np.nan,
            "Free cash flow":           qdata["fcf"]          if is_earnings_week else np.nan,
            "Net debt to EBITDA ratio": nd_ratio              if is_earnings_week else np.nan,
            "EPS estimates":            qdata["eps"]          if is_earnings_week else np.nan,
            "Revenue estimates":        revenue               if is_earnings_week else np.nan,
        })

    df = pd.DataFrame(rows)

    # ── Patch latest-quarter forward estimates onto the most recent earnings row
    if not df.empty and (estimates.get("eps_estimate") or estimates.get("revenue_estimate")):
        # Find the last row where EPS estimates is not NaN (last earnings week)
        mask = df["EPS estimates"].notna()
        if mask.any():
            last_earnings_idx = df[mask].index[-1]
            if estimates.get("eps_estimate") is not None:
                df.loc[last_earnings_idx, "EPS estimates"] = float(estimates["eps_estimate"])
            if estimates.get("revenue_estimate") is not None:
                df.loc[last_earnings_idx, "Revenue estimates"] = float(estimates["revenue_estimate"])

    return df


# ── Public interface ───────────────────────────────────────────────────────────

def download(ticker: str) -> tuple[bytes, str]:
    """
    Fetch quarterly financial data for `ticker`.
    Returns (csv_bytes, filename).
    Raises RuntimeError with a user-friendly message on failure.
    """
    ticker = ticker.upper().strip()
    tv_sym = _tv_symbol(ticker)

    data      = _fetch_yfinance(ticker)
    estimates = _fetch_tv_estimates(tv_sym)
    df        = _build_dataframe(ticker, data, estimates)

    if df.empty:
        raise RuntimeError(
            f"No data could be assembled for '{ticker}'. "
            "This can happen for very new companies, ETFs, or non-US tickers."
        )

    logger.info(f"Built {len(df)} weekly rows for {ticker} "
                f"({df['EBITDA'].notna().sum()} earnings weeks)")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename  = f"NASDAQ_{ticker}__1W_auto.csv"
    return csv_bytes, filename


def get_downloader():
    """Compatibility shim used by bot.py."""
    class _Downloader:
        def download(self, ticker: str):
            return download(ticker)
    return _Downloader()
