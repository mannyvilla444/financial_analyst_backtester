# pipeline.py
"""
Core CSV → HTML analysis pipeline, extracted from bot.py's run_analysis()
so the Flask routes can call it without importing any Telegram code.

It follows bot.py step for step (same parser, same column validation,
same 4-quarter minimum, same TTM/prediction calls), with one correction:

CURRENT PRICE
─────────────
process_financial_data() keeps only the rows that carry earnings data (one
per quarter), so the "latest" row it hands on is the last quarter-end week,
not the latest bar. Using that row made Current Price, Mkt Cap, P/E, the
current valuation multiples and every price-target upside stale by up to a
quarter. Here the latest close (and market cap) is taken from the last row
of the *raw* CSV — or from a live quote when one is supplied — and pushed
into every calculation that needs the current price.
"""
import math
import re

import numpy as np
import pandas as pd

from financial_analysis import (
    parse_csv, validate_columns, process_financial_data,
    calculate_ttm_metrics, get_latest_metrics,
    calculate_price_predictions, extract_ticker_from_filename,
)
from html_report import generate_html

MULTIPLES = {                          # multiple column → TTM denominator
    "Market_Cap_to_Revenue": "Revenue_TTM",
    "Market_Cap_to_EBITDA":  "EBITDA_TTM",
    "Market_Cap_to_FCF":     "FCF_TTM",
}


def _ticker_from_filename(filename: str) -> str:
    """
    Ticker from a TradingView-style filename, using the bot's own extractor.
      'NASDAQ_AMD, 1W.csv'       → 'AMD'
      'NASDAQ_AMD__1W_auto.csv'  → 'AMD'   (tradingview_downloader output)
    If nothing matches, fall back to the cleaned-up filename stem so that
    several unrecognised files in one batch don't all collapse to 'Unknown'.
    """
    ticker = extract_ticker_from_filename(filename)
    if ticker and ticker != "Unknown":
        return ticker
    stem = re.sub(r"\.csv$", "", filename or "", flags=re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").upper()
    return stem[:12] or "UNKNOWN"


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


# ── Latest price ──────────────────────────────────────────────────────────────

def latest_quote(df_raw: pd.DataFrame, current_price: float | None = None) -> dict:
    """
    Latest price, market cap and date from the raw (un-deduplicated) CSV.

    price   : `current_price` if given (live quote), else the close of the last
              row that has one.
    mkt_cap : shares × price, with shares = Mkt cap / close on the latest row
              that has both. For a CSV without a live quote this is exactly the
              last row's own Mkt cap.
    """
    raw = df_raw.copy()
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    for c in ("close", "Mkt cap"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["time"]).sort_values("time")

    priced = raw[raw["close"].notna() & (raw["close"] > 0)]
    if priced.empty:
        raise ValueError("CSV has no close prices.")
    last = priced.iloc[-1]

    with_cap = priced[priced["Mkt cap"].notna() & (priced["Mkt cap"] > 0)]
    shares = (float(with_cap.iloc[-1]["Mkt cap"]) / float(with_cap.iloc[-1]["close"])
              if not with_cap.empty else None)

    live  = _num(current_price)
    price = live if live and live > 0 else float(last["close"])
    date  = pd.Timestamp.now().normalize() if live and live > 0 else pd.Timestamp(last["time"])
    return {
        "price":   price,
        "mkt_cap": shares * price if shares else None,
        "shares":  shares,
        "date":    date,
        "live":    bool(live and live > 0),
    }


def _with_latest_row(df_ttm: pd.DataFrame, q: dict) -> pd.DataFrame:
    """
    df_ttm plus a row at the latest price date: latest close/mkt cap, the most
    recent TTM figures carried forward, and the multiples recomputed on today's
    market cap. html_report then reads its "current" multiples from this row and
    the multiple-history charts end at today's value. Quarterly columns stay NaN
    on this row, so the quarterly bar charts and margin line are unaffected.
    """
    df = df_ttm.copy()
    last_date = pd.Timestamp(df["time"].iloc[-1])
    if q["date"] > last_date:
        new = {c: np.nan for c in df.columns}
        new["time"] = q["date"]
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    i = df.index[-1]
    df.loc[i, "close"] = q["price"]
    if q["mkt_cap"]:
        df.loc[i, "Mkt cap"] = q["mkt_cap"]
        for mult, ttm in MULTIPLES.items():
            if ttm in df.columns:
                vals = df[ttm].dropna()
                denom = _num(vals.iloc[-1]) if len(vals) else None
                df.loc[i, mult] = (q["mkt_cap"] / denom) if denom else np.nan
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_analysis(csv_bytes: bytes, filename: str,
                 current_price: float | None = None) -> tuple[bytes, str, str, dict]:
    """
    Core pipeline: raw CSV bytes → (html_bytes, ticker, output_filename, summary).

    current_price: optional live quote (used for typed tickers). When omitted,
    the latest close in the CSV is used.
    summary: the headline numbers used by the multi-company comparison.
    Raises ValueError with a user-readable message on failure.
    """
    ticker = _ticker_from_filename(filename)

    try:
        df, err = parse_csv(csv_bytes, filename)
    except Exception as e:           # e.g. non-UTF-8 bytes
        df, err = None, str(e)
    if err or df is None:
        raise ValueError(f"Could not parse CSV: {err}")

    missing = validate_columns(df)
    if missing:
        raise ValueError(
            f"Missing columns: {', '.join(missing)}. "
            "Export the chart from TradingView with these indicators added."
        )

    df_clean = process_financial_data(df)
    if len(df_clean) < 4:
        raise ValueError(
            "Not enough quarterly data (need at least 4 rows with complete financials)."
        )

    df_ttm  = calculate_ttm_metrics(df_clean)
    metrics = get_latest_metrics(df_ttm)
    if metrics is None:
        raise ValueError(
            f"Could not compute TTM metrics for {ticker}. "
            "Check that the CSV has Quarterly Revenue and Gross Profit data."
        )

    # ── Use the latest price everywhere a current price is needed ─────────────
    q = latest_quote(df, current_price)
    metrics["Current_Price"] = q["price"]
    metrics["Price_Date"]    = q["date"]
    if q["mkt_cap"]:
        metrics["Current_Mkt_Cap"] = q["mkt_cap"]
    eps = _num(metrics.get("EPS_TTM"))
    metrics["PE_Ratio"] = (q["price"] / eps) if eps else np.nan

    # Price targets: calculate_price_predictions() reads the current price and
    # share count from the last row, so give it the latest close / mkt cap there.
    # The historical multiple ranges it uses are left untouched.
    df_pred = df_ttm.copy()
    df_pred.loc[df_pred.index[-1], "close"] = q["price"]
    if q["mkt_cap"]:
        df_pred.loc[df_pred.index[-1], "Mkt cap"] = q["mkt_cap"]
    predictions = calculate_price_predictions(df_pred)

    df_report = _with_latest_row(df_ttm, q)
    html_str  = generate_html(ticker, df_report, metrics, predictions)

    summary = build_summary(ticker, df_report, metrics, q)
    return html_str.encode("utf-8"), ticker, f"{ticker}_financial_analysis.html", summary


def build_summary(ticker: str, df_report: pd.DataFrame, metrics: dict, q: dict) -> dict:
    """Headline numbers for the comparison report (JSON-safe; None = n/a)."""
    def last(col):
        if col not in df_report.columns:
            return None
        s = df_report[col].dropna()
        return _num(s.iloc[-1]) if len(s) else None

    def median_hist(col):
        """Median of this company's own positive historical multiples."""
        if col not in df_report.columns:
            return None
        s = pd.to_numeric(df_report[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s = s[s > 0]
        return _num(s.median()) if len(s) >= 3 else None

    mkt = _num(metrics.get("Current_Mkt_Cap"))

    def mult(ttm_col):
        d = last(ttm_col)
        return (mkt / d) if (mkt and d and d > 0) else None

    return {
        "ticker":          ticker,
        "price":           _num(q["price"]),
        "price_date":      pd.Timestamp(q["date"]).strftime("%Y-%m-%d"),
        "financials_date": pd.Timestamp(metrics["ticker_date"]).strftime("%Y-%m-%d"),
        "mkt_cap":         mkt,
        "revenue_ttm":     _num(metrics.get("Revenue_TTM")),
        "revenue_growth":  _num(metrics.get("Revenue_Growth_YoY")),
        "gross_margin":    _num(metrics.get("Gross_Profit_Margin")),
        "ebitda_margin":   _num(metrics.get("EBITDA_Margin")),
        "fcf_margin":      _num(metrics.get("FCF_Margin")),
        "fcf_ttm":         _num(metrics.get("FCF_TTM")),
        "net_debt_ebitda": _num(metrics.get("Net_Debt_to_EBITDA")),
        "pe":              _num(metrics.get("PE_Ratio")),
        "ebitda_ttm":      last("EBITDA_TTM"),
        # current multiples on today's market cap; None when TTM ≤ 0 (not meaningful)
        "mult_revenue":    mult("Revenue_TTM"),
        "mult_ebitda":     mult("EBITDA_TTM"),
        "mult_fcf":        mult("FCF_TTM"),
        "median_mult_revenue": median_hist("Market_Cap_to_Revenue"),
        "median_mult_ebitda":  median_hist("Market_Cap_to_EBITDA"),
        "median_mult_fcf":     median_hist("Market_Cap_to_FCF"),
    }
