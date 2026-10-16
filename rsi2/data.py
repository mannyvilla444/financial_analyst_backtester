"""Data loading and validation (spec §7). Validation raises; it never warns."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ("Open", "High", "Low", "Close")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Case-insensitive column names → Open/High/Low/Close/Volume."""
    mapping = {}
    for c in df.columns:
        key = str(c).strip().lower()
        for want in ("Open", "High", "Low", "Close", "Volume"):
            if key == want.lower():
                mapping[c] = want
    return df.rename(columns=mapping)


def validate(df: pd.DataFrame, min_rows: int | None = None) -> pd.DataFrame:
    """Raise ValueError on anything the engine can't trust."""
    df = normalize_columns(df)
    for col in REQUIRED:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")
    if df.index.tz is not None:
        raise ValueError("Index must be tz-naive")
    if df.index.has_duplicates:
        dups = df.index[df.index.duplicated()].strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(f"Duplicated dates in index: {dups}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not sorted ascending")
    px = df[list(REQUIRED)].astype(float)
    if px.isna().any().any():
        bad = px.index[px.isna().any(axis=1)].strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(f"NaN prices on: {bad}")
    if (px <= 0).any().any():
        bad = px.index[(px <= 0).any(axis=1)].strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(f"Non-positive prices on: {bad}")
    hl = px["High"] < px["Low"]
    if hl.any():
        raise ValueError(f"High < Low on: {px.index[hl].strftime('%Y-%m-%d').tolist()[:5]}")
    out = (px["Close"] > px["High"]) | (px["Close"] < px["Low"])
    if out.any():
        raise ValueError(f"Close outside [Low, High] on: {px.index[out].strftime('%Y-%m-%d').tolist()[:5]}")
    if min_rows is not None and len(df) < min_rows:
        raise ValueError(f"Only {len(df)} rows; need at least {min_rows} (trend_len + 250) to say anything")
    df = df.copy()
    df[list(REQUIRED)] = px
    return df


def sanitize_ohlc(df: pd.DataFrame, tol: float = 0.005) -> tuple[pd.DataFrame, int]:
    """
    Repair tiny inconsistencies that dividend/split adjustment rounding leaves
    in vendor data — a Close a hair outside [Low, High], or High < Low by a
    hair — when the error is within `tol` (0.5%) of the price. Larger errors are
    left alone so validate() still rejects them. Returns (df, rows_repaired).
    Used by the web app and by the CLI's --repair-ohlc; never applied silently.
    """
    df = normalize_columns(df).copy()
    h, l, c = (df[k].astype(float) for k in ("High", "Low", "Close"))
    fix_h = (c > h) & ((c - h) / c <= tol)
    fix_l = (c < l) & ((l - c) / c <= tol)
    swap = (h < l) & ((l - h) / l <= tol)
    df.loc[fix_h, "High"] = c[fix_h]
    df.loc[fix_l, "Low"] = c[fix_l]
    hh, ll = df["High"].copy(), df["Low"].copy()
    df.loc[swap, "High"], df.loc[swap, "Low"] = ll[swap], hh[swap]
    return df, int((fix_h | fix_l | swap).sum())


def load_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    date_col = next((c for c in raw.columns if str(c).strip().lower() in ("date", "datetime", "time", "timestamp")), None)
    if date_col is None:
        raise ValueError("CSV needs a Date (or Datetime) column")
    raw.index = pd.to_datetime(raw[date_col], utc=False)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw.drop(columns=[date_col])
    raw.index.name = "Date"
    return normalize_columns(raw)


def load_yfinance(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:        # package works without yfinance installed
        raise ImportError("yfinance is not installed: pip install yfinance") from e
    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No price data returned for {ticker!r}")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "Date"
    return normalize_columns(df)[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]]
