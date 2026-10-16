"""
Indicators with TradingView-parity seeding (spec §3).

Leading NaNs (e.g. the first bar of close.diff()) are skipped: the seed is the
simple mean of the first `length` valid values, placed on the last of them —
exactly what Pine's ta.rma / ta.ema do when their source starts with na.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _first_valid(x: np.ndarray) -> int | None:
    idx = np.flatnonzero(~np.isnan(x))
    return int(idx[0]) if idx.size else None


def _check_no_interior_nan(x: np.ndarray, f: int, name: str) -> None:
    if np.isnan(x[f:]).any():
        bad = int(np.flatnonzero(np.isnan(x[f:]))[0]) + f
        raise ValueError(f"{name}: NaN inside the series at position {bad} (after the first valid value)")


def wilder_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA by explicit recursion (the reference implementation)."""
    if length <= 0:
        raise ValueError("length must be positive")
    x = np.asarray(series, dtype=float)
    out = np.full(x.shape, np.nan)
    f = _first_valid(x)
    if f is None or len(x) - f < length:
        return pd.Series(out, index=series.index)
    _check_no_interior_nan(x, f, "wilder_rma")
    seed = f + length - 1
    out[seed] = x[f:f + length].mean()
    for i in range(seed + 1, len(x)):
        out[i] = (out[i - 1] * (length - 1) + x[i]) / length
    return pd.Series(out, index=series.index)


def wilder_rma_vectorized(series: pd.Series, length: int) -> pd.Series:
    """Same result via a manually-seeded ewm (parity asserted in tests)."""
    x = pd.Series(np.asarray(series, dtype=float), index=series.index)
    out = pd.Series(np.nan, index=series.index)
    f = _first_valid(x.values)
    if f is None or len(x) - f < length:
        return out
    seed = f + length - 1
    tail = x.iloc[seed:].copy()
    tail.iloc[0] = x.iloc[f:f + length].mean()
    out.iloc[seed:] = tail.ewm(alpha=1.0 / length, adjust=False).mean().values
    return out


def rsi(close: pd.Series, length: int) -> pd.Series:
    """
    RSI with the explicit edge cases of spec §3.2:
      avg_loss == 0, avg_gain > 0  → 100
      avg_loss == 0, avg_gain == 0 → NaN (flat stretch: "no signal, no exit")
      avg_gain == 0, avg_loss > 0  → 0
      warmup                       → NaN
    """
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = wilder_rma(gain, length).values
    al = wilder_rma(loss, length).values
    out = np.full(ag.shape, np.nan)
    for i in range(len(ag)):
        g, l = ag[i], al[i]
        if np.isnan(g) or np.isnan(l):
            continue
        if l == 0 and g == 0:
            continue                      # deliberate NaN
        if l == 0:
            out[i] = 100.0
        elif g == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + g / l)
    return pd.Series(out, index=close.index)


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.astype(float).rolling(n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    """TradingView ta.ema: alpha = 2/(n+1), seeded with SMA of the first n values."""
    v = np.asarray(x, dtype=float)
    out = np.full(v.shape, np.nan)
    f = _first_valid(v)
    if f is None or len(v) - f < n:
        return pd.Series(out, index=x.index)
    _check_no_interior_nan(v, f, "ema")
    alpha = 2.0 / (n + 1)
    seed = f + n - 1
    out[seed] = v[f:f + n].mean()
    for i in range(seed + 1, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=x.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = h.iloc[0] - l.iloc[0]
    return tr


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder ATR. Reporting only in the spec engine (spec §3.4); rsi2.ab uses it for its optional stop."""
    return wilder_rma(true_range(df), length)
