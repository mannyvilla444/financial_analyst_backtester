import numpy as np
import pandas as pd


def make_df(close, spread=0.001, start="2015-01-02"):
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"Open": close, "High": close * (1 + spread), "Low": close * (1 - spread),
                         "Close": close, "Volume": 1e6}, index=idx)


def one_signal_series(n=460, s=400, rise=0.1, drop=3.0):
    """Steady uptrend (RSI = 100) with one sharp drop at index s → exactly one signal at s."""
    c = 100 + rise * np.arange(n)
    c[s:] -= drop
    return c


def random_walk(n=1500, seed=7, drift=0.0006, vol=0.013):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, vol, n)))
    o = c * np.exp(rng.normal(0, vol / 3, n))
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c, "Volume": 1e6}, index=idx)
