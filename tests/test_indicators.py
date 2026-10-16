import numpy as np
import pandas as pd
import pytest

from rsi2.indicators import wilder_rma, wilder_rma_vectorized, rsi, sma, ema


@pytest.mark.parametrize("length", [2, 3, 14])
def test_rma_loop_equals_seeded_vectorized(length):
    rng = np.random.default_rng(1)
    s = pd.Series(np.r_[np.nan, rng.normal(0, 1, 500)])
    a, b = wilder_rma(s, length), wilder_rma_vectorized(s, length)
    assert a.isna().equals(b.isna())
    assert np.allclose(a.dropna(), b.dropna(), atol=1e-12, rtol=0)


def test_rma_seed_is_sma_of_first_values():
    s = pd.Series([2.0, 4.0, 6.0, 8.0])
    r = wilder_rma(s, 2)
    assert np.isnan(r[0]) and r[1] == 3.0 and r[2] == (3.0 + 6.0) / 2 and r[3] == (4.5 + 8.0) / 2


def test_rsi2_hand_computed():
    close = pd.Series([10, 11, 10.5, 12, 11], dtype=float)
    # gains: -,1,0,1.5,0  losses: -,0,.5,0,1 ; seed at idx 2: ag=.5, al=.25
    r = rsi(close, 2)
    assert r[:2].isna().all()
    assert r[2] == pytest.approx(100 - 100 / (1 + 0.5 / 0.25), abs=1e-12)          # 66.667
    assert r[3] == pytest.approx(100 - 100 / (1 + 1.0 / 0.125), abs=1e-12)         # 88.889
    assert r[4] == pytest.approx(100 - 100 / (1 + 0.5 / 0.5625), abs=1e-12)        # 47.059


def test_rsi_extremes_and_flat():
    up = rsi(pd.Series(np.arange(1, 50, dtype=float)), 2)
    down = rsi(pd.Series(np.arange(50, 1, -1, dtype=float)), 2)
    flat = rsi(pd.Series(np.full(50, 7.0)), 2)
    assert (up.dropna() == 100).all() and up.notna().sum() == 47
    assert (down.dropna() == 0).all()
    assert flat.isna().all()          # 0/0 is deliberately NaN, never 0


@pytest.mark.parametrize("n", [1, 5, 200])
def test_sma_ema_nan_count(n):
    x = pd.Series(np.linspace(1, 2, 400))
    assert sma(x, n).isna().sum() == n - 1
    assert ema(x, n).isna().sum() == n - 1
    assert ema(x, n).iloc[n - 1] == pytest.approx(x.iloc[:n].mean())
