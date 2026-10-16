"""Optional stops for Strategy A / B: ATR stop, time stop, presets and the stop grid."""
import numpy as np
import pandas as pd
import pytest

from rsi2.ab import (ABConfig, EXIT_ATR, EXIT_MA5, EXIT_RSI, EXIT_TIME, STOP_PRESETS, default_time_stop,
                     main, run_stop_grid, run_strategy, run_suite, with_stop)
from tests.helpers import random_walk
from tests.test_ab import ENTRY, N, inputs, only

# entry: crossunder on bar 11 → buy at bar 12's open (price 1012); ATR on the signal bar (11) = 10
ATR = np.full(N, 10.0)


def with_atr(x, atr=ATR):
    x.atr = np.asarray(atr, float)
    return x


def falling_close(at, level):
    c = np.full(N, 1012.0)
    c[at] = level
    return c


# ── ATR stop ──────────────────────────────────────────────────────────────────

def test_atr_stop_level_and_fill_at_next_open():
    cfg = ABConfig(atr_stop_mult=3.0)                     # stop = 1012 − 3 × 10 = 982
    x = with_atr(inputs(ENTRY, close=falling_close(15, 981.9)))
    f = only(run_strategy(x, cfg, "A")[0])
    assert f.tags["stop_level"] == pytest.approx(982.0)
    assert (f.exit_bar, f.exit_price, f.exit_reason) == (16, 1016.0, EXIT_ATR)
    assert f.tags["exit_type"] == "ATR stop"


def test_atr_stop_needs_a_close_below_not_at_the_level():
    cfg = ABConfig(atr_stop_mult=3.0)
    f = only(run_strategy(with_atr(inputs(ENTRY, close=falling_close(15, 982.0))), cfg, "A")[0])
    assert f.exit_bar is None


def test_atr_stop_uses_signal_bar_atr_not_later_values():
    cfg = ABConfig(atr_stop_mult=2.0)
    atr = ATR.copy()
    atr[12:] = 1000.0                                     # ATR blows up after entry — stop must not move
    f = only(run_strategy(with_atr(inputs(ENTRY, close=falling_close(14, 991.0)), atr), cfg, "A")[0])
    assert f.tags["stop_level"] == pytest.approx(992.0) and f.exit_reason == EXIT_ATR and f.exit_bar == 15


def test_atr_stop_active_after_arming_and_on_b_ma5_path():
    cfg = ABConfig(atr_stop_mult=3.0)
    close = falling_close(16, 900.0)
    # A: armed on bar 14 (RSI 85) and still rising on bar 16 — the stop fires anyway
    f = only(run_strategy(with_atr(inputs({**ENTRY, 13: 70.0, 14: 85.0, 15: 90.0, 16: 95.0}, close=close)), cfg, "A")[0])
    assert (f.exit_bar, f.exit_reason) == (17, EXIT_ATR)
    # B on the MA5 path with close above MA... the stop still fires
    x = with_atr(inputs({**ENTRY, 13: 70.0, 14: 85.0}, macd=[13], ma={16: 800.0}, close=close))
    f = only(run_strategy(x, cfg, "B")[0])
    assert (f.exit_bar, f.exit_reason) == (17, EXIT_ATR)


def test_strategy_exit_takes_priority_over_stop_on_same_bar():
    cfg = ABConfig(atr_stop_mult=3.0)
    x = with_atr(inputs({**ENTRY, 13: 70.0, 14: 85.0, 15: 80.0}, close=falling_close(15, 900.0)))
    f = only(run_strategy(x, cfg, "A")[0])
    assert (f.exit_bar, f.exit_reason) == (16, EXIT_RSI)


def test_atr_stop_off_rides_through_the_drop():
    x = with_atr(inputs(ENTRY, close=falling_close(15, 500.0)))
    assert only(run_strategy(x, ABConfig(), "A")[0]).exit_bar is None


# ── Time stop ─────────────────────────────────────────────────────────────────

def test_time_stop_fires_after_n_bars_when_not_armed():
    cfg = ABConfig(time_stop_bars=5)                      # entry bar 12 = bar 1 → bar 16 = bar 5 → sell at 17's open
    f = only(run_strategy(inputs(ENTRY), cfg, "A")[0])
    assert (f.exit_bar, f.exit_price, f.exit_reason) == (17, 1017.0, EXIT_TIME)
    assert f.tags["exit_type"] == "Time stop"


def test_time_stop_skipped_once_armed():
    cfg = ABConfig(time_stop_bars=5)
    # crosses over 80 on bar 16 (the 5th bar) → armed on that close → no time stop; RSI falls on bar 20
    x = inputs({**ENTRY, 15: 70.0, 16: 85.0, 17: 86.0, 18: 87.0, 19: 88.0, 20: 87.0})
    f = only(run_strategy(x, cfg, "A")[0])
    assert (f.exit_bar, f.exit_reason) == (21, EXIT_RSI)


def test_time_stop_skipped_on_b_ma5_path():
    cfg = ABConfig(time_stop_bars=5)
    x = inputs({**ENTRY, 13: 70.0, 14: 85.0}, macd=[13], ma={25: 200.0})   # MA5 path, close < MA on bar 25
    f = only(run_strategy(x, cfg, "B")[0])
    assert (f.exit_bar, f.exit_reason) == (26, EXIT_MA5)


def test_time_stop_bars_per_rsi_length():
    assert [default_time_stop(L) for L in (2, 3, 4)] == [10, 15, 20]
    assert with_stop(ABConfig(rsi_length=3), "time").time_stop_bars == 15
    c = with_stop(ABConfig(rsi_length=4), "time_atr3")
    assert (c.time_stop_bars, c.atr_stop_mult) == (20, 3.0)
    assert with_stop(c, "none").time_stop_bars is None and with_stop(c, "none").atr_stop_mult is None


@pytest.mark.parametrize("bad", [dict(atr_stop_mult=0), dict(atr_stop_mult=-1), dict(time_stop_bars=0),
                                 dict(time_stop_bars=2.5), dict(atr_len=0)])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        ABConfig(**bad)


def test_unknown_preset():
    with pytest.raises(ValueError):
        with_stop(ABConfig(), "tight")


# ── Suite / grid on random-walk data ──────────────────────────────────────────

@pytest.fixture(scope="module")
def daily():
    return random_walk(n=1500, seed=23, vol=0.02)


def test_none_preset_matches_default_suite(daily):
    a, b = run_suite(daily), run_suite(daily, stop="none")
    pd.testing.assert_frame_equal(a.comparison, b.comparison)


def test_stops_only_ever_shorten_trades(daily):
    """Same entries until a stop closes a trade early; a stop can't make a trade last longer."""
    base = run_suite(daily)
    for k in ("time", "atr2", "time_atr3"):
        s = run_suite(daily, stop=k)
        for key, run in s.runs.items():
            led = run.ledger
            assert set(led["exit_reason"]) <= {EXIT_RSI, EXIT_MA5, EXIT_ATR, EXIT_TIME}
            stops = led[led["exit_reason"].isin([EXIT_ATR, EXIT_TIME])]
            if k == "time":
                L = key[1]
                assert (stops["bars_held"] == default_time_stop(L)).all()
            # the first trade is identical up to its exit, which is never later
            f0, g0 = base.runs[key].fills[0], run.fills[0]
            assert f0.entry_bar == g0.entry_bar
            if f0.exit_bar is not None and g0.exit_bar is not None:
                assert g0.exit_bar <= f0.exit_bar


def test_atr_stop_fills_at_open_after_close_below_level(daily):
    s = run_suite(daily, stop="atr2")
    c, o = daily["Close"].to_numpy(), daily["Open"].to_numpy()
    n = 0
    for run in s.runs.values():
        for f in run.fills:
            if f.exit_reason == EXIT_ATR:
                n += 1
                assert c[f.exit_bar - 1] < f.tags["stop_level"] and f.exit_price == o[f.exit_bar]
                # no earlier close in the trade was below the level (first breach wins)
                assert (c[f.entry_bar:f.exit_bar - 1] >= f.tags["stop_level"]).all()
    assert n > 0


def test_no_lookahead_with_stops(daily):
    full = run_suite(daily, stop="time_atr3")
    for k in (600, 1100):
        part = run_suite(daily.iloc[:k], stop="time_atr3")
        for key in full.runs:
            a = [(f.entry_bar, f.exit_bar, f.exit_reason) for f in part.runs[key].fills if f.exit_bar and f.exit_bar < k]
            b = [(f.entry_bar, f.exit_bar, f.exit_reason) for f in full.runs[key].fills if f.exit_bar and f.exit_bar < k]
            assert a == b


def test_stop_grid_shape_and_baseline(daily):
    g = run_stop_grid(daily)
    assert len(g) == len(STOP_PRESETS) * 6 and g.columns[0] == "Stop-loss"
    assert list(g["Stop-loss"].unique()) == [r.label for r in STOP_PRESETS.values()]
    none = g[g["Stop-loss"] == "No stop"]
    assert np.allclose(none["Return vs no stop (pts)"], 0)
    assert (none["Stop exits %"].fillna(0) == 0).all()
    assert (g.loc[g["Stop-loss"] != "No stop", "Stop exits %"].fillna(0) > 0).any()
    # grid rows equal the standalone suite for that preset
    t = run_suite(daily, stop="atr3").comparison
    sub = g[g["Stop-loss"] == "3× ATR"].reset_index(drop=True)
    assert np.allclose(sub["Total return %"], t["Total return %"])


def test_cli_stop_and_compare(tmp_path, daily):
    p = tmp_path / "x.csv"
    daily.rename_axis("Date").to_csv(p)
    out = tmp_path / "out"
    assert main(["--csv", str(p), "--out-dir", str(out), "--stop", "atr3", "--compare-stops"]) == 0
    comp = pd.read_csv(out / "comparison.csv")
    assert (comp["Stop-loss"] == "3× ATR").all()
    assert len(pd.read_csv(out / "stop_comparison.csv")) == 36
    assert main(["--csv", str(p), "--out-dir", str(out), "--stop", "atr3", "--time-stop-bars", "5"]) == 1
    assert main(["--csv", str(p), "--out-dir", str(tmp_path / "o2"), "--atr-stop-mult", "2.5",
                 "--time-stop-bars", "8"]) == 0
    assert (pd.read_csv(tmp_path / "o2" / "comparison.csv")["Stop-loss"] == "time 8 bars + 2.5× ATR(14)").all()
