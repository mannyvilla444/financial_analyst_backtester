"""Strategy Lab (Backtest tab): shared data/window, weekly candles, report inputs."""
import numpy as np
import pandas as pd
import pytest

import regime_lab as R
from tests.helpers import random_walk


@pytest.fixture(scope="module")
def daily():
    return random_walk(n=1500, seed=17, vol=0.02)


def test_weekly_frame_dated_by_last_trading_day(daily):
    wk = R.momentum_frame(daily, "weekly")
    assert daily.index.get_indexer(wk.index).min() >= 0
    assert (wk["close"].to_numpy() == daily.loc[wk.index, "Close"].to_numpy()).all()
    week = daily.index.to_period("W-FRI")
    last_days = daily.index.to_series().groupby(week).last()
    assert (wk.index == pd.DatetimeIndex(last_days.values)).all()


@pytest.mark.parametrize("kind", ["histogram", "crossover"])
def test_weekly_signals_no_lookahead(daily, kind):
    wk_full = R.momentum_frame(daily, "weekly")
    ev_full = R.momentum_events(wk_full, kind, 12, 26, 9)
    week_ends = daily.index.get_indexer(wk_full.index)
    for w in (60, 120, 200, 280):
        part = daily.iloc[:week_ends[w] + 1]
        ev = R.momentum_events(R.momentum_frame(part, "weekly"), kind, 12, 26, 9)
        pd.testing.assert_frame_equal(ev, ev_full.iloc[:w + 1])


@pytest.mark.parametrize("tf", ["weekly", "daily"])
def test_run_lab_end_to_end(daily, tf):
    lab = R.run_lab("TEST", tf, daily=daily)
    assert list(lab.strategies) == ["histogram", "crossover", "A2", "B2", "A3", "B3", "A4", "B4"] or \
        set(lab.strategies) == {"histogram", "crossover", "A2", "A3", "A4", "B2", "B3", "B4"}
    for s in lab.strategies.values():
        assert s.available and s.equity.index[0] == lab.window_start
        if len(s.ledger):
            assert (s.ledger["entry_date"] >= lab.window_start).all()
            assert (s.ledger["bars_held"] >= 1).all()
    # the comparison table and the scoreboard agree (same fills, same accounting)
    for _, row in lab.comparison.iterrows():
        s = lab.strategies[f"{row['Strategy']}{int(row['RSI length'])}"]
        assert row["Total return %"] == pytest.approx(s.summary["total_return_pct"])
        assert row["Total trades"] == s.summary["trades"]
    assert len(lab.comparison) == 6 and len(lab.trades) > 0
    assert lab.benchmark["exposure_pct"] == 100


def test_report_renders(daily):
    import lab_report
    html = lab_report.render(R.run_lab("TEST", "daily", daily=daily))
    for text in ("Strategy A vs B × RSI Length", "comparison.csv", "Strategy B — Exit Paths",
                 "chartEqA", "chartEqB", "Scoreboard", "MACD Histogram"):
        assert text in html
    assert "Hand-off" not in html and "200-day SMA" not in html


def test_short_history_raises():
    with pytest.raises(RuntimeError, match="Not enough price history"):
        R.run_lab("NEW", "weekly", daily=random_walk(n=300, seed=2))


@pytest.mark.parametrize("stop", ["none", "time_atr3"])
def test_lab_stop_option(daily, stop):
    import lab_report
    from rsi2.ab import STOP_PRESETS
    lab = R.run_lab("TEST", "weekly", daily=daily, stop=stop)
    assert lab.stop == stop and len(lab.stop_grid) == 36
    assert (lab.comparison["Stop-loss"] == STOP_PRESETS[stop].label).all()
    # the A/B runs in the scoreboard are exactly the grid rows for this stop
    sub = lab.stop_grid[lab.stop_grid["Stop-loss"] == STOP_PRESETS[stop].label].reset_index(drop=True)
    assert np.allclose(sub["Total return %"], lab.comparison["Total return %"])
    html = lab_report.render(lab)
    assert "Stop-Loss Comparison" in html and "stop_comparison.csv" in html
    assert STOP_PRESETS[stop].label in html


def test_lab_rejects_unknown_stop(daily):
    with pytest.raises(ValueError):
        R.run_lab("TEST", "weekly", daily=daily, stop="tight")
