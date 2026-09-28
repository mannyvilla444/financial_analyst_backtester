"""Strategy A / B rules, driven by hand-built RSI, MA5 and MACD arrays."""
import numpy as np
import pandas as pd
import pytest

from rsi2.ab import (ABConfig, ABInputs, EXIT_MA5, EXIT_RSI, fresh_rising_streak, run_ab, run_strategy,
                     run_suite, main)
from tests.helpers import random_walk

N = 40


def inputs(rsi_vals, ma=None, macd=None, close=None, n=N):
    """RSI defaults to 50 (neutral); open[i] = 1000 + i so a fill price names its bar."""
    r = np.full(n, 50.0)
    for i, v in rsi_vals.items():
        r[i] = v
    c = np.full(n, 100.0) if close is None else np.asarray(close, float)
    m = np.full(n, 90.0)                     # close (100) above MA by default
    for i, v in (ma or {}).items():
        m[i] = v
    mb = np.zeros(n, bool)
    for i in (macd or []):
        mb[i] = True
    return ABInputs(open=1000.0 + np.arange(n), close=c, rsi=r, ma=m, macd_buy=mb)


CFG = ABConfig()
ENTRY = {10: 20.0, 11: 5.0}                  # crossunder on bar 11 → buy at bar 12's open


def only(fills):
    assert len(fills) == 1
    return fills[0]


# ── Strategy A ────────────────────────────────────────────────────────────────

def test_spec_example_75_83_86_81():
    x = inputs({**ENTRY, 13: 75.0, 14: 83.0, 15: 86.0, 16: 81.0})
    f = only(run_strategy(x, CFG, "A")[0])
    assert (f.entry_bar, f.entry_price) == (12, 1012.0)            # next bar's open
    assert (f.exit_bar, f.exit_price) == (17, 1017.0)              # sell signal on the "81" bar → next open
    assert f.exit_reason == EXIT_RSI and f.tags["exit_type"] == "RSI decline-watch"


def test_watch_stays_armed_through_rises_and_flat_bars():
    x = inputs({**ENTRY, 13: 70.0, 14: 82.0, 15: 88.0, 16: 88.0, 17: 91.0, 18: 97.0, 19: 96.9})
    f = only(run_strategy(x, CFG, "A")[0])
    assert f.exit_bar == 20                                        # equal (88→88) is not a decline


def test_watch_arms_on_crossover_only():
    # never ≤ 80 → > 80 while in the trade: RSI climbs to 80 exactly and falls — no arm, no exit
    x = inputs({**ENTRY, 13: 70.0, 14: 80.0, 15: 60.0, 16: 30.0})
    fills, _ = run_strategy(x, CFG, "A")
    assert only(fills).exit_bar is None
    # 80 → 80.5 is a crossover
    f = only(run_strategy(inputs({**ENTRY, 13: 80.0, 14: 80.5, 15: 79.0}), CFG, "A")[0])
    assert f.exit_bar == 16


def test_no_stop_no_time_exit_rides_forever():
    close = np.r_[np.full(12, 100.0), np.linspace(100, 20, N - 12)]   # price collapses
    x = inputs(ENTRY, close=close)
    f = only(run_strategy(x, CFG, "A")[0])
    assert f.exit_bar is None


def test_crossunder_needs_previous_at_or_above_10():
    below = {i: 9.0 for i in range(12)}
    below[11] = 8.0
    assert run_strategy(inputs(below), CFG, "A")[0] == []                    # below 10 all along
    f = only(run_strategy(inputs({10: 10.0, 11: 9.99}), CFG, "A")[0])        # 10 → 9.99 counts
    assert f.entry_bar == 12


def test_buy_signals_ignored_while_in_trade():
    x = inputs({**ENTRY, 13: 30.0, 14: 4.0, 15: 50.0})                       # 2nd crossunder in trade
    fills, _ = run_strategy(x, CFG, "A")
    assert len(fills) == 1


def test_buy_on_sell_signal_bar_ignored_but_allowed_after_exit_fill():
    # sell signal on 16 (and a crossunder on 16 — ignored, still in the trade),
    # exit fills at 17's open; crossunder on 17 → new entry at 18's open
    x = inputs({**ENTRY, 13: 70.0, 14: 85.0, 15: 12.0, 16: 5.0, 17: 50.0})
    fills, _ = run_strategy(x, CFG, "A")
    assert fills[0].exit_bar == 16                                          # 12.0 < 85 → sell at 16's open
    x2 = inputs({**ENTRY, 13: 70.0, 14: 85.0, 15: 84.0, 16: 30.0, 17: 5.0})
    fills2, _ = run_strategy(x2, CFG, "A")
    assert [f.entry_bar for f in fills2] == [12, 18] and fills2[0].exit_bar == 16


def test_pending_orders_at_end_of_data():
    x = inputs({**ENTRY, 36: 70.0, 37: 85.0, 38: 90.0, 39: 88.0})          # sell signal on the last bar
    f = only(run_strategy(x, CFG, "A")[0])
    assert f.exit_bar is None and f.tags.get("sell_pending") is True
    fills, pending = run_strategy(inputs({38: 20.0, 39: 5.0}), CFG, "A")   # buy signal on the last bar
    assert fills == [] and pending is True


def test_strategy_a_ignores_macd():
    x = inputs({**ENTRY, 13: 70.0, 14: 85.0, 15: 90.0, 16: 89.0}, macd=[12, 13])
    assert only(run_strategy(x, CFG, "A")[0]).exit_reason == EXIT_RSI


# ── Strategy B ────────────────────────────────────────────────────────────────

B_PATH = {**ENTRY, 13: 70.0, 14: 85.0, 15: 90.0, 16: 60.0}   # crossover on 14, RSI decline on 16


def test_b_macd_seen_switches_to_ma5_and_ignores_rsi_decline():
    x = inputs(B_PATH, macd=[13], ma={18: 101.0})             # close (100) < MA on bar 18
    f = only(run_strategy(x, CFG, "B")[0])
    assert f.exit_reason == EXIT_MA5 and f.tags["exit_type"] == "MACD/MA5"
    assert f.exit_bar == 19 and f.exit_price == 1019.0          # decided on 18, filled at 19's open
    assert f.tags["macd_seen_before_80"] is True


def test_b_macd_on_entry_bar_counts_but_signal_bar_does_not():
    on_entry = only(run_strategy(inputs(B_PATH, macd=[12], ma={18: 101.0}), CFG, "B")[0])
    assert on_entry.exit_reason == EXIT_MA5
    before = only(run_strategy(inputs(B_PATH, macd=[11], ma={18: 101.0}), CFG, "B")[0])
    assert before.exit_reason == EXIT_RSI and before.exit_bar == 17


def test_b_macd_after_crossover_does_not_change_path():
    f = only(run_strategy(inputs(B_PATH, macd=[15], ma={18: 101.0}), CFG, "B")[0])
    assert f.exit_reason == EXIT_RSI and f.exit_bar == 17 and f.tags["macd_seen_before_80"] is False


def test_b_macd_on_crossover_bar_counts():
    f = only(run_strategy(inputs(B_PATH, macd=[14], ma={18: 101.0}), CFG, "B")[0])
    assert f.exit_reason == EXIT_MA5


def test_b_ma_check_starts_the_bar_after_the_decision():
    f = only(run_strategy(inputs(B_PATH, macd=[13], ma={14: 101.0, 15: 101.0}), CFG, "B")[0])
    assert f.exit_bar == 16                                      # not 15: bar 14 (decision bar) is skipped


def test_b_without_crossover_rides_even_with_macd():
    f = only(run_strategy(inputs({**ENTRY, 13: 70.0}, macd=[13], ma={20: 101.0}), CFG, "B")[0])
    assert f.exit_bar is None


def test_fresh_rising_streak_only_first_bar():
    hist = [5, 4, 3, 4, 5, 6, 7, 6, 7, 8]
    assert np.flatnonzero(fresh_rising_streak(pd.Series(hist))).tolist() == [4, 9]


# ── Whole-suite behaviour on realistic data ───────────────────────────────────

@pytest.fixture(scope="module")
def daily():
    return random_walk(n=1500, seed=23, vol=0.02)


def test_suite_six_runs_and_columns(daily):
    s = run_suite(daily)
    assert sorted(s.runs) == [(st, L) for st in "AB" for L in (2, 3, 4)]
    assert list(s.comparison.columns) == [
        "RSI length", "Strategy", "Total trades", "Win rate %", "Avg win %", "Avg loss %", "Profit factor",
        "Total return %", "Max drawdown %", "Avg holding (bars)", "MACD/MA5 exits %", "RSI decline exits %",
        "Open at end"]
    assert len(s.comparison) == 6
    a = s.comparison[s.comparison.Strategy == "A"]
    b = s.comparison[s.comparison.Strategy == "B"]
    assert a["MACD/MA5 exits %"].isna().all()
    closed_b = b[b["Total trades"] > 0]
    assert np.allclose(closed_b["MACD/MA5 exits %"] + closed_b["RSI decline exits %"], 100)
    for col in ("Entry date", "Entry price", "Exit date", "Exit price", "RSI length", "Strategy",
                "Exit type", "Trade return %"):
        assert col in s.trades.columns
    assert set(s.trades["Exit type"][s.trades.Status == "closed"]) <= {"RSI decline-watch", "MACD/MA5"}
    assert (s.trades.loc[s.trades.Strategy == "A", "Exit type"].isin(
        ["RSI decline-watch", "open — no exit armed", "armed: RSI decline-watch", "sell pending (next open)"])).all()


def test_longer_rsi_trades_less(daily):
    s = run_suite(daily)
    a = s.comparison[s.comparison.Strategy == "A"].set_index("RSI length")["Total trades"]
    assert a[2] >= a[4]


def test_suite_fills_and_accounting(daily):
    s = run_suite(daily)
    for (strat, L), run in s.runs.items():
        led = run.ledger
        if not len(led):
            continue
        assert (led["bars_held"] >= 1).all()
        o = daily["Open"].to_numpy()
        assert np.allclose(led["entry_price"], o[led["entry_bar"]]) and np.allclose(led["exit_price"], o[led["exit_bar"]])
        last = led["exit_date"].iloc[-1]
        exp = run.config.initial_capital * np.prod(1 + led["net_ret_pct"] / 100)
        assert run.equity.loc[last, "equity"] == pytest.approx(exp, rel=1e-10)
        assert run.equity["in_position"].sum() == led["bars_held"].sum() + sum(t["bars_held"] + 1 for t in run.open_trades)


def test_no_lookahead(daily):
    full = run_suite(daily)
    for k in (500, 900, 1300):
        part = run_suite(daily.iloc[:k])
        for key in full.runs:
            a = [(f.entry_bar, f.entry_price, f.exit_bar, f.exit_price, f.exit_reason)
                 for f in part.runs[key].fills if f.exit_bar is not None and f.exit_bar < k]
            b = [(f.entry_bar, f.entry_price, f.exit_bar, f.exit_price, f.exit_reason)
                 for f in full.runs[key].fills if f.exit_bar is not None and f.exit_bar < k]
            assert a == b and len(a) > 0


def test_b_uses_both_exit_paths_on_real_like_data(daily):
    s = run_suite(daily)
    reasons = set()
    for L in (2, 3, 4):
        reasons |= set(s.runs[("B", L)].ledger["exit_reason"])
    assert reasons == {EXIT_RSI, EXIT_MA5}


def test_cli_writes_csvs(tmp_path, daily):
    p = tmp_path / "x.csv"
    daily.rename_axis("Date").to_csv(p)
    out = tmp_path / "out"
    assert main(["--csv", str(p), "--out-dir", str(out)]) == 0
    comp = pd.read_csv(out / "comparison.csv")
    trades = pd.read_csv(out / "trades.csv")
    assert len(comp) == 6 and len(trades) > 0
