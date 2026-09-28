import numpy as np
import pandas as pd
import pytest

from rsi2 import StrategyConfig
from rsi2.engine import run
from tests.helpers import make_df, one_signal_series, random_walk

S = 400
N = 460


def _inject(df, rsi_vals=None, fast=None, trend=None):
    ind = pd.DataFrame({"trend": np.full(len(df), 1.0), "fast": np.full(len(df), 0.5),
                        "rsi": np.full(len(df), 50.0)}, index=df.index)
    for col, vals in (("rsi", rsi_vals), ("fast", fast), ("trend", trend)):
        for i, v in (vals or {}).items():
            ind.iloc[i, ind.columns.get_loc(col)] = v
    return ind


def _base_df():
    return make_df(np.linspace(100, 110, N))


def test_and_requires_both_conditions_or_does_not():
    df = _base_df()
    c = df["Close"].to_numpy()
    # signal at S, entry S+1; at S+2 RSI 70 but close below fast
    ind = _inject(df, rsi_vals={S: 5.0, S + 2: 70.0}, fast={i: c[i] + 1 for i in range(S, N)})
    r_and = run(df, StrategyConfig(exit_logic="AND"), indicators=ind)
    r_or = run(df, StrategyConfig(exit_logic="OR"), indicators=ind)
    assert r_and.fills[0].entry_bar == S + 1
    assert len(r_and.ledger) == 0 and len(r_and.open_trades) == 1
    assert r_or.ledger["exit_bar"].tolist() == [S + 2] and r_or.ledger["exit_reason"].tolist() == ["target"]


def test_trend_break_on_close_not_intrabar_dip():
    df = _base_df()
    c = df["Close"].to_numpy()
    trend = {i: 1.0 for i in range(N)}
    df.iloc[S + 2, df.columns.get_loc("Low")] = c[S + 2] - 5      # dips below trend intrabar...
    trend[S + 2] = c[S + 2] - 2                                    # ...but closes above it
    trend[S + 3] = c[S + 3] + 1                                    # closes below trend here
    ind = _inject(df, rsi_vals={S: 5.0}, trend=trend)
    res = run(df, StrategyConfig(), indicators=ind)
    assert res.ledger["exit_bar"].tolist() == [S + 3]
    assert res.ledger["exit_reason"].tolist() == ["trend_break"]


def test_gap_down_stop_fills_at_open():
    df = _base_df()
    entry = df["Close"].iloc[S + 1]
    i = S + 2
    df.iloc[i, df.columns.get_loc("Open")] = entry * 0.90
    df.iloc[i, df.columns.get_loc("Low")] = entry * 0.88
    df.iloc[i, df.columns.get_loc("Close")] = entry * 0.95
    df.iloc[i, df.columns.get_loc("High")] = entry * 0.96
    ind = _inject(df, rsi_vals={S: 5.0})
    res = run(df, StrategyConfig(use_stop=True, stop_pct=5, use_trend_stop=False), indicators=ind)
    assert res.ledger["exit_reason"].tolist() == ["stop"]
    assert res.ledger["exit_price"].iloc[0] == pytest.approx(entry * 0.90)


def test_exit_priority_target_before_trend_break():
    df = _base_df()
    c = df["Close"].to_numpy()
    ind = _inject(df, rsi_vals={S: 5.0, S + 2: 70.0}, trend={S + 2: c[S + 2] + 1})
    res = run(df, StrategyConfig(), indicators=ind)
    assert res.ledger["exit_reason"].tolist() == ["target"]


def test_nan_rsi_never_triggers_entry_or_exit():
    df = _base_df()
    ind = _inject(df, rsi_vals={S: np.nan, S + 1: 5.0})   # NaN is not "oversold"
    res = run(df, StrategyConfig(exit_logic="OR"), indicators=ind)
    assert [f.entry_bar for f in res.fills] == []          # NaN → 5 is not a crossunder from >=10


def test_flag_only_on_bar_after_entry():
    df = _base_df()
    ind = _inject(df, rsi_vals={S: 5.0, S + 2: 4.0, S + 3: 3.0})
    res = run(df, StrategyConfig(), indicators=ind)
    assert res.fills[0].flagged is True
    ind2 = _inject(df, rsi_vals={S: 5.0, S + 3: 3.0})       # low RSI only 2 bars after entry
    assert run(df, StrategyConfig(), indicators=ind2).fills[0].flagged is False


def test_time_stop():
    df = _base_df()
    ind = _inject(df, rsi_vals={S: 5.0})
    res = run(df, StrategyConfig(use_timeout=True, max_bars=4), indicators=ind)
    assert res.ledger["exit_reason"].tolist() == ["time_stop"] and res.ledger["bars_held"].tolist() == [4]


def test_recheck_filter_cancels_entry_below_trend():
    df = _base_df()
    c = df["Close"].to_numpy()
    ind = _inject(df, rsi_vals={S: 5.0}, trend={S + 1: c[S + 1] + 1})
    assert run(df, StrategyConfig(recheck_filter=False), indicators=ind).fills[0].entry_bar == S + 1
    assert run(df, StrategyConfig(recheck_filter=True), indicators=ind).fills == []


def test_open_trade_not_force_closed():
    df = _base_df()
    ind = _inject(df, rsi_vals={S: 5.0})
    res = run(df, StrategyConfig(use_trend_stop=False), indicators=ind)
    assert len(res.ledger) == 0 and len(res.open_trades) == 1
    assert res.equity["in_position"].iloc[-1]


def test_start_date_gates_entries():
    df = random_walk(seed=3)
    full = run(df, StrategyConfig())
    later = run(df, StrategyConfig(start_date=str(df.index[900].date())))
    assert (later.ledger["entry_bar"] >= 900).all()
    assert later.equity.index[0] == df.index[900]
    assert len(later.ledger) < len(full.ledger)


# ── Accounting ────────────────────────────────────────────────────────────────

def test_zero_costs_net_equals_gross():
    res = run(random_walk(), StrategyConfig(commission_bps=0, slippage_bps=0))
    assert np.allclose(res.ledger["net_ret_pct"], res.ledger["gross_ret_pct"], atol=1e-12)


def test_costs_make_net_strictly_lower():
    res = run(random_walk(), StrategyConfig())
    assert (res.ledger["net_ret_pct"] < res.ledger["gross_ret_pct"]).all()


def test_final_equity_is_product_of_trade_returns():
    cfg = StrategyConfig()
    res = run(random_walk(seed=5), cfg)
    last_exit = res.ledger["exit_date"].iloc[-1]
    expected = cfg.initial_capital * np.prod(1 + res.ledger["net_ret_pct"] / 100)
    assert res.equity.loc[last_exit, "equity"] == pytest.approx(expected, rel=1e-10)


def test_known_trade_pnl():
    cfg = StrategyConfig()
    df = make_df(one_signal_series(s=S))
    res = run(df, cfg)
    t = res.ledger.iloc[0]
    b = cfg.total_bps / 10_000
    net = (t.exit_price * (1 - b)) / (t.entry_price * (1 + b)) - 1
    assert t.net_ret_pct == pytest.approx(net * 100)
    assert t.pnl_dollars == pytest.approx(cfg.initial_capital * net)
    assert t.shares == pytest.approx(cfg.initial_capital / (t.entry_price * (1 + b)))


def test_whole_shares_and_partial_position():
    res = run(random_walk(), StrategyConfig(whole_shares=True, position_pct=50))
    assert (res.ledger["shares"] == np.floor(res.ledger["shares"])).all()
    assert res.equity["equity"].notna().all()


def test_deterministic():
    df = random_walk(seed=9)
    a, b = run(df, StrategyConfig()).ledger, run(df, StrategyConfig()).ledger
    pd.testing.assert_frame_equal(a, b)


# ── Degenerate input ─────────────────────────────────────────────────────────

def test_flat_series_no_trades():
    res = run(make_df(np.full(N, 50.0), spread=0), StrategyConfig())
    assert len(res.ledger) == 0 and not res.fills


def test_below_trend_no_trades():
    res = run(make_df(np.linspace(200, 100, N)), StrategyConfig())
    assert len(res.fills) == 0


def test_too_few_rows_raises():
    with pytest.raises(ValueError, match="need at least"):
        run(make_df(np.linspace(100, 110, 300)), StrategyConfig())


@pytest.mark.parametrize("breaker, msg", [
    (lambda d: d.drop(columns=["Low"]), "Missing required column: Low"),
    (lambda d: d.iloc[::-1], "sorted"),
    (lambda d: pd.concat([d, d.iloc[[5]]]).sort_index(), "Duplicated"),
    (lambda d: d.assign(High=d["Low"] * 0.99), "High < Low"),
    (lambda d: d.assign(Close=d["High"] * 1.01), "Close outside"),
    (lambda d: d.assign(Open=-1.0), "Non-positive"),
])
def test_validation_raises(breaker, msg):
    with pytest.raises(ValueError, match=msg):
        run(breaker(_base_df()), StrategyConfig())


def test_config_validation():
    for bad in (dict(trend_len=0), dict(entry_level=120), dict(entry_delay=-1),
                dict(position_pct=0), dict(exit_logic="XOR"), dict(trend_type="WMA")):
        with pytest.raises(ValueError):
            StrategyConfig(**bad)


def test_sanitize_repairs_only_tiny_errors():
    from rsi2.data import sanitize_ohlc
    df = _base_df()
    c = df["Close"].to_numpy()
    df.iloc[10, df.columns.get_loc("High")] = c[10] * 0.999      # 0.1% → repaired
    df.iloc[20, df.columns.get_loc("High")] = c[20] * 0.97       # 3% → left for validate() to reject
    fixed, n = sanitize_ohlc(df)
    assert n == 1 and fixed["High"].iloc[10] == c[10]
    with pytest.raises(ValueError, match="High < Low|Close outside"):
        run(fixed, StrategyConfig())
