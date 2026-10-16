import numpy as np
import pandas as pd

from rsi2 import StrategyConfig
from rsi2.engine import run
from tests.helpers import make_df, one_signal_series, random_walk

S = 400


def test_single_signal_entry_next_bar_close():
    df = make_df(one_signal_series(s=S))
    res = run(df, StrategyConfig(commission_bps=0, slippage_bps=0))
    assert np.flatnonzero(res.signals.to_numpy()).tolist() == [S]
    f = res.fills[0]
    assert f.entry_bar == S + 1
    assert f.entry_price == df["Close"].iloc[S + 1]
    assert res.ledger["entry_bar"].tolist() == [S + 1]


def test_entry_delay_zero_enters_on_signal_bar():
    df = make_df(one_signal_series(s=S))
    res = run(df, StrategyConfig(entry_delay=0))
    assert res.fills[0].entry_bar == S


def test_minimum_hold_one_bar_everywhere():
    res = run(random_walk(), StrategyConfig())
    assert len(res.ledger) > 10
    assert (res.ledger["bars_held"] >= 1).all()
    assert (res.ledger["exit_bar"] > res.ledger["entry_bar"]).all()


def test_signal_while_in_position_is_discarded():
    c = one_signal_series(s=S)
    c[S + 3:] -= 3.0          # second crossunder at S+3 while the S+1 trade is open
    df = make_df(c)
    res = run(df, StrategyConfig())
    sig = np.flatnonzero(res.signals.to_numpy()).tolist()
    assert sig == [S, S + 3]
    assert len(res.fills) == 1 and res.fills[0].entry_bar == S + 1
    assert res.fills[0].exit_bar is None or res.fills[0].exit_bar > S + 4 or len(res.fills) == 1


def _key(ledger, k):
    cols = ["entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "flagged"]
    part = ledger[ledger["exit_bar"] < k][cols].reset_index(drop=True)
    return part


def test_no_lookahead():
    df = random_walk(n=1400, seed=11)
    for cfg in (StrategyConfig(), StrategyConfig(use_stop=True, stop_pct=4, exit_logic="OR"),
                StrategyConfig(trend_type="EMA", use_timeout=True, max_bars=5)):
        full = run(df, cfg).ledger
        assert len(full) > 10
        for k in (500, 700, 950, 1200):
            part = run(df.iloc[:k], cfg).ledger
            pd.testing.assert_frame_equal(_key(part, k), _key(full, k))
