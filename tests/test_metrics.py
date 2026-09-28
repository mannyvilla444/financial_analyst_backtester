import json
import os

import numpy as np
import pandas as pd
import pytest

from rsi2 import StrategyConfig
from rsi2 import metrics
from rsi2.cli import main
from rsi2.engine import run
from tests.helpers import make_df, random_walk


def test_t_distribution_p_values():
    assert metrics.t_two_sided_p(2.0, 10) == pytest.approx(0.073388, abs=1e-5)
    assert metrics.t_two_sided_p(3.0, 5) == pytest.approx(0.030099, abs=1e-5)
    assert metrics.t_two_sided_p(1.959964, 1e7) == pytest.approx(0.05, abs=1e-4)


def test_drawdown_from_daily_curve():
    eq = pd.Series([100, 110, 99, 105, 111, 90, 95], index=pd.date_range("2024-01-01", periods=7), dtype=float)
    dd, days = metrics.drawdown_stats(eq)
    assert dd == pytest.approx((90 / 111 - 1) * 100)
    assert days == 3


def test_summary_has_required_fields_and_exposure():
    res = run(random_walk(), StrategyConfig())
    s = metrics.summarize(res.ledger, res.equity, res.open_trades)
    for k in ("total_return_pct", "cagr_pct", "max_drawdown_pct", "max_drawdown_days", "sharpe", "sortino",
              "calmar", "trades", "win_rate_pct", "profit_factor", "t_stat", "exposure_pct",
              "return_per_exposure_pct", "max_consecutive_losses", "exit_mix", "open_trades"):
        assert k in s
    assert 0 < s["exposure_pct"] < 60
    assert s["final_equity"] == pytest.approx(res.equity["equity"].iloc[-1])


def test_drawdown_uses_daily_not_trade_curve():
    res = run(random_walk(seed=21), StrategyConfig())
    s = metrics.summarize(res.ledger, res.equity, res.open_trades)
    trade_curve = np.cumprod(1 + res.ledger["net_ret_pct"].to_numpy() / 100)
    trade_dd = (trade_curve / np.maximum.accumulate(trade_curve) - 1).min() * 100
    assert s["max_drawdown_pct"] <= trade_dd + 1e-9


def test_flagged_comparison_warns_when_small():
    res = run(random_walk(), StrategyConfig())
    fc = metrics.flagged_comparison(res.ledger)
    assert fc["flagged"]["n"] + fc["normal"]["n"] == len(res.ledger)
    assert fc["warning"] and "Underpowered" in fc["warning"]


def test_buy_and_hold():
    df = make_df(np.linspace(100, 200, 500))
    stats, eq = metrics.buy_and_hold(df, 100, 0.0, 1000.0)
    assert eq.iloc[-1] == pytest.approx(1000 * 200 / df["Close"].iloc[100])
    assert stats["exposure_pct"] == 100.0


def test_period_tables():
    eq = pd.Series(np.linspace(100, 121, 600), index=pd.bdate_range("2020-01-01", periods=600))
    y = metrics.yearly_returns(eq)
    assert np.prod(1 + y / 100) == pytest.approx(eq.iloc[-1] / eq.iloc[0])
    g = metrics.monthly_grid(eq)
    assert np.nanprod(1 + g.to_numpy() / 100) == pytest.approx(eq.iloc[-1] / eq.iloc[0])


def test_cli_end_to_end(tmp_path):
    df = random_walk()
    csv = tmp_path / "spy.csv"
    df.rename_axis("Date").to_csv(csv)
    out = tmp_path / "out"
    assert main(["--csv", str(csv), "--out-dir", str(out)]) == 0
    for f in ("trades.csv", "equity_daily.csv", "summary.json", "summary.txt"):
        assert (out / f).exists()
    js = json.loads((out / "summary.json").read_text())
    assert js["config"]["exit_logic"] == "AND" and js["code_version"]
    # byte-identical on re-run
    out2 = tmp_path / "out2"
    main(["--csv", str(csv), "--out-dir", str(out2)])
    assert (out / "trades.csv").read_bytes() == (out2 / "trades.csv").read_bytes()


def test_cli_exit_codes(tmp_path):
    short = make_df(np.linspace(100, 110, 300))
    p = tmp_path / "short.csv"
    short.rename_axis("Date").to_csv(p)
    assert main(["--csv", str(p), "--out-dir", str(tmp_path / "o1")]) == 1
    flat = make_df(np.full(500, 50.0), spread=0)
    p2 = tmp_path / "flat.csv"
    flat.rename_axis("Date").to_csv(p2)
    assert main(["--csv", str(p2), "--out-dir", str(tmp_path / "o2")]) == 2
