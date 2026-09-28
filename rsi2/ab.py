"""
rsi2/ab.py — Strategy A (pure RSI mean reversion) and Strategy B (RSI mean
reversion + MACD filter), run for several RSI lengths.

STRATEGY A — Pure RSI mean reversion
  Buy     RSI crosses UNDER entry_level (prev ≥ 10, now < 10) while flat.
          Filled at the NEXT bar's open. Signals while in a trade are ignored.
  Setup   RSI crosses OVER overbought_level (prev ≤ 80, now > 80) while in the
          trade → arm the "decline watch" (stays armed until it fires).
  Sell    Once armed, the first bar whose RSI is below the previous bar's RSI
          → sell at the NEXT bar's open.
  No stop-loss, no time exit: if RSI never crosses over 80 the trade rides.

  Example: RSI 75, 83, 86, 81 → 83 arms, 86 still rising, 81 < 86 → sell
  signal on the "81" bar, filled at the following bar's open.

STRATEGY B — same entry; exit depends on MACD during the trade
  From the entry bar on, watch the daily MACD histogram for a "BUY state": the
  first bar of a fresh 2-in-a-row rising streak (hist[t] > hist[t-1] and
  hist[t-1] > hist[t-2], but NOT hist[t-2] > hist[t-3]).
  When RSI crosses over 80:
    • a BUY state has occurred since entry → this trade permanently exits
      when the daily close is below MA5 (checked from the next bar), sold at
      the next bar's open; the RSI decline watch is ignored for this trade.
    • otherwise → Strategy A's decline watch.
  Each trade's exit type is logged ("MACD/MA5" or "RSI decline-watch").

Timing: every decision uses bar-close data up to that bar; every fill happens
at a later bar's open. Costs: commission + slippage per side (default 3 + 1
bps). Accounting (100% of equity per trade, daily mark-to-market) is shared
with the rest of the lab via rsi2.portfolio.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np
import pandas as pd

from . import metrics as M
from .indicators import rsi as rsi_ind, sma
from .portfolio import Fill, simulate

EXIT_RSI = "rsi_decline"
EXIT_MA5 = "macd_ma5"
EXIT_LABELS = {EXIT_RSI: "RSI decline-watch", EXIT_MA5: "MACD/MA5"}
STRATEGY_NAMES = {"A": "A · Pure RSI mean reversion", "B": "B · RSI mean reversion + MACD filter"}
DEFAULT_LENGTHS = (2, 3, 4)


@dataclass(frozen=True)
class ABConfig:
    rsi_length: int = 2
    entry_level: float = 10.0
    overbought_level: float = 80.0
    ma_exit_len: int = 5
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    commission_bps: float = 3.0
    slippage_bps: float = 1.0
    initial_capital: float = 10_000.0

    def __post_init__(self):
        for name in ("rsi_length", "ma_exit_len", "macd_fast", "macd_slow", "macd_signal"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive integer (got {v!r})")
        if not 0 <= self.entry_level < self.overbought_level <= 100:
            raise ValueError("need 0 <= entry_level < overbought_level <= 100")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be < macd_slow")
        if self.commission_bps < 0 or self.slippage_bps < 0 or self.initial_capital <= 0:
            raise ValueError("costs must be >= 0 and initial_capital > 0")

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


# ── Indicators ────────────────────────────────────────────────────────────────

def macd_buy_state(close: pd.Series, fast: int, slow: int, signal: int) -> np.ndarray:
    """
    True on the FIRST bar of a fresh 2-in-a-row rising MACD-histogram streak
    (continuations of an existing streak are False). Uses the same EMA MACD as
    macd_backtest (pandas ewm, adjust=False) so it matches the Histogram strategy.
    """
    c = close.astype(float)
    macd = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    hist = macd - macd.ewm(span=signal, adjust=False).mean()
    return fresh_rising_streak(hist)


def fresh_rising_streak(hist: pd.Series) -> np.ndarray:
    """True where hist rose on this bar and the previous one, but not the one before."""
    h = pd.Series(np.asarray(hist, dtype=float))
    up = (h > h.shift(1)).to_numpy(bool)
    out = np.zeros(len(h), dtype=bool)
    for t in range(3, len(h)):
        out[t] = up[t] and up[t - 1] and not up[t - 2]
    return out


@dataclass
class ABInputs:
    """Arrays the loop reads — tests can build these by hand."""
    open: np.ndarray
    close: np.ndarray
    rsi: np.ndarray
    ma: np.ndarray
    macd_buy: np.ndarray


def make_inputs(df: pd.DataFrame, cfg: ABConfig) -> ABInputs:
    c = df["Close"].astype(float)
    return ABInputs(
        open=df["Open"].to_numpy(float), close=c.to_numpy(float),
        rsi=rsi_ind(c, cfg.rsi_length).to_numpy(float), ma=sma(c, cfg.ma_exit_len).to_numpy(float),
        macd_buy=macd_buy_state(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal),
    )


# ── The bar loop ──────────────────────────────────────────────────────────────

@dataclass
class ABRun:
    strategy: str
    config: ABConfig
    fills: list[Fill]
    pending_buy: bool                       # a BUY fired on the last bar (fills at the next open)
    start_bar: int
    ledger: pd.DataFrame | None = None
    equity: pd.DataFrame | None = None
    open_trades: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _cross_under(r, i, level):
    return not (np.isnan(r[i]) or np.isnan(r[i - 1])) and r[i - 1] >= level and r[i] < level


def _cross_over(r, i, level):
    return not (np.isnan(r[i]) or np.isnan(r[i - 1])) and r[i - 1] <= level and r[i] > level


def run_strategy(x: ABInputs, cfg: ABConfig, strategy: str, start_bar: int = 1) -> tuple[list[Fill], bool]:
    """
    Returns (fills, pending_buy). Orders decided at bar i's close fill at bar
    i+1's open; an order still pending after the last bar stays unfilled (an
    open trade with a pending sell is tagged sell_pending).
    """
    if strategy not in ("A", "B"):
        raise ValueError("strategy must be 'A' or 'B'")
    r, n = x.rsi, len(x.close)
    fills: list[Fill] = []
    pos: Fill | None = None
    buy_order = False
    sell_order: str | None = None
    armed = False               # RSI decline watch
    ma_path = False             # Strategy B: exit on close < MA
    ma_from = None              # first bar the MA exit is checked
    macd_seen = False           # Strategy B: BUY state since entry

    for i in range(max(start_bar, 1), n):
        # ── 1. fills at this bar's open ──────────────────────────────────────
        if sell_order is not None and pos is not None:
            pos.exit_bar, pos.exit_price, pos.exit_reason = i, float(x.open[i]), sell_order
            pos.tags["exit_type"] = EXIT_LABELS[sell_order]
            pos, sell_order = None, None
        if buy_order and pos is None:
            pos = Fill(entry_bar=i, entry_price=float(x.open[i]), entry_timing="open", exit_timing="open",
                       tags={"strategy": strategy, "rsi_length": cfg.rsi_length, "exit_type": None,
                             "macd_seen_before_80": None})
            fills.append(pos)
            buy_order = False
            armed, ma_path, ma_from, macd_seen = False, False, None, False

        # ── 2. decisions at this bar's close ─────────────────────────────────
        if pos is not None:
            if sell_order is not None:
                continue                                    # already leaving at the next open
            if strategy == "B" and x.macd_buy[i]:
                macd_seen = True                            # counts from the entry bar on
            if ma_path:
                if i >= ma_from and not np.isnan(x.ma[i]) and x.close[i] < x.ma[i]:
                    sell_order = EXIT_MA5
            elif armed:
                if not (np.isnan(r[i]) or np.isnan(r[i - 1])) and r[i] < r[i - 1]:
                    sell_order = EXIT_RSI
            elif _cross_over(r, i, cfg.overbought_level):
                if strategy == "B":
                    pos.tags["macd_seen_before_80"] = macd_seen
                if strategy == "B" and macd_seen:
                    ma_path, ma_from = True, i + 1
                    pos.tags["exit_type"] = EXIT_LABELS[EXIT_MA5]
                else:
                    armed = True
                    pos.tags["exit_type"] = EXIT_LABELS[EXIT_RSI]
        elif not buy_order and _cross_under(r, i, cfg.entry_level):
            buy_order = True

    if pos is not None and sell_order is not None:
        pos.tags["sell_pending"] = True
    return fills, buy_order


def run_ab(df: pd.DataFrame, cfg: ABConfig, strategy: str, start_bar: int = 1,
           inputs: ABInputs | None = None) -> ABRun:
    x = inputs if inputs is not None else make_inputs(df, cfg)
    fills, pending = run_strategy(x, cfg, strategy, start_bar)
    sim = simulate(df, fills, total_bps=cfg.total_bps, initial_capital=cfg.initial_capital, start_bar=start_bar)
    summary = M.summarize(sim.ledger, sim.equity, sim.open_trades)
    run = ABRun(strategy=strategy, config=cfg, fills=fills, pending_buy=pending, start_bar=start_bar,
                ledger=sim.ledger, equity=sim.equity, open_trades=sim.open_trades, summary=summary)
    return run


# ── Suite: A and B × RSI lengths ──────────────────────────────────────────────

@dataclass
class ABSuite:
    runs: dict                      # (strategy, length) -> ABRun
    comparison: pd.DataFrame
    trades: pd.DataFrame
    start_bar: int


def _exit_share(ledger: pd.DataFrame, reason: str) -> float | None:
    if ledger is None or not len(ledger):
        return None
    return float((ledger["exit_reason"] == reason).mean() * 100)


def comparison_table(runs: dict) -> pd.DataFrame:
    rows = []
    for (strat, length), run in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        s, led = run.summary, run.ledger
        pf = s.get("profit_factor")
        rows.append({
            "RSI length": length,
            "Strategy": strat,
            "Total trades": s.get("trades", 0),
            "Win rate %": s.get("win_rate_pct"),
            "Avg win %": s.get("avg_win_pct"),
            "Avg loss %": s.get("avg_loss_pct"),
            "Profit factor": np.inf if pf == "inf" else pf,
            "Total return %": s.get("total_return_pct"),
            "Max drawdown %": s.get("max_drawdown_pct"),
            "Avg holding (bars)": s.get("avg_bars_held"),
            "MACD/MA5 exits %": _exit_share(led, EXIT_MA5) if strat == "B" else None,
            "RSI decline exits %": _exit_share(led, EXIT_RSI) if strat == "B" else None,
            "Open at end": len(run.open_trades),
        })
    return pd.DataFrame(rows)


def trade_log(runs: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for (strat, length), run in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        for _, t in run.ledger.iterrows():
            rows.append({
                "Entry date": pd.Timestamp(t.entry_date).strftime("%Y-%m-%d"), "Entry price": t.entry_price,
                "Exit date": pd.Timestamp(t.exit_date).strftime("%Y-%m-%d"), "Exit price": t.exit_price,
                "RSI length": length, "Strategy": strat, "Exit type": EXIT_LABELS.get(t.exit_reason, t.exit_reason),
                "Trade return %": t.net_ret_pct, "Gross return %": t.gross_ret_pct,
                "Bars held": int(t.bars_held), "Status": "closed",
            })
        for ot in run.open_trades:
            rows.append({
                "Entry date": pd.Timestamp(ot["entry_date"]).strftime("%Y-%m-%d"), "Entry price": ot["entry_price"],
                "Exit date": "", "Exit price": np.nan, "RSI length": length, "Strategy": strat,
                "Exit type": ("sell pending (next open)" if ot.get("sell_pending") else
                              (f"armed: {ot['exit_type']}" if ot.get("exit_type") else "open — no exit armed")),
                "Trade return %": ot["unrealized_pct"], "Gross return %": np.nan,
                "Bars held": int(ot["bars_held"]), "Status": "open (marked to last close)",
            })
    return pd.DataFrame(rows, columns=["Entry date", "Entry price", "Exit date", "Exit price", "RSI length",
                                       "Strategy", "Exit type", "Trade return %", "Gross return %", "Bars held", "Status"])


def run_suite(df: pd.DataFrame, lengths=DEFAULT_LENGTHS, start_bar: int | None = None,
              base: ABConfig | None = None) -> ABSuite:
    """Run Strategy A and B for every RSI length on the same bars and window."""
    base = base or ABConfig()
    if start_bar is None:
        start_bar = min(len(df) - 1, max(100, 3 * base.macd_slow))   # MACD warm-up
    runs = {}
    for length in lengths:
        cfg = replace(base, rsi_length=int(length))
        x = make_inputs(df, cfg)
        for strat in ("A", "B"):
            runs[(strat, int(length))] = run_ab(df, cfg, strat, start_bar, inputs=x)
    return ABSuite(runs=runs, comparison=comparison_table(runs), trades=trade_log(runs, df.index),
                   start_bar=start_bar)


# ── CLI: python -m rsi2.ab --ticker AMD ───────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    from .data import load_csv, load_yfinance, sanitize_ohlc, validate
    p = argparse.ArgumentParser(prog="rsi2.ab", description="Strategy A / B × RSI length backtest")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker")
    src.add_argument("--csv")
    p.add_argument("--start", help="data start (YYYY-MM-DD) for --ticker")
    p.add_argument("--end", help="data end (YYYY-MM-DD) for --ticker")
    p.add_argument("--lengths", default="2,3,4", help="RSI lengths, comma-separated (default 2,3,4)")
    p.add_argument("--out-dir")
    p.add_argument("--repair-ohlc", action="store_true")
    for f in ("entry_level", "overbought_level", "commission_bps", "slippage_bps", "initial_capital"):
        p.add_argument("--" + f.replace("_", "-"), type=float)
    for f in ("ma_exit_len", "macd_fast", "macd_slow", "macd_signal"):
        p.add_argument("--" + f.replace("_", "-"), type=int)
    a = p.parse_args(argv)
    try:
        overrides = {k: v for k, v in vars(a).items()
                     if k in ABConfig.__dataclass_fields__ and v is not None}
        base = ABConfig(**overrides)
        if a.ticker:
            df, name = load_yfinance(a.ticker, a.start, a.end), a.ticker.upper()
        else:
            df, name = load_csv(a.csv), os.path.splitext(os.path.basename(a.csv))[0]
        if a.repair_ohlc:
            df, _ = sanitize_ohlc(df)
        df = validate(df, min_rows=250)
        lengths = [int(v) for v in a.lengths.split(",") if v.strip()]
        suite = run_suite(df, lengths, base=base)
    except (ValueError, TypeError) as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1
    out = a.out_dir or os.path.join("results", f"{name}_AB_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True)
    suite.comparison.to_csv(os.path.join(out, "comparison.csv"), index=False, float_format="%.4f")
    suite.trades.to_csv(os.path.join(out, "trades.csv"), index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        print(f"{name}: {df.index[suite.start_bar]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}, "
              f"costs {base.total_bps:g} bps/side, fills at next open\n")
        print(suite.comparison.to_string(index=False))
    print(f"\nWrote comparison.csv and trades.csv to {out}")
    return 0 if len(suite.trades) else 2


if __name__ == "__main__":
    sys.exit(main())
