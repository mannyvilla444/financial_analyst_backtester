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
  No stop-loss, no time exit by default: if RSI never crosses over 80 the
  trade rides. Optional stops (below) can be switched on for both A and B.

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

OPTIONAL STOPS (ABConfig.atr_stop_mult / time_stop_bars; presets in STOP_PRESETS)
  ATR stop   level = entry price − mult × ATR(14), using the ATR at the signal
             bar's close (known when the order was placed). Fixed for the life
             of the trade. A daily CLOSE below the level → sell at the next open
             (so a gap can fill below the level). Active in every phase.
  Time stop  if the sell is not armed — RSI hasn't crossed over 80 (A's decline
             watch, or B's MA5 switch) — by the close of the Nth bar in the trade
             (entry bar = 1), sell at the next open.
  The strategy's own exit is checked first; a stop only fires if it didn't.

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
from .indicators import atr as atr_ind, rsi as rsi_ind, sma
from .portfolio import Fill, simulate

EXIT_RSI = "rsi_decline"
EXIT_MA5 = "macd_ma5"
EXIT_ATR = "atr_stop"
EXIT_TIME = "time_stop"
EXIT_LABELS = {EXIT_RSI: "RSI decline-watch", EXIT_MA5: "MACD/MA5", EXIT_ATR: "ATR stop", EXIT_TIME: "Time stop"}
STOP_EXITS = (EXIT_ATR, EXIT_TIME)
STRATEGY_NAMES = {"A": "A · Pure RSI mean reversion", "B": "B · RSI mean reversion + MACD filter"}
DEFAULT_LENGTHS = (2, 3, 4)
# Time-stop length scales with the RSI length: longer RSIs take longer to reach 80.
TIME_STOP_BARS = {2: 10, 3: 15, 4: 20}


def default_time_stop(rsi_length: int) -> int:
    return TIME_STOP_BARS.get(int(rsi_length), 5 * int(rsi_length))


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
    atr_stop_mult: float | None = None     # e.g. 3.0 → stop at entry − 3 × ATR; None = off
    atr_len: int = 14
    time_stop_bars: int | None = None      # e.g. 10 → exit if not armed by the 10th bar; None = off

    def __post_init__(self):
        for name in ("rsi_length", "ma_exit_len", "macd_fast", "macd_slow", "macd_signal", "atr_len"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive integer (got {v!r})")
        if not 0 <= self.entry_level < self.overbought_level <= 100:
            raise ValueError("need 0 <= entry_level < overbought_level <= 100")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be < macd_slow")
        if self.commission_bps < 0 or self.slippage_bps < 0 or self.initial_capital <= 0:
            raise ValueError("costs must be >= 0 and initial_capital > 0")
        if self.atr_stop_mult is not None and not (isinstance(self.atr_stop_mult, (int, float))
                                                   and not isinstance(self.atr_stop_mult, bool)
                                                   and 0 < self.atr_stop_mult < 50):
            raise ValueError(f"atr_stop_mult must be a positive number or None (got {self.atr_stop_mult!r})")
        if self.time_stop_bars is not None and (not isinstance(self.time_stop_bars, int)
                                                or isinstance(self.time_stop_bars, bool) or self.time_stop_bars < 1):
            raise ValueError(f"time_stop_bars must be a positive integer or None (got {self.time_stop_bars!r})")

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps

    @property
    def stop_label(self) -> str:
        bits = []
        if self.time_stop_bars:
            bits.append(f"time {self.time_stop_bars} bars")
        if self.atr_stop_mult:
            bits.append(f"{self.atr_stop_mult:g}× ATR({self.atr_len})")
        return " + ".join(bits) if bits else "no stop"


# ── Stop presets (the Backtest tab compares all of them) ─────────────────────

@dataclass(frozen=True)
class StopRule:
    key: str
    label: str
    atr_mult: float | None = None
    time_stop: bool = False


STOP_PRESETS = {r.key: r for r in (
    StopRule("none", "No stop"),
    StopRule("time", "Time stop", time_stop=True),
    StopRule("atr2", "2× ATR", atr_mult=2.0),
    StopRule("atr2.5", "2.5× ATR", atr_mult=2.5),
    StopRule("atr3", "3× ATR", atr_mult=3.0),
    StopRule("time_atr3", "Time + 3× ATR", atr_mult=3.0, time_stop=True),
)}


def with_stop(cfg: ABConfig, stop: str) -> ABConfig:
    """cfg with the preset's stops applied (time-stop length follows cfg.rsi_length)."""
    if stop not in STOP_PRESETS:
        raise ValueError(f"unknown stop preset {stop!r} (choose from {', '.join(STOP_PRESETS)})")
    rule = STOP_PRESETS[stop]
    return replace(cfg, atr_stop_mult=rule.atr_mult,
                   time_stop_bars=default_time_stop(cfg.rsi_length) if rule.time_stop else None)


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
    atr: np.ndarray | None = None          # needed only when the ATR stop is on


def make_inputs(df: pd.DataFrame, cfg: ABConfig) -> ABInputs:
    c = df["Close"].astype(float)
    return ABInputs(
        open=df["Open"].to_numpy(float), close=c.to_numpy(float),
        rsi=rsi_ind(c, cfg.rsi_length).to_numpy(float), ma=sma(c, cfg.ma_exit_len).to_numpy(float),
        macd_buy=macd_buy_state(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal),
        atr=atr_ind(df, cfg.atr_len).to_numpy(float),
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
    if cfg.atr_stop_mult and x.atr is None:
        raise ValueError("the ATR stop needs inputs.atr")
    fills: list[Fill] = []
    pos: Fill | None = None
    buy_order = False
    sell_order: str | None = None
    armed = False               # RSI decline watch
    ma_path = False             # Strategy B: exit on close < MA
    ma_from = None              # first bar the MA exit is checked
    macd_seen = False           # Strategy B: BUY state since entry
    stop_level = None           # ATR stop price for the open trade

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
            stop_level = None
            if cfg.atr_stop_mult:
                a = x.atr[i - 1]                            # ATR at the signal bar's close
                if not np.isnan(a):
                    stop_level = pos.entry_price - cfg.atr_stop_mult * float(a)
                    pos.tags["stop_level"] = stop_level

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
            # ── optional stops (only if the strategy's own exit didn't fire) ──
            if sell_order is None:
                if stop_level is not None and x.close[i] < stop_level:
                    sell_order = EXIT_ATR
                elif (cfg.time_stop_bars and not armed and not ma_path
                      and i - pos.entry_bar + 1 >= cfg.time_stop_bars):
                    sell_order = EXIT_TIME
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
    stop: str = "none"


def _exit_share(ledger: pd.DataFrame, reasons) -> float | None:
    if ledger is None or not len(ledger):
        return None
    reasons = (reasons,) if isinstance(reasons, str) else tuple(reasons)
    return float(ledger["exit_reason"].isin(reasons).mean() * 100)


COMPARISON_COLUMNS = ["RSI length", "Strategy", "Total trades", "Win rate %", "Avg win %", "Avg loss %",
                      "Profit factor", "Total return %", "Max drawdown %", "Avg holding (bars)",
                      "MACD/MA5 exits %", "RSI decline exits %", "Stop exits %", "Worst trade %",
                      "Open at end", "Stop-loss"]


def comparison_table(runs: dict, stop_label: str = "No stop") -> pd.DataFrame:
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
            "Stop exits %": _exit_share(led, STOP_EXITS),
            "Worst trade %": s.get("largest_loss_pct"),
            "Open at end": len(run.open_trades),
            "Stop-loss": stop_label,
        })
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


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
                "Stop level": t.get("stop_level", np.nan),
            })
        for ot in run.open_trades:
            rows.append({
                "Entry date": pd.Timestamp(ot["entry_date"]).strftime("%Y-%m-%d"), "Entry price": ot["entry_price"],
                "Exit date": "", "Exit price": np.nan, "RSI length": length, "Strategy": strat,
                "Exit type": ("sell pending (next open)" if ot.get("sell_pending") else
                              (f"armed: {ot['exit_type']}" if ot.get("exit_type") else "open — no exit armed")),
                "Trade return %": ot["unrealized_pct"], "Gross return %": np.nan,
                "Bars held": int(ot["bars_held"]), "Status": "open (marked to last close)",
                "Stop level": ot.get("stop_level", np.nan),
            })
    return pd.DataFrame(rows, columns=["Entry date", "Entry price", "Exit date", "Exit price", "RSI length",
                                       "Strategy", "Exit type", "Trade return %", "Gross return %", "Bars held",
                                       "Status", "Stop level"])


def default_start_bar(df: pd.DataFrame, base: ABConfig) -> int:
    return min(len(df) - 1, max(100, 3 * base.macd_slow))          # MACD warm-up


def run_suite(df: pd.DataFrame, lengths=DEFAULT_LENGTHS, start_bar: int | None = None,
              base: ABConfig | None = None, stop: str | None = None, _inputs: dict | None = None) -> ABSuite:
    """
    Run Strategy A and B for every RSI length on the same bars and window.
    stop: a STOP_PRESETS key applied to every run (None = use base's own stop
    fields as given, which default to no stop).
    """
    base = base or ABConfig()
    if start_bar is None:
        start_bar = default_start_bar(df, base)
    runs, label = {}, None
    for length in lengths:
        cfg = replace(base, rsi_length=int(length))
        if stop is not None:
            cfg = with_stop(cfg, stop)
        if _inputs is not None and int(length) in _inputs:
            x = _inputs[int(length)]
        else:
            x = make_inputs(df, cfg)
            if _inputs is not None:
                _inputs[int(length)] = x
        for strat in ("A", "B"):
            runs[(strat, int(length))] = run_ab(df, cfg, strat, start_bar, inputs=x)
    if stop is not None:
        label = STOP_PRESETS[stop].label
    elif base.atr_stop_mult or base.time_stop_bars:
        label = base.stop_label
    else:
        label = STOP_PRESETS["none"].label
    return ABSuite(runs=runs, comparison=comparison_table(runs, label), trades=trade_log(runs, df.index),
                   start_bar=start_bar, stop=stop or "custom")


def run_stop_grid(df: pd.DataFrame, lengths=DEFAULT_LENGTHS, start_bar: int | None = None,
                  base: ABConfig | None = None, stops=tuple(STOP_PRESETS)) -> pd.DataFrame:
    """
    Every stop preset × Strategy A/B × RSI length on the same bars and window
    (6 presets × 6 runs = 36 by default). One comparison row per run, with a
    "Stop-loss" column naming the preset and "Return vs no stop (pts)".
    """
    base = base or ABConfig()
    if start_bar is None:
        start_bar = default_start_bar(df, base)
    cache: dict = {}
    tables = [run_suite(df, lengths, start_bar, base, stop=k, _inputs=cache).comparison for k in stops]
    grid = pd.concat(tables, ignore_index=True)
    grid = grid[["Stop-loss"] + [c for c in grid.columns if c != "Stop-loss"]]
    none = grid[grid["Stop-loss"] == STOP_PRESETS["none"].label].set_index(["RSI length", "Strategy"])["Total return %"]
    grid["Return vs no stop (pts)"] = [
        (r["Total return %"] - none.get((r["RSI length"], r["Strategy"]), np.nan)) if len(none) else np.nan
        for _, r in grid.iterrows()]
    return grid


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
    p.add_argument("--stop", choices=list(STOP_PRESETS), help="stop preset for every run (default: none)")
    p.add_argument("--atr-stop-mult", type=float, help="custom ATR stop multiple, e.g. 3")
    p.add_argument("--time-stop-bars", type=int, help="custom time stop in bars, e.g. 10")
    p.add_argument("--compare-stops", action="store_true",
                   help="also run every stop preset and write stop_comparison.csv")
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
        if a.stop and (a.atr_stop_mult or a.time_stop_bars):
            raise ValueError("use either --stop or --atr-stop-mult/--time-stop-bars, not both")
        suite = run_suite(df, lengths, base=base, stop=a.stop)
        grid = run_stop_grid(df, lengths, base=replace(base, atr_stop_mult=None, time_stop_bars=None)) \
            if a.compare_stops else None
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
        print(f"Stop-loss: {suite.comparison['Stop-loss'].iloc[0] if len(suite.comparison) else '—'}")
        print(suite.comparison.drop(columns=["Stop-loss"]).to_string(index=False))
        if grid is not None:
            grid.to_csv(os.path.join(out, "stop_comparison.csv"), index=False, float_format="%.4f")
            print("\nStop-loss comparison (total return % / max drawdown %):")
            piv = grid.pivot_table(index=["RSI length", "Strategy"], columns="Stop-loss",
                                   values=["Total return %", "Max drawdown %"], sort=False)
            print(piv.round(1).to_string())
    print(f"\nWrote comparison.csv and trades.csv{' and stop_comparison.csv' if grid is not None else ''} to {out}")
    return 0 if len(suite.trades) else 2


if __name__ == "__main__":
    sys.exit(main())
