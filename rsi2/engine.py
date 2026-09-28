"""
The RSI(2) bar loop (spec §4, §5, §6).

Execution model (Pine process_orders_on_close = true):
  bar i            signal: close > trend and RSI crosses under entry_level
  bar i+delay      entry fills at that bar's close
  bar entry+1 ...  first bar an exit can happen (minimum hold = 1 bar)

Everything decided on bar i uses data at indices <= i. The only forward
reference is the deliberate signal → entry delay.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import validate
from .indicators import ema, rsi, sma
from .portfolio import Fill, SimResult, simulate


@dataclass
class Arrays:
    """Plain numpy views the loop reads (so tests can inject crafted indicators)."""
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    trend: np.ndarray
    fast: np.ndarray
    rsi: np.ndarray


@dataclass
class RunResult:
    config: StrategyConfig
    data: pd.DataFrame
    indicators: pd.DataFrame
    signals: pd.Series
    fills: list[Fill]
    sim: SimResult
    start_bar: int          # first bar of the equity curve / entry window

    @property
    def ledger(self) -> pd.DataFrame:
        return self.sim.ledger

    @property
    def equity(self) -> pd.DataFrame:
        return self.sim.equity

    @property
    def open_trades(self) -> list[dict]:
        return self.sim.open_trades


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    c = df["Close"].astype(float)
    trend = ema(c, cfg.trend_len) if cfg.trend_type == "EMA" else sma(c, cfg.trend_len)
    return pd.DataFrame({"trend": trend, "fast": sma(c, cfg.fast_len), "rsi": rsi(c, cfg.rsi_len)}, index=df.index)


def make_arrays(df: pd.DataFrame, ind: pd.DataFrame) -> Arrays:
    return Arrays(
        open=df["Open"].to_numpy(float), high=df["High"].to_numpy(float),
        low=df["Low"].to_numpy(float), close=df["Close"].to_numpy(float),
        trend=ind["trend"].to_numpy(float), fast=ind["fast"].to_numpy(float),
        rsi=ind["rsi"].to_numpy(float),
    )


def entry_signals(a: Arrays, cfg: StrategyConfig) -> np.ndarray:
    """signal[i] = close[i] > trend[i] and a true crossunder of RSI through entry_level."""
    n = len(a.close)
    sig = np.zeros(n, dtype=bool)
    for i in range(1, n):
        r, rp, t = a.rsi[i], a.rsi[i - 1], a.trend[i]
        if np.isnan(r) or np.isnan(rp) or np.isnan(t):
            continue
        sig[i] = a.close[i] > t and r < cfg.entry_level and rp >= cfg.entry_level
    return sig


# ── Per-bar rules, shared with the combined strategies in regime_lab.py ──────

def entry_ok(i: int, a: Arrays, sig: np.ndarray, cfg: StrategyConfig, in_window: bool) -> bool:
    """Flat on bar i: enter at close[i] if a signal fired entry_delay bars ago."""
    j = i - cfg.entry_delay
    if j < 0 or not sig[j] or not in_window:
        return False
    if cfg.recheck_filter:
        t = a.trend[i]
        return not np.isnan(t) and a.close[i] > t
    return True


def is_flagged(i: int, pos: Fill, a: Arrays, cfg: StrategyConfig) -> bool:
    """Pattern flag: RSI still below flag_level on the bar right after the fill."""
    return i == pos.entry_bar + 1 and not np.isnan(a.rsi[i]) and a.rsi[i] < cfg.flag_level


def stop_fill(i: int, pos: Fill, a: Arrays, cfg: StrategyConfig) -> float | None:
    """Intrabar catastrophic stop; a gap below the stop fills at the open."""
    if not cfg.use_stop:
        return None
    stop_price = pos.entry_price * (1 - cfg.stop_pct / 100.0)
    if a.low[i] <= stop_price:
        return min(a.open[i], stop_price)
    return None


def close_exit_reason(i: int, pos: Fill, a: Arrays, cfg: StrategyConfig) -> str | None:
    """Close-based exits in priority order: target, trend_break, time_stop."""
    r = a.rsi[i]
    if not np.isnan(r):                       # NaN RSI = no signal, no exit
        rsi_hit = r >= cfg.exit_level
        above_fast = (not np.isnan(a.fast[i])) and a.close[i] > a.fast[i]
        core = (rsi_hit and above_fast) if cfg.exit_logic == "AND" else (rsi_hit or above_fast)
        if core:
            return "target"
    if cfg.use_trend_stop and not np.isnan(a.trend[i]) and a.close[i] < a.trend[i]:
        return "trend_break"
    if cfg.use_timeout and (i - pos.entry_bar) >= cfg.max_bars:
        return "time_stop"
    return None


def loop_start(cfg: StrategyConfig) -> int:
    return cfg.trend_len + 1


def window_bounds(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[int, np.ndarray]:
    """(first equity bar, entry-window mask)."""
    start = loop_start(cfg)
    in_win = np.ones(len(df), dtype=bool)
    if cfg.start_date is not None:
        in_win &= df.index >= pd.Timestamp(cfg.start_date)
        first = int(np.argmax(df.index >= pd.Timestamp(cfg.start_date))) if in_win.any() else len(df)
        start = max(start, first)
    return start, in_win


def prepare(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = validate(df)
    if cfg.end_date is not None:
        df = df.loc[: pd.Timestamp(cfg.end_date)]
    return validate(df, min_rows=cfg.trend_len + 250)


def run(df: pd.DataFrame, cfg: StrategyConfig | None = None,
        indicators: pd.DataFrame | None = None, validate_data: bool = True) -> RunResult:
    """Backtest RSI(2) mean reversion on daily OHLC data."""
    cfg = cfg or StrategyConfig()
    if validate_data:
        df = prepare(df, cfg)
    ind = indicators if indicators is not None else compute_indicators(df, cfg)
    a = make_arrays(df, ind)
    sig = entry_signals(a, cfg)
    start, in_win = window_bounds(df, cfg)

    fills: list[Fill] = []
    pos: Fill | None = None
    for i in range(start, len(df)):
        if pos is not None:
            if is_flagged(i, pos, a, cfg):
                pos.flagged = True
            px = stop_fill(i, pos, a, cfg)
            if px is not None:
                pos.exit_bar, pos.exit_price, pos.exit_reason = i, px, "stop"
                pos = None
                continue
            reason = close_exit_reason(i, pos, a, cfg)
            if reason is not None:
                pos.exit_bar, pos.exit_price, pos.exit_reason = i, a.close[i], reason
                pos = None
        elif entry_ok(i, a, sig, cfg, bool(in_win[i])):
            pos = Fill(entry_bar=i, entry_price=float(a.close[i]))
            fills.append(pos)

    sim = simulate(df, fills, total_bps=cfg.total_bps, initial_capital=cfg.initial_capital,
                   position_pct=cfg.position_pct, whole_shares=cfg.whole_shares, start_bar=start)
    return RunResult(config=cfg, data=df, indicators=ind, signals=pd.Series(sig, index=df.index),
                     fills=fills, sim=sim, start_bar=start)
