"""
regime_lab.py — backtest momentum and mean-reversion strategies side by side,
on the same data, window, costs and accounting.

STRATEGIES
──────────
Momentum (reference, existing engines in macd_backtest.py):
  • MACD Histogram
  • MACD Crossover + MA50
  Run on the selected timeframe (weekly candles are built from the daily data
  and dated by their last trading day). Parameters: best of the same 12-combo
  sweep as before, shown in full in the report.

Mean reversion (rsi2/ab.py), always on daily bars, for RSI length 2, 3 and 4:
  • Strategy A — Pure RSI mean reversion: buy when RSI crosses under 10 (next
    open); after RSI crosses over 80 arm a decline watch; sell at the next open
    after the first bar whose RSI is below the previous bar's. No stop, no time exit.
  • Strategy B — Same entry. If a MACD-histogram BUY state (first bar of a fresh
    2-rising-bars streak) occurred since entry when RSI crosses over 80, exit
    instead on the first close below MA5 (next open); otherwise as Strategy A.

FAIR COMPARISON
───────────────
All strategies trade only inside one common window (after every indicator has
warmed up), pay the same costs (4 bps per side), invest 100% of equity per
trade and are marked to market daily; buy & hold is the benchmark.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd

import macd_backtest as mb
from rsi2 import metrics as M
from rsi2.ab import ABConfig, EXIT_LABELS, STOP_EXITS, STOP_PRESETS, default_time_stop, run_stop_grid, run_suite
from rsi2.data import sanitize_ohlc, validate
from rsi2.portfolio import Fill, simulate

HISTORY_YEARS = 6
CAPITAL = 10_000.0
COST_BPS = 4.0                        # per side: 3 commission + 1 slippage
MOMENTUM_WARMUP_CANDLES = 53          # MA50 + 3-candle slope check
MACD_WARMUP_BARS = 100                # daily MACD (Strategy B) settles
MIN_TRADES = {"histogram": 10, "crossover": 5}   # same selection rule as macd_backtest.run_all
TIMEFRAMES = {"weekly": ("1wk", "Weekly"), "daily": ("1d", "Daily")}
RSI_LENGTHS = (2, 3, 4)
AB_CONFIG = ABConfig(commission_bps=3.0, slippage_bps=1.0, initial_capital=CAPITAL)

STRATEGY_INFO = {
    "histogram": ("MACD Histogram", "Momentum",
                  "BUY on the first 2-candle rise of the MACD histogram · SELL after 2 falling candles · ATR stop."),
    "crossover": ("MACD Crossover + MA50", "Momentum",
                  "BUY when MACD crosses above signal with MA50 rising, or price reclaims MA50 while MACD is "
                  "bullish · SELL on MACD cross below signal · ATR stop."),
}
for _L in RSI_LENGTHS:
    STRATEGY_INFO[f"A{_L}"] = (
        f"A · RSI({_L})", "Mean reversion",
        f"Buy at the next open when RSI({_L}) crosses under 10. After RSI crosses over 80, sell at the next open "
        f"following the first bar where RSI falls.")
    STRATEGY_INFO[f"B{_L}"] = (
        f"B · RSI({_L}) + MACD", "Mean rev + MACD",
        f"Same entry as A. When RSI({_L}) crosses over 80: if a MACD-histogram BUY state happened since entry, "
        f"sell at the next open after the first close below MA5; otherwise use A's RSI decline watch.")




def stop_rule_text(stop: str, rsi_length: int) -> str:
    """Plain-English stop rule for one preset at one RSI length."""
    rule = STOP_PRESETS[stop]
    bits = []
    if rule.time_stop:
        bits.append(f"Time stop: if RSI hasn't crossed over 80 by the close of bar {default_time_stop(rsi_length)} "
                    f"of the trade, sell at the next open.")
    if rule.atr_mult:
        bits.append(f"ATR stop: a close below entry − {rule.atr_mult:g}× ATR(14) sells at the next open.")
    return " ".join(bits) if bits else "No stop-loss, no time exit."


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_daily(ticker: str, years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Split/dividend-adjusted daily OHLCV (auto_adjust=True), tz-naive, one row per date."""
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")
    start = (datetime.now() - timedelta(days=int(365.25 * years) + 10)).strftime("%Y-%m-%d")
    df = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No price data found for '{ticker}'")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    cols = {c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
    df = df.rename(columns=cols)[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in cols.values()]]
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def momentum_frame(daily: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Candles for the momentum strategies (lowercase columns, as macd_backtest
    expects). Weekly candles are aggregated from the daily bars and indexed by
    the date of their LAST trading day, so every weekly date exists in the
    daily index and a candle is only "known" at that day's close.
    """
    d = daily.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    if timeframe == "daily":
        return d.copy()
    week = d.index.to_period("W-FRI")
    wk = d.groupby(week).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    wk.index = pd.DatetimeIndex(d.index.to_series().groupby(week).last().values)
    return wk


# ── Momentum signals as per-candle events ─────────────────────────────────────

def momentum_events(frame: pd.DataFrame, kind: str, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """BUY / SELL condition on each completed candle — the same rules the
    macd_backtest engines trade on, using only data up to that candle."""
    close = frame["close"].astype(float)
    macd, sig, hist = mb.calc_macd(close, fast, slow, signal)
    atr = mb.calc_atr(frame)
    if kind == "histogram":
        up = (hist > hist.shift(1)).fillna(False)
        down = (hist < hist.shift(1)).fillna(False)
        buy = up & up.shift(1, fill_value=False) & ~up.shift(2, fill_value=True)
        sell = down & down.shift(1, fill_value=False)
    else:
        ma = mb.calc_ma(close, 50)
        cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        cross_down = (macd < sig) & (macd.shift(1) >= sig.shift(1))
        ma_up = ma > ma.shift(3)
        reclaim = (close.shift(1) < ma.shift(1)) & (close > ma)
        buy = (cross_up & ma_up) | ((macd > sig) & reclaim)
        sell = cross_down
    return pd.DataFrame({"buy": buy.astype(bool), "sell": sell.astype(bool), "atr": atr}, index=frame.index)


@dataclass
class DailyEvents:
    buy: np.ndarray
    sell: np.ndarray
    atr: np.ndarray


def events_on_daily(ev: pd.DataFrame, daily_index: pd.DatetimeIndex) -> DailyEvents:
    pos = daily_index.get_indexer(ev.index)
    if (pos < 0).any():
        raise AssertionError("Momentum candle dates must exist in the daily index")
    n = len(daily_index)
    buy, sell, atr = np.zeros(n, bool), np.zeros(n, bool), np.full(n, np.nan)
    buy[pos], sell[pos], atr[pos] = ev["buy"].to_numpy(), ev["sell"].to_numpy(), ev["atr"].to_numpy(float)
    return DailyEvents(buy, sell, atr)


# ── Momentum strategies (existing engines) → fills ────────────────────────────

def _trades_to_fills(trades, daily_index: pd.DatetimeIndex, start_bar: int) -> list[Fill]:
    fills = []
    for t in trades:
        e = daily_index.get_loc(pd.Timestamp(t.entry_date))
        if e < start_bar:
            continue
        if t.exit_reason == "open→close":          # still open at the end: mark to market, don't close
            fills.append(Fill(entry_bar=e, entry_price=float(t.entry_price)))
        else:
            fills.append(Fill(entry_bar=e, entry_price=float(t.entry_price),
                              exit_bar=daily_index.get_loc(pd.Timestamp(t.exit_date)),
                              exit_price=float(t.exit_price), exit_reason=t.exit_reason))
    return fills


def _quick_returns(fills: list[Fill], close: np.ndarray, bps: float) -> np.ndarray:
    c = bps / 10_000
    out = []
    for f in fills:
        px = close[-1] if f.is_open else f.exit_price
        out.append(((px * (1 - c)) / (f.entry_price * (1 + c)) - 1) * 100)
    return np.array(out)


def sweep_momentum(kind: str, frame: pd.DataFrame, daily: pd.DataFrame, ticker: str,
                   start_bar: int, interval: str, bps: float) -> tuple[list[dict], dict, list[Fill]]:
    """Run all 12 parameter sets; rank by net fixed-size PnL inside the window."""
    close = daily["Close"].to_numpy(float)
    runs = []
    for fast, slow, sig, atr_mult in product(mb.PARAM_GRID["fast"], mb.PARAM_GRID["slow"],
                                             mb.PARAM_GRID["signal"], mb.PARAM_GRID["atr_mult"]):
        if fast >= slow:
            continue
        engine = mb.run_backtest if kind == "histogram" else mb.run_crossover_backtest
        res = engine(frame, ticker, fast, slow, sig, atr_mult=atr_mult, position_size=CAPITAL,
                     sizing_mode="fixed", starting_capital=CAPITAL, interval=interval)
        fills = _trades_to_fills(res.trades, daily.index, start_bar)
        r = _quick_returns(fills, close, bps)
        closed = [f for f in fills if not f.is_open]
        pf = M.profit_factor(r) if len(r) else None
        label = res.params_label
        runs.append({
            "params": {"fast": fast, "slow": slow, "signal": sig, "atr_mult": atr_mult, "label": label},
            "fills": fills,
            "row": {"Params": label, "Trades": len(fills),
                    "Win %": f"{(r > 0).mean() * 100:.1f}%" if len(r) else "–",
                    "PnL ($10k/trade, net)": f"${r.sum() * CAPITAL / 100:+,.0f}",
                    "Avg trade": f"{r.mean():+.2f}%" if len(r) else "–",
                    "Profit factor": ("∞" if pf == float("inf") else f"{pf:.2f}") if pf is not None else "–",
                    "Stopped out": f"{sum(f.exit_reason == 'stop' for f in closed) / len(closed) * 100:.0f}%" if closed else "–"},
            "score": float(r.sum()) if len(r) else -1e18,
            "n": len(fills),
        })
    eligible = [x for x in runs if x["n"] >= MIN_TRADES[kind]] or runs
    best = max(eligible, key=lambda x: x["score"])
    rows = []
    for x in sorted(runs, key=lambda x: -x["score"]):
        rows.append({"✓": "★" if x is best else "", **x["row"]})
    return rows, best["params"], best["fills"]


@dataclass
class StrategyResult:
    key: str
    name: str
    family: str
    rules: str
    params: str = ""
    available: bool = True
    error: str | None = None
    summary: dict = field(default_factory=dict)
    ledger: pd.DataFrame | None = None
    equity: pd.Series | None = None
    open_trades: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def evaluate(key: str, daily: pd.DataFrame, fills: list[Fill], start_bar: int, params: str = "") -> StrategyResult:
    name, family, rules = STRATEGY_INFO[key]
    sim = simulate(daily, fills, total_bps=COST_BPS, initial_capital=CAPITAL, start_bar=start_bar)
    summary = M.summarize(sim.ledger, sim.equity, sim.open_trades)
    r = sim.ledger["net_ret_pct"].to_numpy(float) if len(sim.ledger) else np.array([])
    summary["fixed_pnl"] = float(r.sum() * CAPITAL / 100) if len(r) else 0.0
    return StrategyResult(key=key, name=name, family=family, rules=rules, params=params, summary=summary,
                          ledger=sim.ledger, equity=sim.equity["equity"], open_trades=sim.open_trades, fills=fills)


def exit_path_stats(res: StrategyResult) -> dict:
    """Strategy B: how trades split between the MACD/MA5 and RSI decline-watch exits."""
    led = res.ledger
    out = {"closed": 0 if led is None else len(led)}
    if led is None or not len(led):
        return out
    stops = led[led["exit_reason"].isin(STOP_EXITS)]
    out["stop_n"] = len(stops)
    out["stop_pct"] = len(stops) / len(led) * 100
    out["stop_avg"] = float(stops["net_ret_pct"].mean()) if len(stops) else None
    for reason, key in (("macd_ma5", "ma5"), ("rsi_decline", "rsi")):
        part = led[led["exit_reason"] == reason]
        r = part["net_ret_pct"].to_numpy(float)
        out[f"{key}_n"] = len(part)
        out[f"{key}_pct"] = len(part) / len(led) * 100
        out[f"{key}_avg"] = float(r.mean()) if len(r) else None
        out[f"{key}_win"] = float((r > 0).mean() * 100) if len(r) else None
        out[f"{key}_bars"] = float(part["bars_held"].mean()) if len(part) else None
    ma, rs = led[led["exit_reason"] == "macd_ma5"], led[led["exit_reason"] == "rsi_decline"]
    if len(ma) >= 2 and len(rs) >= 2:
        out["welch_t"], out["p_value"] = M.welch_t_test(ma["net_ret_pct"].to_numpy(float),
                                                        rs["net_ret_pct"].to_numpy(float))
    return out


@dataclass
class LabResult:
    ticker: str
    timeframe: str
    tf_label: str
    daily: pd.DataFrame
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    start_bar: int
    strategies: dict
    benchmark: dict
    benchmark_equity: pd.Series
    sweeps: dict
    comparison: pd.DataFrame | None = None     # the A/B × RSI-length table (also offered as CSV)
    trades: pd.DataFrame | None = None         # every A/B trade (also offered as CSV)
    pending: dict = field(default_factory=dict)
    cost_bps: float = COST_BPS
    stop: str = "none"                         # STOP_PRESETS key used for the A/B runs above
    stop_grid: pd.DataFrame | None = None      # every stop preset × A/B × RSI length
    notes: list = field(default_factory=list)

    def ranked(self) -> list[StrategyResult]:
        ok = [s for s in self.strategies.values() if s.available]
        return sorted(ok, key=lambda s: s.summary.get("total_return_pct", -1e9), reverse=True)


def run_lab(ticker: str, timeframe: str = "weekly", daily: pd.DataFrame | None = None,
            stop: str = "none") -> LabResult:
    ticker = ticker.upper().strip()
    if timeframe not in TIMEFRAMES:
        raise ValueError("timeframe must be 'weekly' or 'daily'")
    if stop not in STOP_PRESETS:
        raise ValueError(f"stop must be one of: {', '.join(STOP_PRESETS)}")
    interval, tf_label = TIMEFRAMES[timeframe]
    notes = []

    daily = fetch_daily(ticker) if daily is None else daily
    daily, repaired = sanitize_ohlc(daily)
    if repaired:
        notes.append(f"{repaired} daily bar(s) had tiny OHLC inconsistencies from price adjustment (≤0.5%) "
                     f"and were repaired before testing.")
    daily = validate(daily)
    frame = momentum_frame(daily, timeframe)
    if len(frame) < MOMENTUM_WARMUP_CANDLES + 20 or len(daily) < MACD_WARMUP_BARS + 60:
        raise RuntimeError(f"Not enough price history for {ticker} ({len(daily)} daily bars)")

    mom_ready = daily.index.get_loc(frame.index[MOMENTUM_WARMUP_CANDLES])
    start_bar = max(mom_ready, MACD_WARMUP_BARS)
    window_start = daily.index[start_bar]

    strategies, sweeps = {}, {}
    for kind in ("histogram", "crossover"):
        rows, params, fills = sweep_momentum(kind, frame, daily, ticker, start_bar, interval, COST_BPS)
        sweeps[kind] = rows
        strategies[kind] = evaluate(kind, daily, fills, start_bar, f"{params['label']} ({tf_label.lower()})")

    suite = run_suite(daily, RSI_LENGTHS, start_bar=start_bar, base=AB_CONFIG, stop=stop)
    stop_grid = run_stop_grid(daily, RSI_LENGTHS, start_bar=start_bar, base=AB_CONFIG)
    stop_label = STOP_PRESETS[stop].label
    pending = {}
    for (strat, L), run in suite.runs.items():
        key = f"{strat}{L}"
        res = evaluate(key, daily, run.fills, start_bar,
                       f"RSI({L}) <10 → >80 · fills at next open" + (" · MACD 12/26/9 daily, MA5" if strat == "B" else "")
                       + f" · stop: {stop_label}")
        res.rules = f"{res.rules} {stop_rule_text(stop, L)}"
        if strat == "B":
            res.extra["paths"] = exit_path_stats(res)
        strategies[key] = res
        if run.pending_buy:
            pending[key] = "BUY signal on the last bar — would buy at the next open"
        elif run.open_trades and run.open_trades[0].get("sell_pending"):
            pending[key] = "SELL signal on the last bar — would sell at the next open"

    bench, bench_eq = M.buy_and_hold(daily, start_bar, COST_BPS, CAPITAL)
    return LabResult(ticker=ticker, timeframe=timeframe, tf_label=tf_label, daily=daily,
                     window_start=window_start, window_end=daily.index[-1], start_bar=start_bar,
                     strategies=strategies, benchmark=bench, benchmark_equity=bench_eq, sweeps=sweeps,
                     comparison=suite.comparison, trades=suite.trades, pending=pending, notes=notes,
                     stop=stop, stop_grid=stop_grid)
