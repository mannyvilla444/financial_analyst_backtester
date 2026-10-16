"""
macd_backtest.py

MACD Histogram Strategy Backtester on Weekly candles.

STRATEGY RULES
──────────────
Entry  : MACD histogram increases for 2 consecutive Weekly candles → BUY
Exit 1 : MACD histogram decreases for 2 consecutive Weekly candles → SELL
Exit 2 : ATR stop loss — exit if candle low <= entry - (N × ATR)

THREE POSITION SIZING MODES (all run on best param set)
─────────────────────────────────────────────────────────
1. FIXED       — $10,000 per trade regardless of account size
2. ALL_IN      — start with $10,000, deploy 100% of available capital each trade
3. HALF        — start with $10,000, deploy 50% of available capital each trade
   (for ALL_IN and HALF: cash compounds between trades, buy-only)

PARAMETER SWEEP
───────────────
MACD fast × slow × signal × ATR mult — 12 combos tested, best by total PnL selected.
Best params then used for all three sizing modes.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Data ──────────────────────────────────────────────────────────────────────

# Candles per year for Sharpe annualisation
CANDLES_PER_YEAR = {
    "1d":    252,
    "daily": 252,
    "2d":    126,
    "3d":    84,
    "1wk":   52,
    "weekly":52,
}

# yfinance native interval for each user-facing interval
_YF_INTERVAL = {
    "1d":    "1d",
    "daily": "1d",
    "2d":    "1d",    # fetch daily, resample
    "3d":    "1d",    # fetch daily, resample
    "1wk":   "1wk",
    "weekly":"1wk",
}

# Resample rule where needed
_RESAMPLE_RULE = {
    "2d": "2D",
    "3d": "3D",
}


def fetch_price_data(ticker: str, period: str = "5y",
                     interval: str = "1wk") -> pd.DataFrame:
    """
    Fetch OHLCV data for the given interval.
    Supported: 1d/daily, 2d, 3d, 1wk/weekly
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    iv_key   = interval.lower()
    yf_iv    = _YF_INTERVAL.get(iv_key, "1wk")
    resample = _RESAMPLE_RULE.get(iv_key)

    df = yf.Ticker(ticker).history(period=period, interval=yf_iv)
    if df.empty:
        raise RuntimeError(f"No price data found for '{ticker}'")

    df.index   = df.index.tz_localize(None) if df.index.tz else df.index
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    if resample:
        df = df.resample(resample).agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

    return df


# ── Indicators ────────────────────────────────────────────────────────────────

def calc_macd(close, fast, slow, signal):
    ema_f = close.ewm(span=fast,   adjust=False).mean()
    ema_s = close.ewm(span=slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal,  adjust=False).mean()
    return macd, sig, macd - sig


def calc_atr(df, period=14):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_ma(close: pd.Series, period: int = 50) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(window=period, min_periods=period).mean()



# ── Trade ─────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:    datetime
    entry_price:   float
    stop_price:    float
    exit_date:     Optional[datetime] = None
    exit_price:    Optional[float]    = None
    exit_reason:   str                = ""
    shares:        float              = 0.0
    position_size: float              = 0.0

    @property
    def pnl(self):
        if self.exit_price is None: return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self):
        if self.exit_price is None or self.entry_price == 0: return 0.0
        return (self.exit_price / self.entry_price - 1) * 100

    @property
    def hold_days(self):
        if self.exit_date is None: return 0
        return (self.exit_date - self.entry_date).days


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    ticker:        str
    fast:          int
    slow:          int
    signal:        int
    atr_mult:      float
    position_size: float
    sizing_mode:   str   = "fixed"   # "fixed" | "all_in" | "half"
    interval:      str   = "1wk"
    candles_per_year: int = 52
    trades:        list  = field(default_factory=list)
    # Defaults so a run with zero trades still has these attributes
    # (_compute_stats only sets them when trades exist).
    total_trades:     int   = 0
    winning_trades:   int   = 0
    losing_trades:    int   = 0
    win_rate:         float = 0.0
    total_pnl:        float = 0.0
    final_capital:    float = 0.0
    total_return_pct: float = 0.0   # final_capital / starting_capital - 1
    avg_win:          float = 0.0
    avg_loss:         float = 0.0
    profit_factor:    float = 0.0
    avg_hold_days:    float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe:           float = 0.0
    stop_hit_rate:    float = 0.0
    expectancy:       float = 0.0

    @property
    def params_label(self):
        return f"MACD({self.fast},{self.slow},{self.signal}) ATR×{self.atr_mult}"

    @property
    def is_profitable(self):
        return self.total_pnl > 0 and self.win_rate >= 40


# ── Crossover Result ─────────────────────────────────────────────────────────

@dataclass
class CrossoverResult:
    """Result for MACD Crossover + MA50 trend filter strategy."""
    ticker:        str
    fast:          int
    slow:          int
    signal:        int
    atr_mult:      float
    ma_period:     int   = 50
    position_size: float = 10_000.0
    sizing_mode:   str   = "fixed"
    interval:      str   = "1wk"
    candles_per_year: int = 52
    trades:        list  = field(default_factory=list)
    total_trades:     int   = 0
    winning_trades:   int   = 0
    losing_trades:    int   = 0
    win_rate:         float = 0.0
    total_pnl:        float = 0.0
    final_capital:    float = 0.0
    total_return_pct: float = 0.0
    avg_win:          float = 0.0
    avg_loss:         float = 0.0
    profit_factor:    float = 0.0
    avg_hold_days:    float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe:           float = 0.0
    stop_hit_rate:    float = 0.0
    expectancy:       float = 0.0

    @property
    def params_label(self):
        return (f"MACD({self.fast},{self.slow},{self.signal}) "
                f"ATR×{self.atr_mult} MA{self.ma_period}")

    @property
    def is_profitable(self):
        return self.total_pnl > 0 and self.win_rate >= 40


def _compute_crossover_stats(result: CrossoverResult, starting_capital: float):
    """Reuse the same stats logic — identical math, different type."""
    trades = result.trades
    if not trades:
        return result
    pnls     = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    winners  = [p for p in pnls if p > 0]
    losers   = [p for p in pnls if p <= 0]
    result.total_trades   = len(trades)
    result.winning_trades = len(winners)
    result.losing_trades  = len(losers)
    result.win_rate       = len(winners) / len(trades) * 100
    result.total_pnl      = sum(pnls)
    result.avg_win        = np.mean(winners) if winners else 0.0
    result.avg_loss       = np.mean(losers)  if losers  else 0.0
    result.profit_factor  = (
        sum(winners) / abs(sum(losers))
        if losers and sum(losers) != 0 else float("inf")
    )
    result.avg_hold_days  = np.mean([t.hold_days for t in trades])
    result.stop_hit_rate  = sum(1 for t in trades if t.exit_reason == "stop") / len(trades) * 100
    result.expectancy     = np.mean(pnls)
    result.final_capital  = starting_capital + result.total_pnl
    result.total_return_pct = (result.final_capital / starting_capital - 1) * 100
    cum  = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd   = cum - peak
    result.max_drawdown_pct = (
        (dd.min() / (peak.max() + starting_capital)) * 100
        if peak.max() > 0 else 0.0
    )
    if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
        result.sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(52)
    return result


# ── Core backtest ─────────────────────────────────────────────────────────────

def _compute_stats(result: BacktestResult, starting_capital: float):
    trades = result.trades
    if not trades:
        return result

    pnls     = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    winners  = [p for p in pnls if p > 0]
    losers   = [p for p in pnls if p <= 0]

    result.total_trades   = len(trades)
    result.winning_trades = len(winners)
    result.losing_trades  = len(losers)
    result.win_rate       = len(winners) / len(trades) * 100
    result.total_pnl      = sum(pnls)
    result.avg_win        = np.mean(winners) if winners else 0.0
    result.avg_loss       = np.mean(losers)  if losers  else 0.0
    result.profit_factor  = (
        sum(winners) / abs(sum(losers))
        if losers and sum(losers) != 0 else float("inf")
    )
    result.avg_hold_days  = np.mean([t.hold_days for t in trades])
    result.stop_hit_rate  = sum(1 for t in trades if t.exit_reason == "stop") / len(trades) * 100
    result.expectancy     = np.mean(pnls)
    result.final_capital  = starting_capital + result.total_pnl
    result.total_return_pct = (result.final_capital / starting_capital - 1) * 100

    # Max drawdown
    cum  = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd   = cum - peak
    result.max_drawdown_pct = (
        (dd.min() / (peak.max() + starting_capital)) * 100
        if peak.max() > 0 else 0.0
    )

    # Sharpe (~52 weekly candles/year)
    if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
        result.sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(52)

    return result


def run_backtest(
    df: pd.DataFrame,
    ticker: str,
    fast: int,
    slow: int,
    signal: int,
    atr_period:    int   = 14,
    atr_mult:      float = 2.0,
    position_size: float = 10_000.0,
    sizing_mode:   str   = "fixed",   # "fixed" | "all_in" | "half"
    starting_capital: float = 10_000.0,
    interval:      str   = "1wk",
) -> BacktestResult:
    cpy = CANDLES_PER_YEAR.get(interval.lower(), 52)
    result = BacktestResult(
        ticker=ticker, fast=fast, slow=slow, signal=signal,
        atr_mult=atr_mult, position_size=position_size,
        sizing_mode=sizing_mode, interval=interval, candles_per_year=cpy,
    )

    df = df.copy()
    df["macd"], df["sig"], df["hist"] = calc_macd(df["close"], fast, slow, signal)
    df["atr"] = calc_atr(df, atr_period)
    df = df.dropna()

    if len(df) < slow + signal + 5:
        return result

    df["hist_up"]   = df["hist"] > df["hist"].shift(1)
    df["hist_down"] = df["hist"] < df["hist"].shift(1)

    in_trade     = False
    entry_price  = 0.0
    stop_price   = 0.0
    shares       = 0.0
    entry_date   = None
    cash         = starting_capital   # used for compounding modes
    trades       = []

    for i in range(3, len(df)):      # start at 3 so we can look back 3 candles
        row        = df.iloc[i]
        prev       = df.iloc[i - 1]
        prior_prev = df.iloc[i - 2]  # candle before the two-candle setup
        close = row["close"]
        date  = df.index[i]

        if not in_trade:
            # Entry: histogram rises 2 candles in a row FOR THE FIRST TIME —
            # meaning the candle before those two was flat or declining.
            # This prevents entering mid-run when histogram has been rising
            # for many candles already.
            first_upturn = (
                row["hist_up"] and
                prev["hist_up"] and
                not prior_prev["hist_up"]   # candle before was NOT rising
            )
            if first_upturn:
                # Determine capital to deploy
                if sizing_mode == "fixed":
                    deploy = position_size
                elif sizing_mode == "all_in":
                    deploy = cash
                else:  # half
                    deploy = cash * 0.5

                if deploy < 1.0:
                    continue   # not enough cash

                entry_price = close
                stop_price  = entry_price - atr_mult * row["atr"]
                shares      = deploy / entry_price
                entry_date  = date
                in_trade    = True

                if sizing_mode != "fixed":
                    cash -= deploy   # capital is now in the position

        else:
            exit_price  = None
            exit_reason = ""

            if row["low"] <= stop_price:
                exit_price  = stop_price
                exit_reason = "stop"
            elif row["hist_down"] and prev["hist_down"]:
                exit_price  = close
                exit_reason = "signal"

            if exit_price is not None:
                t = Trade(
                    entry_date    = entry_date,
                    entry_price   = entry_price,
                    stop_price    = stop_price,
                    exit_date     = date,
                    exit_price    = exit_price,
                    exit_reason   = exit_reason,
                    shares        = shares,
                    position_size = shares * entry_price,
                )
                trades.append(t)

                if sizing_mode != "fixed":
                    cash += exit_price * shares   # return proceeds to cash

                in_trade = False

    # Close open trade at last price
    if in_trade:
        lp = df.iloc[-1]["close"]
        t  = Trade(
            entry_date    = entry_date,
            entry_price   = entry_price,
            stop_price    = stop_price,
            exit_date     = df.index[-1],
            exit_price    = lp,
            exit_reason   = "open→close",
            shares        = shares,
            position_size = shares * entry_price,
        )
        trades.append(t)
        if sizing_mode != "fixed":
            cash += lp * shares

    result.trades       = trades
    result.final_capital = cash if sizing_mode != "fixed" else starting_capital + sum(t.pnl for t in trades)
    return _compute_stats(result, starting_capital)


# ── MACD Crossover + MA50 Trend Filter Backtest ──────────────────────────────

def run_crossover_backtest(
    df: pd.DataFrame,
    ticker: str,
    fast: int,
    slow: int,
    signal: int,
    atr_period:    int   = 14,
    atr_mult:      float = 2.0,
    ma_period:     int   = 50,
    position_size: float = 10_000.0,
    sizing_mode:   str   = "fixed",
    starting_capital: float = 10_000.0,
    interval:      str   = "1wk",
) -> CrossoverResult:
    """
    MACD Crossover Strategy with MA50 trend filter.

    ENTRY : MACD line crosses ABOVE signal line AND MA50 is trending up
            (MA50 trending up = MA50[i] > MA50[i-3], a 3-week slope check)
    EXIT  : MACD line crosses BELOW signal line
    STOP  : Same ATR-based stop as histogram strategy
    """
    cpy = CANDLES_PER_YEAR.get(interval.lower(), 52)
    result = CrossoverResult(
        ticker=ticker, fast=fast, slow=slow, signal=signal,
        atr_mult=atr_mult, ma_period=ma_period,
        position_size=position_size, sizing_mode=sizing_mode,
        interval=interval, candles_per_year=cpy,
    )

    df = df.copy()
    df["macd"], df["sig"], df["hist"] = calc_macd(df["close"], fast, slow, signal)
    df["atr"]  = calc_atr(df, atr_period)
    df["ma50"] = calc_ma(df["close"], ma_period)
    df = df.dropna()

    if len(df) < max(slow + signal, ma_period) + 5:
        return result

    # Pre-compute crossover signals
    df["cross_up"]   = (df["macd"] > df["sig"]) & (df["macd"].shift(1) <= df["sig"].shift(1))
    df["cross_down"] = (df["macd"] < df["sig"]) & (df["macd"].shift(1) >= df["sig"].shift(1))
    # MA50 trending up: current MA50 > MA50 from 3 candles ago
    df["ma_up"] = df["ma50"] > df["ma50"].shift(3)
    # MACD already bullish (above signal, regardless of when it crossed)
    df["macd_bullish"] = df["macd"] > df["sig"]
    # Price crosses above MA50 this candle (prev close below, current close above)
    df["price_cross_ma50"] = (df["close"].shift(1) < df["ma50"].shift(1)) & (df["close"] > df["ma50"])

    in_trade    = False
    entry_price = 0.0
    stop_price  = 0.0
    shares      = 0.0
    entry_date  = None
    cash        = starting_capital
    trades      = []

    for i in range(1, len(df)):
        row   = df.iloc[i]
        close = row["close"]
        date  = df.index[i]

        if not in_trade:
            # Trigger A: MACD crossover + MA50 trending up
            trig_a = bool(row["cross_up"]) and bool(row["ma_up"])
            # Trigger B: MACD already bullish + price reclaims MA50 (missed entry recovery)
            trig_b = bool(row["macd_bullish"]) and bool(row["price_cross_ma50"])

            if trig_a or trig_b:
                if sizing_mode == "fixed":
                    deploy = position_size
                elif sizing_mode == "all_in":
                    deploy = cash
                else:
                    deploy = cash * 0.5

                if deploy < 1.0:
                    continue

                entry_price = close
                stop_price  = entry_price - atr_mult * float(row["atr"])
                shares      = deploy / entry_price
                entry_date  = date
                in_trade    = True

                if sizing_mode != "fixed":
                    cash -= deploy

        else:
            exit_price  = None
            exit_reason = ""

            # ATR stop loss (same rule as histogram strategy)
            if row["low"] <= stop_price:
                exit_price  = stop_price
                exit_reason = "stop"
            # SELL: MACD crosses below signal
            elif row["cross_down"]:
                exit_price  = close
                exit_reason = "signal"

            if exit_price is not None:
                t = Trade(
                    entry_date    = entry_date,
                    entry_price   = entry_price,
                    stop_price    = stop_price,
                    exit_date     = date,
                    exit_price    = exit_price,
                    exit_reason   = exit_reason,
                    shares        = shares,
                    position_size = shares * entry_price,
                )
                trades.append(t)

                if sizing_mode != "fixed":
                    cash += exit_price * shares

                in_trade = False

    # Close open trade at last price
    if in_trade:
        lp = df.iloc[-1]["close"]
        t  = Trade(
            entry_date    = entry_date,
            entry_price   = entry_price,
            stop_price    = stop_price,
            exit_date     = df.index[-1],
            exit_price    = lp,
            exit_reason   = "open→close",
            shares        = shares,
            position_size = shares * entry_price,
        )
        trades.append(t)
        if sizing_mode != "fixed":
            cash += lp * shares

    result.trades        = trades
    result.final_capital = cash if sizing_mode != "fixed" else starting_capital + sum(t.pnl for t in trades)
    return _compute_crossover_stats(result, starting_capital)


def run_crossover_all(
    ticker: str,
    price_df: pd.DataFrame,
    fixed_position_size: float = 10_000.0,
    starting_capital:    float = 10_000.0,
    interval:            str   = "1wk",
) -> tuple:
    """
    Sweep all 12 MACD param combos for the crossover strategy.
    Returns (summary_df, best_fixed, allin_result, half_result, trades_df)
    """
    fixed_results = []
    for fast, slow, sig, atr_mult in product(
        PARAM_GRID["fast"], PARAM_GRID["slow"],
        PARAM_GRID["signal"], PARAM_GRID["atr_mult"],
    ):
        if fast >= slow:
            continue
        r = run_crossover_backtest(
            price_df, ticker, fast, slow, sig,
            atr_mult=atr_mult,
            position_size=fixed_position_size,
            sizing_mode="fixed",
            starting_capital=starting_capital,
            interval=interval,
        )
        fixed_results.append(r)

    rows = []
    for r in fixed_results:
        rows.append({
            "Params":        r.params_label,
            "Trades":        r.total_trades,
            "Win %":         f"{r.win_rate:.1f}%",
            "Total PnL":     f"${r.total_pnl:+,.0f}",
            "Profit Factor": f"{r.profit_factor:.2f}",
            "Avg Win":       f"${r.avg_win:+,.0f}",
            "Avg Loss":      f"${r.avg_loss:+,.0f}",
            "Expectancy":    f"${r.expectancy:+,.0f}",
            "Sharpe":        f"{r.sharpe:.2f}",
            "Max DD":        f"{r.max_drawdown_pct:.1f}%",
            "Stop %":        f"{r.stop_hit_rate:.1f}%",
            "✓":             "✅" if r.is_profitable else "❌",
            "_pnl":          r.total_pnl,
        })

    summary_df = (pd.DataFrame(rows)
                  .sort_values("_pnl", ascending=False)
                  .drop("_pnl", axis=1))

    ranked = sorted([r for r in fixed_results if r.total_trades >= 5],
                    key=lambda r: r.total_pnl, reverse=True)
    best_fixed = ranked[0] if ranked else max(fixed_results, key=lambda r: r.total_pnl)

    allin_result = run_crossover_backtest(
        price_df, ticker, best_fixed.fast, best_fixed.slow, best_fixed.signal,
        atr_mult=best_fixed.atr_mult,
        position_size=starting_capital,
        sizing_mode="all_in",
        starting_capital=starting_capital,
        interval=interval,
    )
    half_result = run_crossover_backtest(
        price_df, ticker, best_fixed.fast, best_fixed.slow, best_fixed.signal,
        atr_mult=best_fixed.atr_mult,
        position_size=starting_capital,
        sizing_mode="half",
        starting_capital=starting_capital,
        interval=interval,
    )

    trade_rows = []
    for t in best_fixed.trades:
        trade_rows.append({
            "Entry":   t.entry_date.strftime("%Y-%m-%d") if t.entry_date else "",
            "Exit":    t.exit_date.strftime("%Y-%m-%d")  if t.exit_date  else "",
            "Entry $": f"${t.entry_price:.2f}",
            "Exit $":  f"${t.exit_price:.2f}" if t.exit_price else "open",
            "Stop $":  f"${t.stop_price:.2f}",
            "Reason":  t.exit_reason,
            "PnL $":   f"${t.pnl:+,.2f}",
            "PnL %":   f"{t.pnl_pct:+.2f}%",
            "Days":    t.hold_days,
        })
    trades_df = pd.DataFrame(trade_rows)

    return summary_df, best_fixed, allin_result, half_result, trades_df


# ── Compounding equity curve helpers ─────────────────────────────────────────

def equity_curve(result: BacktestResult, starting_capital: float) -> pd.Series:
    """
    Returns a pd.Series of portfolio value over time for a BacktestResult.
    For fixed sizing: starting_capital + cumulative PnL.
    For compounding: actual capital after each trade.
    """
    if not result.trades:
        return pd.Series(dtype=float)

    cap = starting_capital
    dates = []
    values = []
    for t in result.trades:
        cap += t.pnl
        dates.append(t.exit_date)
        values.append(cap)
    return pd.Series(values, index=dates)


# ── Parameter sweep ───────────────────────────────────────────────────────────

PARAM_GRID = {
    "fast":     [8, 12],
    "slow":     [21, 26],
    "signal":   [9],
    "atr_mult": [1.5, 2.0, 3.0],
}


def run_all(
    ticker: str,
    fixed_position_size: float = 10_000.0,
    starting_capital:    float = 10_000.0,
    period:              str   = "5y",
    interval:            str   = "1wk",
):
    """
    1. Sweep all 12 param combos using FIXED sizing → find best params
    2. Re-run best params with ALL_IN compounding
    3. Re-run best params with HALF compounding

    Returns:
        summary_df    — comparison table (all 12 combos, fixed sizing)
        fixed_result  — best BacktestResult (fixed)
        allin_result  — best params, all-in compounding
        half_result   — best params, 50% compounding
        trades_df     — trade log for fixed best
        price_df      — raw Weekly OHLCV DataFrame
    """
    price_df = fetch_price_data(ticker, period=period, interval=interval)

    # ── Step 1: Fixed-sizing sweep ────────────────────────────────────────────
    fixed_results = []
    for fast, slow, sig, atr_mult in product(
        PARAM_GRID["fast"], PARAM_GRID["slow"],
        PARAM_GRID["signal"], PARAM_GRID["atr_mult"],
    ):
        if fast >= slow:
            continue
        r = run_backtest(price_df, ticker, fast, slow, sig,
                         atr_mult=atr_mult,
                         position_size=fixed_position_size,
                         sizing_mode="fixed",
                         starting_capital=starting_capital,
                         interval=interval)
        fixed_results.append(r)

    # Build summary table
    rows = []
    for r in fixed_results:
        rows.append({
            "Params":        r.params_label,
            "Trades":        r.total_trades,
            "Win %":         f"{r.win_rate:.1f}%",
            "Total PnL":     f"${r.total_pnl:+,.0f}",
            "Profit Factor": f"{r.profit_factor:.2f}",
            "Avg Win":       f"${r.avg_win:+,.0f}",
            "Avg Loss":      f"${r.avg_loss:+,.0f}",
            "Expectancy":    f"${r.expectancy:+,.0f}",
            "Sharpe":        f"{r.sharpe:.2f}",
            "Max DD":        f"{r.max_drawdown_pct:.1f}%",
            "Stop %":        f"{r.stop_hit_rate:.1f}%",
            "✓":             "✅" if r.is_profitable else "❌",
            "_pnl":          r.total_pnl,
        })

    summary_df = (pd.DataFrame(rows)
                  .sort_values("_pnl", ascending=False)
                  .drop("_pnl", axis=1))

    # Pick best (highest PnL, at least 10 trades)
    ranked = sorted([r for r in fixed_results if r.total_trades >= 10],
                    key=lambda r: r.total_pnl, reverse=True)
    best_fixed = ranked[0] if ranked else max(fixed_results, key=lambda r: r.total_pnl)

    # ── Step 2: ALL_IN compounding with best params ───────────────────────────
    allin_result = run_backtest(
        price_df, ticker,
        best_fixed.fast, best_fixed.slow, best_fixed.signal,
        atr_mult=best_fixed.atr_mult,
        position_size=starting_capital,
        sizing_mode="all_in",
        starting_capital=starting_capital,
        interval=interval,
    )

    # ── Step 3: HALF compounding with best params ─────────────────────────────
    half_result = run_backtest(
        price_df, ticker,
        best_fixed.fast, best_fixed.slow, best_fixed.signal,
        atr_mult=best_fixed.atr_mult,
        position_size=starting_capital,
        sizing_mode="half",
        starting_capital=starting_capital,
        interval=interval,
    )

    # Trade log for best fixed
    trade_rows = []
    for t in best_fixed.trades:
        trade_rows.append({
            "Entry":      t.entry_date.strftime("%Y-%m-%d") if t.entry_date else "",
            "Exit":       t.exit_date.strftime("%Y-%m-%d")  if t.exit_date  else "",
            "Entry $":    f"${t.entry_price:.2f}",
            "Exit $":     f"${t.exit_price:.2f}"  if t.exit_price else "open",
            "Stop $":     f"${t.stop_price:.2f}",
            "Reason":     t.exit_reason,
            "PnL $":      f"${t.pnl:+,.2f}",
            "PnL %":      f"{t.pnl_pct:+.2f}%",
            "Days":       t.hold_days,
        })
    trades_df = pd.DataFrame(trade_rows)

    # ── Step 4: Run crossover strategy with same param sweep ────────────────
    cx_summary, cx_best, cx_allin, cx_half, cx_trades_df = run_crossover_all(
        ticker, price_df,
        fixed_position_size=fixed_position_size,
        starting_capital=starting_capital,
        interval=interval,
    )

    # ── Step 5: Determine winner ──────────────────────────────────────────────
    winner = "histogram" if best_fixed.total_pnl >= cx_best.total_pnl else "crossover"

    return (summary_df, best_fixed, allin_result, half_result, trades_df, price_df,
            cx_summary, cx_best, cx_allin, cx_half, cx_trades_df, winner)


def format_telegram_sizing_summary(
    ticker:           str,
    fixed:            BacktestResult,
    allin:            BacktestResult,
    half:             BacktestResult,
    cx_fixed:         "CrossoverResult | None" = None,
    cx_allin:         "CrossoverResult | None" = None,
    cx_half:          "CrossoverResult | None" = None,
    winner:           str = "histogram",
    starting_capital: float = 10_000.0,
) -> str:
    """Build both-strategy comparison for the Telegram message."""
    def verdict(r): return "✅" if r.is_profitable else "❌"
    def row(r):
        return (f"  PnL: ${r.total_pnl:+,.0f}  |  "
                f"Win: {r.win_rate:.1f}%  |  "
                f"Trades: {r.total_trades}  |  "
                f"PF: {r.profit_factor:.2f}")

    winner_emoji = "🏆"
    hist_tag  = f" {winner_emoji}" if winner == "histogram"  else ""
    cross_tag = f" {winner_emoji}" if winner == "crossover"  else ""

    lines = [
        f"📈 *{ticker} — Dual MACD Backtest (Weekly)*",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        f"〽️ *Strategy 1: Histogram*{hist_tag}",
        f"  Params: {fixed.params_label}",
        row(fixed) + f"  {verdict(fixed)}",
        f"  All-In: ${allin.final_capital:,.0f}  ({allin.total_return_pct:+.1f}%)  "
        f"| 50%: ${half.final_capital:,.0f}  ({half.total_return_pct:+.1f}%)",
    ]

    if cx_fixed:
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            f"📊 *Strategy 2: Crossover + MA50*{cross_tag}",
            f"  Params: {cx_fixed.params_label}",
            row(cx_fixed) + f"  {verdict(cx_fixed)}",
        ]
        if cx_allin and cx_half:
            lines.append(
                f"  All-In: ${cx_allin.final_capital:,.0f}  ({cx_allin.total_return_pct:+.1f}%)  "
                f"| 50%: ${cx_half.final_capital:,.0f}  ({cx_half.total_return_pct:+.1f}%)"
            )

    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🏆 *Winner: {'Histogram' if winner == 'histogram' else 'Crossover + MA50'}*"
        f"  (by total PnL, fixed sizing)",
        "",
        "⚠️ _Past performance ≠ future results_",
    ]
    return "\n".join(lines)


# ── Notification prompt (appended to backtest results) ────────────────────────

def prompt_notification_question(ticker: str, best: "BacktestResult") -> str:
    """
    Returns the follow-up Telegram message asking the user if they want
    to turn on live signal notifications for this ticker/strategy.
    Called by bot.py after the backtest PDF is sent.
    """
    return (
        f"📡 *Turn on trade notifications for {ticker}?*\n"
        f"\n"
        f"I'll monitor the live MACD signal using your best parameters:\n"
        f"  *{best.params_label}*\n"
        f"\n"
        f"You'll get a Telegram message the moment a:\n"
        f"  🟢 *BUY* signal fires  (histogram ↑↑ 2 candles)\n"
        f"  🔴 *SELL* signal fires  (histogram ↓↓ 2 candles)\n"
        f"  🛑 *STOP* is triggered  (price hits ATR stop)\n"
        f"\n"
        f"Checks run every hour on 2-day candles.\n"
        f"\n"
        f"Reply with:\n"
        f"  /notify on {ticker}   — start notifications\n"
        f"  /notify off {ticker}  — stop notifications\n"
        f"  /notify list          — see all active watchers\n"
        f"  /notify check {ticker} — check signal right now"
    )


# ── Live signal check (used by the web app's signal worker) ───────────────────

def check_current_signal(
    ticker:   str,
    fast:     int   = 12,
    slow:     int   = 26,
    signal:   int   = 9,
    atr_mult: float = 2.0,
    strategy: str   = "histogram",   # "histogram" | "crossover"
    interval: str   = "1wk",
    in_trade: bool  = False,
) -> Optional[dict]:
    """
    Evaluate the latest candle for `ticker` and return a signal dict if a
    BUY or SELL condition is met right now, else None.

    The rules are the same ones the backtests trade on (and that the Telegram
    bot's signal_monitor.py alerts on):

      histogram — BUY  : histogram rises 2 candles in a row for the FIRST time
                         (the candle before the two-candle setup was not rising)
                  SELL : histogram falls 2 candles in a row
      crossover — BUY  : MACD crosses above signal AND MA50 trending up
                         (MA50 now > MA50 three candles ago), OR
                         MACD already above signal AND price reclaims MA50
                         (only when not already in a trade)
                  SELL : MACD crosses below signal

    De-duplication ("is this new since the last check?") is handled by the
    caller, which compares against the last signal it logged for the ticker.

    Returned shape:
        {"ticker", "type": "BUY"|"SELL", "price", "timestamp", "message",
         "candle_date", "strategy", "interval", "params", "stop"}
    Raises RuntimeError if price data can't be fetched.
    """
    import time

    df = fetch_price_data(ticker, period="5y", interval=interval)
    if len(df) < 60:
        raise RuntimeError(f"Not enough price history for {ticker} ({len(df)} candles)")

    df = df.copy()
    df["macd"], df["sig"], df["hist"] = calc_macd(df["close"], fast, slow, signal)
    df["atr"]  = calc_atr(df)
    df["ma50"] = calc_ma(df["close"], 50)
    df = df.dropna(subset=["macd", "sig", "hist", "atr"])
    if len(df) < 5:
        return None

    c0, c1, c2, c3 = df.iloc[-1], df.iloc[-2], df.iloc[-3], df.iloc[-4]
    close = float(c0["close"])
    atr   = float(c0["atr"])
    candle_date = df.index[-1].strftime("%Y-%m-%d")
    tf_label = {"1wk": "weekly", "weekly": "weekly", "1d": "daily", "daily": "daily",
                "2d": "2-day", "3d": "3-day"}.get(interval.lower(), interval)

    kind, detail = None, ""

    if strategy == "crossover":
        if pd.isna(c0["ma50"]) or pd.isna(c1["ma50"]) or pd.isna(c3["ma50"]):
            return None
        macd_now, sig_now   = float(c0["macd"]), float(c0["sig"])
        macd_prev, sig_prev = float(c1["macd"]), float(c1["sig"])
        ma_now, ma_old      = float(c0["ma50"]), float(c3["ma50"])
        cross_up   = macd_now > sig_now and macd_prev <= sig_prev
        cross_down = macd_now < sig_now and macd_prev >= sig_prev
        price_reclaims_ma = float(c1["close"]) < float(c1["ma50"]) and close > ma_now

        if cross_up and ma_now > ma_old:
            kind   = "BUY"
            detail = (f"MACD crossed above signal with MA50 trending up "
                      f"(${ma_old:.2f} → ${ma_now:.2f}) on {tf_label}")
        elif not in_trade and macd_now > sig_now and price_reclaims_ma:
            kind   = "BUY"
            detail = f"Price reclaimed MA50 (${ma_now:.2f}) while MACD bullish on {tf_label}"
        elif cross_down:
            kind   = "SELL"
            detail = f"MACD crossed below signal ({macd_prev:.4f} → {macd_now:.4f}) on {tf_label}"
    else:
        h0, h1, h2, h3 = (float(c["hist"]) for c in (c0, c1, c2, c3))
        if h0 > h1 and h1 > h2 and not (h2 > h3):
            kind   = "BUY"
            detail = f"MACD histogram first upturn ({h1:.4f} → {h0:.4f} ↑↑) on {tf_label}"
        elif h0 < h1 and h1 < h2:
            kind   = "SELL"
            detail = f"MACD histogram falling 2 candles ({h1:.4f} → {h0:.4f} ↓↓) on {tf_label}"

    if kind is None:
        return None

    stop = close - atr_mult * atr if kind == "BUY" else None
    params = f"MACD({fast},{slow},{signal}) ATR×{atr_mult}"
    message = detail
    if stop is not None:
        message += f" · ATR stop ${stop:.2f} ({(stop / close - 1) * 100:+.1f}%)"
    message += f" · {params}"

    return {
        "ticker":      ticker.upper(),
        "type":        kind,
        "price":       round(close, 2),
        "timestamp":   time.time(),
        "message":     message,
        "candle_date": candle_date,
        "strategy":    strategy,
        "interval":    interval,
        "params":      params,
        "stop":        round(stop, 2) if stop is not None else None,
    }
