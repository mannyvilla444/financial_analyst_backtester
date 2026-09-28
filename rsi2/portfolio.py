"""
Shared fill → ledger + daily mark-to-market accounting.

Every strategy (RSI(2), the MACD strategies, the combined ones) produces a list
of Fills; this module turns them into a trade ledger and a daily equity curve
with one cost model, so results are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LEDGER_COLUMNS = [
    "entry_date", "entry_bar", "entry_price", "exit_date", "exit_bar", "exit_price",
    "shares", "bars_held", "gross_ret_pct", "net_ret_pct", "pnl_dollars", "exit_reason",
    "flagged", "mae_pct", "mfe_pct",
]


@dataclass
class Fill:
    entry_bar: int
    entry_price: float
    exit_bar: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    flagged: bool = False
    tags: dict = field(default_factory=dict)   # strategy-specific extras (e.g. handoff)
    entry_timing: str = "close"                # "close" | "open" — when on entry_bar the fill happened
    exit_timing: str = "close"                 # "close" | "open" — when on exit_bar the fill happened

    @property
    def is_open(self) -> bool:
        return self.exit_bar is None


@dataclass
class SimResult:
    ledger: pd.DataFrame          # closed trades only
    equity: pd.DataFrame          # date-indexed: equity, drawdown, in_position
    open_trades: list[dict]       # marked-to-market, excluded from trade stats


def simulate(df: pd.DataFrame, fills: list[Fill], *, total_bps: float, initial_capital: float,
             position_pct: float = 100.0, whole_shares: bool = False, start_bar: int = 0) -> SimResult:
    """
    Fills execute at their own prices; the position is marked at each daily Close.
    in_position is True on bars entry_bar+1 .. exit_bar (the bars whose price
    change the position actually carried), which is what exposure counts.
    """
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    dates = df.index
    n = len(df)
    cost = total_bps / 10_000.0

    fills = sorted((f for f in fills if f.entry_bar >= start_bar), key=lambda f: f.entry_bar)
    for a, b in zip(fills, fills[1:]):
        if a.exit_bar is None or b.entry_bar <= a.exit_bar:
            raise ValueError(f"Overlapping trades: entry at bar {b.entry_bar} before exit of trade at {a.entry_bar}")

    equity = np.full(n, np.nan)
    in_pos = np.zeros(n, dtype=bool)
    cash = float(initial_capital)
    shares = 0.0
    cur: Fill | None = None
    k = 0
    rows, open_trades = [], []

    for i in range(start_bar, n):
        if cur is not None:
            in_pos[i] = True
            if cur.exit_bar == i:
                eff_sell = cur.exit_price * (1 - cost)
                eff_buy = cur.entry_price * (1 + cost)
                cash += shares * eff_sell
                # bars the position actually lived through (after the fill, before the exit)
                a = cur.entry_bar if cur.entry_timing == "open" else cur.entry_bar + 1
                b = i - 1 if cur.exit_timing == "open" else i
                seg_lo = low[a:b + 1].min() if b >= a else cur.exit_price
                seg_hi = high[a:b + 1].max() if b >= a else cur.exit_price
                row = {
                    "entry_date": dates[cur.entry_bar], "entry_bar": cur.entry_bar,
                    "entry_price": cur.entry_price, "exit_date": dates[i], "exit_bar": i,
                    "exit_price": cur.exit_price, "shares": shares, "bars_held": i - cur.entry_bar,
                    "gross_ret_pct": (cur.exit_price / cur.entry_price - 1) * 100,
                    "net_ret_pct": (eff_sell / eff_buy - 1) * 100,
                    "pnl_dollars": shares * (eff_sell - eff_buy),
                    "exit_reason": cur.exit_reason, "flagged": bool(cur.flagged),
                    "mae_pct": (min(seg_lo, cur.exit_price) / cur.entry_price - 1) * 100,
                    "mfe_pct": (max(seg_hi, cur.exit_price) / cur.entry_price - 1) * 100,
                }
                row.update(cur.tags)
                rows.append(row)
                if row["bars_held"] < 1:
                    raise AssertionError(f"Trade with bars_held < 1 at bar {i}")
                shares, cur = 0.0, None
        if cur is None and k < len(fills) and fills[k].entry_bar == i:
            cur = fills[k]
            k += 1
            eff_buy = cur.entry_price * (1 + cost)
            alloc = cash * position_pct / 100.0
            shares = alloc / eff_buy
            if whole_shares:
                shares = float(np.floor(shares))
            cash -= shares * eff_buy
        equity[i] = cash + shares * close[i]

    # Exposure: a fill at the OPEN of a bar is exposed during that bar (entry) or
    # not at all (exit) — so bars in the market always equals bars_held.
    for f in fills:
        if f.entry_timing == "open" and f.entry_bar >= start_bar:
            in_pos[f.entry_bar] = True
        if f.exit_bar is not None and f.exit_timing == "open":
            in_pos[f.exit_bar] = False
    if cur is not None:
        mark = close[n - 1]
        open_trades.append({
            "entry_date": dates[cur.entry_bar], "entry_bar": cur.entry_bar,
            "entry_price": cur.entry_price, "last_date": dates[n - 1], "last_close": mark,
            "unrealized_pct": ((mark * (1 - cost)) / (cur.entry_price * (1 + cost)) - 1) * 100,
            "bars_held": n - 1 - cur.entry_bar, "flagged": bool(cur.flagged), **cur.tags,
        })
    if k < len(fills):
        raise ValueError(f"Fill at bar {fills[k].entry_bar} was never reached")

    eq = pd.Series(equity[start_bar:], index=dates[start_bar:], name="equity")
    if eq.isna().any():
        raise AssertionError("NaN reached the equity curve")
    dd = eq / eq.cummax() - 1
    eq_df = pd.DataFrame({"equity": eq, "drawdown": dd * 100, "in_position": in_pos[start_bar:]})
    extra_cols = sorted({k for r in rows for k in r} - set(LEDGER_COLUMNS))
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS + extra_cols)
    return SimResult(ledger=ledger, equity=eq_df, open_trades=open_trades)
