"""
Statistics (spec §9). Consumes a trade ledger and a daily equity curve; knows
nothing about any strategy's rules.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── Student-t p-values without scipy ─────────────────────────────────────────

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    if not np.isfinite(t) or df <= 0:
        return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(t, two-sided p) for a difference in means with unequal variances."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, t_two_sided_p(t, df)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def profit_factor(returns: np.ndarray) -> float | None:
    wins = returns[returns > 0].sum()
    losses = -returns[returns <= 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else None
    return wins / losses


def max_consecutive_losses(returns: np.ndarray) -> int:
    best = run = 0
    for r in returns:
        run = run + 1 if r <= 0 else 0
        best = max(best, run)
    return best


def drawdown_stats(equity: pd.Series) -> tuple[float, int]:
    """(max drawdown %, longest peak-to-recovery duration in calendar days; an
    unrecovered drawdown counts up to the last date)."""
    eq = equity.astype(float)
    peak = eq.cummax()
    max_dd = float((eq / peak - 1).min() * 100)
    longest, peak_date, under = 0, eq.index[0], False
    for d, e, p in zip(eq.index, eq.to_numpy(), peak.to_numpy()):
        if e >= p:
            if under:
                longest = max(longest, (d - peak_date).days)
                under = False
            peak_date = d
        else:
            under = True
    if under:
        longest = max(longest, (eq.index[-1] - peak_date).days)
    return max_dd, longest


def curve_stats(equity: pd.Series) -> dict:
    """Return / risk statistics from a daily mark-to-market curve."""
    eq = equity.astype(float)
    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = rets.std(ddof=1) * math.sqrt(TRADING_DAYS) if len(rets) > 1 else float("nan")
    sd = rets.std(ddof=1)
    sharpe = rets.mean() / sd * math.sqrt(TRADING_DAYS) if len(rets) > 1 and sd > 0 else float("nan")
    downside = math.sqrt((np.minimum(rets, 0) ** 2).mean()) if len(rets) else float("nan")
    sortino = rets.mean() / downside * math.sqrt(TRADING_DAYS) if downside and downside > 0 else float("nan")
    max_dd, dd_days = drawdown_stats(eq)
    calmar = (cagr * 100) / abs(max_dd) if max_dd < 0 else float("nan")
    return {
        "start": eq.index[0].strftime("%Y-%m-%d"), "end": eq.index[-1].strftime("%Y-%m-%d"),
        "years": years, "initial_equity": float(eq.iloc[0]), "final_equity": float(eq.iloc[-1]),
        "total_return_pct": total * 100, "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_dd, "max_drawdown_days": dd_days,
        "ann_volatility_pct": vol * 100 if _f(vol) is not None else None,
        "sharpe": _f(sharpe), "sortino": _f(sortino), "calmar": _f(calmar), "risk_free_rate": 0.0,
    }


def trade_stats(ledger: pd.DataFrame, years: float) -> dict:
    r = ledger["net_ret_pct"].to_numpy(float) if len(ledger) else np.array([])
    n = len(r)
    out = {"trades": n, "trades_per_year": n / years if years > 0 else None}
    if n == 0:
        return out
    wins, losses = r[r > 0], r[r <= 0]
    sd = r.std(ddof=1) if n > 1 else float("nan")
    out.update({
        "win_rate_pct": len(wins) / n * 100,
        "avg_net_ret_pct": float(r.mean()), "median_net_ret_pct": float(np.median(r)),
        "avg_win_pct": float(wins.mean()) if len(wins) else None,
        "avg_loss_pct": float(losses.mean()) if len(losses) else None,
        "largest_win_pct": float(r.max()), "largest_loss_pct": float(r.min()),
        "profit_factor": _f(profit_factor(r)) if profit_factor(r) != float("inf") else "inf",
        "expectancy_pct": float(r.mean()),
        "avg_bars_held": float(ledger["bars_held"].mean()),
        "max_consecutive_losses": max_consecutive_losses(r),
        "t_stat": _f(r.mean() / sd * math.sqrt(n)) if n > 1 and sd > 0 else None,
        "sum_net_ret_pct": float(r.sum()),
    })
    return out


def exit_mix(ledger: pd.DataFrame) -> list[dict]:
    if not len(ledger):
        return []
    g = ledger.groupby("exit_reason")["net_ret_pct"]
    return [{"exit_reason": k, "count": int(v.count()), "avg_net_ret_pct": float(v.mean())}
            for k, v in sorted(g, key=lambda kv: -kv[1].count())]


def summarize(ledger: pd.DataFrame, equity: pd.DataFrame, open_trades: list[dict]) -> dict:
    cs = curve_stats(equity["equity"])
    ts = trade_stats(ledger, cs["years"])
    exposure = float(equity["in_position"].mean() * 100) if len(equity) else 0.0
    return {**cs, **ts, "exposure_pct": exposure,
            "return_per_exposure_pct": cs["total_return_pct"] / (exposure / 100) if exposure > 0 else None,
            "open_trades": len(open_trades), "exit_mix": exit_mix(ledger)}


def buy_and_hold(df: pd.DataFrame, start_bar: int, total_bps: float, initial_capital: float) -> tuple[dict, pd.Series]:
    """Buy at the close of start_bar (entry cost applied), marked to market daily."""
    close = df["Close"].astype(float).iloc[start_bar:]
    shares = initial_capital / (close.iloc[0] * (1 + total_bps / 10_000))
    eq = (close * shares).rename("equity")
    eq.iloc[0] = initial_capital * (close.iloc[0] / (close.iloc[0] * (1 + total_bps / 10_000)))
    stats = curve_stats(eq)
    stats.update({"exposure_pct": 100.0, "return_per_exposure_pct": stats["total_return_pct"]})
    return stats, eq


def group_stats(r: np.ndarray) -> dict:
    n = len(r)
    if n == 0:
        return {"n": 0}
    pf = profit_factor(r)
    return {"n": n, "win_rate_pct": float((r > 0).mean() * 100), "avg_net_ret_pct": float(r.mean()),
            "median_net_ret_pct": float(np.median(r)),
            "profit_factor": "inf" if pf == float("inf") else _f(pf)}


def flagged_comparison(ledger: pd.DataFrame, min_n: int = 30) -> dict:
    if not len(ledger) or "flagged" not in ledger:
        return {"flagged": {"n": 0}, "normal": {"n": 0}, "welch_t": None, "p_value": None,
                "warning": "No closed trades."}
    f = ledger.loc[ledger["flagged"].astype(bool), "net_ret_pct"].to_numpy(float)
    nf = ledger.loc[~ledger["flagged"].astype(bool), "net_ret_pct"].to_numpy(float)
    t, p = welch_t_test(f, nf)
    warn = None
    if len(f) < min_n or len(nf) < min_n:
        warn = (f"Underpowered: flagged n={len(f)}, normal n={len(nf)} (need ≥{min_n} each). "
                f"Differences at this size are usually noise.")
    return {"flagged": group_stats(f), "normal": group_stats(nf), "welch_t": _f(t), "p_value": _f(p), "warning": warn}


def yearly_returns(equity: pd.Series) -> pd.Series:
    """Calendar-year returns (%) from the daily curve; first year from the curve's start."""
    ye = equity.groupby(equity.index.year).last()
    prev = pd.Series([equity.iloc[0]] + ye.iloc[:-1].tolist(), index=ye.index)
    return (ye / prev - 1) * 100


def monthly_grid(equity: pd.Series) -> pd.DataFrame:
    me = equity.groupby([equity.index.year, equity.index.month]).last()
    prev = pd.Series([equity.iloc[0]] + me.iloc[:-1].tolist(), index=me.index)
    m = (me / prev - 1) * 100
    grid = m.unstack(level=1)
    grid.index.name, grid.columns.name = "year", "month"
    return grid
