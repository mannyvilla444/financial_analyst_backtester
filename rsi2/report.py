"""Console report and output files (spec §10). Knows nothing about the rules."""
from __future__ import annotations

import json
import math
import os

import pandas as pd

from . import __version__


def _fmt(v, spec=".2f", suffix=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, str):
        return v
    return f"{v:{spec}}{suffix}"


def summary_text(ticker: str, summary: dict, bench: dict, flagged: dict, yearly: pd.Series, cfg: dict) -> str:
    s, b = summary, bench
    L = [
        f"RSI(2) mean reversion — {ticker}   ({s['start']} → {s['end']}, {s['years']:.1f} years)",
        f"Costs: {cfg['commission_bps'] + cfg['slippage_bps']:.1f} bps per side (net figures throughout)",
        f"Exit logic: {cfg['exit_logic']}   trend {cfg['trend_type']}({cfg['trend_len']})   "
        f"RSI({cfg['rsi_len']}) < {cfg['entry_level']} → exit ≥ {cfg['exit_level']}",
        "",
        "RETURNS                              strategy      buy & hold",
        f"  Total return                  {_fmt(s['total_return_pct'], '>10.2f', '%')}   {_fmt(b['total_return_pct'], '>10.2f', '%')}",
        f"  CAGR                          {_fmt(s['cagr_pct'], '>10.2f', '%')}   {_fmt(b['cagr_pct'], '>10.2f', '%')}",
        f"  Max drawdown (daily MTM)      {_fmt(s['max_drawdown_pct'], '>10.2f', '%')}   {_fmt(b['max_drawdown_pct'], '>10.2f', '%')}",
        f"  Sharpe (rf=0)                 {_fmt(s['sharpe'], '>10.2f')}   {_fmt(b['sharpe'], '>10.2f')}",
        f"  Exposure                      {_fmt(s['exposure_pct'], '>10.1f', '%')}   {_fmt(100.0, '>10.1f', '%')}",
        f"  Return ÷ exposure             {_fmt(s['return_per_exposure_pct'], '>10.2f', '%')}   {_fmt(b['return_per_exposure_pct'], '>10.2f', '%')}",
        f"  Final equity                  {_fmt(s['final_equity'], '>10,.0f')}",
        f"  Sortino / Calmar              {_fmt(s['sortino'])} / {_fmt(s['calmar'])}",
        f"  Ann. volatility               {_fmt(s['ann_volatility_pct'], '.2f', '%')}",
        f"  Longest drawdown              {s['max_drawdown_days']} days",
        "",
        "TRADES",
        f"  Closed trades {s['trades']}  ({_fmt(s.get('trades_per_year'), '.1f')}/yr)   open at end: {s['open_trades']}",
    ]
    if s["trades"]:
        L += [
            f"  >>> t-stat of mean trade: {_fmt(s.get('t_stat'))}   (below ~2 the edge is not distinguishable from noise)",
            f"  Win rate {_fmt(s['win_rate_pct'], '.1f', '%')}   avg {_fmt(s['avg_net_ret_pct'], '.3f', '%')}   "
            f"median {_fmt(s['median_net_ret_pct'], '.3f', '%')}   profit factor {_fmt(s['profit_factor'])}",
            f"  Avg win {_fmt(s['avg_win_pct'], '.2f', '%')}   avg loss {_fmt(s['avg_loss_pct'], '.2f', '%')}   "
            f"largest {_fmt(s['largest_win_pct'], '.2f', '%')} / {_fmt(s['largest_loss_pct'], '.2f', '%')}",
            f"  Avg bars held {_fmt(s['avg_bars_held'], '.1f')}   max consecutive losses {s['max_consecutive_losses']}",
            "", "EXIT MIX",
        ]
        L += [f"  {m['exit_reason']:<14} {m['count']:>5}   avg {m['avg_net_ret_pct']:+.3f}%" for m in s["exit_mix"]]
    fl, nm = flagged["flagged"], flagged["normal"]
    L += ["", "FLAGGED vs NORMAL (RSI still < flag level the bar after entry)",
          f"  flagged: n={fl.get('n', 0)}  win {_fmt(fl.get('win_rate_pct'), '.1f', '%')}  avg {_fmt(fl.get('avg_net_ret_pct'), '.3f', '%')}  "
          f"median {_fmt(fl.get('median_net_ret_pct'), '.3f', '%')}  PF {_fmt(fl.get('profit_factor'))}",
          f"  normal : n={nm.get('n', 0)}  win {_fmt(nm.get('win_rate_pct'), '.1f', '%')}  avg {_fmt(nm.get('avg_net_ret_pct'), '.3f', '%')}  "
          f"median {_fmt(nm.get('median_net_ret_pct'), '.3f', '%')}  PF {_fmt(nm.get('profit_factor'))}",
          f"  Welch t {_fmt(flagged.get('welch_t'))}   p-value {_fmt(flagged.get('p_value'), '.4f')}"]
    if flagged.get("warning"):
        L.append(f"  WARNING: {flagged['warning']}")
    L += ["", "RETURNS BY YEAR"] + [f"  {y}  {r:+8.2f}%" for y, r in yearly.items()]
    L += ["", "Survivorship note: backtesting a currently-listed ticker is survivorship-biased."]
    return "\n".join(L)


def write_outputs(out_dir: str, ticker: str, run, summary: dict, bench: dict, flagged: dict,
                  yearly: pd.Series, monthly: pd.DataFrame, text: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ledger = run.ledger.copy()
    for c in ("entry_date", "exit_date"):
        ledger[c] = pd.to_datetime(ledger[c]).dt.strftime("%Y-%m-%d")
    ledger.to_csv(os.path.join(out_dir, "trades.csv"), index=False, float_format="%.10g")
    eq = run.equity.copy()
    eq.index = eq.index.strftime("%Y-%m-%d")
    eq.index.name = "date"
    eq.to_csv(os.path.join(out_dir, "equity_daily.csv"), float_format="%.10g")
    payload = {
        "ticker": ticker, "code_version": __version__, "config": run.config.to_dict(),
        "summary": summary, "benchmark": bench, "flagged_vs_normal": flagged,
        "open_trades": [{k: (v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else v) for k, v in t.items()}
                        for t in run.open_trades],
        "yearly_returns_pct": {str(k): v for k, v in yearly.items()},
        "monthly_returns_pct": {str(y): {str(m): (None if pd.isna(v) else v) for m, v in row.items()}
                                for y, row in monthly.iterrows()},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
