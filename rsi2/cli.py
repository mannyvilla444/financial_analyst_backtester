"""
python -m rsi2.cli --ticker SPY --start 2005-01-01 --end 2025-01-01
python -m rsi2.cli --csv data/spy.csv --exit-logic OR --no-use-trend-stop
python -m rsi2.cli --ticker QQQ --use-stop --stop-pct 6 --out-dir results/qqq

Exit codes: 0 success, 1 validation failure, 2 no trades generated.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime

from .config import StrategyConfig
from . import data as data_mod, engine, metrics, report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rsi2", description="RSI(2) mean-reversion backtester")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker", help="download daily data with yfinance (auto_adjust=True)")
    src.add_argument("--csv", help="CSV with Date, Open, High, Low, Close[, Volume]")
    p.add_argument("--start", help="data start date for --ticker (YYYY-MM-DD)")
    p.add_argument("--end", help="data end date for --ticker (YYYY-MM-DD)")
    p.add_argument("--json-config", help="JSON file of StrategyConfig fields (CLI flags override it)")
    p.add_argument("--out-dir", help="output folder (default ./results/<ticker>_<timestamp>/)")
    p.add_argument("--repair-ohlc", action="store_true",
                   help="fix tiny (≤0.5%%) OHLC inconsistencies left by price adjustment before validating")
    for f in dataclasses.fields(StrategyConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            names = [flag]
            if f.name.startswith("use_"):              # --trend-stop / --no-trend-stop aliases
                names.append("--" + f.name[4:].replace("_", "-"))
            p.add_argument(*names, dest=f.name, action=argparse.BooleanOptionalAction, default=None)
        else:
            conv = {"int": int, "float": float}.get(str(f.type), str)
            p.add_argument(flag, dest=f.name, type=conv, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        values = {}
        if args.json_config:
            with open(args.json_config) as f:
                values.update(json.load(f))
        for name in StrategyConfig.field_names():
            v = getattr(args, name, None)
            if v is not None:
                values[name] = v
        cfg = StrategyConfig(**values)
        if args.ticker:
            df = data_mod.load_yfinance(args.ticker, args.start, args.end)
            name = args.ticker.upper()
        else:
            df = data_mod.load_csv(args.csv)
            name = os.path.splitext(os.path.basename(args.csv))[0]
        if args.repair_ohlc:
            df, fixed = data_mod.sanitize_ohlc(df)
            if fixed:
                print(f"Repaired {fixed} bar(s) with tiny OHLC inconsistencies (--repair-ohlc)")
        run = engine.run(df, cfg)
    except (ValueError, TypeError) as e:
        hint = ""
        if "outside [Low, High]" in str(e) or "High < Low" in str(e):
            hint = "\n(If these are tiny adjustment-rounding errors, re-run with --repair-ohlc.)"
        print(f"Validation error: {e}{hint}", file=sys.stderr)
        return 1

    summary = metrics.summarize(run.ledger, run.equity, run.open_trades)
    bench, _ = metrics.buy_and_hold(run.data, run.start_bar, cfg.total_bps, cfg.initial_capital)
    flagged = metrics.flagged_comparison(run.ledger)
    yearly = metrics.yearly_returns(run.equity["equity"])
    monthly = metrics.monthly_grid(run.equity["equity"])
    text = report.summary_text(name, summary, bench, flagged, yearly, cfg.to_dict())
    out_dir = args.out_dir or os.path.join("results", f"{name}_{datetime.now():%Y%m%d_%H%M%S}")
    report.write_outputs(out_dir, name, run, summary, bench, flagged, yearly, monthly, text)
    print(text)
    print(f"\nOutputs written to {out_dir}")
    return 2 if summary["trades"] == 0 and not run.open_trades else 0


if __name__ == "__main__":
    sys.exit(main())
