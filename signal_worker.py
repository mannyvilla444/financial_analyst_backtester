# signal_worker.py
"""
Background MACD signal monitoring for the web app (replaces the Telegram
bot's signal_monitor.py — alerts land in the Signals tab instead of chat).

signals.json shape:
{
  "subscribed": ["AMD", ...],
  "watchers":   {"AMD": {"strategy": "histogram", "fast": 12, "slow": 26,
                         "signal": 9, "atr_mult": 2.0, "interval": "1wk",
                         "params_label": "...", "backtest_pnl": ..., ...}},
  "log":        [ newest-first signal dicts, max 500 ]
}
"""
import json
import os
import threading
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
POLL_INTERVAL = int(os.environ.get("SIGNAL_POLL_INTERVAL_S", 3600))
MAX_LOG = 500

# Shared by the worker thread and the Flask request handlers so that
# read-modify-write cycles on signals.json never interleave.
state_lock = threading.Lock()


def load_state(signals_file: str) -> dict:
    try:
        with open(signals_file) as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("not an object")
    except FileNotFoundError:
        state = {}
    except Exception as e:
        logger.warning(f"signals.json unreadable ({e}); starting fresh")
        state = {}
    state.setdefault("subscribed", [])
    state.setdefault("watchers", {})
    state.setdefault("log", [])
    return state


def save_state(signals_file: str, state: dict):
    """Atomic write: a crash mid-write can't leave a truncated JSON file."""
    tmp = signals_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, signals_file)


# ── Strategy selection (same as the bot's /notify on) ─────────────────────────

def choose_watcher(ticker: str, interval: str = "1wk") -> dict:
    """
    Backtest both MACD strategies for `ticker` and return the winner's
    settings to monitor with. Raises RuntimeError if there's no price data.
    """
    import macd_backtest
    (_, h_best, _, _, _, price_df,
     _, cx_best, _, _, _, winner) = macd_backtest.run_all(
        ticker, fixed_position_size=10_000, starting_capital=10_000, interval=interval)
    best = h_best if winner == "histogram" else cx_best
    return {
        "strategy":     winner,
        "fast":         int(best.fast),
        "slow":         int(best.slow),
        "signal":       int(best.signal),
        "atr_mult":     float(best.atr_mult),
        "interval":     interval,
        "params_label": best.params_label,
        "backtest_pnl": round(float(getattr(best, "total_pnl", 0.0)), 2),
        "win_rate":     round(float(getattr(best, "win_rate", 0.0)), 1),
        "trades":       int(getattr(best, "total_trades", 0)),
        "added_at":     datetime.now().isoformat(timespec="seconds"),
    }


# ── Signal checks ─────────────────────────────────────────────────────────────

def _last_signal_type(log: list, ticker: str):
    for s in log:                      # log is newest-first
        if s.get("ticker") == ticker:
            return s.get("type")
    return None


def check_signal(ticker: str, watcher: dict | None = None, last_type: str | None = None) -> dict | None:
    """
    Check whether a new MACD signal has fired for ticker.
    Returns a signal dict or None.

    Shape:
    {
        "ticker":    "AMD",
        "type":      "BUY" | "SELL",
        "price":     142.30,
        "timestamp": 1712000000.0,
        "message":   "MACD histogram first upturn … on weekly",
        ...extra fields: candle_date, strategy, interval, params, stop
    }
    """
    w = watcher or {}
    try:
        import macd_backtest
        return macd_backtest.check_current_signal(
            ticker,
            fast=w.get("fast", 12), slow=w.get("slow", 26), signal=w.get("signal", 9),
            atr_mult=w.get("atr_mult", 2.0),
            strategy=w.get("strategy", "histogram"),
            interval=w.get("interval", "1wk"),
            in_trade=(last_type == "BUY"),
        )
    except Exception as e:
        logger.warning(f"Signal check failed for {ticker}: {e}")
        return None


def _check_tickers(signals_file: str, tickers: list[str]) -> list[dict]:
    """Check `tickers` and append any genuinely new signals to the log."""
    with state_lock:
        state = load_state(signals_file)
        watchers = dict(state["watchers"])
        last_types = {t: _last_signal_type(state["log"], t) for t in tickers}

    # Network calls happen outside the lock so the web UI stays responsive.
    new_signals = []
    for ticker in tickers:
        sig = check_signal(ticker, watchers.get(ticker), last_types.get(ticker))
        # De-duplicate like the bot: never log the same direction twice in a
        # row for a ticker. This also stops the still-forming weekly candle
        # from re-alerting on every hourly poll.
        if sig and sig["type"] != last_types.get(ticker):
            new_signals.append(sig)
            logger.info(f"Signal: {sig}")

    if not new_signals:
        return []

    # Re-read before writing so subscribe/unsubscribe changes made while the
    # checks were running are not overwritten; drop signals for tickers that
    # were unsubscribed in the meantime.
    with state_lock:
        state = load_state(signals_file)
        keep = [s for s in new_signals if s["ticker"] in state["subscribed"]]
        if keep:
            state["log"] = (list(reversed(keep)) + state["log"])[:MAX_LOG]
            save_state(signals_file, state)
    return keep


def check_ticker_now(signals_file: str, ticker: str) -> list[dict]:
    """Immediate one-off check (used right after subscribing)."""
    return _check_tickers(signals_file, [ticker])


def run_forever(signals_file: str):
    """Infinite loop — runs as a daemon thread inside the Flask process."""
    logger.info(f"Signal worker started (every {POLL_INTERVAL}s)")
    while True:
        try:
            _run_once(signals_file)
        except Exception as e:
            logger.error(f"Signal worker error: {e}")
        time.sleep(POLL_INTERVAL)


def _run_once(signals_file: str):
    with state_lock:
        tickers = list(load_state(signals_file)["subscribed"])
    if tickers:
        _check_tickers(signals_file, tickers)
