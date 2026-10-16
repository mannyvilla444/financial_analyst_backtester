# app.py
import hmac
import logging
import os
import re
import threading

from dotenv import load_dotenv, dotenv_values
from flask import Flask, Response, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Settings ──────────────────────────────────────────────────────────────────
# 1. .env next to this file (always this folder's, whatever the working dir is)
# 2. Telegram settings fall back to the Telegram bot's own .env, so the web app
#    works without copying the token over. Default location is the sibling
#    folder ../financial_bot_agent/.env; override with BOT_ENV_PATH.
ENV_FILE = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_FILE)
BOT_ENV_FILE = os.environ.get("BOT_ENV_PATH") or os.path.join(
    os.path.dirname(BASE_DIR), "financial_bot_agent", ".env")
TELEGRAM_SETTINGS_SOURCE = {}
for _key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
    if os.environ.get(_key, "").strip():
        TELEGRAM_SETTINGS_SOURCE[_key] = ENV_FILE if os.path.exists(ENV_FILE) else "environment"
    elif os.path.exists(BOT_ENV_FILE):
        _val = (dotenv_values(BOT_ENV_FILE).get(_key) or "").strip()
        if _val:
            os.environ[_key] = _val
            TELEGRAM_SETTINGS_SOURCE[_key] = BOT_ENV_FILE

from pipeline import run_analysis                          # noqa: E402
from tradingview_downloader import download as tv_download # noqa: E402
import web_reports                                         # noqa: E402
import compare_report                                      # noqa: E402
import signal_worker                                       # noqa: E402
import telegram_share                                      # noqa: E402
import report_store                                        # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB upload limit
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-not-secret")

SIGNALS_FILE = os.environ.get("SIGNALS_FILE", os.path.join(BASE_DIR, "signals.json"))
TELEGRAM_LINKS_FILE = os.environ.get("TELEGRAM_LINKS_FILE", os.path.join(BASE_DIR, "telegram_links.json"))

# ── Saved Reports library ─────────────────────────────────────────────────────
# Every generated report is parked in PENDING_REPORTS under a token that goes
# back with the report; POST /api/reports {report_token, ticker} saves the
# server's own copy. The library lives in REPORTS_DIR — on Render, mount a
# persistent disk there or the library is wiped on every deploy.
REPORTS_DIR = os.environ.get("REPORTS_DIR") or os.path.join(BASE_DIR, "saved_reports")
REPORTS_TZ = report_store.load_tz(os.environ.get("REPORTS_TZ"))
try:
    REPORTS_MAX = int(os.environ.get("REPORTS_MAX") or 2000)
except ValueError:
    REPORTS_MAX = 2000
REPORTS = report_store.LocalReportStore(REPORTS_DIR, REPORTS_TZ, REPORTS_MAX)
PENDING_REPORTS = report_store.PendingCache()


def _admin_token() -> str:
    return (os.environ.get("REPORTS_ADMIN_TOKEN") or "").strip()


def _report_offer(html: str, rtype: str, ticker: str, source: dict, params: dict | None = None) -> dict:
    """Extra response fields so the browser can offer to save this report (empty if saving is off)."""
    if not REPORTS.available:
        return {"report_save": {"available": False}}
    token, rep = PENDING_REPORTS.put(html, rtype, ticker, source, params)
    return {"report_token": token,
            "report_save": {"available": True, "type": rtype, "type_part": report_store.REPORT_TYPES[rtype][0],
                            "run_date": report_store.run_date(rep.created_at, REPORTS_TZ),
                            "created_at": report_store._iso(rep.created_at),
                            "suggested_ticker": report_store.suggest_ticker(ticker)}}


# Tickers like AMD, BRK.B, BF-B, ^GSPC, EURUSD=X
TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,14}$")


def _ticker_from_request():
    data   = request.get_json(force=True, silent=True) or {}
    ticker = str(data.get("ticker", "")).strip().upper()
    if not ticker:
        return None, data, (jsonify(error="ticker is required"), 400)
    if not TICKER_RE.match(ticker):
        return None, data, (jsonify(error=f"'{ticker}' doesn't look like a ticker symbol"), 400)
    return ticker, data, None


# ── Start background signal thread on app startup ─────────────────────────────
# Runs once per process. render.yaml uses a single Gunicorn worker so there is
# exactly one monitor; set DISABLE_SIGNAL_WORKER=1 to turn it off (e.g. tests).
_signal_thread = None
if os.environ.get("DISABLE_SIGNAL_WORKER") != "1":
    _signal_thread = threading.Thread(
        target=signal_worker.run_forever,
        kwargs={"signals_file": SIGNALS_FILE},
        daemon=True,
        name="signal-worker",
    )
    _signal_thread.start()


# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify(status="ok",
                   signal_worker=bool(_signal_thread and _signal_thread.is_alive()))


# ── CSV upload → HTML report ───────────────────────────────────────────────────
@app.route("/api/analyze/csv", methods=["POST"])
def analyze_csv():
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".csv"):
        return jsonify(error="File must be a .csv"), 400
    try:
        html_bytes, ticker, filename, summary = run_analysis(f.read(), f.filename)
        html = html_bytes.decode("utf-8")
        return jsonify(ticker=ticker, html=html, filename=filename, summary=summary,
                       **_report_offer(html, "financial", ticker, {"kind": "csv", "filename": f.filename[:200]}))
    except ValueError as e:
        return jsonify(error=str(e)), 422
    except Exception as e:
        logger.exception(f"CSV analysis failed for {f.filename}")
        return jsonify(error=f"Analysis failed: {e}"), 500


# ── Ticker text → HTML report (auto-fetches via yfinance) ─────────────────────
def _live_price(ticker: str) -> float | None:
    """Latest traded price from yfinance (None if unavailable)."""
    try:
        import yfinance as yf
        fi = yf.Ticker(ticker).fast_info
        p = getattr(fi, "last_price", None)
        if p is None and hasattr(fi, "get"):
            p = fi.get("lastPrice")
        p = float(p)
        return p if p > 0 else None
    except Exception as e:
        logger.warning(f"Live price unavailable for {ticker}: {e}")
        return None


@app.route("/api/analyze/ticker", methods=["POST"])
def analyze_ticker():
    ticker, _, err = _ticker_from_request()
    if err:
        return err
    try:
        csv_bytes, filename = tv_download(ticker)
        # Use the live quote for the current price (falls back to the latest
        # weekly close inside run_analysis if yfinance can't provide one).
        html_bytes, ticker, html_name, summary = run_analysis(
            csv_bytes, filename, current_price=_live_price(ticker))
        html = html_bytes.decode("utf-8")
        return jsonify(ticker=ticker, html=html, filename=html_name, summary=summary,
                       **_report_offer(html, "financial", ticker, {"kind": "ticker"}))
    except (RuntimeError, ValueError) as e:
        return jsonify(error=str(e)), 422
    except Exception as e:
        logger.exception(f"Ticker analysis failed for {ticker}")
        return jsonify(error=f"Fetch failed: {e}"), 500


# ── Multi-company comparison ───────────────────────────────────────────────────
@app.route("/api/analyze/compare", methods=["POST"])
def analyze_compare():
    data = request.get_json(force=True, silent=True) or {}
    companies = data.get("companies") or []
    if not isinstance(companies, list) or len(companies) < 2:
        return jsonify(error="Need at least two analysed companies to compare"), 400
    if len(companies) > 60:
        return jsonify(error="Too many companies (max 60)"), 400
    try:
        market_size = data.get("market_size_b")
        market_size = float(market_size) if market_size not in (None, "") else None
    except (TypeError, ValueError):
        market_size = None
    try:
        html_str = compare_report.generate_comparison_html(companies, market_size)
    except ValueError as e:
        return jsonify(error=str(e)), 422
    except Exception as e:
        logger.exception("Comparison failed")
        return jsonify(error=f"Comparison failed: {e}"), 500
    return jsonify(html=html_str, filename="peer_comparison.html",
                   tickers=[c.get("ticker") for c in companies if isinstance(c, dict)])


# ── Send a report to Telegram ──────────────────────────────────────────────────
@app.route("/api/telegram/status")
def telegram_status():
    token = telegram_share.get_token()
    info  = telegram_share.bot_info(token) if token else None
    return jsonify(
        configured=bool(token),
        bot_username=(info or {}).get("username"),
        error=(info or {}).get("error"),
        links=telegram_share.load_links(TELEGRAM_LINKS_FILE),
    )


@app.route("/api/telegram/diagnose", methods=["POST"])
def telegram_diagnose():
    data = request.get_json(force=True, silent=True) or {}
    result = telegram_share.diagnose(str(data.get("link", "")))
    result["app_folder"] = BASE_DIR
    result["env_file"] = ENV_FILE if os.path.exists(ENV_FILE) else None
    result["token_source"] = TELEGRAM_SETTINGS_SOURCE.get("TELEGRAM_BOT_TOKEN")
    result["chat_id_source"] = TELEGRAM_SETTINGS_SOURCE.get("TELEGRAM_CHAT_ID")
    return jsonify(result)


@app.route("/api/telegram/links", methods=["DELETE"])
def telegram_forget_link():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(links=telegram_share.forget_link(TELEGRAM_LINKS_FILE, str(data.get("link", ""))))


@app.route("/api/telegram/send", methods=["POST"])
def telegram_send():
    data     = request.get_json(force=True, silent=True) or {}
    link     = str(data.get("link", ""))
    reports  = data.get("reports")
    if not isinstance(reports, list):
        reports = [{"html": data.get("html"), "filename": data.get("filename"), "caption": data.get("caption")}]
    reports = [r for r in reports if isinstance(r, dict) and r.get("html")]
    if not reports:
        return jsonify(error="No report to send"), 400
    if len(reports) > 30:
        return jsonify(error="Too many reports in one send (max 30)"), 400

    sent, result = [], None
    try:
        for r in reports:
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(r.get("filename") or "report.html"))[:80]
            if not name.lower().endswith(".html"):
                name += ".html"
            caption = str(r.get("caption") or name.replace("_", " ").rsplit(".", 1)[0])
            result = telegram_share.send_document(link, name, str(r["html"]).encode("utf-8"), caption)
            sent.append(name)
    except telegram_share.TelegramError as e:
        return jsonify(error=str(e), sent=sent), 422 if not sent else 207
    except Exception as e:
        logger.exception("Telegram send failed")
        return jsonify(error=f"Send failed: {e}", sent=sent), 500

    links = telegram_share.remember_link(TELEGRAM_LINKS_FILE, result["link"], result["chat_title"])
    return jsonify(ok=True, sent=sent, chat_title=result["chat_title"], link=result["link"], links=links)


# ── Estimates analysis ─────────────────────────────────────────────────────────
@app.route("/api/analyze/estimates", methods=["POST"])
def analyze_estimates():
    ticker, _, err = _ticker_from_request()
    if err:
        return err
    try:
        # Never raises for missing data — returns a "no estimates" page instead.
        html_str = web_reports.render_estimates(ticker)
        return jsonify(ticker=ticker, html=html_str, filename=f"{ticker}_estimates.html",
                       **_report_offer(html_str, "estimates", ticker, {"kind": "ticker"}))
    except Exception as e:
        logger.exception(f"Estimates failed for {ticker}")
        return jsonify(error=f"Estimates failed: {e}"), 500


# ── MACD backtest (synchronous — runs in ~5–30 s) ─────────────────────────────
@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    ticker, data, err = _ticker_from_request()
    if err:
        return err
    timeframe = data.get("timeframe", "weekly")   # "weekly" | "daily"
    if timeframe not in web_reports.TIMEFRAMES:
        return jsonify(error="timeframe must be 'weekly' or 'daily'"), 400
    stop = data.get("stop") or "none"              # a rsi2.ab.STOP_PRESETS key
    if stop not in web_reports.STOP_PRESETS:
        return jsonify(error=f"stop must be one of: {', '.join(web_reports.STOP_PRESETS)}"), 400
    try:
        html_str = web_reports.render_backtest(ticker, timeframe, stop)
        suffix = "" if stop == "none" else f"_stop-{stop}"
        return jsonify(ticker=ticker, html=html_str, timeframe=timeframe, stop=stop,
                       filename=f"{ticker}_backtest_{timeframe}{suffix}.html",
                       **_report_offer(html_str, "backtest", ticker, {"kind": "ticker"},
                                       {"timeframe": timeframe, "stop": stop,
                                        "stop_label": web_reports.STOP_PRESETS[stop].label}))
    except RuntimeError as e:
        return jsonify(error=str(e)), 422
    except Exception as e:
        logger.exception(f"Backtest failed for {ticker}")
        return jsonify(error=f"Backtest failed: {e}"), 500


# ── Saved Reports library ──────────────────────────────────────────────────────
@app.errorhandler(report_store.ReportError)
def _report_error(e):
    return jsonify(error=str(e)), e.status


def _is_admin() -> bool:
    want, got = _admin_token(), (request.headers.get("X-Admin-Token") or "").strip()
    return bool(want) and bool(got) and hmac.compare_digest(want.encode(), got.encode())


@app.route("/api/reports/status")
def reports_status():
    out = {"available": REPORTS.available, "delete_enabled": bool(_admin_token()),
           "timezone": str(getattr(REPORTS_TZ, "key", "UTC")), "max": REPORTS_MAX}
    if REPORTS.available:
        out["count"] = REPORTS.count()
    else:
        out["message"] = "Report library is unavailable on this server."
    return jsonify(out)


@app.route("/api/reports", methods=["POST"])
def save_report():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send JSON: {report_token, ticker}"), 400
    if "html" in data:
        # Reports are only ever saved from the server's own copy (see report_store.PendingCache).
        return jsonify(error="Reports can't be uploaded — run the report, then save it."), 400
    if not REPORTS.available:
        return jsonify(error="Report library is unavailable on this server."), 503
    token = data.get("report_token")
    if not isinstance(token, str) or not token:
        return jsonify(error="report_token is required"), 400
    pending = PENDING_REPORTS.get(token)
    if pending is None:
        return jsonify(error="This report expired — run it again to save it."), 410
    with pending.lock:                                   # one save per generated report (double-clicks)
        if pending.saved_id:
            try:
                return jsonify({**REPORTS.meta(pending.saved_id), "already_saved": True}), 200
            except report_store.ReportError:
                pending.saved_id = None                      # deleted since — allow saving again
        meta = REPORTS.save(pending.html, pending.rtype, data.get("ticker"), created_at=pending.created_at,
                            source={**pending.source, "detected_ticker": pending.detected_ticker},
                            params=pending.params)
        pending.saved_id = meta["id"]
    logger.info(f"Saved report {meta['name']} ({meta['size_bytes']:,} bytes)")
    return jsonify(meta), 201


def _int_arg(name, default, lo, hi):
    try:
        v = int(request.args.get(name, default))
    except (TypeError, ValueError):
        raise report_store.ReportError(f"{name} must be a whole number")
    return max(lo, min(hi, v))


@app.route("/api/reports", methods=["GET"])
def list_reports():
    if not REPORTS.available:
        return jsonify(error="Report library is unavailable on this server."), 503
    rtype = (request.args.get("type") or "").strip().lower() or None
    if rtype in ("all",):
        rtype = None
    rows, total = REPORTS.list(rtype=rtype, ticker=(request.args.get("ticker") or "")[:20] or None,
                               q=(request.args.get("q") or "")[:100] or None,
                               limit=_int_arg("limit", 25, 1, 100), offset=_int_arg("offset", 0, 0, 10**6))
    return jsonify(reports=rows, total=total)


@app.route("/api/reports/<rid>", methods=["GET"])
def get_report(rid):
    meta, html = REPORTS.get(rid)
    return jsonify({**meta, "html": html})


@app.route("/api/reports/<rid>/download")
def download_report(rid):
    meta, html = REPORTS.get(rid)
    return Response(html.encode("utf-8"), mimetype="text/html; charset=utf-8", headers={
        "Content-Disposition": f'attachment; filename="{meta["name"]}.html"',
        "X-Content-Type-Options": "nosniff",
    })


@app.route("/api/reports/admin", methods=["POST"])
def reports_admin_check():
    if not _admin_token():
        return jsonify(error="Deleting reports is turned off on this server (REPORTS_ADMIN_TOKEN isn't set)."), 403
    if not _is_admin():
        return jsonify(error="Wrong admin token."), 403
    return jsonify(ok=True)


@app.route("/api/reports/<rid>", methods=["DELETE"])
def delete_report(rid):
    if not _admin_token():
        return jsonify(error="Deleting reports is turned off on this server (REPORTS_ADMIN_TOKEN isn't set)."), 403
    if not _is_admin():
        return jsonify(error="Wrong admin token."), 403
    meta = REPORTS.meta(rid)
    REPORTS.delete(rid)
    logger.info(f"Deleted report {meta['name']}")
    return jsonify(ok=True, id=rid, name=meta["name"])


# ── Signal monitoring ──────────────────────────────────────────────────────────
def _load_signals_state() -> dict:
    return signal_worker.load_state(SIGNALS_FILE)


def _save_signals_state(state: dict):
    signal_worker.save_state(SIGNALS_FILE, state)


def _signals_payload(state: dict, signals=None):
    return dict(subscribed=state["subscribed"],
                watchers={t: state["watchers"].get(t) for t in state["subscribed"]},
                **({"signals": signals} if signals is not None else {}))


@app.route("/api/signals/subscribe", methods=["POST"])
def subscribe_signal():
    ticker, _, err = _ticker_from_request()
    if err:
        return err

    with signal_worker.state_lock:
        state = _load_signals_state()
        if ticker in state["subscribed"]:
            return jsonify(**_signals_payload(state), new_signals=[])

    # Backtest both strategies to pick what to monitor (like /notify on).
    # Slow (a few seconds) — done outside the lock.
    try:
        watcher = signal_worker.choose_watcher(ticker)
    except RuntimeError as e:
        return jsonify(error=f"Couldn't subscribe to {ticker}: {e}"), 422
    except Exception as e:
        logger.exception(f"Subscribe failed for {ticker}")
        return jsonify(error=f"Couldn't subscribe to {ticker}: {e}"), 500

    with signal_worker.state_lock:
        state = _load_signals_state()
        if ticker not in state["subscribed"]:
            state["subscribed"].append(ticker)
        state["watchers"][ticker] = watcher
        _save_signals_state(state)

    # First check right away rather than waiting for the next hourly poll.
    new = signal_worker.check_ticker_now(SIGNALS_FILE, ticker)

    with signal_worker.state_lock:
        state = _load_signals_state()
    return jsonify(**_signals_payload(state), new_signals=new)


@app.route("/api/signals/subscribe", methods=["DELETE"])
def unsubscribe_signal():
    data   = request.get_json(force=True, silent=True) or {}
    ticker = str(data.get("ticker", "")).strip().upper()
    with signal_worker.state_lock:
        state = _load_signals_state()
        state["subscribed"] = [t for t in state["subscribed"] if t != ticker]
        state["watchers"].pop(ticker, None)
        _save_signals_state(state)
    return jsonify(**_signals_payload(state))


@app.route("/api/signals")
def get_signals():
    try:
        since = float(request.args.get("since", 0))
    except ValueError:
        since = 0.0
    with signal_worker.state_lock:
        state = _load_signals_state()
    log = [s for s in state["log"] if s.get("timestamp", 0) > since]
    return jsonify(**_signals_payload(state, signals=log))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File too large (20 MB max)"), 413


if __name__ == "__main__":
    # use_reloader=False: the reloader imports this module twice, which would
    # start two signal threads.
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)), use_reloader=False)
