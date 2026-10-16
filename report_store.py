"""
report_store.py — the shared Saved Reports library.

Two pieces:

PendingCache   Every report the app generates is parked here (in memory, bounded,
               TTL) under a random token that goes back to the browser with the
               report. Saving sends only {token, ticker}: the server saves ITS OWN
               copy of the HTML, so nobody can plant arbitrary HTML in the library.

LocalReportStore
               The library itself, on disk under REPORTS_DIR:
                 reports.sqlite3     metadata index (one row per report)
                 files/<uuid>.html.gz  the report HTML, gzip-compressed
               A save writes the file first (tmp file + fsync + os.replace) and
               only then inserts the index row, inside one IMMEDIATE transaction
               that also picks the unique name and enforces REPORTS_MAX. So a
               crash mid-save can leave an unreferenced file (cleaned at startup)
               but never a listed report without its file.

Names: {TICKER}_financial_analysis_{YYYY-MM-DD}, {TICKER}_estimates_report_{date},
{TICKER}_backtest_report_{date}; the date is the day the report was RUN, in
REPORTS_TZ. A repeat on the same day gets _2, _3, … Names are labels only —
files are keyed by server-generated UUIDs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# type key → (name part, display label)
REPORT_TYPES = {
    "financial": ("financial_analysis", "Financial analysis"),
    "estimates": ("estimates_report", "Estimates"),
    "backtest": ("backtest_report", "Backtest"),
}
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
ID_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_REPORT_BYTES = 10 * 1024 * 1024          # one report's HTML (the largest today is ~0.3 MB)
DEFAULT_TZ = "America/Los_Angeles"


class ReportError(Exception):
    """An error with an HTTP status the API can return as-is."""
    status = 400

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        if status is not None:
            self.status = status


# ── Naming ────────────────────────────────────────────────────────────────────

def normalize_ticker(raw) -> str:
    """Trim + upper-case; must match TICKER_RE (1–10 of A-Z 0-9 . -, starting with a letter/digit)."""
    t = str(raw or "").strip().upper()
    if not t:
        raise ReportError("Enter the ticker for this report.")
    if not TICKER_RE.match(t):
        raise ReportError(f"“{t[:20]}” isn't a valid ticker for a report name — use 1–10 letters, digits, "
                          f"dots or dashes, e.g. AAPL or BRK.B.")
    return t


def suggest_ticker(detected) -> str:
    """Best-effort prefill from a detected symbol (e.g. ^GSPC → GSPC, EURUSD=X → EURUSD-X)."""
    t = str(detected or "").strip().upper().replace("^", "").replace("=", "-")
    t = re.sub(r"[^A-Z0-9.\-]", "", t).lstrip(".-")[:10]
    return t if TICKER_RE.match(t) else ""


def load_tz(name: str | None):
    """ZoneInfo for name; falls back to UTC with a warning (Windows needs the tzdata package)."""
    from zoneinfo import ZoneInfo
    name = (name or DEFAULT_TZ).strip() or DEFAULT_TZ
    try:
        return ZoneInfo(name)
    except Exception as e:
        logger.warning(f"Time zone {name!r} unavailable ({e}); report dates will use UTC. "
                       f"On Windows run: pip install tzdata")
        return timezone.utc


def run_date(created_at: datetime, tz) -> str:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return created_at.astimezone(tz).strftime("%Y-%m-%d")


def base_name(rtype: str, ticker: str, date: str) -> str:
    if rtype not in REPORT_TYPES:
        raise ReportError(f"Unknown report type {rtype!r}")
    return f"{ticker}_{REPORT_TYPES[rtype][0]}_{date}"


# ── Pending (not yet saved) reports ───────────────────────────────────────────

@dataclass
class PendingReport:
    html: str
    rtype: str
    detected_ticker: str
    created_at: datetime
    source: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    size: int = 0
    saved_id: str | None = None                        # set once saved — a token saves at most once
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class PendingCache:
    """Thread-safe LRU of freshly generated reports, bounded by count, bytes and age."""

    def __init__(self, max_entries: int = 50, max_bytes: int = 50 * 1024 * 1024, ttl_s: float = 2 * 3600,
                 clock=time.monotonic):
        self.max_entries, self.max_bytes, self.ttl_s, self.clock = max_entries, max_bytes, ttl_s, clock
        self._d: OrderedDict[str, tuple[float, PendingReport]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        for tok in [t for t, (ts, _) in self._d.items() if now - ts > self.ttl_s]:
            self._bytes -= self._d.pop(tok)[1].size
        while self._d and (len(self._d) > self.max_entries or self._bytes > self.max_bytes):
            self._bytes -= self._d.popitem(last=False)[1][1].size

    def put(self, html: str, rtype: str, detected_ticker: str = "", source: dict | None = None,
            params: dict | None = None, created_at: datetime | None = None) -> tuple[str, PendingReport]:
        if rtype not in REPORT_TYPES:
            raise ValueError(f"unknown report type {rtype!r}")
        rep = PendingReport(html=html, rtype=rtype, detected_ticker=str(detected_ticker or ""),
                            created_at=created_at or datetime.now(timezone.utc),
                            source=dict(source or {}), params=dict(params or {}),
                            size=len(html.encode("utf-8")))
        tok = secrets.token_urlsafe(24)
        with self._lock:
            now = self.clock()
            self._d[tok] = (now, rep)
            self._bytes += rep.size
            self._evict(now)
        return tok, rep

    def get(self, token) -> PendingReport | None:
        if not isinstance(token, str) or not token:
            return None
        with self._lock:
            now = self.clock()
            self._evict(now)
            hit = self._d.get(token)
            return hit[1] if hit else None

    def __len__(self):
        with self._lock:
            return len(self._d)

    @property
    def total_bytes(self) -> int:
        return self._bytes


# ── The library ───────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    base_name      TEXT NOT NULL,
    type           TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    saved_at_utc   TEXT NOT NULL,
    run_date_local TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    source         TEXT NOT NULL DEFAULT '{}',
    params         TEXT NOT NULL DEFAULT '{}',
    sha256         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_reports_created ON reports (created_at_utc DESC);
CREATE INDEX IF NOT EXISTS ix_reports_type_ticker ON reports (type, ticker);
CREATE INDEX IF NOT EXISTS ix_reports_base ON reports (base_name);
"""
_COLS = ("id", "name", "type", "ticker", "created_at_utc", "saved_at_utc", "run_date_local",
         "size_bytes", "source", "params", "sha256")


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _row(r: sqlite3.Row) -> dict:
    d = {k: r[k] for k in _COLS}
    d["source"] = json.loads(d["source"] or "{}")
    d["params"] = json.loads(d["params"] or "{}")
    d["type_label"] = REPORT_TYPES.get(d["type"], (None, d["type"]))[1]
    d["run_date"] = d["run_date_local"]
    d["created_at"] = d["created_at_utc"]
    return d


class LocalReportStore:
    """Reports on local disk (use a Render persistent disk in production)."""

    def __init__(self, root: str, tz=None, max_reports: int = 2000):
        self.root = os.path.abspath(root)
        self.files = os.path.join(self.root, "files")
        self.db_path = os.path.join(self.root, "reports.sqlite3")
        self.tz = tz or timezone.utc
        self.max_reports = int(max_reports)
        self.available, self.reason = False, None
        try:
            os.makedirs(self.files, exist_ok=True)
            probe = os.path.join(self.root, f".probe-{uuid.uuid4().hex}")
            with open(probe, "wb") as f:
                f.write(b"ok")
            os.remove(probe)
            with self._conn() as c:
                c.executescript(_SCHEMA)
            self.available = True
            self._clean_orphans()
        except (OSError, sqlite3.Error) as e:
            self.reason = f"{type(e).__name__}: {e}"
            logger.warning(f"Report library disabled — {self.root} is not usable ({self.reason})")

    # ── internals ────────────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)   # explicit transactions
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout = 30000")
        return c

    @contextmanager
    def _conn(self):
        c = self._connect()
        try:
            yield c
        finally:
            c.close()

    def _path(self, rid: str) -> str:
        if not ID_RE.match(rid or ""):
            raise ReportError("Report not found.", 404)
        return os.path.join(self.files, f"{rid}.html.gz")

    def _require(self):
        if not self.available:
            raise ReportError("Report library is unavailable on this server.", 503)

    def _clean_orphans(self, min_age_s: float = 3600) -> None:
        """Remove files with no index row (left by a crash between file write and insert)."""
        try:
            with self._conn() as c:
                ids = {r[0] for r in c.execute("SELECT id FROM reports")}
            now = time.time()
            for fn in os.listdir(self.files):
                rid = fn.split(".", 1)[0]
                p = os.path.join(self.files, fn)
                if (fn.endswith(".tmp") or (fn.endswith(".html.gz") and rid not in ids)) \
                        and now - os.path.getmtime(p) > min_age_s:
                    os.remove(p)
        except OSError as e:
            logger.warning(f"Orphan cleanup skipped: {e}")

    def _write_file(self, rid: str, data: bytes) -> str:
        final = self._path(rid)
        tmp = final + f".{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp, "wb") as f:
                with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
                    gz.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, final)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return final

    # ── public API ───────────────────────────────────────────────────────────
    def save(self, html: str, rtype: str, ticker: str, created_at: datetime | None = None,
             source: dict | None = None, params: dict | None = None) -> dict:
        self._require()
        if rtype not in REPORT_TYPES:
            raise ReportError(f"Unknown report type {rtype!r}")
        ticker = normalize_ticker(ticker)
        if not isinstance(html, str) or not html:
            raise ReportError("Empty report.")
        data = html.encode("utf-8")
        if len(data) > MAX_REPORT_BYTES:
            raise ReportError("This report is too large to save.", 413)
        created_at = created_at or datetime.now(timezone.utc)
        date = run_date(created_at, self.tz)
        base = base_name(rtype, ticker, date)
        rid = uuid.uuid4().hex
        sha = hashlib.sha256(data).hexdigest()

        path = self._write_file(rid, data)                 # 1. the file (atomic)
        try:                                               # 2. the index row (atomic, serialized)
            c = self._connect()
            try:
                c.execute("BEGIN IMMEDIATE")
                n = c.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
                if n >= self.max_reports:
                    raise ReportError(f"The report library is full ({self.max_reports} reports). "
                                      f"Ask the admin to delete old reports.", 507)
                taken = {r[0] for r in c.execute("SELECT name FROM reports WHERE base_name = ?", (base,))}
                name, k = base, 2
                while name in taken:
                    name, k = f"{base}_{k}", k + 1
                now = _iso(datetime.now(timezone.utc))
                c.execute(f"INSERT INTO reports ({', '.join(_COLS)}, base_name) VALUES ({', '.join('?' * (len(_COLS) + 1))})",
                          (rid, name, rtype, ticker, _iso(created_at), now, date, len(data),
                           json.dumps(source or {}), json.dumps(params or {}), sha, base))
                c.execute("COMMIT")
            except BaseException:
                if c.in_transaction:
                    c.execute("ROLLBACK")
                raise
            finally:
                c.close()
        except BaseException:
            try:
                os.remove(path)
            except OSError:
                pass
            raise
        return self.meta(rid)

    def meta(self, rid: str) -> dict:
        self._require()
        self._path(rid)
        with self._conn() as c:
            r = c.execute(f"SELECT {', '.join(_COLS)} FROM reports WHERE id = ?", (rid,)).fetchone()
        if r is None:
            raise ReportError("Report not found.", 404)
        return _row(r)

    def get(self, rid: str) -> tuple[dict, str]:
        m = self.meta(rid)
        try:
            with gzip.open(self._path(rid), "rb") as f:
                data = f.read()
        except FileNotFoundError:
            raise ReportError("This report's file is missing on the server.", 404)
        if hashlib.sha256(data).hexdigest() != m["sha256"]:
            raise ReportError("This report's file is damaged on the server.", 500)
        return m, data.decode("utf-8")

    def list(self, rtype: str | None = None, ticker: str | None = None, q: str | None = None,
             limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        self._require()
        where, args = [], []
        if rtype:
            if rtype not in REPORT_TYPES:
                raise ReportError(f"Unknown report type {rtype!r}")
            where.append("type = ?"); args.append(rtype)
        if ticker:
            where.append("ticker LIKE ? ESCAPE '\\'"); args.append(_like(str(ticker).strip().upper()) + "%")
        if q:
            where.append("name LIKE ? ESCAPE '\\'"); args.append("%" + _like(str(q).strip()) + "%")
        w = (" WHERE " + " AND ".join(where)) if where else ""
        limit, offset = max(1, min(int(limit), 200)), max(0, int(offset))
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM reports{w}", args).fetchone()[0]
            rows = c.execute(f"SELECT {', '.join(_COLS)} FROM reports{w} "
                             f"ORDER BY created_at_utc DESC, saved_at_utc DESC LIMIT ? OFFSET ?",
                             args + [limit, offset]).fetchall()
        return [_row(r) for r in rows], total

    def delete(self, rid: str) -> None:
        self._require()
        path = self._path(rid)
        with self._conn() as c:
            cur = c.execute("DELETE FROM reports WHERE id = ?", (rid,))
        if cur.rowcount == 0:
            raise ReportError("Report not found.", 404)
        try:
            os.remove(path)
        except OSError:
            pass

    def count(self) -> int:
        self._require()
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM reports").fetchone()[0]


def _like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
