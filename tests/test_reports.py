"""Saved Reports library: naming, pending tokens, the on-disk store, the API and the report routes."""
import gzip
import io
import os
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone

import pytest

os.environ.setdefault("DISABLE_SIGNAL_WORKER", "1")
os.environ.setdefault("REPORTS_DIR", tempfile.mkdtemp(prefix="finbot-reports-test-"))   # importing app must not touch the real library

import report_store as R                                    # noqa: E402

LA = R.load_tz("America/Los_Angeles")
T0 = datetime(2026, 10, 16, 17, 42, tzinfo=timezone.utc)    # 10:42 AM in Los Angeles
HTML = "<!doctype html><title>t</title><p>−12.5% ★ 📊 naïve</p>"


@pytest.fixture
def store(tmp_path):
    return R.LocalReportStore(str(tmp_path / "lib"), LA, max_reports=100)


# ── Naming ────────────────────────────────────────────────────────────────────

def test_names_per_type(store):
    assert store.save(HTML, "financial", "aapl", created_at=T0)["name"] == "AAPL_financial_analysis_2026-10-16"
    assert store.save(HTML, "estimates", "AAPL", created_at=T0)["name"] == "AAPL_estimates_report_2026-10-16"
    assert store.save(HTML, "backtest", " AAPL ", created_at=T0)["name"] == "AAPL_backtest_report_2026-10-16"


def test_date_is_run_day_in_reports_tz(tmp_path):
    late = datetime(2026, 10, 17, 6, 30, tzinfo=timezone.utc)   # 23:30 on the 16th in Los Angeles
    assert R.run_date(late, LA) == "2026-10-16"
    utc = R.LocalReportStore(str(tmp_path / "utc"), timezone.utc)
    la = R.LocalReportStore(str(tmp_path / "la"), LA)
    assert utc.save(HTML, "backtest", "AAPL", created_at=late)["name"].endswith("2026-10-17")
    m = la.save(HTML, "backtest", "AAPL", created_at=late)
    assert m["name"].endswith("2026-10-16") and m["created_at_utc"].startswith("2026-10-17T06:30:00")


def test_bad_zone_falls_back_to_utc():
    assert R.load_tz("Not/AZone") is timezone.utc


@pytest.mark.parametrize("raw,ok", [("aapl", "AAPL"), (" brk.b ", "BRK.B"), ("BF-B", "BF-B"), ("7203", "7203"),
                                    ("ABCDEFGHIJ", "ABCDEFGHIJ")])
def test_ticker_normalization(raw, ok):
    assert R.normalize_ticker(raw) == ok


@pytest.mark.parametrize("raw", ["", "  ", "A B", "^GSPC", "EURUSD=X", "ABCDEFGHIJK", "../etc", "AAPL/1", ".AAPL", "-X"])
def test_ticker_rejected(raw):
    with pytest.raises(R.ReportError):
        R.normalize_ticker(raw)


def test_suggested_ticker_prefill():
    assert R.suggest_ticker("^GSPC") == "GSPC"
    assert R.suggest_ticker("eurusd=x") == "EURUSD-X"
    assert R.suggest_ticker("AMD") == "AMD"
    assert R.suggest_ticker("") == ""


def test_collisions_get_suffixes_under_concurrency(store):
    names, errors = [], []

    def worker():
        try:
            names.append(store.save(HTML, "financial", "AAPL", created_at=T0)["name"])
        except Exception as e:          # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    base = "AAPL_financial_analysis_2026-10-16"
    assert sorted(names) == sorted([base] + [f"{base}_{k}" for k in range(2, 21)])


def test_other_types_and_days_do_not_collide(store):
    store.save(HTML, "financial", "AAPL", created_at=T0)
    assert store.save(HTML, "estimates", "AAPL", created_at=T0)["name"] == "AAPL_estimates_report_2026-10-16"
    nxt = datetime(2026, 10, 17, 17, 0, tzinfo=timezone.utc)
    assert store.save(HTML, "financial", "AAPL", created_at=nxt)["name"] == "AAPL_financial_analysis_2026-10-17"


# ── Pending cache ─────────────────────────────────────────────────────────────

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_pending_ttl():
    clk = Clock()
    c = R.PendingCache(ttl_s=60, clock=clk)
    tok, _ = c.put(HTML, "estimates", "AMD")
    assert c.get(tok).html == HTML
    clk.t = 61
    assert c.get(tok) is None and len(c) == 0


def test_pending_bounds():
    c = R.PendingCache(max_entries=3, max_bytes=10_000)
    toks = [c.put(f"<p>{i}</p>", "estimates", "AMD")[0] for i in range(5)]
    assert [c.get(t) is not None for t in toks] == [False, False, True, True, True]
    big = R.PendingCache(max_entries=50, max_bytes=2_500)
    t1, _ = big.put("x" * 1000, "backtest", "A")
    t2, _ = big.put("y" * 1000, "backtest", "A")
    t3, _ = big.put("z" * 1000, "backtest", "A")          # 3000 bytes > 2500 → the oldest goes
    assert big.get(t1) is None and big.get(t2) and big.get(t3) and big.total_bytes == 2000


def test_pending_tokens_are_random_and_unguessable():
    c = R.PendingCache()
    toks = {c.put(HTML, "financial", "A")[0] for _ in range(50)}
    assert len(toks) == 50 and all(len(t) >= 32 for t in toks)
    assert c.get("") is None and c.get(None) is None and c.get(123) is None


# ── Store ─────────────────────────────────────────────────────────────────────

def test_round_trip_is_byte_identical(store):
    m = store.save(HTML, "backtest", "AMD", created_at=T0, source={"kind": "ticker"},
                   params={"timeframe": "weekly", "stop": "atr3"})
    meta, html = store.get(m["id"])
    assert html == HTML and html.encode() == HTML.encode()
    assert meta["sha256"] == R.hashlib.sha256(HTML.encode()).hexdigest()
    assert meta["params"] == {"timeframe": "weekly", "stop": "atr3"} and meta["source"] == {"kind": "ticker"}
    assert meta["size_bytes"] == len(HTML.encode())
    with gzip.open(os.path.join(store.files, m["id"] + ".html.gz"), "rb") as f:   # stored compressed
        assert f.read() == HTML.encode()


def test_file_named_by_id_never_by_ticker(store):
    m = store.save(HTML, "financial", "AAPL", created_at=T0)
    assert os.listdir(store.files) == [m["id"] + ".html.gz"]


def test_failed_index_insert_leaves_nothing(store, monkeypatch):
    real = store._connect
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(store, "_connect", flaky)
    with pytest.raises(sqlite3.OperationalError):
        store.save(HTML, "financial", "AAPL", created_at=T0)
    monkeypatch.setattr(store, "_connect", real)
    assert store.list()[1] == 0 and os.listdir(store.files) == []          # no row, no file


def test_failed_file_write_leaves_nothing(store, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(R.os, "replace", boom)
    with pytest.raises(OSError):
        store.save(HTML, "financial", "AAPL", created_at=T0)
    monkeypatch.undo()
    assert store.list()[1] == 0 and os.listdir(store.files) == []          # no tmp file either


def test_orphan_files_cleaned_at_startup(tmp_path):
    root = str(tmp_path / "lib")
    s = R.LocalReportStore(root, LA)
    keep = s.save(HTML, "financial", "AAPL", created_at=T0)["id"]
    orphan = os.path.join(s.files, "f" * 32 + ".html.gz")
    tmp = os.path.join(s.files, "e" * 32 + ".html.gz.abc.tmp")
    for p in (orphan, tmp):
        open(p, "wb").close()
        os.utime(p, (0, 0))                                                 # old
    fresh = os.path.join(s.files, "d" * 32 + ".html.gz")
    open(fresh, "wb").close()                                               # recent → may be mid-save
    R.LocalReportStore(root, LA)
    assert sorted(os.listdir(s.files)) == sorted([keep + ".html.gz", os.path.basename(fresh)])


def test_list_filters_search_paging_and_order(store):
    days = [datetime(2026, 10, d, 17, 0, tzinfo=timezone.utc) for d in (10, 12, 14)]
    for d in days:
        store.save(HTML, "financial", "AAPL", created_at=d)
        store.save(HTML, "backtest", "AMD", created_at=d)
    store.save(HTML, "estimates", "AAP", created_at=days[0])
    rows, total = store.list()
    assert total == 7 and [r["created_at_utc"] for r in rows] == sorted((r["created_at_utc"] for r in rows), reverse=True)
    assert store.list(rtype="backtest")[1] == 3
    assert store.list(ticker="aap")[1] == 4                                 # prefix: AAPL ×3 + AAP
    assert store.list(ticker="AAPL")[1] == 3
    assert {r["name"] for r in store.list(q="2026-10-12")[0]} == {"AAPL_financial_analysis_2026-10-12",
                                                                  "AMD_backtest_report_2026-10-12"}
    assert store.list(q="%")[1] == 0 and store.list(q="_report_")[1] == 4  # LIKE wildcards are literal
    p1, _ = store.list(limit=3, offset=0)
    p2, _ = store.list(limit=3, offset=3)
    p3, _ = store.list(limit=3, offset=6)
    ids = [r["id"] for r in p1 + p2 + p3]
    assert len(ids) == 7 == len(set(ids))
    with pytest.raises(R.ReportError):
        store.list(rtype="nope")


def test_reports_max(tmp_path):
    s = R.LocalReportStore(str(tmp_path / "lib"), LA, max_reports=2)
    s.save(HTML, "financial", "A", created_at=T0)
    s.save(HTML, "financial", "B", created_at=T0)
    with pytest.raises(R.ReportError) as e:
        s.save(HTML, "financial", "C", created_at=T0)
    assert e.value.status == 507 and "full" in str(e.value)
    assert s.count() == 2 and len(os.listdir(s.files)) == 2                 # rejected file removed


def test_too_large(store, monkeypatch):
    monkeypatch.setattr(R, "MAX_REPORT_BYTES", 10)
    with pytest.raises(R.ReportError) as e:
        store.save(HTML, "financial", "A", created_at=T0)
    assert e.value.status == 413


def test_unwritable_dir_disables_saving(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    s = R.LocalReportStore(str(blocker / "lib"), LA)
    assert not s.available and s.reason
    with pytest.raises(R.ReportError) as e:
        s.save(HTML, "financial", "A")
    assert e.value.status == 503


def test_damaged_or_missing_file(store):
    m = store.save(HTML, "financial", "AAPL", created_at=T0)
    p = os.path.join(store.files, m["id"] + ".html.gz")
    with gzip.open(p, "wb") as f:
        f.write(b"tampered")
    with pytest.raises(R.ReportError) as e:
        store.get(m["id"])
    assert e.value.status == 500
    os.remove(p)
    with pytest.raises(R.ReportError) as e:
        store.get(m["id"])
    assert e.value.status == 404


def test_delete(store):
    m = store.save(HTML, "financial", "AAPL", created_at=T0)
    store.delete(m["id"])
    assert store.count() == 0 and os.listdir(store.files) == []
    with pytest.raises(R.ReportError):
        store.delete(m["id"])


@pytest.mark.parametrize("bad", ["../../etc/passwd", "x" * 32, "ABCDEF" * 6, ""])
def test_ids_validated(store, bad):
    with pytest.raises(R.ReportError) as e:
        store.get(bad)
    assert e.value.status == 404


# ── API ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def api(tmp_path, monkeypatch):
    import app as A
    monkeypatch.setattr(A, "REPORTS", R.LocalReportStore(str(tmp_path / "lib"), LA))
    monkeypatch.setattr(A, "PENDING_REPORTS", R.PendingCache())
    monkeypatch.setattr(A, "REPORTS_TZ", LA)
    monkeypatch.delenv("REPORTS_ADMIN_TOKEN", raising=False)
    # no network: stub every report generator
    monkeypatch.setattr(A.web_reports, "render_estimates", lambda t: f"<p>estimates {t}</p>")
    monkeypatch.setattr(A.web_reports, "render_backtest", lambda t, tf, stop: f"<p>backtest {t} {tf} {stop}</p>")
    monkeypatch.setattr(A, "tv_download", lambda t: (b"csv", f"NASDAQ_{t}, 1W.csv"))
    monkeypatch.setattr(A, "_live_price", lambda t: 100.0)
    monkeypatch.setattr(A, "run_analysis", lambda data, name, current_price=None:
                        (f"<p>analysis {name}</p>".encode(), "AMD", "AMD_report.html", {"ticker": "AMD"}))
    return A, A.app.test_client()


def _gen(client, kind="estimates", ticker="AMD"):
    if kind == "estimates":
        return client.post("/api/analyze/estimates", json={"ticker": ticker}).json
    if kind == "backtest":
        return client.post("/api/backtest", json={"ticker": ticker, "timeframe": "daily", "stop": "atr3"}).json
    if kind == "ticker":
        return client.post("/api/analyze/ticker", json={"ticker": ticker}).json
    return client.post("/api/analyze/csv", data={"file": (io.BytesIO(b"x"), "NASDAQ_AMD, 1W.csv")},
                       content_type="multipart/form-data").json


@pytest.mark.parametrize("kind,old_fields,rtype", [
    ("estimates", {"ticker", "html", "filename"}, "estimates"),
    ("backtest", {"ticker", "html", "timeframe", "stop", "filename"}, "backtest"),
    ("ticker", {"ticker", "html", "filename", "summary"}, "financial"),
    ("csv", {"ticker", "html", "filename", "summary"}, "financial"),
])
def test_routes_return_token_and_keep_old_fields(api, kind, old_fields, rtype):
    A, c = api
    d = _gen(c, kind)
    assert old_fields <= set(d) and d["report_token"]
    s = d["report_save"]
    assert s["available"] and s["type"] == rtype and s["suggested_ticker"] == "AMD"
    assert s["type_part"] == R.REPORT_TYPES[rtype][0] and len(s["run_date"]) == 10
    assert A.PENDING_REPORTS.get(d["report_token"]).html == d["html"]


def test_save_flow_end_to_end(api):
    A, c = api
    d = _gen(c, "backtest")
    r = c.post("/api/reports", json={"report_token": d["report_token"], "ticker": "amd"})
    assert r.status_code == 201
    m = r.json
    assert m["name"].startswith("AMD_backtest_report_") and m["params"]["stop"] == "atr3"
    assert m["params"]["timeframe"] == "daily" and m["params"]["stop_label"] == "3× ATR"
    g = c.get(f"/api/reports/{m['id']}").json
    assert g["html"] == d["html"] and g["name"] == m["name"]
    lst = c.get("/api/reports?type=backtest").json
    assert lst["total"] == 1 and "html" not in lst["reports"][0]
    dl = c.get(f"/api/reports/{m['id']}/download")
    assert dl.status_code == 200 and dl.data.decode() == d["html"]
    assert dl.headers["Content-Disposition"] == f'attachment; filename="{m["name"]}.html"'
    assert dl.headers["X-Content-Type-Options"] == "nosniff"


def test_csv_source_recorded(api):
    A, c = api
    d = _gen(c, "csv")
    m = c.post("/api/reports", json={"report_token": d["report_token"], "ticker": "AMD"}).json
    assert m["source"] == {"kind": "csv", "filename": "NASDAQ_AMD, 1W.csv", "detected_ticker": "AMD"}


def test_second_report_same_day_gets_suffix(api):
    A, c = api
    n1 = c.post("/api/reports", json={"report_token": _gen(c)["report_token"], "ticker": "AMD"}).json["name"]
    n2 = c.post("/api/reports", json={"report_token": _gen(c)["report_token"], "ticker": "AMD"}).json["name"]
    assert n2 == n1 + "_2"


def test_same_token_saves_once(api):
    A, c = api
    tok = _gen(c)["report_token"]
    a = c.post("/api/reports", json={"report_token": tok, "ticker": "AMD"})
    b = c.post("/api/reports", json={"report_token": tok, "ticker": "AMD"})
    assert a.status_code == 201 and b.status_code == 200 and b.json["already_saved"]
    assert a.json["id"] == b.json["id"] and c.get("/api/reports").json["total"] == 1


def test_user_can_correct_the_ticker(api):
    A, c = api
    m = c.post("/api/reports", json={"report_token": _gen(c, "csv")["report_token"], "ticker": "nvda"}).json
    assert m["ticker"] == "NVDA" and m["name"].startswith("NVDA_financial_analysis_")


def test_save_errors(api, monkeypatch):
    A, c = api
    tok = _gen(c)["report_token"]
    assert c.post("/api/reports", json={"report_token": "nope", "ticker": "AMD"}).status_code == 410
    assert c.post("/api/reports", json={"ticker": "AMD"}).status_code == 400
    assert c.post("/api/reports", json={"report_token": tok, "ticker": "A B"}).status_code == 400
    assert c.post("/api/reports", data="junk", content_type="text/plain").status_code == 400
    # the browser never supplies HTML
    r = c.post("/api/reports", json={"report_token": tok, "ticker": "AMD", "html": "<script>x</script>"})
    assert r.status_code == 400 and c.get("/api/reports").json["total"] == 0
    # expired
    clk = Clock()
    monkeypatch.setattr(A, "PENDING_REPORTS", R.PendingCache(ttl_s=10, clock=clk))
    tok2 = _gen(c)["report_token"]
    clk.t = 11
    r = c.post("/api/reports", json={"report_token": tok2, "ticker": "AMD"})
    assert r.status_code == 410 and "expired" in r.json["error"]


def test_list_validation_and_404s(api):
    A, c = api
    assert c.get("/api/reports?limit=x").status_code == 400
    assert c.get("/api/reports?type=bogus").status_code == 400
    assert c.get("/api/reports/" + "0" * 32).status_code == 404
    assert c.get("/api/reports/not-an-id").status_code == 404
    assert c.get("/api/reports/not-an-id/download").status_code == 404
    assert c.get("/api/reports?type=all").status_code == 200


def test_delete_requires_admin_token(api, monkeypatch):
    A, c = api
    m = c.post("/api/reports", json={"report_token": _gen(c)["report_token"], "ticker": "AMD"}).json
    url = f"/api/reports/{m['id']}"
    assert c.delete(url).status_code == 403                                  # disabled: env unset
    assert c.post("/api/reports/admin").status_code == 403
    assert c.get("/api/reports/status").json["delete_enabled"] is False
    monkeypatch.setenv("REPORTS_ADMIN_TOKEN", "s3cret")
    assert c.get("/api/reports/status").json["delete_enabled"] is True
    assert c.delete(url).status_code == 403
    assert c.delete(url, headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert c.post("/api/reports/admin", headers={"X-Admin-Token": "s3cret"}).json == {"ok": True}
    assert c.delete(url, headers={"X-Admin-Token": "s3cret"}).status_code == 200
    assert c.get(url).status_code == 404
    assert c.delete(url, headers={"X-Admin-Token": "s3cret"}).status_code == 404


def test_library_unavailable_app_still_works(api, monkeypatch, tmp_path):
    A, c = api
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(A, "REPORTS", R.LocalReportStore(str(blocker / "lib"), LA))
    st = c.get("/api/reports/status").json
    assert st["available"] is False and st["message"] == "Report library is unavailable on this server."
    d = _gen(c)
    assert d["html"] and "report_token" not in d and d["report_save"] == {"available": False}
    assert c.post("/api/reports", json={"report_token": "x", "ticker": "AMD"}).status_code == 503
    assert c.get("/api/reports").status_code == 503
    assert c.get("/").status_code == 200


def test_status_reports_count_and_tz(api):
    A, c = api
    c.post("/api/reports", json={"report_token": _gen(c)["report_token"], "ticker": "AMD"})
    st = c.get("/api/reports/status").json
    assert st["available"] and st["count"] == 1 and st["timezone"] == "America/Los_Angeles"
