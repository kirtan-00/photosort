import os
import time
from pathlib import Path
from fastapi.testclient import TestClient
from photosort.index import index_folder
from photosort.server import create_app


def test_api(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    assert c.get("/").status_code == 200 and "photosort" in c.get("/").text.lower()
    st = c.get("/api/stats").json()
    assert st["photos"] == 1
    res = c.get("/api/search").json()["results"]
    assert res[0]["rel"] == "a.jpg"
    assert c.get(f"/api/thumb/{res[0]['qhash']}?size=grid").headers["content-type"] == "image/jpeg"
    assert c.post("/api/export", json={"ids": [res[0]["id"]], "name": "t"}).json()["started"]
    for _ in range(100):
        p = c.get("/api/export/progress").json()
        if not p["running"]: break
        time.sleep(0.05)
    assert p["error"] is None and p["done"] == 1
    ex = Path(p["path"])
    assert ex.is_dir() and (ex / "a.jpg").is_file() and not str(ex).startswith(str(tmp_path))
    assert c.get("/api/people").json() == []
    assert sorted(os.listdir(tmp_path)) == ["a.jpg"]


def test_export_refuses_when_disk_is_short(tmp_path, monkeypatch):
    from conftest import make_image
    import photosort.server as srv
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    pid = c.get("/api/search").json()["results"][0]["id"]
    class Usage: free = 10
    monkeypatch.setattr(srv.shutil, "disk_usage", lambda p: Usage)
    r = c.post("/api/export", json={"ids": [pid], "name": "t", "mode": "copy"})
    assert r.status_code == 400 and "free" in r.json()["detail"]
    assert c.post("/api/export", json={"ids": [pid], "name": "t", "mode": "symlink"}).json()["started"]
    for _ in range(100):
        p2 = c.get("/api/export/progress").json()
        if not p2["running"]: break
        time.sleep(0.05)
    assert p2["error"] is None
    assert sorted(os.listdir(tmp_path)) == ["a.jpg"]


def test_folder_switch_refused_while_export_running(tmp_path, monkeypatch):
    from conftest import make_image
    import photosort.server as srv
    from photosort.export import export_ids as real_export_ids
    for i in range(5):
        make_image(tmp_path, f"p{i}.jpg", seed=i)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    ids = c.get("/api/search/ids").json()["ids"]

    def slow_export_ids(root, ids_, name, mode="copy", progress=None, base=None):
        def slow_progress(d):
            time.sleep(0.1)
            if progress: progress(d)
        return real_export_ids(root, ids_, name, mode, progress=slow_progress, base=base)

    monkeypatch.setattr(srv, "export_ids", slow_export_ids)
    assert c.post("/api/export", json={"ids": ids, "name": "t", "mode": "symlink"}).json()["started"]
    assert c.post("/api/folder", json={"path": str(tmp_path)}).status_code == 409
    for _ in range(200):
        p = c.get("/api/export/progress").json()
        if not p["running"]: break
        time.sleep(0.02)
    assert p["error"] is None
    assert c.post("/api/folder", json={"path": str(tmp_path)}).status_code == 200


def test_search_by_missing_image_id_is_404(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    r = c.get("/api/search", params={"image_id": 999})
    assert r.status_code == 404


def test_search_total_and_ids_endpoint(tmp_path):
    from conftest import make_image
    for i in range(5): make_image(tmp_path, f"p{i}.jpg", seed=i)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    page = c.get("/api/search", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 5 and page["offset"] == 2 and [r["rel"] for r in page["results"]] == ["p2.jpg", "p3.jpg"]
    ids = c.get("/api/search/ids").json()
    assert ids["total"] == 5 and len(ids["ids"]) == 5 and all(isinstance(i, int) for i in ids["ids"])


def test_ui_static_app_js_served(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    r = c.get("/ui/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_ui_unknown_static_file_404s(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    assert c.get("/ui/does-not-exist.js").status_code == 404


def test_search_after_reindex_and_person_filter_dont_500(tmp_path):
    """Index() opens its sqlite connection on the thread that builds the app; FastAPI
    runs sync endpoints in a worker thread. refresh() and person-filtered search must
    not reuse that connection cross-thread or sqlite raises ProgrammingError -> 500."""
    from conftest import make_image
    import time
    make_image(tmp_path, "a.jpg", seed=1)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    make_image(tmp_path, "b.jpg", seed=2)
    r = c.post("/api/index", json={"faces": False})
    assert r.status_code == 200
    for _ in range(200):
        p = c.get("/api/progress").json()
        if not p["running"]:
            break
        time.sleep(0.05)
    assert p["running"] is False
    r2 = c.get("/api/search")
    assert r2.status_code == 200
    assert len(r2.json()["results"]) == 2
    r3 = c.get("/api/search", params={"person": 1})
    assert r3.status_code == 200


def test_index_and_progress_cycle(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    r = c.post("/api/index", json={"faces": False})
    assert r.status_code == 200 and r.json()["started"] is True
    r2 = c.post("/api/index", json={"faces": False})
    assert r2.status_code == 409
    import time
    for _ in range(200):
        p = c.get("/api/progress").json()
        if not p["running"]:
            break
        time.sleep(0.05)
    assert p["running"] is False


def test_export_bad_name_is_400_and_writes_nothing(tmp_path):
    import os
    from conftest import make_image
    from photosort.config import export_root
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    pid = c.get("/api/search").json()["results"][0]["id"]
    for bad in ["../../x", "/tmp/x", ".."]:
        r = c.post("/api/export", json={"ids": [pid], "name": bad})
        assert r.status_code == 400, bad
    assert os.listdir(export_root()) == [] and sorted(os.listdir(tmp_path)) == ["a.jpg"]
    assert c.post("/api/export", json={"ids": [pid], "name": "fine"}).json()["started"]
    for _ in range(100):
        p = c.get("/api/export/progress").json()
        if not p["running"]: break
        time.sleep(0.05)
    assert p["error"] is None
    assert Path(p["path"]).resolve().is_relative_to(export_root().resolve())


def test_index_failure_is_reported_as_error_stage(tmp_path, monkeypatch):
    import time
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    def boom(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr("photosort.index.index_folder", boom)
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/index", json={"faces": False}).status_code == 200
    for _ in range(200):
        p = c.get("/api/progress").json()
        if not p["running"]:
            break
        time.sleep(0.05)
    assert p["running"] is False and p["stage"] == "error" and "disk on fire" in p["error"]
    assert c.post("/api/index", json={"faces": False}).status_code == 200   # not wedged


def test_free_port_skips_busy_port():
    import socket
    from photosort.cli import free_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0)); busy = s.getsockname()[1]; s.listen(1)
        got = free_port(busy)
        assert got != busy and busy < got < busy + 10


# ---------- folder switching ----------

def test_no_folder_open_by_default_and_endpoints_degrade():
    c = TestClient(create_app(None))
    f = c.get("/api/folder").json()
    assert f == {"root": None, "name": None, "indexed": False}
    assert c.get("/api/stats").json()["photos"] == 0
    assert c.get("/api/search").json() == {"results": [], "total": 0, "offset": 0, "limit": 200}
    assert c.get("/api/people").json() == []
    assert c.post("/api/index", json={"faces": False}).status_code == 400
    assert c.post("/api/export", json={"ids": [], "name": "t"}).status_code == 400


def test_post_folder_switches_root(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(None))
    r = c.post("/api/folder", json={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["root"] == str(tmp_path) and body["indexed"] is True
    st = c.get("/api/stats").json()
    assert st["photos"] == 1
    res = c.get("/api/search").json()["results"]
    assert res[0]["rel"] == "a.jpg"


def test_post_folder_non_directory_is_400(tmp_path):
    c = TestClient(create_app(None))
    r = c.post("/api/folder", json={"path": str(tmp_path / "does-not-exist")})
    assert r.status_code == 400


def test_post_folder_409_while_indexing(tmp_path):
    c = TestClient(create_app(tmp_path))
    c.app.state.photosort["running"] = True
    try:
        r = c.post("/api/folder", json={"path": str(tmp_path)})
        assert r.status_code == 409
    finally:
        c.app.state.photosort["running"] = False


def test_folder_recent_lists_switched_path(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(None))
    c.post("/api/folder", json={"path": str(tmp_path)})
    recent = c.get("/api/folder/recent").json()["recent"]
    assert any(r["path"] == str(tmp_path) for r in recent)


def test_folder_choose_returns_204_when_picker_gives_nothing(tmp_path, monkeypatch):
    import subprocess as sp
    def fake_run(*a, **k):
        return sp.CompletedProcess(a, returncode=1, stdout="", stderr="")
    monkeypatch.setattr("photosort.server.subprocess.run", fake_run)
    c = TestClient(create_app(None))
    r = c.post("/api/folder/choose")
    assert r.status_code == 204


# ---------- categories ----------

def test_categories_endpoint_returns_fixed_and_discovered(tmp_path):
    from conftest import make_image
    from photosort import db as db_mod
    make_image(tmp_path, "a.jpg"); make_image(tmp_path, "b.jpg", seed=2)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    assert c.get("/api/categories").json() == {"fixed": {"unclassified": 2}, "discovered": {}}
    conn = db_mod.connect(tmp_path)
    conn.execute("UPDATE photos SET category='beach', cluster='excavator', cluster_score=0.9 WHERE rel='a.jpg'")
    conn.execute("UPDATE photos SET cluster='excavator', cluster_score=0.2 WHERE rel='b.jpg'")
    conn.commit()
    assert c.get("/api/categories").json() == {"fixed": {"beach": 1, "unclassified": 1}, "discovered": {"excavator": 2}}
    assert TestClient(create_app(None)).get("/api/categories").json() == {"fixed": {}, "discovered": {}}


def test_search_by_cluster(tmp_path):
    from conftest import make_image
    from photosort import db as db_mod
    make_image(tmp_path, "a.jpg", seed=1); make_image(tmp_path, "b.jpg", seed=2)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    conn = db_mod.connect(tmp_path)
    conn.execute("UPDATE photos SET cluster='havan fire', cluster_score=0.8 WHERE rel='b.jpg'"); conn.commit()
    c = TestClient(create_app(tmp_path))
    r = c.get("/api/search", params={"cluster": "havan fire"})
    assert r.status_code == 200 and [x["rel"] for x in r.json()["results"]] == ["b.jpg"] and r.json()["total"] == 1
    assert c.get("/api/search/ids", params={"cluster": "havan fire"}).json()["total"] == 1
    assert c.get("/api/search", params={"cluster": "crane"}).json()["results"] == []


def _wait_idle(c, n=100):
    import time
    for _ in range(n):
        if not c.get("/api/progress").json()["running"]: return
        time.sleep(0.1)

def test_errors_listed_and_retryable(tmp_path):
    """A transient read failure on an unchanged file: a plain re-index leaves it alone (same size+mtime),
    retry_errors re-processes it."""
    from conftest import make_image
    from photosort import db
    make_image(tmp_path, "a.jpg"); make_image(tmp_path, "b.jpg", seed=2)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    conn = db.connect(tmp_path); conn.execute("UPDATE photos SET status='error' WHERE rel='a.jpg'"); conn.commit()
    c = TestClient(create_app(tmp_path))
    assert c.get("/api/stats").json()["errors"] == 1
    listed = c.get("/api/errors").json()["errors"]
    assert [e["rel"] for e in listed] == ["a.jpg"] and listed[0]["indexed_at"]
    assert c.post("/api/index", json={"faces": False}).json()["started"]; _wait_idle(c)
    assert c.get("/api/stats").json()["errors"] == 1          # unchanged file, not retried by default
    assert c.post("/api/index", json={"faces": False, "retry_errors": True}).json()["started"]; _wait_idle(c)
    assert c.get("/api/stats").json()["errors"] == 0


def test_search_by_category(tmp_path):
    """category is another agent's concurrent work (db column + Filters field + the
    actual filtering in search.Index). We degrade to an empty list if Filters doesn't
    support category yet; otherwise we check /api/search only returns matching rows.
    Both photos are written BEFORE the app/Index is built so a fresh Index sees them
    (no stale cache)."""
    from conftest import make_image
    from photosort import db as db_mod
    make_image(tmp_path, "a.jpg", seed=1)
    make_image(tmp_path, "b.jpg", seed=2)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    conn = db_mod.connect(tmp_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(photos)")]
    if "category" in cols:
        a_id = conn.execute("SELECT id FROM photos WHERE rel='a.jpg'").fetchone()[0]
        b_id = conn.execute("SELECT id FROM photos WHERE rel='b.jpg'").fetchone()[0]
        conn.execute("UPDATE photos SET category='beach' WHERE id=?", (a_id,))
        conn.execute("UPDATE photos SET category='ocean' WHERE id=?", (b_id,))
        conn.commit()
    c = TestClient(create_app(tmp_path))
    r = c.get("/api/search", params={"category": "beach"})
    assert r.status_code == 200
    if "category" in cols:
        res = r.json()["results"]
        assert [row["rel"] for row in res] == ["a.jpg"]
        assert res[0]["category"] == "beach"
    else:
        assert r.json()["results"] == []


def test_search_by_category_flags_less_sure_and_sure_only_drops_them(tmp_path):
    from conftest import make_image
    from photosort import db as db_mod
    for i, n in enumerate("abcd"): make_image(tmp_path, f"{n}.jpg", seed=i)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    conn = db_mod.connect(tmp_path)
    conn.execute("UPDATE photos SET category='building', category_score=0.8, category_guess='building', category_guess_score=0.8 WHERE rel='a.jpg'")
    conn.execute("UPDATE photos SET category='building', category_score=0.4, category_guess='building', category_guess_score=0.4 WHERE rel='b.jpg'")
    conn.execute("UPDATE photos SET category='other', category_score=0.45, category_guess='building', category_guess_score=0.45 WHERE rel='c.jpg'")
    conn.execute("UPDATE photos SET category='other', category_score=0.9, category_guess='road', category_guess_score=0.9 WHERE rel='d.jpg'")
    conn.commit()
    c = TestClient(create_app(tmp_path))
    body = c.get("/api/search", params={"category": "building"}).json()
    assert body["total"] == 3
    assert [(r["rel"], r["sure"], r["confidence"]) for r in body["results"]] == [("a.jpg", True, 0.8), ("c.jpg", False, 0.45), ("b.jpg", False, 0.4)]
    assert c.get("/api/search/ids", params={"category": "building"}).json()["ids"] == [r["id"] for r in body["results"]]
    sure = c.get("/api/search", params={"category": "building", "sure_only": 1}).json()
    assert sure["total"] == 1 and [r["rel"] for r in sure["results"]] == ["a.jpg"]
    assert c.get("/api/search/ids", params={"category": "building", "sure_only": 1}).json()["total"] == 1


def test_export_categories_sure_only_by_default(tmp_path, tmp_path_factory):
    """Export ticked categories leaves the less-sure band out unless include_unsure is set; then a photo
    filed under "other" whose guess was beach lands in beach/ (and still in other/ when other is ticked)."""
    from test_export import _two_category_shoot
    from photosort import db as db_mod
    before = _two_category_shoot(tmp_path)
    conn = db_mod.connect(tmp_path)
    conn.execute("UPDATE photos SET category='other', category_score=0.3, category_guess='beach', category_guess_score=0.3 WHERE rel='c.jpg'")
    conn.execute("UPDATE photos SET category_score=0.2 WHERE rel='b.jpg'")       # ocean, but barely
    conn.commit()
    c = TestClient(create_app(tmp_path))
    disk = tmp_path_factory.mktemp("disk")
    assert c.post("/api/export/destination", json={"path": str(disk)}).status_code == 200
    r = c.post("/api/export/categories", json={"categories": ["beach", "ocean", "other"], "mode": "symlink"})
    assert r.json()["total"] == 2 and _wait_export(c)["error"] is None
    out = disk.resolve() / tmp_path.resolve().name / "categories"
    assert sorted(x.name for x in (out / "beach").iterdir()) == ["a.jpg"]
    assert sorted(x.name for x in (out / "other").iterdir()) == ["c.jpg"]
    assert not (out / "ocean").exists()
    r = c.post("/api/export/categories", json={"categories": ["beach", "ocean", "other"], "mode": "symlink", "include_unsure": True})
    assert r.json()["total"] == 4 and _wait_export(c)["error"] is None
    assert sorted(x.name for x in (out / "beach").iterdir()) == ["a.jpg", "c.jpg"]
    assert sorted(x.name for x in (out / "ocean").iterdir()) == ["b.jpg"]
    assert sorted(os.listdir(tmp_path)) == before


# final-review fixes

def test_export_people_refuses_copy_when_disk_is_short(tmp_path, monkeypatch):
    from conftest import make_image
    from photosort import db
    import photosort.server as srv
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    conn = db.connect(tmp_path); conn.execute("UPDATE photos SET n_faces=1"); conn.commit()   # one solo shot to export
    c = TestClient(create_app(tmp_path))
    class Usage: free = 10
    monkeypatch.setattr(srv.shutil, "disk_usage", lambda p: Usage)
    r = c.post("/api/export/people", json={"mode": "copy"})
    assert r.status_code == 400 and "free" in r.json()["detail"]
    r2 = c.post("/api/export/people", json={"mode": "symlink"})
    assert r2.status_code == 200
    assert (Path(r2.json()["path"]) / "solo" / "a.jpg").is_symlink()
    assert sorted(os.listdir(tmp_path)) == ["a.jpg"]


def test_export_start_is_serialised(tmp_path, monkeypatch):
    """Two rapid export POSTs: the check-then-set window spans export_bytes + disk_usage + export_dir,
    so without a lock both pass the 'already running' check."""
    import threading
    from conftest import make_image
    import photosort.server as srv
    from photosort.export import export_bytes as real_export_bytes, export_ids as real_export_ids
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    app = create_app(tmp_path)
    pid = TestClient(app).get("/api/search").json()["results"][0]["id"]

    def slow_export_bytes(root, ids):
        time.sleep(0.3)
        return real_export_bytes(root, ids)

    def slow_export_ids(root, ids_, name, mode="copy", progress=None, base=None):
        time.sleep(0.5)
        return real_export_ids(root, ids_, name, mode, progress=progress, base=base)

    monkeypatch.setattr(srv, "export_bytes", slow_export_bytes)
    monkeypatch.setattr(srv, "export_ids", slow_export_ids)
    codes = []
    def post():
        codes.append(TestClient(app).post("/api/export", json={"ids": [pid], "name": "race", "mode": "copy"}).status_code)
    ts = [threading.Thread(target=post) for _ in range(2)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert sorted(codes) == [200, 409]
    c = TestClient(app)
    for _ in range(200):
        p = c.get("/api/export/progress").json()
        if not p["running"]: break
        time.sleep(0.02)
    assert p["error"] is None and p["done"] == 1
    assert sorted(os.listdir(tmp_path)) == ["a.jpg"]


def test_folder_choose_409_while_export_running(tmp_path, monkeypatch):
    import subprocess as sp
    called = []
    def fake_run(*a, **k):
        called.append(a)
        return sp.CompletedProcess(a, returncode=0, stdout=str(tmp_path) + "\n", stderr="")
    monkeypatch.setattr("photosort.server.subprocess.run", fake_run)
    c = TestClient(create_app(tmp_path))
    c.app.state.photosort["export"]["running"] = True
    try:
        assert c.post("/api/folder/choose").status_code == 409
        assert called == []            # refused before the picker was even opened
    finally:
        c.app.state.photosort["export"]["running"] = False
    assert c.post("/api/folder/choose").status_code == 200


def _fake_index_folder(stats):
    def fake(root, faces=True, progress=None, retry_errors=False, **kw):
        if progress: progress({"stage": "done", "done": stats["total"], "total": stats["total"], "stage_started": time.time()})
        return dict(stats)
    return fake


def test_index_run_categorises_when_something_changed(tmp_path, monkeypatch):
    from conftest import make_image
    import photosort.server as srv
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    calls = []
    def fake_classify(root, people_by_faces=True):
        calls.append(Path(root)); return {"other": 1}
    monkeypatch.setattr(srv.classify_mod, "classify_and_store", fake_classify)
    monkeypatch.setattr("photosort.index.index_folder",
                        _fake_index_folder(dict(total=1, skipped=0, indexed=1, errors=0, embedded=1, seconds=0.1)))
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/index", json={"faces": False}).json()["started"]; _wait_idle(c)
    assert calls == [tmp_path]
    cp = c.get("/api/classify/progress").json()
    assert cp["running"] is False and cp["error"] is None and cp["counts"] == {"other": 1}
    p = c.get("/api/progress").json()
    assert p["stage"] == "done" and p["done"] == 1 and p["total"] == 1 and p["running"] is False
    # nothing changed on the next run: no second classify pass
    monkeypatch.setattr("photosort.index.index_folder",
                        _fake_index_folder(dict(total=1, skipped=1, indexed=0, errors=0, embedded=0, seconds=0.1)))
    assert c.post("/api/index", json={"faces": False}).json()["started"]; _wait_idle(c)
    assert calls == [tmp_path]
    assert c.get("/api/progress").json()["stage"] == "done"


def test_classify_endpoint_leaves_the_index_fresh_after_discovery(tmp_path, monkeypatch):
    """Categorise writes the fixed categories, then the discovered ones. A search that lands between the two
    refreshes the Index and clears the stale flag; the cluster columns written after that must still reach
    the next search, otherwise a tile says "x 1" and clicking it finds nothing."""
    from conftest import make_image
    from photosort import db as db_mod
    import photosort.server as srv
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    monkeypatch.setattr(srv.classify_mod, "classify_and_store", lambda root, people_by_faces=True: {"other": 1})
    def fake_discover(root, k=None):
        c.app.state.photosort["stale"] = False              # a search refreshed the Index in between
        conn = db_mod.connect(root); conn.execute("UPDATE photos SET cluster='x', cluster_score=1.0"); conn.commit()
        return {"x": 1}
    monkeypatch.setattr(srv.classify_mod, "discover_and_store", fake_discover)
    assert c.post("/api/classify").json()["started"]
    for _ in range(100):
        if not c.get("/api/classify/progress").json()["running"]: break
        time.sleep(0.05)
    cp = c.get("/api/classify/progress").json()
    assert cp["counts"] == {"other": 1} and cp["discovered"] == {"x": 1} and cp["error"] is None
    assert c.get("/api/categories").json()["discovered"] == {"x": 1}
    assert c.get("/api/search", params={"cluster": "x"}).json()["total"] == 1


def test_index_run_survives_a_classify_failure(tmp_path, monkeypatch):
    from conftest import make_image
    import photosort.server as srv
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    def boom(root, people_by_faces=True):
        raise RuntimeError("no embeddings")
    monkeypatch.setattr(srv.classify_mod, "classify_and_store", boom)
    monkeypatch.setattr("photosort.index.index_folder",
                        _fake_index_folder(dict(total=1, skipped=0, indexed=1, errors=0, embedded=1, seconds=0.1)))
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/index", json={"faces": False}).json()["started"]; _wait_idle(c)
    p = c.get("/api/progress").json()
    assert p["stage"] == "done" and p["running"] is False
    cp = c.get("/api/classify/progress").json()
    assert cp["running"] is False and "no embeddings" in cp["error"]
    assert c.post("/api/classify").status_code == 200        # the manual button still works afterwards
    for _ in range(100):
        if not c.get("/api/classify/progress").json()["running"]: break
        time.sleep(0.05)


# find a person from a reference photo

def test_people_find_returns_ranked_photos(tmp_path, monkeypatch):
    from test_people import _fake_shoot, _p0_reference
    from photosort import people
    conn = _fake_shoot(tmp_path, n_people=3, per=4)
    before = sorted(os.listdir(tmp_path))
    ref = _p0_reference(conn)   # built here: the endpoint runs on a worker thread, sqlite conns don't cross
    monkeypatch.setattr(people, "_reference_faces", lambda path: [ref])
    c = TestClient(create_app(tmp_path))
    r = c.post("/api/people/find", json={"path": str(tmp_path / "p1_0.jpg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 4 and len(body["results"]) == 4 and body["faces_in_reference"] == 1
    assert body["results"][0]["score"] >= body["results"][-1]["score"]
    assert all("qhash" in x and "rel" in x for x in body["results"])
    assert all(x["rel"].startswith("p0_") for x in body["results"])
    assert all(x["sure"] is True and x["confidence"] == x["score"] for x in body["results"])
    assert c.post("/api/people/find", json={"path": str(tmp_path / "p1_0.jpg"), "min_sim": 0.99}).json()["total"] == 0
    assert c.post("/api/people/find", json={"path": "/nope.jpg"}).status_code == 400
    assert sorted(os.listdir(tmp_path)) == before


def test_people_find_returns_a_less_sure_band_below_the_slider(tmp_path, monkeypatch):
    """Matches from max(min_sim - 0.1, 0.4) up to min_sim come back too, flagged sure: false, after the
    sure ones and sorted by similarity; total counts both bands."""
    from test_people import _fake_shoot, _p0_reference
    from photosort import people
    conn = _fake_shoot(tmp_path, n_people=3, per=4)
    ref = _p0_reference(conn)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [ref])
    c = TestClient(create_app(tmp_path))
    at_default = c.post("/api/people/find", json={"path": str(tmp_path / "p1_0.jpg")}).json()
    sims = sorted((x["score"] for x in at_default["results"]), reverse=True)
    assert len(sims) == 4
    cut = (sims[1] + sims[2]) / 2                    # two above the slider, two in the band below it
    body = c.post("/api/people/find", json={"path": str(tmp_path / "p1_0.jpg"), "min_sim": cut}).json()
    assert body["total"] == 4
    assert [x["sure"] for x in body["results"]] == [True, True, False, False]
    scores = [x["score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True) and all(x["confidence"] == x["score"] for x in body["results"])

def test_people_find_no_face_is_200_empty(tmp_path, monkeypatch):
    from test_people import _fake_shoot
    from photosort import people
    _fake_shoot(tmp_path)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [])
    c = TestClient(create_app(tmp_path))
    body = c.post("/api/people/find", json={"path": str(tmp_path / "p1_0.jpg")}).json()
    assert body == {"faces_in_reference": 0, "person_id": None, "total": 0, "results": []}

def test_people_find_choose_204_on_cancel_and_400_without_folder(tmp_path, monkeypatch):
    import subprocess as sp
    def fake_run(*a, **k):
        return sp.CompletedProcess(a, returncode=1, stdout="", stderr="")
    monkeypatch.setattr("photosort.server.subprocess.run", fake_run)
    assert TestClient(create_app(None)).post("/api/people/find/choose").status_code == 400
    from test_people import _fake_shoot, _p0_reference
    from photosort import people
    conn = _fake_shoot(tmp_path)
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/people/find/choose").status_code == 204
    ref = _p0_reference(conn)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [ref])
    chosen = str(tmp_path / "p2_1.jpg")
    monkeypatch.setattr("photosort.server.subprocess.run",
                        lambda *a, **k: sp.CompletedProcess(a, returncode=0, stdout=chosen + "\n", stderr=""))
    body = c.post("/api/people/find/choose").json()
    assert body["path"] == chosen and body["total"] == 4 and body["results"][0]["rel"].startswith("p0_")

def test_people_find_tiny_face_and_unreadable_reference(tmp_path, monkeypatch):
    from test_people import _fake_shoot, _p0_reference
    from photosort import people
    conn = _fake_shoot(tmp_path)
    c = TestClient(create_app(tmp_path))
    r = c.post("/api/people/find", json={"path": str(tmp_path / "p1_0.jpg")})   # real decode of a byte stub
    assert r.status_code == 400 and "could not read" in r.json()["detail"]
    tiny = _p0_reference(conn, w=20, h=20)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [tiny])
    body = c.post("/api/people/find", json={"path": str(tmp_path / "p1_0.jpg")}).json()
    assert body["reference_face_too_small"] is True and body["total"] == 0 and body["faces_in_reference"] == 1


# export destination (another disk)

def _shoot_client(tmp_path, n=1):
    from conftest import make_image
    for i in range(n):
        make_image(tmp_path, f"p{i}.jpg", seed=i)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    return TestClient(create_app(tmp_path))


def _wait_export(c, n=200):
    for _ in range(n):
        p = c.get("/api/export/progress").json()
        if not p["running"]: return p
        time.sleep(0.02)
    return p


def test_export_destination_default_payload(tmp_path):
    from photosort.config import export_root
    c = _shoot_client(tmp_path)
    d = c.get("/api/export/destination").json()
    assert d["path"] == str(export_root()) and d["default"] is True and d["mounted"] is True
    assert isinstance(d["free_gb"], float) and d["free_gb"] > 0


def test_export_destination_set_get_and_reset(tmp_path, tmp_path_factory):
    from photosort import settings
    from photosort.config import export_root
    c = _shoot_client(tmp_path)
    disk = tmp_path_factory.mktemp("disk")
    r = c.post("/api/export/destination", json={"path": str(disk)})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["path"] == str(disk.resolve()) and d["default"] is False and d["mounted"] is True and d["free_gb"] > 0
    assert c.get("/api/export/destination").json() == d
    assert settings.get_export_base() == disk.resolve()            # survives a restart
    assert c.post("/api/export/destination", json={"path": str(disk / "nope")}).status_code == 400
    pid = c.get("/api/search").json()["results"][0]["id"]
    assert c.post("/api/export", json={"ids": [pid], "name": "t"}).json()["started"]
    p = _wait_export(c)
    assert p["error"] is None and p["done"] == 1
    assert Path(p["path"]) == disk.resolve() / tmp_path.resolve().name / "t" and (Path(p["path"]) / "p0.jpg").is_file()
    assert os.listdir(export_root()) == []
    d2 = c.request("DELETE", "/api/export/destination").json()
    assert d2["path"] == str(export_root()) and d2["default"] is True
    assert settings.get_export_base() is None
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg"]


def test_export_destination_inside_source_is_400(tmp_path):
    c = _shoot_client(tmp_path)
    (tmp_path / "sub").mkdir()
    for bad in [tmp_path, tmp_path / "sub"]:
        r = c.post("/api/export/destination", json={"path": str(bad)})
        assert r.status_code == 400 and "inside the source folder" in r.json()["detail"], bad
    assert c.get("/api/export/destination").json()["default"] is True
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg", "sub"]


def test_export_refuses_when_destination_not_mounted(tmp_path, tmp_path_factory):
    import shutil as sh
    c = _shoot_client(tmp_path)
    disk = tmp_path_factory.mktemp("disk")
    assert c.post("/api/export/destination", json={"path": str(disk)}).status_code == 200
    sh.rmtree(disk)                                                  # the disk got unplugged
    d = c.get("/api/export/destination").json()
    assert d["mounted"] is False and d["free_gb"] == 0.0 and d["default"] is False
    pid = c.get("/api/search").json()["results"][0]["id"]
    for mode in ["copy", "symlink"]:
        r = c.post("/api/export", json={"ids": [pid], "name": "t", "mode": mode})
        assert r.status_code == 400 and "not mounted" in r.json()["detail"], mode
    r = c.post("/api/export/people", json={"mode": "symlink"})
    assert r.status_code == 400 and "not mounted" in r.json()["detail"]
    assert c.request("DELETE", "/api/export/destination").json()["default"] is True
    assert c.post("/api/export", json={"ids": [pid], "name": "t", "mode": "symlink"}).json()["started"]
    assert _wait_export(c)["error"] is None
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg"]


def test_export_preflight_checks_the_destination_disk(tmp_path, tmp_path_factory, monkeypatch):
    import photosort.server as srv
    c = _shoot_client(tmp_path)
    disk = tmp_path_factory.mktemp("disk")
    assert c.post("/api/export/destination", json={"path": str(disk)}).status_code == 200
    asked = []
    class Usage: free = 10
    def fake_usage(p):
        asked.append(Path(p)); return Usage
    monkeypatch.setattr(srv.shutil, "disk_usage", fake_usage)
    pid = c.get("/api/search").json()["results"][0]["id"]
    r = c.post("/api/export", json={"ids": [pid], "name": "t", "mode": "copy"})
    assert r.status_code == 400 and "on that disk" in r.json()["detail"]
    assert asked and asked[-1] == disk.resolve()
    assert c.request("DELETE", "/api/export/destination").json()["default"] is True
    r2 = c.post("/api/export", json={"ids": [pid], "name": "t", "mode": "copy"})
    assert r2.status_code == 400 and "on this Mac" in r2.json()["detail"]
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg"]


def test_export_destination_choose_mirrors_folder_picker(tmp_path, tmp_path_factory, monkeypatch):
    import subprocess as sp
    c = _shoot_client(tmp_path)
    monkeypatch.setattr("photosort.server.subprocess.run",
                        lambda *a, **k: sp.CompletedProcess(a, returncode=1, stdout="", stderr=""))
    assert c.post("/api/export/destination/choose").status_code == 204
    disk = tmp_path_factory.mktemp("disk")
    monkeypatch.setattr("photosort.server.subprocess.run",
                        lambda *a, **k: sp.CompletedProcess(a, returncode=0, stdout=str(disk) + "\n", stderr=""))
    d = c.post("/api/export/destination/choose").json()
    assert d["path"] == str(disk.resolve()) and d["default"] is False and d["mounted"] is True
    monkeypatch.setattr("photosort.server.subprocess.run",
                        lambda *a, **k: sp.CompletedProcess(a, returncode=0, stdout=str(tmp_path) + "\n", stderr=""))
    r = c.post("/api/export/destination/choose")
    assert r.status_code == 400 and "inside the source folder" in r.json()["detail"]
    assert c.get("/api/export/destination").json()["path"] == str(disk.resolve())   # the bad pick changed nothing


# export selected categories, one folder each

def test_export_categories_endpoint_runs_to_completion(tmp_path, tmp_path_factory):
    from test_export import _two_category_shoot
    before = _two_category_shoot(tmp_path)
    c = TestClient(create_app(tmp_path))
    disk = tmp_path_factory.mktemp("disk")
    assert c.post("/api/export/destination", json={"path": str(disk)}).status_code == 200
    r = c.post("/api/export/categories", json={"categories": ["beach", "ocean"], "mode": "copy", "include_raw": True})
    assert r.status_code == 200, r.text
    assert r.json() == {"started": True, "total": 2}
    p = _wait_export(c)
    assert p["error"] is None and p["done"] == 3 and p["total"] == 3 and p["failed"] == 0
    out = disk.resolve() / tmp_path.resolve().name / "categories"
    assert Path(p["path"]) == out
    assert sorted(x.name for x in (out / "beach").iterdir()) == ["a.ARW", "a.jpg"]
    assert sorted(x.name for x in (out / "ocean").iterdir()) == ["b.jpg"]
    assert c.post("/api/export/categories", json={"categories": ["beach"], "mode": "csv"}).status_code == 400
    assert c.post("/api/export/categories", json={"categories": [], "mode": "copy"}).status_code == 400
    r2 = c.post("/api/export/categories", json={"categories": None, "mode": "symlink"})
    assert r2.json()["total"] == 2
    assert _wait_export(c)["error"] is None
    assert sorted(os.listdir(tmp_path)) == before


def test_export_categories_endpoint_exports_discovered_names_too(tmp_path, tmp_path_factory):
    """Discovered names land under categories/discovered/<name>/; fixed and discovered go out in one job,
    and a request with no fixed categories ticked but a discovered one is fine."""
    from test_export import _two_category_shoot
    from photosort import db as db_mod
    before = _two_category_shoot(tmp_path)
    conn = db_mod.connect(tmp_path)
    conn.execute("UPDATE photos SET cluster='excavator', cluster_score=0.9 WHERE rel IN ('b.jpg', 'c.jpg')"); conn.commit()
    c = TestClient(create_app(tmp_path))
    disk = tmp_path_factory.mktemp("disk")
    assert c.post("/api/export/destination", json={"path": str(disk)}).status_code == 200
    r = c.post("/api/export/categories", json={"categories": ["beach"], "discovered": ["excavator"], "mode": "symlink"})
    assert r.status_code == 200, r.text
    assert r.json() == {"started": True, "total": 3}
    p = _wait_export(c)
    assert p["error"] is None and p["done"] == 3 and p["failed"] == 0
    out = disk.resolve() / tmp_path.resolve().name / "categories"
    assert sorted(x.name for x in (out / "beach").iterdir()) == ["a.jpg"]
    assert sorted(x.name for x in (out / "discovered" / "excavator").iterdir()) == ["b.jpg", "c.jpg"]
    assert not (out / "ocean").exists()
    r2 = c.post("/api/export/categories", json={"categories": [], "discovered": ["excavator"], "mode": "symlink"})
    assert r2.status_code == 200 and r2.json()["total"] == 2
    assert _wait_export(c)["error"] is None
    assert c.post("/api/export/categories", json={"categories": [], "discovered": [], "mode": "symlink"}).status_code == 400
    assert c.post("/api/export/categories", json={"categories": [], "discovered": None, "mode": "symlink"}).status_code == 400
    assert sorted(os.listdir(tmp_path)) == before


def test_export_categories_endpoint_preflight_and_lock(tmp_path, tmp_path_factory, monkeypatch):
    import photosort.server as srv
    from test_export import _two_category_shoot
    before = _two_category_shoot(tmp_path)
    c = TestClient(create_app(tmp_path))
    disk = tmp_path_factory.mktemp("disk")
    assert c.post("/api/export/destination", json={"path": str(disk)}).status_code == 200
    class Usage: free = 10
    monkeypatch.setattr(srv.shutil, "disk_usage", lambda p: Usage)
    r = c.post("/api/export/categories", json={"categories": None, "mode": "copy"})
    assert r.status_code == 400 and "on that disk" in r.json()["detail"]
    c.app.state.photosort["export"]["running"] = True
    try:
        assert c.post("/api/export/categories", json={"categories": None, "mode": "symlink"}).status_code == 409
    finally:
        c.app.state.photosort["export"]["running"] = False
    assert c.post("/api/export/categories", json={"categories": None, "mode": "symlink"}).status_code == 200
    assert _wait_export(c)["error"] is None
    assert sorted(os.listdir(tmp_path)) == before


# named people: save a reference, list, show, rename, delete, export per person

def _named_shoot(tmp_path, monkeypatch):
    """Three fake people, references saved for two of them (Arya twice, Priest once)."""
    from test_people import _fake_shoot, _two_named_people
    conn = _fake_shoot(tmp_path, n_people=3, per=4)
    before = sorted(os.listdir(tmp_path))
    _two_named_people(tmp_path, monkeypatch, conn)
    return conn, before


def test_people_references_save_list_show_rename_delete(tmp_path, monkeypatch):
    from test_people import _fake_shoot, _person_reference
    from photosort import people
    conn = _fake_shoot(tmp_path, n_people=3, per=4)
    before = sorted(os.listdir(tmp_path))
    c = TestClient(create_app(tmp_path))
    assert c.get("/api/people/references").json() == {"people": []}
    ref = _person_reference(conn, 0)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [ref])
    r = c.post("/api/people/references", json={"name": "Arya", "path": str(tmp_path / "p0_0.jpg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["id"], int) and body["name"] == "Arya" and body["faces_in_reference"] == 1
    assert c.post("/api/people/references", json={"name": "  ", "path": str(tmp_path / "p0_0.jpg")}).status_code == 400
    assert c.post("/api/people/references", json={"name": "X", "path": "/nope.jpg"}).status_code == 400
    monkeypatch.setattr(people, "_reference_faces", lambda path: [])
    r = c.post("/api/people/references", json={"name": "X", "path": str(tmp_path / "p0_0.jpg")})
    assert r.status_code == 400 and "no face" in r.json()["detail"]
    tiny = _person_reference(conn, 0, w=20, h=20)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [tiny])
    r = c.post("/api/people/references", json={"name": "X", "path": str(tmp_path / "p0_0.jpg")})
    assert r.status_code == 400 and "too small" in r.json()["detail"]
    monkeypatch.setattr(people, "_reference_faces", lambda path: [ref])
    for bad in (".", ".."):
        r = c.post("/api/people/references", json={"name": bad, "path": str(tmp_path / "p0_0.jpg")})
        assert r.status_code == 400 and "cannot be used as a folder" in r.json()["detail"], bad
    monkeypatch.delattr(people, "_reference_faces")   # restore the real one: a byte stub is unreadable
    r = c.post("/api/people/references", json={"name": "X", "path": str(tmp_path / "p0_0.jpg")})
    assert r.status_code == 400 and "could not read" in r.json()["detail"]

    lst = c.get("/api/people/references").json()["people"]
    assert lst == [{"name": "Arya", "reference_ids": [body["id"]], "sources": [str(tmp_path / "p0_0.jpg")], "count": 4}]
    assert c.get("/api/people/references", params={"min_sim": 0.99}).json()["people"][0]["count"] == 0

    r = c.post("/api/people/references/Arya/find", json={})
    assert r.status_code == 200, r.text
    found = r.json()
    assert found["total"] == 4 and len(found["results"]) == 4
    assert all(x["rel"].startswith("p0_") and "qhash" in x for x in found["results"])
    scores = [x["score"] for x in found["results"]]
    assert scores == sorted(scores, reverse=True)
    assert c.post("/api/people/references/Arya/find", json={"min_sim": 0.99}).json()["total"] == 0
    assert c.post("/api/people/references/Nobody/find", json={}).status_code == 404

    r = c.post("/api/people/references/Arya/rename", json={"name": "Arya Mehta"})
    assert r.status_code == 200, r.text
    assert [p["name"] for p in c.get("/api/people/references").json()["people"]] == ["Arya Mehta"]
    assert c.post("/api/people/references/Arya/rename", json={"name": "Z"}).status_code == 404
    assert c.post("/api/people/references/Arya%20Mehta/rename", json={"name": " "}).status_code == 400
    for bad in (".", ".."):
        r = c.post("/api/people/references/Arya%20Mehta/rename", json={"name": bad})
        assert r.status_code == 400 and "cannot be used as a folder" in r.json()["detail"], bad
    assert [p["name"] for p in c.get("/api/people/references").json()["people"]] == ["Arya Mehta"]   # rename refused, unchanged
    assert c.post("/api/people/references/Arya%20Mehta/find", json={}).json()["total"] == 4

    assert c.delete("/api/people/references/" + str(body["id"])).status_code == 200
    assert c.delete("/api/people/references/" + str(body["id"])).status_code == 404
    assert c.get("/api/people/references").json() == {"people": []}
    assert sorted(os.listdir(tmp_path)) == before


def test_people_references_list_sorted_by_count_and_no_folder(tmp_path, monkeypatch):
    assert TestClient(create_app(None)).get("/api/people/references").json() == {"people": []}
    assert TestClient(create_app(None)).post("/api/people/references", json={"name": "A", "path": "/x.jpg"}).status_code == 400
    conn, before = _named_shoot(tmp_path, monkeypatch)
    conn.execute("DELETE FROM faces WHERE photo_id IN (SELECT id FROM photos WHERE rel='p0_3.jpg')"); conn.commit()
    c = TestClient(create_app(tmp_path))
    lst = c.get("/api/people/references").json()["people"]
    assert [(p["name"], p["count"], len(p["reference_ids"])) for p in lst] == [("Priest", 4, 1), ("Arya", 3, 2)]
    assert sorted(os.listdir(tmp_path)) == before


def test_export_references_endpoint_runs_to_completion(tmp_path, tmp_path_factory, monkeypatch):
    conn, before = _named_shoot(tmp_path, monkeypatch)
    c = TestClient(create_app(tmp_path))
    disk = tmp_path_factory.mktemp("disk")
    assert c.post("/api/export/destination", json={"path": str(disk)}).status_code == 200
    r = c.post("/api/export/references", json={"names": None, "mode": "copy", "include_raw": False})
    assert r.status_code == 200, r.text
    assert r.json() == {"started": True, "total": 8}
    p = _wait_export(c)
    assert p["error"] is None and p["done"] == 8 and p["total"] == 8 and p["failed"] == 0 and p["skipped"] == 0
    out = disk.resolve() / tmp_path.resolve().name / "people"
    assert Path(p["path"]) == out
    assert sorted(x.name for x in (out / "Arya").iterdir()) == [f"p0_{j}.jpg" for j in range(4)]
    assert sorted(x.name for x in (out / "Priest").iterdir()) == [f"p1_{j}.jpg" for j in range(4)]
    assert all((out / "Arya" / f).is_file() and not (out / "Arya" / f).is_symlink() for f in os.listdir(out / "Arya"))
    assert not (out / "failed.txt").exists()
    # a re-export, even under a different mode, finds Priest's photos already there and skips
    # them rather than failing or duplicating
    r2 = c.post("/api/export/references", json={"names": ["Priest"], "mode": "symlink"})
    assert r2.json()["total"] == 4
    p2 = _wait_export(c)
    assert p2["error"] is None and p2["skipped"] == 4
    assert len(os.listdir(out / "Priest")) == 4 and not any(x.is_symlink() for x in (out / "Priest").iterdir())
    for bad in [{"names": ["Nobody"]}, {"names": []}, {"names": ["Arya"], "min_sim": 0.99}]:
        r = c.post("/api/export/references", json=dict(bad, mode="symlink"))
        assert r.status_code == 400, bad
    assert c.post("/api/export/references", json={"names": None, "mode": "csv"}).status_code == 400
    assert sorted(os.listdir(tmp_path)) == before


def test_export_references_endpoint_preflight_and_lock(tmp_path, tmp_path_factory, monkeypatch):
    import photosort.server as srv
    conn, before = _named_shoot(tmp_path, monkeypatch)
    c = TestClient(create_app(tmp_path))
    disk = tmp_path_factory.mktemp("disk")
    assert c.post("/api/export/destination", json={"path": str(disk)}).status_code == 200
    class Usage: free = 10
    monkeypatch.setattr(srv.shutil, "disk_usage", lambda p: Usage)
    r = c.post("/api/export/references", json={"names": None, "mode": "copy"})
    assert r.status_code == 400 and "on that disk" in r.json()["detail"]
    c.app.state.photosort["export"]["running"] = True
    try:
        assert c.post("/api/export/references", json={"names": None, "mode": "symlink"}).status_code == 409
    finally:
        c.app.state.photosort["export"]["running"] = False
    assert c.post("/api/export/references", json={"names": None, "mode": "symlink"}).status_code == 200
    p = _wait_export(c)
    assert p["error"] is None and p["skipped"] == 0
    assert TestClient(create_app(None)).post("/api/export/references", json={"names": None}).status_code == 400
    assert sorted(os.listdir(tmp_path)) == before


# index bundles: pack the index into one zip, install one on another Mac

def test_bundle_export_endpoint_runs_to_completion(tmp_path, tmp_path_factory):
    from photosort.bundle import inspect_bundle
    c = _shoot_client(tmp_path, n=3)
    disk = tmp_path_factory.mktemp("disk")
    assert c.post("/api/export/destination", json={"path": str(disk)}).status_code == 200
    r = c.post("/api/bundle/export")
    assert r.status_code == 200, r.text
    assert r.json() == {"started": True, "total": 6}
    p = _wait_export(c)
    assert p["error"] is None and p["done"] == 6 and p["total"] == 6 and p["failed"] == 0
    z = Path(p["path"])
    assert z == disk.resolve() / f"{tmp_path.resolve().name}.photosort-index.zip" and z.is_file()
    assert inspect_bundle(z)["photos"] == 3
    assert TestClient(create_app(None)).post("/api/bundle/export").status_code == 400
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg", "p1.jpg", "p2.jpg"]


def test_bundle_export_409_while_busy_and_preflight(tmp_path, tmp_path_factory, monkeypatch):
    import photosort.server as srv
    c = _shoot_client(tmp_path, n=2)
    st = c.app.state.photosort
    st["export"]["running"] = True
    try:
        assert c.post("/api/bundle/export").status_code == 409
    finally:
        st["export"]["running"] = False
    st["running"] = True
    try:
        assert c.post("/api/bundle/export").status_code == 409
    finally:
        st["running"] = False
    class Usage: free = 10
    monkeypatch.setattr(srv.shutil, "disk_usage", lambda p: Usage)
    r = c.post("/api/bundle/export")
    assert r.status_code == 400 and "free" in r.json()["detail"]
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg", "p1.jpg"]


def test_bundle_import_endpoint_switches_to_the_shoot(tmp_path, tmp_path_factory):
    from photosort.bundle import export_bundle
    c = _shoot_client(tmp_path, n=3)
    z = export_bundle(tmp_path, tmp_path_factory.mktemp("out"))
    other = TestClient(create_app(None))                                 # a second app with no folder open
    r = other.post("/api/bundle/import", json={"zip": str(z), "root": str(tmp_path)})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["imported"] is True and info["root"] == str(tmp_path.resolve()) and info["indexed"] is True
    assert info["name"] == tmp_path.name and info["photos"] == 3
    assert other.get("/api/stats").json()["photos"] == 3
    assert other.get("/api/folder").json()["root"] == str(tmp_path.resolve())
    assert len(other.get("/api/search").json()["results"]) == 3
    # root omitted: the bundle's own root, which is a directory on this Mac
    r2 = other.post("/api/bundle/import", json={"zip": str(z)})
    assert r2.status_code == 200 and r2.json()["imported"] is True
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg", "p1.jpg", "p2.jpg"]


def test_bundle_import_endpoint_400s_and_409s(tmp_path, tmp_path_factory):
    import zipfile
    from photosort.bundle import export_bundle
    c = _shoot_client(tmp_path, n=1)
    out = tmp_path_factory.mktemp("out")
    z = export_bundle(tmp_path, out)
    plain = out / "plain.zip"
    with zipfile.ZipFile(plain, "w") as zf:
        zf.writestr("hello.txt", "hi")
    r = c.post("/api/bundle/import", json={"zip": str(plain), "root": str(tmp_path)})
    assert r.status_code == 400 and "not a photosort index bundle" in r.json()["detail"]
    r = c.post("/api/bundle/import", json={"zip": str(out / "missing.zip"), "root": str(tmp_path)})
    assert r.status_code == 400
    r = c.post("/api/bundle/import", json={"zip": str(z), "root": str(tmp_path / "nope")})
    assert r.status_code == 400 and "not a directory" in r.json()["detail"]
    st = c.app.state.photosort
    st["export"]["running"] = True
    try:
        assert c.post("/api/bundle/import", json={"zip": str(z), "root": str(tmp_path)}).status_code == 409
        assert c.post("/api/bundle/import/choose").status_code == 409
        assert c.post("/api/bundle/import/choose-root", json={"zip": str(z)}).status_code == 409
    finally:
        st["export"]["running"] = False
    st["running"] = True
    try:
        assert c.post("/api/bundle/import", json={"zip": str(z), "root": str(tmp_path)}).status_code == 409
    finally:
        st["running"] = False
    assert c.post("/api/bundle/import", json={"zip": str(z), "root": str(tmp_path)}).status_code == 200
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg"]


def test_bundle_import_choose_imports_or_asks_for_the_root(tmp_path, tmp_path_factory, monkeypatch):
    import json as js
    import subprocess as sp
    import zipfile
    from photosort.bundle import export_bundle
    c = _shoot_client(tmp_path, n=2)
    out = tmp_path_factory.mktemp("out")
    z = export_bundle(tmp_path, out)
    monkeypatch.setattr("photosort.server.subprocess.run",
                        lambda *a, **k: sp.CompletedProcess(a, returncode=1, stdout="", stderr=""))
    assert c.post("/api/bundle/import/choose").status_code == 204
    assert c.post("/api/bundle/import/choose-root", json={"zip": str(z)}).status_code == 204
    # the bundle's root is here: imported straight away
    monkeypatch.setattr("photosort.server.subprocess.run",
                        lambda *a, **k: sp.CompletedProcess(a, returncode=0, stdout=str(z) + "\n", stderr=""))
    info = c.post("/api/bundle/import/choose").json()
    assert info["imported"] is True and info["root"] == str(tmp_path.resolve()) and info["photos"] == 2
    # the bundle's root is not here: nothing installed, the UI must ask for the folder
    away = out / "away.photosort-index.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(away, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "bundle.json":
                b = js.loads(data); b["root"] = "/Volumes/not-here-photosort-test/shoot"; data = js.dumps(b).encode()
            dst.writestr(item, data)
    monkeypatch.setattr("photosort.server.subprocess.run",
                        lambda *a, **k: sp.CompletedProcess(a, returncode=0, stdout=str(away) + "\n", stderr=""))
    r = c.post("/api/bundle/import/choose").json()
    assert r["needs_root"] is True and r["zip"] == str(away) and r["bundle"]["photos"] == 2
    assert r["bundle"]["root"] == "/Volumes/not-here-photosort-test/shoot"
    from photosort.config import app_home, shoot_slug
    assert not (app_home() / shoot_slug(Path("/Volumes/not-here-photosort-test/shoot"))).exists()
    # second step: the folder picker names the folder, then it is imported under that root
    here = tmp_path_factory.mktemp("disk") / "shoot"; here.mkdir()
    monkeypatch.setattr("photosort.server.subprocess.run",
                        lambda *a, **k: sp.CompletedProcess(a, returncode=0, stdout=str(here) + "\n", stderr=""))
    r2 = c.post("/api/bundle/import/choose-root", json={"zip": str(away)}).json()
    assert r2["imported"] is True and r2["root"] == str(here.resolve()) and r2["photos"] == 2
    assert c.get("/api/folder").json()["root"] == str(here.resolve())
    assert c.get("/api/stats").json()["photos"] == 2
    # a picked zip that is not a bundle is a 400, not a crash
    plain = out / "plain.zip"
    with zipfile.ZipFile(plain, "w") as zf:
        zf.writestr("hello.txt", "hi")
    monkeypatch.setattr("photosort.server.subprocess.run",
                        lambda *a, **k: sp.CompletedProcess(a, returncode=0, stdout=str(plain) + "\n", stderr=""))
    assert c.post("/api/bundle/import/choose").status_code == 400
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg", "p1.jpg"]
    assert sorted(os.listdir(here)) == []


def test_export_destination_above_source_is_400(tmp_path):
    c = _shoot_client(tmp_path)
    r = c.post("/api/export/destination", json={"path": str(tmp_path.parent)})
    assert r.status_code == 400 and "contains the source folder" in r.json()["detail"]
    assert c.get("/api/export/destination").json()["default"] is True
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg"]


def test_bundle_import_with_an_unreadable_index_is_400_and_installs_nothing(tmp_path, tmp_path_factory):
    import json as js
    import zipfile
    from photosort.config import app_home, shoot_slug
    c = _shoot_client(tmp_path, n=1)
    other = tmp_path_factory.mktemp("disk") / "shoot"; other.mkdir()          # a root with no index yet
    out = tmp_path_factory.mktemp("out")
    bad = out / "bad.photosort-index.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("bundle.json", js.dumps({"format": "photosort-index/1", "root": str(other), "name": "shoot", "photos": 1}))
        zf.writestr("index.db", os.urandom(4096))                              # not SQLite
    r = c.post("/api/bundle/import", json={"zip": str(bad), "root": str(other)})
    assert r.status_code == 400 and "not readable" in r.json()["detail"]
    assert not (app_home() / shoot_slug(other)).exists()
    assert not list(app_home().glob(".*import*"))
    assert c.get("/api/folder").json()["root"] == str(tmp_path.resolve())      # still on the old shoot
    assert sorted(os.listdir(tmp_path)) == ["p0.jpg"] and sorted(os.listdir(other)) == []
