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

    def slow_export_ids(root, ids_, name, mode="copy", progress=None):
        def slow_progress(d):
            time.sleep(0.1)
            if progress: progress(d)
        return real_export_ids(root, ids_, name, mode, progress=slow_progress)

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

def test_categories_endpoint_returns_a_dict(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    r = c.get("/api/categories")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


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
