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

    def slow_export_ids(root, ids_, name, mode="copy", progress=None):
        time.sleep(0.5)
        return real_export_ids(root, ids_, name, mode, progress=progress)

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
    assert c.post("/api/people/find", json={"path": str(tmp_path / "p1_0.jpg"), "min_sim": 0.99}).json()["total"] == 0
    assert c.post("/api/people/find", json={"path": "/nope.jpg"}).status_code == 400
    assert sorted(os.listdir(tmp_path)) == before

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
