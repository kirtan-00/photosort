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
    ex = c.post("/api/export", json={"ids": [res[0]["id"]], "name": "t"}).json()
    assert Path(ex["path"]).is_dir() and (Path(ex["path"]) / "a.jpg").is_file() and not str(Path(ex["path"])).startswith(str(tmp_path))
    assert c.get("/api/people").json() == []


def test_search_by_missing_image_id_is_404(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    r = c.get("/api/search", params={"image_id": 999})
    assert r.status_code == 404


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
    ok = c.post("/api/export", json={"ids": [pid], "name": "fine"}).json()
    assert Path(ok["path"]).resolve().is_relative_to(export_root().resolve())


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
    assert c.get("/api/search").json() == {"results": []}
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
