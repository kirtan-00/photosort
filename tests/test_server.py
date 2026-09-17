from pathlib import Path
from fastapi.testclient import TestClient
from photosort.index import index_folder
from photosort.server import create_app


def test_api(tmp_path):
    from tests.conftest import make_image
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
    from tests.conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    r = c.get("/api/search", params={"image_id": 999})
    assert r.status_code == 404


def test_ui_static_app_js_served(tmp_path):
    from tests.conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    r = c.get("/ui/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_ui_unknown_static_file_404s(tmp_path):
    from tests.conftest import make_image
    make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    c = TestClient(create_app(tmp_path))
    assert c.get("/ui/does-not-exist.js").status_code == 404


def test_search_after_reindex_and_person_filter_dont_500(tmp_path):
    """Index() opens its sqlite connection on the thread that builds the app; FastAPI
    runs sync endpoints in a worker thread. refresh() and person-filtered search must
    not reuse that connection cross-thread or sqlite raises ProgrammingError -> 500."""
    from tests.conftest import make_image
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
    from tests.conftest import make_image
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
