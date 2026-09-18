from __future__ import annotations
import json
import subprocess
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from . import db
from .config import app_home
from .search import Index, Filters
from .export import export_ids

UI = Path(__file__).parent / "ui"
RECENT_FILE = "recent.json"
RECENT_MAX = 8


class ExportReq(BaseModel):
    ids: list[int]
    name: str
    mode: str = "copy"


class NameReq(BaseModel):
    name: str


class ClusterReq(BaseModel):
    eps: float = 0.5


class IndexReq(BaseModel):
    faces: bool = True


class ModeReq(BaseModel):
    mode: str = "copy"


class FolderReq(BaseModel):
    path: str


def _load_recent() -> list[str]:
    p = app_home() / RECENT_FILE
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text())
        return [str(x) for x in data] if isinstance(data, list) else []
    except Exception:
        return []


def _save_recent(path_str: str) -> None:
    p = app_home() / RECENT_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    recent = [r for r in _load_recent() if r != path_str]
    recent.insert(0, path_str)
    p.write_text(json.dumps(recent[:RECENT_MAX]))


def create_app(root: Path | None = None) -> FastAPI:
    root = Path(root) if root is not None else None
    app = FastAPI(title="photosort")
    state = {
        "root": root,
        "index": Index(root) if root is not None else None,
        "progress": {"stage": "idle", "done": 0, "total": 0},
        "running": False,
        "stale": False,
        "classify": {"running": False, "counts": {}, "error": None},
    }
    app.state.photosort = state

    def _folder_info() -> dict:
        r = state["root"]
        if r is None:
            return {"root": None, "name": None, "indexed": False}
        conn = db.connect(r)
        n = conn.execute("SELECT count(*) FROM photos WHERE status='ok'").fetchone()[0]
        return {"root": str(r), "name": r.name or str(r), "indexed": n > 0}

    def _switch_root(new_root: Path) -> dict:
        if state["running"]:
            raise HTTPException(409, "cannot switch folders while indexing")
        state["root"] = new_root
        state["index"] = Index(new_root)
        state["progress"] = {"stage": "idle", "done": 0, "total": 0}
        state["stale"] = False
        state["classify"] = {"running": False, "counts": {}, "error": None}
        _save_recent(str(new_root))
        return _folder_info()

    def _run(root_at_start: Path, faces: bool):
        from .index import index_folder
        def prog(d):
            state["progress"] = d
        try:
            index_folder(root_at_start, faces=faces, progress=prog)
        except Exception as e:
            state["progress"] = {"stage": "error", "error": f"{type(e).__name__}: {e}", "done": 0, "total": 0}
        finally:
            state["running"] = False
            state["stale"] = True

    def ix() -> Index:
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        if state["stale"]:
            state["index"].refresh()
            state["stale"] = False
        return state["index"]

    @app.get("/")
    def home():
        return FileResponse(UI / "index.html")

    @app.get("/ui/{name}")
    def ui(name: str):
        p = UI / name
        if not p.is_file():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/folder")
    def get_folder():
        return _folder_info()

    @app.post("/api/folder/choose")
    def choose_folder():
        if state["running"]:
            raise HTTPException(409, "cannot switch folders while indexing")
        try:
            result = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose folder with prompt "Pick the photo folder")'],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "folder picker timed out")
        except FileNotFoundError:
            raise HTTPException(501, "folder picker unavailable (osascript not found)")
        path_str = result.stdout.strip()
        if result.returncode != 0 or not path_str:
            return Response(status_code=204)
        p = Path(path_str)
        if not p.is_dir():
            raise HTTPException(400, f"not a directory: {path_str}")
        return _switch_root(p)

    @app.post("/api/folder")
    def set_folder(req: FolderReq):
        p = Path(req.path).expanduser()
        if not p.is_dir():
            raise HTTPException(400, f"not a directory: {req.path}")
        return _switch_root(p.resolve())

    @app.get("/api/folder/recent")
    def recent_folders():
        out = []
        for path_str in _load_recent():
            p = Path(path_str)
            out.append({"path": path_str, "name": p.name or path_str})
        return {"recent": out}

    @app.get("/api/stats")
    def stats():
        root = state["root"]
        if root is None:
            return dict(root=None, photos=0, faces=0, people=0, last_index=None, indexing=state["running"])
        conn = db.connect(root)
        n = lambda q: conn.execute(q).fetchone()[0]
        last = conn.execute("SELECT value FROM meta WHERE key='last_index'").fetchone()
        return dict(
            root=str(root),
            photos=n("SELECT count(*) FROM photos WHERE status='ok'"),
            faces=n("SELECT count(*) FROM faces"),
            people=n("SELECT count(*) FROM people"),
            last_index=last[0] if last else None,
            indexing=state["running"],
        )

    @app.post("/api/index")
    def start_index(req: IndexReq):
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        if state["running"]:
            raise HTTPException(409, "already indexing")
        state["running"] = True
        threading.Thread(target=_run, args=(state["root"], req.faces), daemon=True).start()
        return {"started": True}

    @app.get("/api/progress")
    def progress():
        return dict(state["progress"], running=state["running"])

    @app.get("/api/search")
    def search(q: str | None = None, image_id: int | None = None, sharp: float | None = None, faces: str | None = None,
               person: int | None = None, taken_from: str | None = None, taken_to: str | None = None,
               category: str | None = None, limit: int = 200):
        if state["root"] is None:
            return {"results": []}
        kwargs = dict(sharp_min_pct=sharp, faces=faces or None, person_id=person, taken_from=taken_from, taken_to=taken_to)
        cat_supported = "category" in getattr(Filters, "__dataclass_fields__", {})
        if category:
            if not cat_supported:
                return {"results": []}
            kwargs["category"] = category
        f = Filters(**kwargs)
        try:
            return {"results": ix().search(text=q or None, image_id=image_id, filters=f, limit=limit)}
        except LookupError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/thumb/{qhash}")
    def thumb(qhash: str, size: str = "grid"):
        if state["root"] is None:
            raise HTTPException(404)
        p = db.index_dir(state["root"]) / ("grid" if size == "grid" else "thumbs") / f"{qhash}.jpg"
        if not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, media_type="image/jpeg")

    @app.get("/api/people")
    def people():
        if state["root"] is None:
            return []
        from .people import list_people
        return list_people(state["root"])

    @app.post("/api/people/cluster")
    def cluster(req: ClusterReq):
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        from .people import cluster_faces
        out = cluster_faces(state["root"], eps=req.eps)
        state["stale"] = True
        return out

    @app.post("/api/people/{pid}/name")
    def name(pid: int, req: NameReq):
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        from .people import name_person
        name_person(state["root"], pid, req.name)
        return {"ok": True}

    @app.get("/api/categories")
    def categories():
        if state["root"] is None:
            return {}
        conn = db.connect(state["root"])
        if not hasattr(db, "category_counts"):
            return {}
        try:
            return db.category_counts(conn)
        except Exception:
            return {}

    @app.post("/api/classify")
    def start_classify():
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        from . import classify as classify_mod
        if not hasattr(classify_mod, "classify_and_store"):
            raise HTTPException(501, "categorisation not available yet")
        if state["classify"]["running"]:
            raise HTTPException(409, "already categorising")
        state["classify"] = {"running": True, "counts": {}, "error": None}
        root_at_start = state["root"]

        def _run_classify():
            try:
                counts = classify_mod.classify_and_store(root_at_start)
                state["classify"]["counts"] = counts
                state["stale"] = True
            except Exception as e:
                state["classify"]["error"] = f"{type(e).__name__}: {e}"
            finally:
                state["classify"]["running"] = False

        threading.Thread(target=_run_classify, daemon=True).start()
        return {"started": True}

    @app.get("/api/classify/progress")
    def classify_progress():
        return state["classify"]

    @app.post("/api/export")
    def export(req: ExportReq):
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        try:
            return {"path": str(export_ids(state["root"], req.ids, req.name, req.mode))}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/export/people")
    def export_people_api(req: ModeReq):
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        from .people import export_people
        try:
            return {"path": str(export_people(state["root"], req.mode))}
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app
