from __future__ import annotations
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from . import db
from .search import Index, Filters
from .export import export_ids

UI = Path(__file__).parent / "ui"


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


def create_app(root: Path) -> FastAPI:
    root = Path(root)
    app = FastAPI(title="photosort")
    state = {"index": Index(root), "progress": {"stage": "idle", "done": 0, "total": 0}, "running": False, "stale": False}

    def _run(faces: bool):
        from .index import index_folder
        def prog(d):
            state["progress"] = d
        try:
            index_folder(root, faces=faces, progress=prog)
        finally:
            state["running"] = False
            state["stale"] = True

    def ix() -> Index:
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

    @app.get("/api/stats")
    def stats():
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
        if state["running"]:
            raise HTTPException(409, "already indexing")
        state["running"] = True
        threading.Thread(target=_run, args=(req.faces,), daemon=True).start()
        return {"started": True}

    @app.get("/api/progress")
    def progress():
        return dict(state["progress"], running=state["running"])

    @app.get("/api/search")
    def search(q: str | None = None, image_id: int | None = None, sharp: float | None = None, faces: str | None = None,
               person: int | None = None, taken_from: str | None = None, taken_to: str | None = None, limit: int = 200):
        f = Filters(sharp_min_pct=sharp, faces=faces or None, person_id=person, taken_from=taken_from, taken_to=taken_to)
        try:
            return {"results": ix().search(text=q or None, image_id=image_id, filters=f, limit=limit)}
        except LookupError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/thumb/{qhash}")
    def thumb(qhash: str, size: str = "grid"):
        p = db.index_dir(root) / ("grid" if size == "grid" else "thumbs") / f"{qhash}.jpg"
        if not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, media_type="image/jpeg")

    @app.get("/api/people")
    def people():
        from .people import list_people
        return list_people(root)

    @app.post("/api/people/cluster")
    def cluster(req: ClusterReq):
        from .people import cluster_faces
        out = cluster_faces(root, eps=req.eps)
        state["stale"] = True
        return out

    @app.post("/api/people/{pid}/name")
    def name(pid: int, req: NameReq):
        from .people import name_person
        name_person(root, pid, req.name)
        return {"ok": True}

    @app.post("/api/export")
    def export(req: ExportReq):
        return {"path": str(export_ids(root, req.ids, req.name, req.mode))}

    @app.post("/api/export/people")
    def export_people_api(req: ModeReq):
        from .people import export_people
        return {"path": str(export_people(root, req.mode))}

    return app
