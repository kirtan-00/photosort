from __future__ import annotations
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from . import db
from . import classify as classify_mod
from .config import app_home
from .search import Index, Filters
from .export import export_ids, export_bytes

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
    retry_errors: bool = False


class ModeReq(BaseModel):
    mode: str = "copy"


class FolderReq(BaseModel):
    path: str


class FindReq(BaseModel):
    path: str
    min_sim: float | None = None


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
        "export": {"running": False, "done": 0, "total": 0, "failed": 0, "path": None, "error": None},
        # Held from the "already running" check through setting running=True, and around a folder
        # switch, so two rapid export POSTs (or a switch during the preflight) cannot both pass.
        "export_lock": threading.Lock(),
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
        with state["export_lock"]:
            if state["export"]["running"]:
                raise HTTPException(409, "cannot switch folders while an export is running")
            state["root"] = new_root
            state["index"] = Index(new_root)
            state["progress"] = {"stage": "idle", "done": 0, "total": 0}
            state["stale"] = False
            state["classify"] = {"running": False, "counts": {}, "error": None}
            state["export"] = {"running": False, "done": 0, "total": 0, "failed": 0, "path": None, "error": None}
        _save_recent(str(new_root))
        return _folder_info()

    def _auto_classify(root_at_start: Path, stats: dict) -> None:
        """Categorise at the end of an index run that changed something. A classify failure is
        recorded on state["classify"] and never marks the index run itself as failed."""
        if stats.get("indexed", 0) == 0 and stats.get("embedded", 0) == 0:
            return
        if state["classify"]["running"]:          # a manual Categorise is already on it
            return
        total = stats.get("total", 0)
        state["classify"] = {"running": True, "counts": {}, "error": None}
        state["progress"] = {"stage": "categorise", "done": 0, "total": 0, "stage_started": time.time()}
        try:
            state["classify"]["counts"] = classify_mod.classify_and_store(root_at_start)
        except Exception as e:
            state["classify"]["error"] = f"{type(e).__name__}: {e}"
        finally:
            state["classify"]["running"] = False
            state["progress"] = {"stage": "done", "done": total, "total": total, "stage_started": time.time()}

    def _run(root_at_start: Path, faces: bool, retry_errors: bool):
        from .index import index_folder
        def prog(d):
            state["progress"] = d
        try:
            stats = index_folder(root_at_start, faces=faces, progress=prog, retry_errors=retry_errors)
            _auto_classify(root_at_start, stats)
        except Exception as e:
            from .index import SourceUnavailable
            msg = str(e) if isinstance(e, SourceUnavailable) else f"{type(e).__name__}: {e}"
            state["progress"] = {"stage": "error", "error": msg, "done": 0, "total": 0}
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
        # Same gates as _switch_root, checked up front so nobody sits through the picker for a 409.
        if state["running"]:
            raise HTTPException(409, "cannot switch folders while indexing")
        if state["export"]["running"]:
            raise HTTPException(409, "cannot switch folders while an export is running")
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
            return dict(root=None, photos=0, faces=0, people=0, errors=0, last_index=None, indexing=state["running"])
        conn = db.connect(root)
        n = lambda q: conn.execute(q).fetchone()[0]
        last = conn.execute("SELECT value FROM meta WHERE key='last_index'").fetchone()
        return dict(
            root=str(root),
            photos=n("SELECT count(*) FROM photos WHERE status='ok'"),
            faces=n("SELECT count(*) FROM faces"),
            people=n("SELECT count(*) FROM people"),
            errors=n("SELECT count(*) FROM photos WHERE status='error'"),
            last_index=last[0] if last else None,
            indexing=state["running"],
        )

    @app.get("/api/errors")
    def errors():
        if state["root"] is None:
            return {"errors": []}
        conn = db.connect(state["root"])
        rows = conn.execute("SELECT rel, indexed_at FROM photos WHERE status='error' ORDER BY rel").fetchall()
        return {"errors": [{"rel": r[0], "indexed_at": r[1]} for r in rows]}

    @app.post("/api/index")
    def start_index(req: IndexReq):
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        if state["running"]:
            raise HTTPException(409, "already indexing")
        state["running"] = True
        threading.Thread(target=_run, args=(state["root"], req.faces, req.retry_errors), daemon=True).start()
        return {"started": True}

    @app.get("/api/progress")
    def progress():
        return dict(state["progress"], running=state["running"])

    def _filters(sharp, faces, person, taken_from, taken_to, category) -> Filters:
        return Filters(sharp_min_pct=sharp, faces=faces or None, person_id=person, taken_from=taken_from,
                       taken_to=taken_to, category=category or None)

    @app.get("/api/search")
    def search(q: str | None = None, image_id: int | None = None, sharp: float | None = None, faces: str | None = None,
               person: int | None = None, taken_from: str | None = None, taken_to: str | None = None,
               category: str | None = None, limit: int = 200, offset: int = 0):
        if state["root"] is None:
            return {"results": [], "total": 0, "offset": 0, "limit": limit}
        limit = max(1, min(limit, 1000)); offset = max(0, offset)
        try:
            rows = ix().query(text=q or None, image_id=image_id, filters=_filters(sharp, faces, person, taken_from, taken_to, category))
        except LookupError as e:
            raise HTTPException(404, str(e))
        return {"results": [dict(p) for p in rows[offset:offset + limit]], "total": len(rows), "offset": offset, "limit": limit}

    @app.get("/api/search/ids")
    def search_ids(q: str | None = None, image_id: int | None = None, sharp: float | None = None, faces: str | None = None,
                   person: int | None = None, taken_from: str | None = None, taken_to: str | None = None,
                   category: str | None = None):
        if state["root"] is None:
            return {"ids": [], "total": 0}
        try:
            rows = ix().query(text=q or None, image_id=image_id, filters=_filters(sharp, faces, person, taken_from, taken_to, category))
        except LookupError as e:
            raise HTTPException(404, str(e))
        return {"ids": [p["id"] for p in rows], "total": len(rows)}

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

    def _find_person(p: Path, min_sim: float | None) -> dict:
        # The reference is only read; results are index rows in the same shape as /api/search
        # plus score = cosine sim, so the grid can show them unchanged.
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        if not p.is_file():
            raise HTTPException(400, f"not a readable file: {p}")
        from .people import find_by_reference, ReferenceUnreadable
        from .config import FACE_MATCH_MIN_SIM
        try:
            found = find_by_reference(state["root"], p, FACE_MATCH_MIN_SIM if min_sim is None else min_sim)
        except ReferenceUnreadable:
            raise HTTPException(400, "could not read that image")
        photos = ix().photos
        results = [dict(photos[m["photo_id"]], score=m["sim"]) for m in found["matches"] if m["photo_id"] in photos]
        out = {"faces_in_reference": found["faces_in_reference"], "person_id": found["person_id"],
               "total": len(results), "results": results}
        if found.get("reference_face_too_small"):
            out["reference_face_too_small"] = True
        return out

    @app.post("/api/people/find")
    def find_person(req: FindReq):
        return _find_person(Path(req.path).expanduser(), req.min_sim)

    @app.post("/api/people/find/choose")
    def find_person_choose():
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        try:
            result = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose file with prompt "Pick a photo of the person" of type {"public.image"})'],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "photo picker timed out")
        except FileNotFoundError:
            raise HTTPException(501, "photo picker unavailable (osascript not found)")
        path_str = result.stdout.strip()
        if result.returncode != 0 or not path_str:
            return Response(status_code=204)
        # path rides along so the UI can re-run /api/people/find at another min_sim without the picker.
        return dict(_find_person(Path(path_str), None), path=path_str)

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

    EXPORT_HEADROOM = 1 << 30   # keep 1 GiB free on the Mac after a copy

    def _check_free(need: int) -> None:
        """400 when a copy of `need` bytes would leave less than EXPORT_HEADROOM on the Mac."""
        from .config import export_root
        base = export_root(); base.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(base).free
        if need + EXPORT_HEADROOM > free:
            raise HTTPException(400, f"copy needs {need / 1e9:.1f} GB but only {free / 1e9:.1f} GB is free on this Mac. Use links, or export fewer photos.")

    @app.post("/api/export")
    def export(req: ExportReq):
        with state["export_lock"]:
            root_at_start = state["root"]
            if root_at_start is None:
                raise HTTPException(400, "no folder open")
            if state["export"]["running"]:
                raise HTTPException(409, "an export is already running")
            if req.mode == "copy":
                _check_free(export_bytes(root_at_start, req.ids))
            try:
                from .export import export_dir
                export_dir(root_at_start, req.name)      # validate the name now so a bad one is a 400, not a background error
            except ValueError as e:
                raise HTTPException(400, str(e))
            state["export"] = {"running": True, "done": 0, "total": len(req.ids), "failed": 0, "path": None, "error": None}

        def prog(d):
            state["export"].update(d)

        def _run_export():
            try:
                state["export"]["path"] = str(export_ids(root_at_start, req.ids, req.name, req.mode, progress=prog))
            except Exception as e:
                state["export"]["error"] = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            finally:
                state["export"]["running"] = False

        threading.Thread(target=_run_export, daemon=True).start()
        return {"started": True, "total": len(req.ids)}

    @app.get("/api/export/progress")
    def export_progress():
        return state["export"]

    # People/groups/solo export stays synchronous in this pass, it is the small-shoot
    # bundle, not the main Diu-scale export path that /api/export now backgrounds.
    @app.post("/api/export/people")
    def export_people_api(req: ModeReq):
        root_at_start = state["root"]
        if root_at_start is None:
            raise HTTPException(400, "no folder open")
        from .people import export_people, export_people_ids
        if req.mode == "copy":
            # One photo lands in several folders (each person, plus groups or solo) and each is a
            # separate copy, so size every folder, not the distinct set of photos.
            _check_free(sum(export_bytes(root_at_start, ids) for ids in export_people_ids(root_at_start).values()))
        try:
            return {"path": str(export_people(root_at_start, req.mode))}
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app
