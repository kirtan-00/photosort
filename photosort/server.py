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
from . import settings
from .config import app_home, export_root
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


class DestinationReq(BaseModel):
    path: str


class CategoriesExportReq(BaseModel):
    categories: list[str] | None = None
    mode: str = "copy"
    include_raw: bool = False


class ReferenceReq(BaseModel):
    name: str
    path: str


class MinSimReq(BaseModel):
    min_sim: float | None = None


class ReferencesExportReq(BaseModel):
    names: list[str] | None = None
    mode: str = "copy"
    include_raw: bool = False
    min_sim: float | None = None


class BundleImportReq(BaseModel):
    zip: str
    root: str | None = None


class BundleZipReq(BaseModel):
    zip: str


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
        "export": {"running": False, "done": 0, "total": 0, "failed": 0, "skipped": 0, "path": None, "error": None},
        # Where exports land instead of export_root() (another disk), or None for the default.
        "export_base": settings.get_export_base(),
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
            state["export"] = {"running": False, "done": 0, "total": 0, "failed": 0, "skipped": 0, "path": None, "error": None}
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

    # Named people: reference photos saved under a name, matched on demand at the slider's min_sim.

    def _min_sim(v: float | None) -> float:
        from .config import FACE_MATCH_MIN_SIM
        return FACE_MATCH_MIN_SIM if v is None else v

    @app.post("/api/people/references")
    def save_reference_api(req: ReferenceReq):
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        p = Path(req.path).expanduser()
        if not p.is_file():
            raise HTTPException(400, f"not a readable file: {p}")
        from .people import save_reference, ReferenceUnreadable
        try:
            out = save_reference(state["root"], req.name, p)
        except ReferenceUnreadable:
            raise HTTPException(400, "could not read that image")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"id": out["id"], "name": out["name"], "faces_in_reference": out["faces_in_reference"]}

    @app.get("/api/people/references")
    def list_references_api(min_sim: float | None = None):
        if state["root"] is None:
            return {"people": []}
        from .people import match_references
        conn = db.connect(state["root"])
        groups: dict[str, dict] = {}
        for r in db.list_references(conn):      # every saved name, even one with no match at this min_sim
            g = groups.setdefault(r["name"], {"name": r["name"], "reference_ids": [], "sources": [], "count": 0})
            g["reference_ids"].append(r["id"]); g["sources"].append(r["source"])
        matched = match_references(state["root"], _min_sim(min_sim))
        for name, g in groups.items():
            g["count"] = len(matched.get(name, []))
        return {"people": sorted(groups.values(), key=lambda g: (-g["count"], g["name"]))}

    def _reference_names() -> set[str]:
        return {r["name"] for r in db.list_references(db.connect(state["root"]))}

    @app.post("/api/people/references/{name:path}/find")
    def find_reference_api(name: str, req: MinSimReq):
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        if name not in _reference_names():
            raise HTTPException(404, f"no saved person called {name!r}")
        from .people import match_references
        photos = ix().photos
        matches = match_references(state["root"], _min_sim(req.min_sim)).get(name, [])
        results = [dict(photos[m["photo_id"]], score=m["sim"]) for m in matches if m["photo_id"] in photos]
        return {"name": name, "total": len(results), "results": results}

    @app.post("/api/people/references/{name:path}/rename")
    def rename_reference_api(name: str, req: NameReq):
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        new = req.name.strip()
        if not new:
            raise HTTPException(400, "give the person a name")
        from .export import safe_segment
        try:
            safe_segment(new)
        except ValueError:
            raise HTTPException(400, "that name cannot be used as a folder")
        moved = db.rename_reference(db.connect(state["root"]), name, new)
        if moved == 0:
            raise HTTPException(404, f"no saved person called {name!r}")
        return {"ok": True, "name": new, "moved": moved}

    @app.delete("/api/people/references/{ref_id}")
    def delete_reference_api(ref_id: int):
        if state["root"] is None:
            raise HTTPException(400, "no folder open")
        if not db.delete_reference(db.connect(state["root"]), ref_id):
            raise HTTPException(404, "no such reference")
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

    EXPORT_HEADROOM = 1 << 30   # keep 1 GiB free on the destination disk after a copy

    def _is_default_base(base: Path) -> bool:
        return Path(base).resolve() == export_root().resolve()

    def _resolve_base() -> Path:
        """The folder exports go under right now. The default is created on demand; a chosen
        destination (another disk) must already be there or the export is refused."""
        base = state["export_base"]
        if base is None:
            base = export_root(); base.mkdir(parents=True, exist_ok=True)
            return base
        if not base.is_dir():
            raise HTTPException(400, "export destination is not mounted; plug that disk in or reset the destination")
        return base

    def _check_free(need: int, base: Path, hint: str = "Use links, or export fewer photos.") -> None:
        """400 when a copy of `need` bytes would leave less than EXPORT_HEADROOM on the disk holding base."""
        free = shutil.disk_usage(base).free
        where = "on this Mac" if _is_default_base(base) else "on that disk"
        if need + EXPORT_HEADROOM > free:
            raise HTTPException(400, f"copy needs {need / 1e9:.1f} GB but only {free / 1e9:.1f} GB is free {where}. {hint}")

    def _destination_info() -> dict:
        base = state["export_base"]
        default = base is None
        if default:
            base = export_root(); base.mkdir(parents=True, exist_ok=True)
        mounted = base.is_dir()
        free_gb = round(shutil.disk_usage(base).free / 1e9, 1) if mounted else 0.0
        return {"path": str(base), "default": default, "mounted": mounted, "free_gb": free_gb}

    def _set_destination(p: Path) -> dict:
        p = p.expanduser()
        if not p.is_dir():
            raise HTTPException(400, f"not a directory: {p}")
        p = p.resolve()
        root = state["root"]
        if root is not None:
            r = root.resolve()
            if p == r or p.is_relative_to(r):
                raise HTTPException(400, "destination is inside the source folder")
            if r.is_relative_to(p):
                raise HTTPException(400, "destination contains the source folder; pick a folder that is not above it")
        with state["export_lock"]:
            if state["export"]["running"]:
                raise HTTPException(409, "cannot change the destination while an export is running")
            state["export_base"] = p
        settings.set_export_base(p)
        return _destination_info()

    @app.get("/api/export/destination")
    def get_export_destination():
        return _destination_info()

    @app.post("/api/export/destination")
    def set_export_destination(req: DestinationReq):
        return _set_destination(Path(req.path))

    @app.post("/api/export/destination/choose")
    def choose_export_destination():
        try:
            result = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose folder with prompt "Pick where exports go")'],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "folder picker timed out")
        except FileNotFoundError:
            raise HTTPException(501, "folder picker unavailable (osascript not found)")
        path_str = result.stdout.strip()
        if result.returncode != 0 or not path_str:
            return Response(status_code=204)
        return _set_destination(Path(path_str))

    @app.delete("/api/export/destination")
    def reset_export_destination():
        with state["export_lock"]:
            if state["export"]["running"]:
                raise HTTPException(409, "cannot change the destination while an export is running")
            state["export_base"] = None
        settings.set_export_base(None)
        return _destination_info()

    @app.post("/api/export")
    def export(req: ExportReq):
        with state["export_lock"]:
            root_at_start = state["root"]
            if root_at_start is None:
                raise HTTPException(400, "no folder open")
            if state["export"]["running"]:
                raise HTTPException(409, "an export is already running")
            base = _resolve_base()
            if req.mode == "copy":
                _check_free(export_bytes(root_at_start, req.ids), base)
            try:
                from .export import export_dir
                export_dir(root_at_start, req.name, base)   # validate now so a bad name or base is a 400, not a background error
            except ValueError as e:
                raise HTTPException(400, str(e))
            state["export"] = {"running": True, "done": 0, "total": len(req.ids), "failed": 0, "skipped": 0, "path": None, "error": None}

        def prog(d):
            state["export"].update(d)

        def _run_export():
            try:
                state["export"]["path"] = str(export_ids(root_at_start, req.ids, req.name, req.mode, progress=prog, base=base))
            except Exception as e:
                state["export"]["error"] = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            finally:
                state["export"]["running"] = False

        threading.Thread(target=_run_export, daemon=True).start()
        return {"started": True, "total": len(req.ids)}

    @app.get("/api/export/progress")
    def export_progress():
        return state["export"]

    @app.post("/api/export/categories")
    def export_categories_api(req: CategoriesExportReq):
        """One folder per ticked category under <destination>/<shoot>/categories/. Same job
        machinery as /api/export: one export at a time, preflight for copies, progress polled
        from /api/export/progress. total in the reply counts photos; progress counts RAW siblings too."""
        from .export import export_dir, export_categories, category_rows, categories_bytes
        if req.mode not in ("copy", "symlink"):
            raise HTTPException(400, "mode must be copy or symlink")
        if req.categories is not None and not req.categories:
            raise HTTPException(400, "tick at least one category")
        with state["export_lock"]:
            root_at_start = state["root"]
            if root_at_start is None:
                raise HTTPException(400, "no folder open")
            if state["export"]["running"]:
                raise HTTPException(409, "an export is already running")
            base = _resolve_base()
            n_photos = len(category_rows(root_at_start, req.categories))
            if req.mode == "copy":
                _check_free(categories_bytes(root_at_start, req.categories, req.include_raw), base)
            try:
                export_dir(root_at_start, "categories", base)
            except ValueError as e:
                raise HTTPException(400, str(e))
            state["export"] = {"running": True, "done": 0, "total": n_photos, "failed": 0, "skipped": 0, "path": None, "error": None}

        def prog(d):
            state["export"].update(d)

        def _run_export():
            try:
                state["export"]["path"] = str(export_categories(root_at_start, req.categories, req.mode, req.include_raw,
                                                                base=base, progress=prog))
            except Exception as e:
                state["export"]["error"] = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            finally:
                state["export"]["running"] = False

        threading.Thread(target=_run_export, daemon=True).start()
        return {"started": True, "total": n_photos}

    @app.post("/api/export/references")
    def export_references_api(req: ReferencesExportReq):
        """One folder per saved (or listed) person under <destination>/<shoot>/people/, matched at
        min_sim. Same job machinery as /api/export/categories. total in the reply counts photo
        placements (a frame with two people counts twice); progress counts RAW siblings too."""
        from .export import export_dir
        from .people import export_references, export_references_ids, references_bytes
        if req.mode not in ("copy", "symlink"):
            raise HTTPException(400, "mode must be copy or symlink")
        if req.names is not None and not req.names:
            raise HTTPException(400, "tick at least one person")
        min_sim = _min_sim(req.min_sim)
        with state["export_lock"]:
            root_at_start = state["root"]
            if root_at_start is None:
                raise HTTPException(400, "no folder open")
            if state["export"]["running"]:
                raise HTTPException(409, "an export is already running")
            base = _resolve_base()
            folders = export_references_ids(root_at_start, req.names, min_sim)
            n_photos = sum(len(ids) for ids in folders.values())
            if n_photos == 0:
                raise HTTPException(400, "no saved person matches any photo at this match level")
            if req.mode == "copy":
                _check_free(references_bytes(root_at_start, req.names, req.include_raw, min_sim), base)
            try:
                export_dir(root_at_start, "people", base)
            except ValueError as e:
                raise HTTPException(400, str(e))
            state["export"] = {"running": True, "done": 0, "total": n_photos, "failed": 0, "skipped": 0, "path": None, "error": None}

        def prog(d):
            state["export"].update(d)

        def _run_export():
            try:
                state["export"]["path"] = str(export_references(root_at_start, req.names, req.mode, req.include_raw,
                                                                base=base, progress=prog, min_sim=min_sim))
            except Exception as e:
                state["export"]["error"] = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            finally:
                state["export"]["running"] = False

        threading.Thread(target=_run_export, daemon=True).start()
        return {"started": True, "total": n_photos}

    # Index bundles: the whole index (db with saved people, thumbs, grid) as one zip on the export
    # destination, and the reverse: install such a zip here and open the shoot without re-indexing.

    @app.post("/api/bundle/export")
    def export_bundle_api():
        from . import bundle
        with state["export_lock"]:
            root_at_start = state["root"]
            if root_at_start is None:
                raise HTTPException(400, "no folder open")
            if state["running"]:
                raise HTTPException(409, "cannot pack the index while indexing")
            if state["export"]["running"]:
                raise HTTPException(409, "an export is already running")
            base = _resolve_base()
            try:
                bundle.bundle_path(root_at_start, base)
            except ValueError as e:
                raise HTTPException(400, str(e))
            n_files = len(bundle.bundle_files(root_at_start))
            _check_free(bundle.bundle_bytes(root_at_start), base, hint="Free some space there first.")
            state["export"] = {"running": True, "done": 0, "total": n_files, "failed": 0, "skipped": 0, "path": None, "error": None}

        def prog(d):
            state["export"].update(d)

        def _run_export():
            try:
                state["export"]["path"] = str(bundle.export_bundle(root_at_start, base, progress=prog))
            except Exception as e:
                state["export"]["error"] = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            finally:
                state["export"]["running"] = False

        threading.Thread(target=_run_export, daemon=True).start()
        return {"started": True, "total": n_files}

    def _import_gates() -> None:
        # Same gates as _switch_root, checked up front so nobody sits through a picker for a 409.
        if state["running"]:
            raise HTTPException(409, "cannot import an index while indexing")
        if state["export"]["running"]:
            raise HTTPException(409, "cannot import an index while an export is running")

    def _run_picker(script: str, what: str) -> str | None:
        """POSIX path from a macOS picker, or None when the user cancelled."""
        try:
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise HTTPException(504, f"{what} picker timed out")
        except FileNotFoundError:
            raise HTTPException(501, f"{what} picker unavailable (osascript not found)")
        path_str = result.stdout.strip()
        if result.returncode != 0 or not path_str:
            return None
        return path_str

    def _inspect_zip(zip_path: Path) -> dict:
        from .bundle import inspect_bundle
        try:
            return inspect_bundle(zip_path)
        except ValueError as e:
            raise HTTPException(400, str(e))

    def _import_and_switch(zip_path: Path, root: Path, info: dict) -> dict:
        """Install the bundle for root, switch to it, and return the folder payload plus what was imported."""
        from .bundle import import_bundle
        _import_gates()
        try:
            import_bundle(zip_path, root)
        except ValueError as e:
            raise HTTPException(400, str(e))
        out = _switch_root(root)
        return dict(out, imported=True, root=str(root), photos=info.get("photos"))

    @app.post("/api/bundle/import/choose")
    def import_bundle_choose():
        _import_gates()
        path_str = _run_picker('POSIX path of (choose file with prompt "Pick a photosort index bundle" of type {"public.zip-archive"})', "bundle")
        if path_str is None:
            return Response(status_code=204)
        zip_path = Path(path_str)
        info = _inspect_zip(zip_path)
        root = Path(info["root"])
        if not root.is_dir():
            # Made on a Mac where the disk sat elsewhere: the UI asks for the folder, then calls choose-root.
            return {"needs_root": True, "bundle": info, "zip": str(zip_path)}
        return _import_and_switch(zip_path, root.resolve(), info)

    @app.post("/api/bundle/import")
    def import_bundle_api(req: BundleImportReq):
        _import_gates()
        zip_path = Path(req.zip).expanduser()
        info = _inspect_zip(zip_path)
        root = Path(req.root).expanduser() if req.root else Path(info["root"])
        if not root.is_dir():
            if req.root:
                raise HTTPException(400, f"not a directory: {req.root}")
            raise HTTPException(400, f"that bundle was made for {info['root']}, which is not here; pick the photo folder")
        return _import_and_switch(zip_path, root.resolve(), info)

    @app.post("/api/bundle/import/choose-root")
    def import_bundle_choose_root(req: BundleZipReq):
        _import_gates()
        zip_path = Path(req.zip).expanduser()
        info = _inspect_zip(zip_path)
        path_str = _run_picker('POSIX path of (choose folder with prompt "Pick the photo folder this index was made for")', "folder")
        if path_str is None:
            return Response(status_code=204)
        root = Path(path_str)
        if not root.is_dir():
            raise HTTPException(400, f"not a directory: {path_str}")
        return _import_and_switch(zip_path, root.resolve(), info)

    # People/groups/solo export stays synchronous in this pass, it is the small-shoot
    # bundle, not the main Diu-scale export path that /api/export now backgrounds.
    @app.post("/api/export/people")
    def export_people_api(req: ModeReq):
        root_at_start = state["root"]
        if root_at_start is None:
            raise HTTPException(400, "no folder open")
        from .people import export_people, export_people_ids
        with state["export_lock"]:
            base = _resolve_base()
        if req.mode == "copy":
            # One photo lands in several folders (each person, plus groups or solo) and each is a
            # separate copy, so size every folder, not the distinct set of photos.
            _check_free(sum(export_bytes(root_at_start, ids) for ids in export_people_ids(root_at_start).values()), base)
        try:
            return {"path": str(export_people(root_at_start, req.mode, base=base))}
        except ValueError as e:
            raise HTTPException(400, str(e))

    return app
