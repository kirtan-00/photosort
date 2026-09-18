from __future__ import annotations
import json, multiprocessing as mp, time
from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image
from . import db
from .config import PREVIEW_EDGE, GRID_EDGE, THUMB_QUALITY, JPEG_WORKERS, RAW_WORKERS, EMBED_BATCH, YUNET_PATH, SFACE_PATH
from .walk import find_images, quick_hash
from .decode import load_preview
from .features import phash, exif_info, sharpness_tiles, to_gray

class SourceUnavailable(RuntimeError):
    """The shoot root is not there (disk unplugged, wrong mount) while the index already holds photos.
    Raised before any write so the saved index is left exactly as it was."""

_ENGINE = None
def _face_engine():
    global _ENGINE
    if _ENGINE is None:
        from .faces import FaceEngine
        _ENGINE = FaceEngine()
    return _ENGINE

def process_one(args: tuple[str, str, bool]) -> dict:
    root, rel, want_faces = args
    path = Path(root) / rel
    out = {"rel": rel, "row": None, "faces": [], "error": None}
    try:
        qh = quick_hash(path)
        im = load_preview(path, PREVIEW_EDGE)
        idx = db.index_dir(Path(root))
        im.save(idx / "thumbs" / f"{qh}.jpg", quality=THUMB_QUALITY)
        g = im.copy(); g.thumbnail((GRID_EDGE, GRID_EDGE)); g.save(idx / "grid" / f"{qh}.jpg", quality=80)
        gray = to_gray(im)
        p90, mx = sharpness_tiles(gray)
        info = exif_info(path)
        faces = _face_engine().detect(im) if want_faces else []
        eye = max((f.eye_sharp for f in faces), default=None)
        st = path.stat()
        # n_faces stays NULL when faces were not looked for, so a later faces=True
        # run knows to come back for this photo.
        out["row"] = dict(rel=rel, size=st.st_size, mtime=st.st_mtime, qhash=qh, sibling=None,
            width=info["width"] or im.width, height=info["height"] or im.height, taken_at=info["taken_at"],
            camera=info["camera"], phash=phash(im), sharp_tile=p90, sharp_max=mx, sharp_eye=eye,
            sharp=eye if eye is not None else p90, n_faces=len(faces) if want_faces else None, status="ok")
        out["faces"] = [dict(x=f.x, y=f.y, w=f.w, h=f.h, score=f.score, landmarks=json.dumps(f.landmarks.tolist()),
                             eye_sharp=f.eye_sharp, embed=f.embed.astype(np.float32).tobytes()) for f in faces]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out

def index_folder(root: Path, faces: bool = True, workers: int | None = None,
                 progress: Callable[[dict], None] | None = None, embed: bool = True,
                 retry_errors: bool = False) -> dict:
    t0 = time.time(); root = Path(root)
    _raw = progress or (lambda d: None)
    stage = {"name": None, "t": t0}
    def notify(d: dict) -> None:
        if d["stage"] != stage["name"]:
            stage["name"], stage["t"] = d["stage"], time.time()
        _raw(dict(d, stage_started=stage["t"]))
    if faces and not (YUNET_PATH.exists() and SFACE_PATH.exists()):
        raise FileNotFoundError("face models missing; run scripts/fetch_models.sh")
    conn = db.connect(root)
    notify({"stage": "scan", "done": 0, "total": 0})
    n_ok = conn.execute("SELECT count(*) FROM photos WHERE status='ok'").fetchone()[0]
    if not root.is_dir():
        raise SourceUnavailable(f"{root} is not there. Plug the disk in; the saved index ({n_ok} photos) was left untouched.")
    files = find_images(root)
    if not files and n_ok > 0:
        raise SourceUnavailable(f"{root} has no photos right now. Is the disk mounted? The saved index ({n_ok} photos) was left untouched.")
    known = db.known_files(conn, retry_errors=retry_errors)
    missing = db.missing_files(conn)
    idx = db.index_dir(root)
    # A file that went missing and came back unchanged, with its thumb still on the Mac, needs no re-decode.
    restore = [f.rel for f in files if f.rel in missing and missing[f.rel][:2] == (f.size, f.mtime)
               and (idx / "thumbs" / f"{missing[f.rel][2]}.jpg").is_file()]
    if restore:
        db.restore_missing(conn, restore)
        known.update({r: missing[r][:2] for r in restore})
    need_faces = db.photos_without_faces(conn) if faces else set()
    todo = [f for f in files if known.get(f.rel) != (f.size, f.mtime) or f.rel in need_faces]
    stats = dict(total=len(files), skipped=len(files) - len(todo), indexed=0, errors=0, embedded=0)
    db.mark_missing(conn, {f.rel for f in files})
    if todo:
        sib = {f.rel: f.sibling for f in todo}
        meta = {f.rel: (f.size, f.mtime) for f in todo}
        ctx = mp.get_context("spawn")
        done = 0
        def _store(res):
            nonlocal done
            if res["error"]:
                stats["errors"] += 1
                db.mark_error(conn, res["rel"], meta[res["rel"]][0], meta[res["rel"]][1])
            else:
                res["row"]["sibling"] = sib.get(res["rel"])
                pid = db.upsert_photo(conn, res["row"])
                db.replace_faces(conn, pid, res["faces"])
                stats["indexed"] += 1
            done += 1
            notify({"stage": "features", "done": done, "total": len(todo)})
        # RAW decodes hold ~10x the memory of a JPEG preview, so RAWs always run in a
        # smaller pool no matter what the caller asked for. Two sequential pools, one counter.
        std = [f for f in todo if not f.is_raw]; raw = [f for f in todo if f.is_raw]
        for group, cap in ((std, JPEG_WORKERS), (raw, RAW_WORKERS)):
            if not group:
                continue
            n = min(workers or cap, cap, len(group))
            with ctx.Pool(n) as pool:
                for res in pool.imap_unordered(process_one, [(str(root), f.rel, faces) for f in group], chunksize=2):
                    _store(res)
    if embed:
        from .embed import get_embedder
        pending = db.photos_missing_embed(conn)
        qh = {r[0]: r[1] for r in conn.execute("SELECT id, qhash FROM photos WHERE embed IS NULL AND status='ok'")}
        E = get_embedder()
        for i in range(0, len(pending), EMBED_BATCH):
            batch = pending[i:i + EMBED_BATCH]
            ims = [Image.open(idx / "thumbs" / f"{qh[pid]}.jpg") for pid, _ in batch]
            vecs = E.encode_images(ims)
            for (pid, _), v in zip(batch, vecs):
                db.set_embed(conn, pid, v)
            conn.commit()
            stats["embedded"] += len(batch)
            notify({"stage": "embed", "done": min(i + EMBED_BATCH, len(pending)), "total": len(pending)})
    stats["seconds"] = round(time.time() - t0, 1)
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('last_index', datetime('now'))"); conn.commit()
    notify({"stage": "done", "done": stats["total"], "total": stats["total"]})
    return stats
