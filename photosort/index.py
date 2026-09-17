from __future__ import annotations
import json, multiprocessing as mp, time
from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image
from . import db
from .config import PREVIEW_EDGE, GRID_EDGE, THUMB_QUALITY, JPEG_WORKERS, RAW_WORKERS, EMBED_BATCH
from .walk import find_images, quick_hash
from .decode import load_preview, DecodeError
from .features import phash, exif_info, sharpness_tiles, to_gray

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
        out["row"] = dict(rel=rel, size=st.st_size, mtime=st.st_mtime, qhash=qh, sibling=None,
            width=info["width"] or im.width, height=info["height"] or im.height, taken_at=info["taken_at"],
            camera=info["camera"], phash=phash(im), sharp_tile=p90, sharp_max=mx, sharp_eye=eye,
            sharp=eye if eye is not None else p90, n_faces=len(faces), status="ok")
        out["faces"] = [dict(x=f.x, y=f.y, w=f.w, h=f.h, score=f.score, landmarks=json.dumps(f.landmarks.tolist()),
                             eye_sharp=f.eye_sharp, embed=f.embed.astype(np.float32).tobytes()) for f in faces]
    except (DecodeError, Exception) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out

def index_folder(root: Path, faces: bool = True, workers: int | None = None,
                 progress: Callable[[dict], None] | None = None, embed: bool = True) -> dict:
    t0 = time.time(); root = Path(root)
    notify = progress or (lambda d: None)
    conn = db.connect(root)
    notify({"stage": "scan", "done": 0, "total": 0})
    files = find_images(root)
    known = db.known_files(conn)
    todo = [f for f in files if known.get(f.rel) != (f.size, f.mtime)]
    stats = dict(total=len(files), skipped=len(files) - len(todo), indexed=0, errors=0, embedded=0)
    db.mark_missing(conn, {f.rel for f in files})
    if todo:
        n_raw = sum(f.is_raw for f in todo)
        workers = workers or (RAW_WORKERS if n_raw > len(todo) / 2 else JPEG_WORKERS)
        sib = {f.rel: f.sibling for f in todo}
        meta = {f.rel: (f.size, f.mtime) for f in todo}
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_one, [(str(root), f.rel, faces) for f in todo], chunksize=2), 1):
                if res["error"]:
                    stats["errors"] += 1
                    db.upsert_photo(conn, dict(rel=res["rel"], size=meta[res["rel"]][0], mtime=meta[res["rel"]][1], status="error", n_faces=0))
                else:
                    res["row"]["sibling"] = sib.get(res["rel"])
                    pid = db.upsert_photo(conn, res["row"])
                    db.replace_faces(conn, pid, res["faces"])
                    stats["indexed"] += 1
                notify({"stage": "features", "done": i, "total": len(todo)})
    if embed:
        from .embed import get_embedder
        pending = db.photos_missing_embed(conn)
        idx = db.index_dir(root)
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
