from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from . import db
from .config import FACE_CLUSTER_EPS, FACE_MIN_SAMPLES, GROUP_MIN_FACES

def cluster_faces(root: Path, eps: float = FACE_CLUSTER_EPS, min_samples: int = FACE_MIN_SAMPLES) -> list[dict]:
    conn = db.connect(root)
    fids, pids, F = db.load_face_embeds(conn)
    conn.execute("UPDATE faces SET person_id=NULL"); conn.execute("DELETE FROM people"); conn.commit()
    if len(fids) == 0:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=1).fit_predict(F)
    for lab in sorted(set(labels) - {-1}):
        idx = np.where(labels == lab)[0]
        n_photos = len(set(pids[idx].tolist()))
        best = conn.execute(f"SELECT id FROM faces WHERE id IN ({','.join('?'*len(idx))}) ORDER BY score DESC LIMIT 1", fids[idx].tolist()).fetchone()[0]
        cur = conn.execute("INSERT INTO people(name, cover_face_id, n) VALUES(NULL, ?, ?)", (best, n_photos))
        conn.executemany("UPDATE faces SET person_id=? WHERE id=?", [(cur.lastrowid, int(f)) for f in fids[idx]])
    conn.commit()
    return list_people(root)

def list_people(root: Path) -> list[dict]:
    conn = db.connect(root)
    rows = conn.execute("""SELECT pe.id, pe.name, pe.n, pe.cover_face_id, p.qhash, f.x, f.y, f.w, f.h
                           FROM people pe JOIN faces f ON f.id=pe.cover_face_id JOIN photos p ON p.id=f.photo_id
                           ORDER BY pe.n DESC, pe.id""").fetchall()
    return [dict(id=r[0], name=r[1], n=r[2], cover_face_id=r[3], cover_qhash=r[4], cover_box=[r[5], r[6], r[7], r[8]]) for r in rows]

def name_person(root: Path, person_id: int, name: str) -> None:
    conn = db.connect(root); conn.execute("UPDATE people SET name=? WHERE id=?", (name.strip() or None, person_id)); conn.commit()

def assign_from_reference(root: Path, image_path: Path) -> int | None:
    from .decode import load_preview
    from .faces import FaceEngine
    faces = FaceEngine().detect(load_preview(image_path))
    if not faces:
        return None
    q = max(faces, key=lambda f: f.w * f.h).embed
    conn = db.connect(root); fids, pids, F = db.load_face_embeds(conn)
    labels = np.array([r[0] or -1 for r in conn.execute(
        "SELECT f.person_id FROM faces f JOIN photos p ON p.id=f.photo_id WHERE p.status='ok' ORDER BY f.id")])
    best, best_sim = None, 0.5
    for lab in set(labels.tolist()) - {-1}:
        c = F[labels == lab].mean(axis=0); c /= np.linalg.norm(c)
        s = float(c @ q)
        if s > best_sim: best, best_sim = lab, s
    return best

def export_people(root: Path, mode: str = "copy") -> Path:
    from .export import export_ids, safe_segment
    from .config import export_root
    root = Path(root); conn = db.connect(root)
    for p in list_people(root):
        ids = [r[0] for r in conn.execute("SELECT DISTINCT photo_id FROM faces WHERE person_id=?", (p["id"],))]
        nm = safe_segment(p["name"] or f"person_{p['id']:02d}")
        export_ids(root, ids, f"people/{nm}", mode)
    groups = [r[0] for r in conn.execute("SELECT id FROM photos WHERE status='ok' AND n_faces>=?", (GROUP_MIN_FACES,))]
    solo = [r[0] for r in conn.execute("SELECT id FROM photos WHERE status='ok' AND n_faces=1")]
    export_ids(root, groups, "groups", mode); export_ids(root, solo, "solo", mode)
    return export_root() / root.resolve().name
