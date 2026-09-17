from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from . import db
from .config import FACE_CLUSTER_EPS, FACE_MIN_SAMPLES, GROUP_MIN_FACES

def cluster_faces(root: Path, eps: float = FACE_CLUSTER_EPS, min_samples: int = FACE_MIN_SAMPLES) -> list[dict]:
    conn = db.connect(root)
    fids, pids, F = db.load_face_embeds(conn)
    # Names survive a recluster: remember which face belonged to a named person,
    # then hand each new cluster the majority name among its faces.
    old_names = {r[0]: r[1] for r in conn.execute(
        "SELECT f.id, pe.name FROM faces f JOIN people pe ON pe.id=f.person_id WHERE pe.name IS NOT NULL")}
    conn.execute("UPDATE faces SET person_id=NULL"); conn.execute("DELETE FROM people"); conn.commit()
    if len(fids) == 0:
        return []
    # working_memory caps the pairwise-distance chunks DBSCAN builds (MiB); the default
    # 1024 can spike RSS on a big shoot.
    with sklearn.config_context(working_memory=128):
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=1).fit_predict(F)
    for lab in sorted(set(labels) - {-1}):
        idx = np.where(labels == lab)[0]
        members = [int(f) for f in fids[idx]]
        n_photos = len(set(pids[idx].tolist()))
        best = conn.execute(f"SELECT id FROM faces WHERE id IN ({','.join('?'*len(members))}) ORDER BY score DESC LIMIT 1", members).fetchone()[0]
        votes = Counter(old_names[f] for f in members if f in old_names)
        name = votes.most_common(1)[0][0] if votes else None
        cur = conn.execute("INSERT INTO people(name, cover_face_id, n) VALUES(?, ?, ?)", (name, best, n_photos))
        conn.executemany("UPDATE faces SET person_id=? WHERE id=?", [(cur.lastrowid, f) for f in members])
    conn.commit()
    return list_people(root)

def list_people(root: Path) -> list[dict]:
    conn = db.connect(root)
    # Heal covers whose face row is gone (photo re-indexed or culled): fall back to the
    # best-scoring face still attached to that person.
    conn.execute("""UPDATE people SET cover_face_id = (SELECT id FROM faces WHERE person_id=people.id ORDER BY score DESC LIMIT 1)
                    WHERE cover_face_id IS NULL OR cover_face_id NOT IN (SELECT id FROM faces)""")
    conn.commit()
    rows = conn.execute("""SELECT pe.id, pe.name, pe.n, pe.cover_face_id, p.qhash, f.x, f.y, f.w, f.h
                           FROM people pe LEFT JOIN faces f ON f.id=pe.cover_face_id LEFT JOIN photos p ON p.id=f.photo_id
                           ORDER BY pe.n DESC, pe.id""").fetchall()
    return [dict(id=r[0], name=r[1], n=r[2], cover_face_id=r[3], cover_qhash=r[4],
                 cover_box=[r[5] or 0, r[6] or 0, r[7] or 0, r[8] or 0]) for r in rows]

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
