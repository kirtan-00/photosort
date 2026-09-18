from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from . import db
from .config import FACE_CLUSTER_EPS, FACE_MIN_SAMPLES, GROUP_MIN_FACES, FACE_MATCH_MIN_SIM, FACE_REF_MIN_EDGE

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

class ReferenceUnreadable(Exception):
    """The reference image exists but could not be decoded or scanned for faces."""

def _reference_faces(image_path: Path) -> list:
    """Faces in a reference image. One seam so tests can hand in synthetic faces."""
    from .decode import load_preview
    from .faces import FaceEngine
    return FaceEngine().detect(load_preview(image_path))

def find_by_reference(root: Path, image_path: Path, min_sim: float = FACE_MATCH_MIN_SIM) -> dict:
    """Match the largest face in image_path against every indexed face (not just cluster
    centroids, so it works before clustering and survives a bad cluster). One match per
    photo, the best face in it, sim >= min_sim, sorted by sim desc. person_id is the
    cluster of the single best face, if it has one."""
    try:   # only the decode/detect path; DB errors below stay loud
        faces = _reference_faces(image_path)
    except Exception as e:
        raise ReferenceUnreadable(str(e)) from e
    out = {"faces_in_reference": len(faces), "matches": [], "person_id": None}
    if not faces:
        return out
    ref = max(faces, key=lambda f: f.w * f.h)
    if max(ref.w, ref.h) < FACE_REF_MIN_EDGE:
        # A tiny "face" is usually a false positive; matching it floods the grid with strangers.
        out["reference_face_too_small"] = True
        return out
    q = ref.embed
    conn = db.connect(root); fids, pids, F = db.load_face_embeds(conn)
    if len(fids) == 0:
        return out
    sims = F @ q
    keep = np.where(sims >= min_sim)[0]
    keep = keep[np.argsort(-sims[keep], kind="stable")]
    best: dict[int, dict] = {}
    for i in keep:   # first sight of a photo is its best face
        pid = int(pids[i])
        if pid not in best:
            best[pid] = {"photo_id": pid, "sim": float(sims[i]), "face_id": int(fids[i])}
    out["matches"] = list(best.values())
    if out["matches"]:
        row = conn.execute("SELECT person_id FROM faces WHERE id=?", (out["matches"][0]["face_id"],)).fetchone()
        out["person_id"] = int(row[0]) if row and row[0] is not None else None
    return out

def assign_from_reference(root: Path, image_path: Path) -> int | None:
    faces = _reference_faces(image_path)
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

def export_people_ids(root: Path) -> dict[str, list[int]]:
    """Export folder name -> photo ids for the people/groups/solo bundle. One photo can appear
    under several folders (each person in it, plus groups or solo), and each appearance is a
    separate copy, so callers sizing the export sum over every folder. Two people whose names
    sanitise to the same segment share a folder rather than one silently dropping the other."""
    from .export import safe_segment
    root = Path(root); conn = db.connect(root); out: dict[str, list[int]] = {}
    for p in list_people(root):
        ids = [r[0] for r in conn.execute("SELECT DISTINCT photo_id FROM faces WHERE person_id=?", (p["id"],))]
        nm = safe_segment(p["name"] or f"person_{p['id']:02d}")
        out.setdefault(f"people/{nm}", []).extend(ids)
    out["groups"] = [r[0] for r in conn.execute("SELECT id FROM photos WHERE status='ok' AND n_faces>=?", (GROUP_MIN_FACES,))]
    out["solo"] = [r[0] for r in conn.execute("SELECT id FROM photos WHERE status='ok' AND n_faces=1")]
    return out

def export_people(root: Path, mode: str = "copy") -> Path:
    from .export import export_ids
    from .config import export_root
    root = Path(root)
    for name, ids in export_people_ids(root).items():
        export_ids(root, ids, name, mode)
    return export_root() / root.resolve().name
