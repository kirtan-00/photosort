from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from . import db
from .config import GROUP_MIN_FACES

@dataclass
class Filters:
    sharp_min_pct: float | None = None
    faces: str | None = None
    person_id: int | None = None
    taken_from: str | None = None
    taken_to: str | None = None
    category: str | None = None

class Index:
    def __init__(self, root: Path):
        self.root = Path(root); self.refresh()

    def refresh(self):
        # Open a connection local to the calling thread: sqlite3 connections
        # (check_same_thread=True by default) can't cross threads, and this
        # Index is often built on one thread (app startup) then queried from
        # FastAPI's worker threadpool.
        conn = db.connect(self.root)
        rows = conn.execute("SELECT id, rel, qhash, sharp, n_faces, taken_at, width, height, category FROM photos WHERE status='ok' ORDER BY id").fetchall()
        self.photos = {r["id"]: dict(r) for r in rows}
        sharp = np.array([r["sharp"] or 0.0 for r in rows], float)
        order = sharp.argsort().argsort()
        for r, rank in zip(rows, order):
            self.photos[r["id"]]["sharp_pct"] = float(rank) / max(len(rows) - 1, 1) * 100
        self.ids, self.M = db.load_embeds(conn)
        self.pos = {pid: i for i, pid in enumerate(self.ids.tolist())}

    def _person_photo_ids(self, person_id: int) -> set[int]:
        conn = db.connect(self.root)
        return {r[0] for r in conn.execute("SELECT DISTINCT photo_id FROM faces WHERE person_id=?", (person_id,))}

    def _passes(self, p: dict, f: Filters, person_ids: set[int] | None) -> bool:
        if f.sharp_min_pct is not None and p["sharp_pct"] < f.sharp_min_pct: return False
        n = p["n_faces"] or 0
        if f.faces == "none" and n != 0: return False
        if f.faces == "one" and n != 1: return False
        if f.faces == "two" and n != 2: return False
        if f.faces == "group" and n < GROUP_MIN_FACES: return False
        if person_ids is not None and p["id"] not in person_ids: return False
        if f.category is not None:
            # "unclassified" mirrors db.category_counts' label for a NULL category (never classified).
            if f.category == "unclassified":
                if p["category"] is not None: return False
            elif p["category"] != f.category: return False
        t = p["taken_at"] or ""
        if f.taken_from and t < f.taken_from: return False
        if f.taken_to and t > f.taken_to: return False
        return True

    def query(self, text: str | None = None, image_id: int | None = None, filters: Filters = Filters()) -> list[dict]:
        """Every photo that passes the filters, sorted by similarity (text or image query) or by capture time."""
        person_ids = self._person_photo_ids(filters.person_id) if filters.person_id is not None else None
        cands = [p for p in self.photos.values() if self._passes(p, filters, person_ids)]
        if text or image_id is not None:
            if image_id is not None:
                i = self.pos.get(image_id)
                if i is None:
                    raise LookupError(f"no embedding for photo {image_id}")
                q = self.M[i]
            else:
                from .embed import get_embedder
                q = get_embedder().encode_text([text])[0]
            scores = self.M @ q
            for p in cands:
                i = self.pos.get(p["id"]); p["score"] = float(scores[i]) if i is not None else -1.0
            cands.sort(key=lambda p: -p["score"])
        else:
            for p in cands: p["score"] = 0.0
            cands.sort(key=lambda p: ((p["taken_at"] or "~"), p["rel"]))
        return cands

    def search(self, text: str | None = None, image_id: int | None = None, filters: Filters = Filters(),
               limit: int = 200, offset: int = 0) -> list[dict]:
        return [dict(p) for p in self.query(text, image_id, filters)[offset:offset + limit]]
