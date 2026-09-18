from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
from .config import DB_NAME, EMBED_DIM, app_home, shoot_slug

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos(
  id INTEGER PRIMARY KEY, rel TEXT UNIQUE NOT NULL, size INTEGER, mtime REAL, qhash TEXT,
  sibling TEXT, width INTEGER, height INTEGER, taken_at TEXT, camera TEXT, phash TEXT,
  sharp_tile REAL, sharp_max REAL, sharp_eye REAL, sharp REAL, n_faces INTEGER DEFAULT 0,
  embed BLOB, status TEXT DEFAULT 'ok', indexed_at TEXT DEFAULT (datetime('now')),
  category TEXT, category_score REAL);
CREATE TABLE IF NOT EXISTS faces(
  id INTEGER PRIMARY KEY, photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  x INTEGER, y INTEGER, w INTEGER, h INTEGER, score REAL, landmarks TEXT, eye_sharp REAL,
  embed BLOB, person_id INTEGER);
CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY, name TEXT, cover_face_id INTEGER, n INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS ref_faces(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, embed BLOB NOT NULL, source TEXT,
  created_at TEXT DEFAULT (datetime('now')));
CREATE INDEX IF NOT EXISTS faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
"""

PHOTO_COLS = ["rel","size","mtime","qhash","sibling","width","height","taken_at","camera","phash",
              "sharp_tile","sharp_max","sharp_eye","sharp","n_faces","status"]

def index_dir(root: Path) -> Path:
    d = app_home() / shoot_slug(root)
    (d / "thumbs").mkdir(parents=True, exist_ok=True)
    (d / "grid").mkdir(parents=True, exist_ok=True)
    return d

def connect(root: Path) -> sqlite3.Connection:
    d = index_dir(root)
    conn = sqlite3.connect(d / DB_NAME, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    # Idempotent migration: CREATE TABLE IF NOT EXISTS above only takes effect on a brand-new DB, so a
    # photos table created before category/category_score existed needs them added by hand.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(photos)")}
    if "category" not in cols:
        conn.execute("ALTER TABLE photos ADD COLUMN category TEXT")
    if "category_score" not in cols:
        conn.execute("ALTER TABLE photos ADD COLUMN category_score REAL")
    conn.commit()
    return conn

def upsert_photo(conn, row: dict) -> int:
    cols = ",".join(PHOTO_COLS); ph = ",".join("?" * len(PHOTO_COLS))
    upd = ",".join(f"{c}=excluded.{c}" for c in PHOTO_COLS if c != "rel")
    # embed is cleared so a changed file gets re-embedded; category/category_score are cleared with it
    # since they were derived from that embedding and would otherwise show a stale label.
    conn.execute(f"INSERT INTO photos({cols}) VALUES({ph}) ON CONFLICT(rel) DO UPDATE SET {upd}, embed=NULL, category=NULL, category_score=NULL, indexed_at=datetime('now')",
                 [row.get(c) for c in PHOTO_COLS])
    conn.commit()
    return conn.execute("SELECT id FROM photos WHERE rel=?", (row["rel"],)).fetchone()[0]

def mark_error(conn, rel: str, size: int, mtime: float) -> None:
    """Flag a file that could not be read this pass. Only status/size/mtime move: qhash, embed,
    category and faces from an earlier good pass stay, so a retry after a disk hiccup does not
    have to re-decode and re-embed. A file never seen before gets a minimal error row."""
    cur = conn.execute("UPDATE photos SET status='error', size=?, mtime=?, indexed_at=datetime('now') WHERE rel=?",
                       (size, mtime, rel))
    if cur.rowcount == 0:
        conn.execute("INSERT INTO photos(rel, size, mtime, status, n_faces) VALUES(?, ?, ?, 'error', 0)", (rel, size, mtime))
    conn.commit()

def replace_faces(conn, photo_id: int, faces: list[dict]) -> None:
    conn.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))
    conn.executemany("INSERT INTO faces(photo_id,x,y,w,h,score,landmarks,eye_sharp,embed) VALUES(?,?,?,?,?,?,?,?,?)",
        [(photo_id, f["x"], f["y"], f["w"], f["h"], f["score"], f["landmarks"], f["eye_sharp"], f["embed"]) for f in faces])
    conn.commit()

def set_embed(conn, photo_id: int, vec: np.ndarray) -> None:
    conn.execute("UPDATE photos SET embed=? WHERE id=?", (np.asarray(vec, np.float16).tobytes(), photo_id))

def photos_missing_embed(conn) -> list[tuple[int, str]]:
    return [(r[0], r[1]) for r in conn.execute("SELECT id, rel FROM photos WHERE embed IS NULL AND status='ok' ORDER BY id")]

def load_embeds(conn):
    rows = conn.execute("SELECT id, embed FROM photos WHERE embed IS NOT NULL AND status='ok' ORDER BY id").fetchall()
    if not rows:
        return np.zeros(0, np.int64), np.zeros((0, EMBED_DIM), np.float32)
    ids = np.array([r[0] for r in rows], np.int64)
    M = np.stack([np.frombuffer(r[1], np.float16).astype(np.float32) for r in rows])
    return ids, M

def load_face_embeds(conn):
    rows = conn.execute("SELECT f.id, f.photo_id, f.embed FROM faces f JOIN photos p ON p.id=f.photo_id WHERE p.status='ok' ORDER BY f.id").fetchall()
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 128), np.float32)
    return (np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
            np.stack([np.frombuffer(r[2], np.float32) for r in rows]))

def add_reference(conn, name: str, embed: np.ndarray, source: str) -> int:
    """One saved reference face (a named person). Several rows may share a name; matching
    takes the best of them. The embed is stored float32 like the faces table."""
    cur = conn.execute("INSERT INTO ref_faces(name, embed, source) VALUES(?, ?, ?)",
                       (name, np.asarray(embed, np.float32).tobytes(), source))
    conn.commit()
    return int(cur.lastrowid)

def list_references(conn) -> list[dict]:
    rows = conn.execute("SELECT id, name, source, created_at FROM ref_faces ORDER BY id").fetchall()
    return [dict(id=r[0], name=r[1], source=r[2], created_at=r[3]) for r in rows]

def load_reference_embeds(conn):
    """(ids, names, R) for every saved reference, R float32 (n, 128), rows in id order."""
    rows = conn.execute("SELECT id, name, embed FROM ref_faces ORDER BY id").fetchall()
    if not rows:
        return np.zeros(0, np.int64), [], np.zeros((0, 128), np.float32)
    return (np.array([r[0] for r in rows], np.int64), [r[1] for r in rows],
            np.stack([np.frombuffer(r[2], np.float32) for r in rows]))

def delete_reference(conn, ref_id: int) -> bool:
    cur = conn.execute("DELETE FROM ref_faces WHERE id=?", (ref_id,)); conn.commit()
    return cur.rowcount > 0

def rename_reference(conn, name_old: str, name_new: str) -> int:
    """Every reference saved under name_old now answers to name_new. Returns the rows moved."""
    cur = conn.execute("UPDATE ref_faces SET name=? WHERE name=?", (name_new, name_old)); conn.commit()
    return cur.rowcount

def known_files(conn, retry_errors: bool = False) -> dict[str, tuple[int, float]]:
    """rel -> (size, mtime) for rows that count as already indexed. Missing rows are excluded here
    and handled by missing_files() so a returning file can be restored without a re-decode."""
    q = "SELECT rel, size, mtime FROM photos WHERE status != 'missing'"
    if retry_errors:
        q += " AND status != 'error'"
    return {r[0]: (r[1], r[2]) for r in conn.execute(q)}

def missing_files(conn) -> dict[str, tuple[int, float, str]]:
    """rel -> (size, mtime, qhash) for rows the last scan could not find."""
    return {r[0]: (r[1], r[2], r[3]) for r in conn.execute("SELECT rel, size, mtime, qhash FROM photos WHERE status='missing'")}

def restore_missing(conn, rels: list[str]) -> None:
    conn.executemany("UPDATE photos SET status='ok' WHERE rel=? AND status='missing'", [(r,) for r in rels])
    conn.commit()

def photos_without_faces(conn) -> set[str]:
    """Photos indexed with faces off (n_faces NULL). They need a second pass when faces are wanted."""
    return {r[0] for r in conn.execute("SELECT rel FROM photos WHERE n_faces IS NULL AND status='ok'")}

def category_counts(conn) -> dict[str, int]:
    """category -> count for status='ok' photos; NULL (never classified) is reported as 'unclassified'."""
    rows = conn.execute("SELECT COALESCE(category, 'unclassified') AS c, COUNT(*) FROM photos WHERE status='ok' GROUP BY c").fetchall()
    return {r[0]: r[1] for r in rows}

def mark_missing(conn, present: set[str]) -> None:
    for (rel,) in conn.execute("SELECT rel FROM photos WHERE status='ok'").fetchall():
        if rel not in present:
            conn.execute("UPDATE photos SET status='missing' WHERE rel=?", (rel,))
    conn.commit()
