import os
import numpy as np
from photosort import db
from photosort.embed import get_embedder
from photosort.classify import classify, classify_and_store, write_manifest, CATEGORIES, FALLBACK

BASE_ROW = dict(size=1, mtime=1.0, qhash="h", sibling=None, width=10, height=10, taken_at=None,
                 camera=None, phash="0" * 16, sharp_tile=1.0, sharp_max=2.0, sharp_eye=None, sharp=1.0,
                 n_faces=0, status="ok")

def _row(rel, **kw):
    r = dict(BASE_ROW); r.update(rel=rel); r.update(kw)
    return r

def test_classify_categories_from_text_embeddings(tmp_path):
    """A row whose embedding matches a category's text prompts is put in that category; a row with
    faces is always 'people' regardless of what it looks like; a row with no signal at all is 'other'."""
    conn = db.connect(tmp_path)
    beach_vec = get_embedder().encode_text(["a sandy beach"])[0]

    beach_id = db.upsert_photo(conn, _row("beach.jpg"))
    db.set_embed(conn, beach_id, beach_vec)

    people_id = db.upsert_photo(conn, _row("face.jpg", n_faces=2))
    db.set_embed(conn, people_id, beach_vec)   # looks exactly like the beach photo except for faces

    rng = np.random.default_rng(0)
    junk_vec = rng.normal(size=512).astype(np.float32)
    junk_vec /= np.linalg.norm(junk_vec)
    junk_id = db.upsert_photo(conn, _row("junk.jpg"))
    db.set_embed(conn, junk_id, junk_vec)
    conn.commit()

    results = {r["id"]: r for r in classify(tmp_path)}
    assert results[beach_id]["category"] == "beach"
    assert results[people_id]["category"] == "people"   # face rule overrides the (identical) beach embedding
    assert results[junk_id]["category"] == FALLBACK

def test_classify_and_store_persists_and_counts(tmp_path):
    conn = db.connect(tmp_path)
    beach_vec = get_embedder().encode_text(["a sandy beach"])[0]
    pid = db.upsert_photo(conn, _row("beach.jpg"))
    db.set_embed(conn, pid, beach_vec)
    conn.commit()

    counts = classify_and_store(tmp_path)
    assert counts == {"beach": 1}

    conn2 = db.connect(tmp_path)
    row = conn2.execute("SELECT category, category_score FROM photos WHERE id=?", (pid,)).fetchone()
    assert row["category"] == "beach"
    assert row["category_score"] is not None and 0.0 < row["category_score"] <= 1.0
    assert db.category_counts(conn2) == {"beach": 1}

def test_category_counts_reports_unclassified(tmp_path):
    conn = db.connect(tmp_path)
    db.upsert_photo(conn, _row("never_classified.jpg"))
    assert db.category_counts(conn) == {"unclassified": 1}

def test_write_manifest_symlinks_not_copies_and_nothing_under_root(tmp_path):
    """A shoot with per-day/per-location subfolders can repeat a filename across folders, so
    write_manifest names each symlink after the full relative path ('/' -> '__'), and it must
    never touch the (read-only) shoot root: only the export dir gets new files."""
    conn = db.connect(tmp_path)
    beach_vec = get_embedder().encode_text(["a sandy beach"])[0]
    pid = db.upsert_photo(conn, _row("day2/beach/IMG_0001.jpg", sibling="day2/beach/IMG_0001.raw"))
    db.set_embed(conn, pid, beach_vec)
    conn.commit()

    results = classify(tmp_path)
    base = write_manifest(tmp_path, results)

    link = base / "beach" / "day2__beach__IMG_0001.jpg"
    raw_link = base / "beach" / "day2__beach__IMG_0001.raw"
    assert link.is_symlink() and os.path.realpath(link) == str((tmp_path / "day2/beach/IMG_0001.jpg").resolve())
    assert raw_link.is_symlink()
    assert not link.is_file()   # dangling: the source file was never actually created on disk

    assert not base.resolve().is_relative_to(tmp_path.resolve())   # export dir lives outside the shoot
    assert list(tmp_path.rglob("*")) == []   # nothing was ever written under the (read-only) shoot root

def test_classify_and_store_labels_segments_in_the_same_pass(tmp_path):
    """A video row is categorised like a photo (whole-clip embedding); each of its segments gets its own
    category from its own embedding. Counts stay per photo/video row so the tab agrees with /api/categories."""
    conn = db.connect(tmp_path)
    E = get_embedder()
    beach_vec = E.encode_text(["a sandy beach"])[0]
    road_vec = E.encode_text(["a road with vehicles"])[0]
    vid = db.upsert_photo(conn, _row("clip.mp4", kind="video", duration=4.0))
    db.set_embed(conn, vid, beach_vec)
    db.replace_segments(conn, vid, [dict(idx=0, start=0.0, end=2.0, frame="h_0.jpg"), dict(idx=1, start=2.0, end=4.0, frame="h_1.jpg")])
    segs = conn.execute("SELECT id FROM segments WHERE photo_id=? ORDER BY idx", (vid,)).fetchall()
    db.set_segment_embed(conn, segs[0]["id"], beach_vec)
    db.set_segment_embed(conn, segs[1]["id"], road_vec)
    conn.commit()
    counts = classify_and_store(tmp_path)
    assert counts == {"beach": 1}
    conn2 = db.connect(tmp_path)
    assert conn2.execute("SELECT category FROM photos WHERE id=?", (vid,)).fetchone()[0] == "beach"
    rows = conn2.execute("SELECT category, category_score FROM segments WHERE photo_id=? ORDER BY idx", (vid,)).fetchall()
    assert [r["category"] for r in rows] == ["beach", "road"]
    assert all(r["category_score"] is not None and 0.0 < r["category_score"] <= 1.0 for r in rows)
