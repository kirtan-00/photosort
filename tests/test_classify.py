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

def test_classify_and_store_persists_the_best_guess_for_an_other_photo(tmp_path):
    """A photo the gates sent to "other" still records its best real category and that category's
    probability, so the search can show it under that category as "less sure". A face-forced "people"
    photo is people with score 1.0: the detector decided, not the softmax."""
    conn = db.connect(tmp_path)
    rng = np.random.default_rng(0)
    junk = rng.normal(size=512).astype(np.float32); junk /= np.linalg.norm(junk)
    junk_id = db.upsert_photo(conn, _row("junk.jpg")); db.set_embed(conn, junk_id, junk)
    face_id = db.upsert_photo(conn, _row("face.jpg", n_faces=1)); db.set_embed(conn, face_id, junk)
    conn.commit()
    classify_and_store(tmp_path)
    conn2 = db.connect(tmp_path)
    r = conn2.execute("SELECT category, category_score, category_guess, category_guess_score FROM photos WHERE id=?", (junk_id,)).fetchone()
    assert r["category"] == FALLBACK
    assert r["category_guess"] in CATEGORIES and 0.0 < r["category_guess_score"] <= 1.0
    f = conn2.execute("SELECT category, category_score, category_guess, category_guess_score FROM photos WHERE id=?", (face_id,)).fetchone()
    assert (f["category"], f["category_score"], f["category_guess"], f["category_guess_score"]) == ("people", 1.0, "people", 1.0)

def test_categories_are_the_documentary_set_in_order():
    """The fixed list, in tile and export order (then "other"). Five of these came out of the first video
    shoot: interview, night, food and sky were what "other" was hiding, road grew a car interior."""
    assert list(CATEGORIES) == ["ocean", "beach", "people", "interview", "building", "road", "night", "food", "sky", "birds-animals"]
    assert all(prompts and all(isinstance(t, str) and t for t in prompts) for prompts in CATEGORIES.values())

def test_negative_prompts_describe_content_never_image_quality():
    """"a blurry or badly lit photograph" matched cinematic shallow-focus and flat log footage and became a
    sink (35 of the 49 "other" items on DAY-4). Negatives name content that is off the list, nothing about
    the picture's quality."""
    from photosort.classify import NEGATIVE_PROMPTS
    assert NEGATIVE_PROMPTS
    for p in NEGATIVE_PROMPTS:
        low = p.lower()
        assert not any(w in low for w in ("blur", "focus", "lit", "lighting", "dark", "noisy", "grainy")), p

def test_food_is_a_category_not_the_other_bin(tmp_path):
    """The calibration food image used to be expected as "other" because food was a negative prompt; food
    is a real category now, so a food-shaped embedding is filed under it, sure."""
    conn = db.connect(tmp_path)
    food_vec = get_embedder().encode_text(["a plate of food on a table"])[0]
    pid = db.upsert_photo(conn, _row("thali.jpg")); db.set_embed(conn, pid, food_vec)
    conn.commit()
    res = {r["id"]: r for r in classify(tmp_path)}
    assert res[pid]["category"] == "food" and res[pid]["score"] >= 0.5

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


# Drone shots from cameras that leave no DJI trace: a zero-shot aerial/ground pair, run after the categories

def test_aerial_probability_from_the_prompt_pair():
    """An embedding sitting on the aerial prompts' centroid is aerial; one on the ground centroid is not.
    The pair is its own two-way softmax, not part of the category one: a drone shot of a beach stays beach."""
    from photosort.classify import AERIAL_PROMPTS, GROUND_PROMPTS, AERIAL_MIN_PROB, aerial_probs, CATEGORIES, NEGATIVE_PROMPTS
    E = get_embedder()
    A = E.encode_text(AERIAL_PROMPTS); G = E.encode_text(GROUND_PROMPTS)
    a = A.mean(axis=0); a /= np.linalg.norm(a)
    g = G.mean(axis=0); g /= np.linalg.norm(g)
    p = aerial_probs(np.stack([a, g]), E)
    assert p.shape == (2,) and p[0] >= AERIAL_MIN_PROB and p[1] < AERIAL_MIN_PROB
    assert AERIAL_MIN_PROB == 0.7
    assert not any(t in prompts for prompts in CATEGORIES.values() for t in AERIAL_PROMPTS)
    assert not any(t in NEGATIVE_PROMPTS for t in AERIAL_PROMPTS)

def test_classify_and_store_flags_aerial_rows_without_touching_metadata_ones(tmp_path):
    """Rows with aerial=0 get the zero-shot verdict persisted; a row already 1 from the index (DJI metadata)
    is never re-decided, and a drone shot of a beach is still filed under beach."""
    from photosort.classify import AERIAL_PROMPTS
    conn = db.connect(tmp_path)
    E = get_embedder()
    aerial_vec = E.encode_text(AERIAL_PROMPTS).mean(axis=0); aerial_vec /= np.linalg.norm(aerial_vec)
    beach_vec = E.encode_text(["a sandy beach"])[0]
    top = db.upsert_photo(conn, _row("top.jpg")); db.set_embed(conn, top, aerial_vec)
    beach = db.upsert_photo(conn, _row("beach.jpg")); db.set_embed(conn, beach, beach_vec)
    dji = db.upsert_photo(conn, _row("DJI_0001.MP4", kind="video", aerial=1)); db.set_embed(conn, dji, beach_vec)
    conn.commit()
    classify_and_store(tmp_path)
    conn2 = db.connect(tmp_path)
    got = {r[0]: (r[1], r[2]) for r in conn2.execute("SELECT rel, aerial, category FROM photos")}
    assert got["top.jpg"][0] == 1
    assert got["beach.jpg"] == (0, "beach")
    assert got["DJI_0001.MP4"] == (1, "beach")
    assert db.aerial_count(conn2) == 2

# Discovered categories: k-means over the shoot, named from a fixed vocabulary

def test_vocab_is_large_lowercase_and_unique():
    from photosort.vocab import VOCAB
    assert 250 <= len(VOCAB) <= 400
    assert len(set(VOCAB)) == len(VOCAB)
    assert all(v == v.strip().lower() and v for v in VOCAB)
    for must in ("havan fire", "excavator", "priest", "garland", "scaffolding", "drone aerial view", "wedding couple",
                 "tea cup", "rangoli", "safety helmet", "sunset", "palm tree", "night street", "whiteboard"):
        assert must in VOCAB

def _unit(rng, n=1):
    v = rng.normal(size=(n, 512)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _clustered_shoot(tmp_path, sizes=(20, 20, 20), noise=0.05, seed=0):
    """sizes[i] photos around centre i (well separated random unit vectors). Returns (centres, [[ids of cluster i]])."""
    rng = np.random.default_rng(seed)
    centres = _unit(rng, len(sizes))
    conn = db.connect(tmp_path); groups = []
    n = 0
    for i, size in enumerate(sizes):
        ids = []
        for _ in range(size):
            v = centres[i] + rng.normal(scale=noise, size=512).astype(np.float32); v /= np.linalg.norm(v)
            pid = db.upsert_photo(conn, _row(f"c{i}_{n:03d}.jpg")); db.set_embed(conn, pid, v); ids.append(pid); n += 1
        groups.append(ids)
    conn.commit()
    return centres, groups

def _fake_vocab(centres, labels):
    """A stand-in for the CLIP text matrix: one row per label, the first len(centres) rows are the
    cluster centres themselves so cluster i is named labels[i] without loading the model."""
    rng = np.random.default_rng(99)
    T = np.vstack([centres, _unit(rng, len(labels) - len(centres))]).astype(np.float32)
    return lambda embedder: (list(labels), T)

def test_discover_default_k_grows_with_the_shoot():
    """sqrt(n / 6), clamped to 4..DISCOVER_MAX_K. On DAY-4 (270 items) k=4 gave four coarse clusters and k=8
    split out the vendor, the scooter, the shore and the panel discussion; DISCOVER_MIN_SIZE folding keeps
    the tiny ones away at the finer k."""
    from photosort.classify import discover_k
    assert discover_k(270) == 7
    assert discover_k(3677) == 24
    assert discover_k(30) == 4
    assert discover_k(16) == 4
    assert discover_k(100000) == 24

def test_discover_finds_the_clusters_names_them_and_is_deterministic(tmp_path, monkeypatch):
    from photosort import classify as cm
    centres, groups = _clustered_shoot(tmp_path)
    monkeypatch.setattr(cm, "_vocab_matrix", _fake_vocab(centres, ["excavator", "havan fire", "beach", "dog", "car", "sunset"]))
    out = cm.discover(tmp_path, k=3)
    assert len(out) == 3
    assert sorted(c["name"] for c in out) == ["beach", "excavator", "havan fire"]
    assert [sorted(c["photo_ids"]) for c in sorted(out, key=lambda c: c["name"])] == [sorted(groups[2]), sorted(groups[0]), sorted(groups[1])]
    assert all(c["size"] == 20 and 0.0 < c["score"] <= 1.0 and isinstance(c["id"], int) for c in out)
    assert all(len(c["photo_scores"]) == c["size"] and min(c["photo_scores"]) == 0.0 and max(c["photo_scores"]) == 1.0 for c in out)
    again = cm.discover(tmp_path, k=3)
    assert [(c["name"], c["photo_ids"], c["photo_scores"]) for c in again] == [(c["name"], c["photo_ids"], c["photo_scores"]) for c in out]

def test_discover_same_top_label_takes_the_next_unused_one(tmp_path, monkeypatch):
    """Two clusters whose best vocabulary label is the same word: the later (smaller) one moves to
    its next-best unused label, so every discovered category has a distinct name."""
    from photosort import classify as cm
    centres, groups = _clustered_shoot(tmp_path, sizes=(24, 16))
    rng = np.random.default_rng(5)
    # "crane" sits between the two centres (both clusters score it best); "excavator" is next-best for cluster 1 only
    between = centres[0] + centres[1]; between /= np.linalg.norm(between)
    near1 = centres[1] + 0.3 * rng.normal(size=512).astype(np.float32); near1 /= np.linalg.norm(near1)
    T = np.vstack([between, near1, _unit(rng, 2)]).astype(np.float32)
    monkeypatch.setattr(cm, "_vocab_matrix", lambda e: (["crane", "excavator", "dog", "cat"], T))
    out = cm.discover(tmp_path, k=2)
    assert [c["name"] for c in out] == ["crane", "excavator"]
    assert sorted(out[0]["photo_ids"]) == sorted(groups[0]) and sorted(out[1]["photo_ids"]) == sorted(groups[1])

def test_discover_folds_small_clusters_into_the_nearest_neighbour(tmp_path, monkeypatch):
    from photosort import classify as cm
    centres, groups = _clustered_shoot(tmp_path, sizes=(30, 25, 3))
    monkeypatch.setattr(cm, "_vocab_matrix", _fake_vocab(centres, ["a", "b", "c", "d"]))
    out = cm.discover(tmp_path, k=3)
    assert len(out) == 2 and [c["size"] for c in out] in ([33, 25], [30, 28])
    assert sum(c["size"] for c in out) == 58
    assert set(groups[2]) <= set(out[0]["photo_ids"]) | set(out[1]["photo_ids"])

def test_discover_and_store_writes_cluster_and_counts(tmp_path, monkeypatch):
    from photosort import classify as cm
    centres, groups = _clustered_shoot(tmp_path, sizes=(20, 12))
    monkeypatch.setattr(cm, "_vocab_matrix", _fake_vocab(centres, ["excavator", "havan fire", "dog"]))
    counts = cm.discover_and_store(tmp_path, k=2)
    assert counts == {"excavator": 20, "havan fire": 12}
    conn = db.connect(tmp_path)
    assert db.cluster_counts(conn) == {"excavator": 20, "havan fire": 12}
    rows = conn.execute("SELECT cluster, cluster_score FROM photos WHERE id IN (%s)" % ",".join(map(str, groups[1]))).fetchall()
    assert all(r["cluster"] == "havan fire" and 0.0 <= r["cluster_score"] <= 1.0 for r in rows)

def test_discover_needs_sixteen_embedded_photos(tmp_path, monkeypatch):
    from photosort import classify as cm
    centres, groups = _clustered_shoot(tmp_path, sizes=(8, 7))
    def boom(embedder):
        raise AssertionError("text scoring must not run on a shoot this small")
    monkeypatch.setattr(cm, "_vocab_matrix", boom)
    assert cm.discover(tmp_path) == []
    assert cm.discover_and_store(tmp_path) == {}
    conn = db.connect(tmp_path)
    assert db.cluster_counts(conn) == {}
    assert conn.execute("SELECT count(*) FROM photos WHERE cluster IS NOT NULL").fetchone()[0] == 0

def test_discover_with_the_real_embedder_names_from_the_vocabulary(tmp_path):
    """Three clusters built from CLIP text embeddings of vocabulary words get three distinct vocabulary
    names (the exact words are not asserted: a text embedding is only a proxy for a photo)."""
    from photosort import classify as cm
    from photosort.vocab import VOCAB
    E = get_embedder()
    centres = E.encode_text(["excavator", "havan fire", "a sandy beach"])
    rng = np.random.default_rng(3); conn = db.connect(tmp_path); groups = []
    for i in range(3):
        ids = []
        for j in range(12):
            v = centres[i] + rng.normal(scale=0.02, size=512).astype(np.float32); v /= np.linalg.norm(v)
            pid = db.upsert_photo(conn, _row(f"r{i}_{j}.jpg")); db.set_embed(conn, pid, v); ids.append(pid)
        groups.append(ids)
    conn.commit()
    out = cm.discover(tmp_path, k=3)
    names = [c["name"] for c in out]
    assert len(out) == 3 and len(set(names)) == 3 and all(n in VOCAB for n in names)
    assert sorted(sorted(c["photo_ids"]) for c in out) == sorted(sorted(g) for g in groups)
