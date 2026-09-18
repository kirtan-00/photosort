from photosort import db
from photosort.index import index_folder
from photosort.search import Index, Filters
from PIL import Image

def test_search_and_filters(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "sharp.jpg", kind="sharp"); make_image(tmp_path, "soft.jpg", kind="blurry")
    Image.new("RGB", (900, 600), (200, 30, 30)).save(tmp_path / "red.jpg")
    index_folder(tmp_path, faces=False, workers=1)
    ix = Index(tmp_path)
    allp = ix.search(); assert len(allp) == 3 and all("sharp_pct" in r for r in allp)
    top = ix.search(text="a red wall")[0]; assert top["rel"] == "red.jpg"
    # red.jpg is a flat colour so it scores 0 sharpness; only sharp.jpg is above the 60th percentile
    sharp_only = ix.search(filters=Filters(sharp_min_pct=60)); assert [r["rel"] for r in sharp_only] == ["sharp.jpg"]
    like = ix.search(image_id=top["id"]); assert like[0]["id"] == top["id"]
    assert ix.search(filters=Filters(faces="one")) == []

def test_category_filter(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg", kind="sharp"); make_image(tmp_path, "b.jpg", kind="blurry")
    index_folder(tmp_path, faces=False, workers=1)
    conn = db.connect(tmp_path)
    ids = [r[0] for r in conn.execute("SELECT id FROM photos ORDER BY rel")]
    conn.execute("UPDATE photos SET category='beach' WHERE id=?", (ids[0],))
    conn.commit()   # b.jpg's category stays NULL: never classified

    ix = Index(tmp_path)
    beach = ix.search(filters=Filters(category="beach"))
    assert [r["rel"] for r in beach] == ["a.jpg"] and beach[0]["category"] == "beach"
    assert ix.search(filters=Filters(category="road")) == []
    unclassified = ix.search(filters=Filters(category="unclassified"))
    assert [r["rel"] for r in unclassified] == ["b.jpg"]

def test_search_offset_pages_through_query(tmp_path):
    from conftest import make_image
    from photosort.index import index_folder
    from photosort.search import Index
    for i in range(5): make_image(tmp_path, f"p{i}.jpg", seed=i)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    ix = Index(tmp_path)
    everything = ix.query()
    assert [r["rel"] for r in everything] == [f"p{i}.jpg" for i in range(5)]
    assert [r["rel"] for r in ix.search(limit=2, offset=0)] == ["p0.jpg", "p1.jpg"]
    assert [r["rel"] for r in ix.search(limit=2, offset=4)] == ["p4.jpg"]
    assert ix.search(limit=2, offset=99) == []

def test_cluster_filter(tmp_path):
    """Filters(cluster=name) selects on the stored photos.cluster column (a discovered category)."""
    from conftest import make_image
    make_image(tmp_path, "a.jpg", kind="sharp"); make_image(tmp_path, "b.jpg", kind="blurry")
    index_folder(tmp_path, faces=False, workers=1)
    conn = db.connect(tmp_path)
    ids = [r[0] for r in conn.execute("SELECT id FROM photos ORDER BY rel")]
    conn.execute("UPDATE photos SET cluster='excavator', cluster_score=0.9 WHERE id=?", (ids[0],))
    conn.commit()
    ix = Index(tmp_path)
    hits = ix.search(filters=Filters(cluster="excavator"))
    assert [r["rel"] for r in hits] == ["a.jpg"] and hits[0]["cluster"] == "excavator"
    assert ix.search(filters=Filters(cluster="crane")) == []

def _shoot_with_scores(tmp_path, rows):
    """rows: (rel, category, category_score, category_guess, category_guess_score, cluster, cluster_score)."""
    from conftest import make_image
    for i, r in enumerate(rows): make_image(tmp_path, r[0], seed=i)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    conn = db.connect(tmp_path)
    for r in rows:
        conn.execute("UPDATE photos SET category=?, category_score=?, category_guess=?, category_guess_score=?, cluster=?, cluster_score=? WHERE rel=?",
                     r[1:] + (r[0],))
    conn.commit()

def test_category_search_puts_sure_first_then_less_sure_by_confidence(tmp_path):
    """category=X is the union of photos filed under X and photos filed under "other" whose best guess
    was X. Sure ones (score >= 0.5) come first in capture order; the rest follow sorted by confidence.
    A NULL score (classified before scores were stored) counts as sure; "other" itself is a bin, never
    less sure; a guess is never sure, whatever its probability: the gates already said no."""
    _shoot_with_scores(tmp_path, [
        ("a.jpg", "building", 0.9, "building", 0.9, None, None),
        ("b.jpg", "building", 0.41, "building", 0.41, None, None),
        ("c.jpg", "other", 0.3, "building", 0.3, None, None),
        ("d.jpg", "other", 0.45, "building", 0.45, None, None),
        ("e.jpg", "other", 0.6, "building", 0.6, None, None),
        ("f.jpg", "building", None, None, None, None, None),
        ("g.jpg", "other", 0.9, "road", 0.9, None, None),
    ])
    ix = Index(tmp_path)
    got = ix.search(filters=Filters(category="building"))
    assert [r["rel"] for r in got] == ["a.jpg", "f.jpg", "e.jpg", "d.jpg", "b.jpg", "c.jpg"]
    assert [r["sure"] for r in got] == [True, True, False, False, False, False]
    assert [r["confidence"] for r in got] == [0.9, 1.0, 0.6, 0.45, 0.41, 0.3]
    sure = ix.search(filters=Filters(category="building", sure_only=True))
    assert [r["rel"] for r in sure] == ["a.jpg", "f.jpg"]
    other = ix.search(filters=Filters(category="other"))
    assert [r["rel"] for r in other] == ["c.jpg", "d.jpg", "e.jpg", "g.jpg"] and all(r["sure"] for r in other)
    assert all(r["sure"] and r["confidence"] == 1.0 for r in ix.search())

def test_cluster_search_uses_cluster_score_for_the_divider(tmp_path):
    _shoot_with_scores(tmp_path, [
        ("a.jpg", None, None, None, None, "excavator", 0.2),
        ("b.jpg", None, None, None, None, "excavator", 1.0),
        ("c.jpg", None, None, None, None, "excavator", 0.49),
        ("d.jpg", None, None, None, None, "crane", 0.8),
    ])
    ix = Index(tmp_path)
    got = ix.search(filters=Filters(cluster="excavator"))
    assert [(r["rel"], r["sure"], r["confidence"]) for r in got] == [("b.jpg", True, 1.0), ("c.jpg", False, 0.49), ("a.jpg", False, 0.2)]
    assert [r["rel"] for r in ix.search(filters=Filters(cluster="excavator", sure_only=True))] == ["b.jpg"]
