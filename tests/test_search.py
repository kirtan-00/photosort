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
