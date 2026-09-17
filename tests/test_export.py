from photosort.index import index_folder
from photosort.search import Index
from photosort.export import export_ids

def test_export_copy_default_symlink_and_csv(tmp_path):
    import os
    from tests.conftest import make_image
    from photosort.config import export_root
    make_image(tmp_path, "a.jpg"); index_folder(tmp_path, faces=False, workers=1, embed=False)
    ids = [r["id"] for r in Index(tmp_path).search()]
    out = export_ids(tmp_path, ids, "test")
    assert out == export_root() / tmp_path.resolve().name / "test"
    assert (out / "a.jpg").is_file() and not (out / "a.jpg").is_symlink()
    assert (out / "a.jpg").read_bytes() == (tmp_path / "a.jpg").read_bytes()
    ln = export_ids(tmp_path, ids, "links", "symlink")
    assert (ln / "a.jpg").is_symlink() and (ln / "a.jpg").resolve() == (tmp_path / "a.jpg").resolve()
    out2 = export_ids(tmp_path, ids, "csv", "csv")
    assert "a.jpg" in (out2 / "photos.csv").read_text()
    assert sorted(os.listdir(tmp_path)) == ["a.jpg"]   # source folder untouched
