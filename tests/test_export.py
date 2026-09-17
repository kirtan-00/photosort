from pathlib import Path
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

def _one_photo(tmp_path):
    from tests.conftest import make_image
    make_image(tmp_path, "a.jpg"); index_folder(tmp_path, faces=False, workers=1, embed=False)
    return [r["id"] for r in Index(tmp_path).search()]

def test_export_name_cannot_escape_export_root(tmp_path):
    import os, pytest
    from photosort.config import export_root
    ids = _one_photo(tmp_path)
    before = sorted(os.listdir(export_root()))
    for bad in ["../../x", "/tmp/x", "..", ".", "a//b", str(tmp_path / "inside")]:
        with pytest.raises(ValueError):
            export_ids(tmp_path, ids, bad)
    assert sorted(os.listdir(export_root())) == before          # nothing created anywhere
    assert sorted(os.listdir(tmp_path)) == ["a.jpg"]            # source untouched
    assert not (tmp_path.parent / "x").exists() and not Path("/tmp/x").exists()

def test_export_nested_name_is_sanitised_per_segment(tmp_path):
    from photosort.config import export_root
    ids = _one_photo(tmp_path)
    out = export_ids(tmp_path, ids, "people/Ar/ya")
    assert out.resolve().is_relative_to(export_root().resolve())
    assert out == export_root() / tmp_path.resolve().name / "people" / "Ar" / "ya"
    assert (out / "a.jpg").is_file()
    assert export_ids(tmp_path, ids, "").name == "export" and export_ids(tmp_path, ids, "  ").name == "export"
    out2 = export_ids(tmp_path, ids, "people/Ar:ya\\bad")
    assert out2.name == "Ar_ya_bad" and out2.parent.name == "people"

def test_export_skips_missing_photos(tmp_path):
    from photosort import db
    ids = _one_photo(tmp_path)
    (tmp_path / "a.jpg").unlink()
    index_folder(tmp_path, faces=False, workers=1, embed=False)   # marks a.jpg missing
    assert db.connect(tmp_path).execute("SELECT status FROM photos").fetchone()[0] == "missing"
    out = export_ids(tmp_path, ids, "culled")                    # must not raise
    assert out.is_dir() and list(out.iterdir()) == []
