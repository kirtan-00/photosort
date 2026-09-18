import os
from pathlib import Path
from photosort.index import index_folder
from photosort.search import Index
from photosort.export import export_ids

def test_export_copy_default_symlink_and_csv(tmp_path):
    import os
    from conftest import make_image
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
    from conftest import make_image
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
    from conftest import make_image
    ids = _one_photo(tmp_path)
    make_image(tmp_path, "b.jpg", seed=2); index_folder(tmp_path, faces=False, workers=1, embed=False)
    (tmp_path / "a.jpg").unlink()
    index_folder(tmp_path, faces=False, workers=1, embed=False)   # marks a.jpg missing, b.jpg keeps the folder non-empty
    assert db.connect(tmp_path).execute("SELECT status FROM photos WHERE rel='a.jpg'").fetchone()[0] == "missing"
    out = export_ids(tmp_path, ids, "culled")                    # must not raise
    assert out.is_dir() and list(out.iterdir()) == []

def test_export_reports_progress_and_survives_a_bad_file(tmp_path):
    import os
    from conftest import make_image
    from photosort.export import export_bytes
    make_image(tmp_path, "a.jpg", seed=1); make_image(tmp_path, "b.jpg", seed=2)
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    ids = [r["id"] for r in Index(tmp_path).search()]
    assert export_bytes(tmp_path, ids) == (tmp_path / "a.jpg").stat().st_size + (tmp_path / "b.jpg").stat().st_size
    # b.jpg vanishes from the disk after indexing: copy must finish a.jpg and report one failure
    (tmp_path / "b.jpg").unlink()
    seen = []
    out = export_ids(tmp_path, ids, "partial", "copy", progress=seen.append)
    assert (out / "a.jpg").is_file() and not (out / "b.jpg").exists()
    assert seen[-1] == {"done": 2, "total": 2, "failed": 1}
    assert "b.jpg" in (out / "failed.txt").read_text()
    # links: os.symlink happily points at a missing file, so the missing source must be caught explicitly
    seen2 = []
    ln = export_ids(tmp_path, ids, "partial-links", "symlink", progress=seen2.append)
    assert (ln / "a.jpg").is_symlink() and not (ln / "b.jpg").exists() and not (ln / "b.jpg").is_symlink()
    assert seen2[-1] == {"done": 2, "total": 2, "failed": 1}
    assert "b.jpg" in (ln / "failed.txt").read_text()
    assert sorted(os.listdir(tmp_path)) == ["a.jpg"]


# export destination (another disk)

def test_export_dir_honours_an_explicit_base(tmp_path, tmp_path_factory):
    import pytest
    from photosort.export import export_dir
    ids = _one_photo(tmp_path)
    other = tmp_path_factory.mktemp("disk")
    assert export_dir(tmp_path, "sel", base=other) == other / tmp_path.resolve().name / "sel"
    with pytest.raises(ValueError, match="inside the source folder"):
        export_dir(tmp_path, "sel", base=tmp_path)
    with pytest.raises(ValueError, match="inside the source folder"):
        export_dir(tmp_path, "sel", base=tmp_path / "sub")
    with pytest.raises(ValueError):
        export_dir(tmp_path, "../../x", base=other)
    assert sorted(os.listdir(tmp_path)) == ["a.jpg"] and ids


def test_export_ids_copies_into_the_other_base(tmp_path, tmp_path_factory):
    from photosort.config import export_root
    ids = _one_photo(tmp_path)
    other = tmp_path_factory.mktemp("disk")
    before = sorted(os.listdir(export_root()))
    out = export_ids(tmp_path, ids, "sel", base=other)
    assert out == other / tmp_path.resolve().name / "sel"
    assert (out / "a.jpg").is_file() and not (out / "a.jpg").is_symlink()
    assert sorted(os.listdir(export_root())) == before          # nothing under the default
    assert sorted(os.listdir(tmp_path)) == ["a.jpg"]


# export selected categories, one folder each

def _two_category_shoot(tmp_path):
    """a.jpg (beach) with a RAW sibling a.ARW, b.jpg (ocean), c.jpg left unclassified."""
    from conftest import make_image
    from photosort import db
    make_image(tmp_path, "a.jpg", seed=1); make_image(tmp_path, "b.jpg", seed=2); make_image(tmp_path, "c.jpg", seed=3)
    (tmp_path / "a.ARW").write_bytes(b"raw bytes, never decoded")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    conn = db.connect(tmp_path)
    assert conn.execute("SELECT sibling FROM photos WHERE rel='a.jpg'").fetchone()[0] == "a.ARW"
    conn.execute("UPDATE photos SET category='beach' WHERE rel='a.jpg'")
    conn.execute("UPDATE photos SET category='ocean' WHERE rel='b.jpg'")
    conn.commit()
    return sorted(os.listdir(tmp_path))


def test_export_categories_one_folder_per_category(tmp_path, tmp_path_factory):
    from photosort.export import export_categories
    before = _two_category_shoot(tmp_path)
    disk = tmp_path_factory.mktemp("disk")
    seen = []
    out = export_categories(tmp_path, ["beach", "ocean"], base=disk, progress=seen.append)
    assert out == disk / tmp_path.resolve().name / "categories"
    assert sorted(p.name for p in (out / "beach").iterdir()) == ["a.jpg"]
    assert sorted(p.name for p in (out / "ocean").iterdir()) == ["b.jpg"]
    assert (out / "beach" / "a.jpg").is_file() and not (out / "beach" / "a.jpg").is_symlink()
    assert seen[-1] == {"done": 2, "total": 2, "failed": 0} and not (out / "failed.txt").exists()
    assert sorted(os.listdir(tmp_path)) == before


def test_export_categories_include_raw_and_links(tmp_path, tmp_path_factory):
    from photosort.export import export_categories
    before = _two_category_shoot(tmp_path)
    disk = tmp_path_factory.mktemp("disk")
    seen = []
    out = export_categories(tmp_path, ["beach"], mode="symlink", include_raw=True, base=disk, progress=seen.append)
    assert sorted(p.name for p in (out / "beach").iterdir()) == ["a.ARW", "a.jpg"]
    assert (out / "beach" / "a.ARW").is_symlink() and (out / "beach" / "a.ARW").resolve() == (tmp_path / "a.ARW").resolve()
    assert seen[-1] == {"done": 2, "total": 2, "failed": 0}       # the RAW sibling counts
    assert not (out / "ocean").exists()
    assert sorted(os.listdir(tmp_path)) == before


def test_export_categories_none_means_every_classified_one(tmp_path, tmp_path_factory):
    from photosort.export import export_categories
    before = _two_category_shoot(tmp_path)
    disk = tmp_path_factory.mktemp("disk")
    out = export_categories(tmp_path, None, base=disk)
    assert sorted(p.name for p in out.iterdir()) == ["beach", "ocean"]          # unclassified skipped
    out2 = export_categories(tmp_path, ["unclassified"], base=disk)
    assert sorted(p.name for p in (out2 / "unclassified").iterdir()) == ["c.jpg"]
    assert sorted(os.listdir(tmp_path)) == before


def test_export_categories_collision_and_failed_file(tmp_path, tmp_path_factory):
    from photosort import db
    from photosort.export import export_categories
    before = _two_category_shoot(tmp_path)
    disk = tmp_path_factory.mktemp("disk")
    a_id = db.connect(tmp_path).execute("SELECT id FROM photos WHERE rel='a.jpg'").fetchone()[0]
    out = export_categories(tmp_path, ["beach"], base=disk)
    assert sorted(p.name for p in (out / "beach").iterdir()) == ["a.jpg"]
    out = export_categories(tmp_path, ["beach"], base=disk)                    # second run: name taken
    assert sorted(p.name for p in (out / "beach").iterdir()) == sorted(["a.jpg", f"{a_id}_a.jpg"])
    (tmp_path / "b.jpg").unlink()                                               # ocean's only photo vanished
    seen = []
    export_categories(tmp_path, ["ocean"], base=disk, progress=seen.append)
    assert seen[-1] == {"done": 1, "total": 1, "failed": 1}
    assert "b.jpg" in (out / "failed.txt").read_text()
    (tmp_path / "b.jpg").write_bytes(b"")                                        # restore the listing for the check
    assert sorted(os.listdir(tmp_path)) == before


def test_categories_bytes_counts_the_sibling(tmp_path):
    from photosort.export import categories_bytes
    _two_category_shoot(tmp_path)
    a = (tmp_path / "a.jpg").stat().st_size; raw = (tmp_path / "a.ARW").stat().st_size
    b = (tmp_path / "b.jpg").stat().st_size
    assert categories_bytes(tmp_path, ["beach"], False) == a
    assert categories_bytes(tmp_path, ["beach"], True) == a + raw
    assert categories_bytes(tmp_path, None, True) == a + raw + b
    (tmp_path / "a.ARW").unlink()
    assert categories_bytes(tmp_path, ["beach"], True) == a                     # a sibling that fails to stat is skipped

def test_export_dir_refuses_a_base_above_the_shoot(tmp_path):
    import pytest
    from photosort.export import export_dir
    ids = _one_photo(tmp_path)
    with pytest.raises(ValueError, match="contains the source folder"):
        export_dir(tmp_path, "sel", base=tmp_path.parent)
    with pytest.raises(ValueError, match="contains the source folder"):
        export_dir(tmp_path, "sel", base=tmp_path.parent.parent)
    assert sorted(os.listdir(tmp_path)) == ["a.jpg"] and ids
