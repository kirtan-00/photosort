from pathlib import Path
from photosort.walk import find_images, quick_hash

def test_finds_and_pairs(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    (tmp_path / "a.ARW").write_bytes(b"y" * 10)
    (tmp_path / "b.nef").write_bytes(b"z" * 10)
    (tmp_path / ".photosort").mkdir(); (tmp_path / ".photosort" / "t.jpg").write_bytes(b"q")
    (tmp_path / "notes.txt").write_text("no")
    files = find_images(tmp_path)
    rels = sorted(f.rel for f in files)
    assert rels == ["a.jpg", "b.nef"]
    a = next(f for f in files if f.rel == "a.jpg")
    assert a.sibling == "a.ARW" and a.is_raw is False
    assert next(f for f in files if f.rel == "b.nef").is_raw is True

def test_quick_hash_changes_with_content(tmp_path):
    p = tmp_path / "x.jpg"; p.write_bytes(b"a" * 200_000)
    h1 = quick_hash(p); p.write_bytes(b"a" * 199_999 + b"b"); h2 = quick_hash(p)
    assert h1 != h2 and len(h1) == 40

def test_unstatable_file_is_skipped(tmp_path):
    (tmp_path / "ok.jpg").write_bytes(b"x" * 10)
    (tmp_path / "gone.jpg").symlink_to(tmp_path / "does-not-exist.jpg")   # stat() raises OSError
    assert [f.rel for f in find_images(tmp_path)] == ["ok.jpg"]
