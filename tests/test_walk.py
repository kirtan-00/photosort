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


def test_pairs_raw_in_sibling_folder(tmp_path):
    (tmp_path / "Day1" / "JPG").mkdir(parents=True); (tmp_path / "Day1" / "RAW").mkdir()
    (tmp_path / "Day1" / "JPG" / "DSC01.jpg").write_bytes(b"x")
    (tmp_path / "Day1" / "RAW" / "DSC01.ARW").write_bytes(b"y")
    (tmp_path / "Day2").mkdir()
    (tmp_path / "Day2" / "DSC02.jpg").write_bytes(b"x"); (tmp_path / "Day2" / "DSC02.jpg.bak").write_bytes(b"q")
    (tmp_path / "Day2" / "DSC03.nef").write_bytes(b"z")           # no JPEG anywhere: stays as a RAW entry
    files = find_images(tmp_path)
    rels = sorted(f.rel for f in files)
    assert rels == ["Day1/JPG/DSC01.jpg", "Day2/DSC02.jpg", "Day2/DSC03.nef"]
    assert next(f for f in files if f.rel == "Day1/JPG/DSC01.jpg").sibling == "Day1/RAW/DSC01.ARW"


def test_videos_are_found_and_never_paired_with_a_raw(tmp_path):
    from photosort.walk import find_images
    (tmp_path / "clip.MP4").write_bytes(b"v" * 10)
    (tmp_path / "clip.ARW").write_bytes(b"y" * 10)       # same stem as the video: the RAW stays its own entry
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    (tmp_path / "b.mov").write_bytes(b"w" * 10)
    (tmp_path / ".hidden.mp4").write_bytes(b"h")
    (tmp_path / "photosort-out").mkdir(); (tmp_path / "photosort-out" / "old.mp4").write_bytes(b"o")
    files = {f.rel: f for f in find_images(tmp_path)}
    assert sorted(files) == ["a.jpg", "b.mov", "clip.ARW", "clip.MP4"]
    assert files["clip.MP4"].is_video is True and files["clip.MP4"].is_raw is False and files["clip.MP4"].sibling is None
    assert files["clip.ARW"].is_raw is True and files["clip.ARW"].sibling is None
    assert files["a.jpg"].is_video is False and files["b.mov"].is_video is True
