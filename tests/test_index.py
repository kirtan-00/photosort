import numpy as np
from photosort.index import index_folder
from photosort import db

def test_index_then_incremental(tmp_path):
    from conftest import make_image
    for i in range(6):
        make_image(tmp_path, f"p{i}.jpg", kind="sharp" if i < 4 else "blurry", seed=i)
    (tmp_path / "junk.jpg").write_bytes(b"nope")
    s1 = index_folder(tmp_path, faces=True, workers=2)
    assert s1["indexed"] == 6 and s1["errors"] == 1 and s1["embedded"] == 6
    conn = db.connect(tmp_path)
    rows = conn.execute("SELECT rel, sharp, qhash FROM photos WHERE status='ok' ORDER BY rel").fetchall()
    assert len(rows) == 6
    idx = db.index_dir(tmp_path)
    assert (idx / "thumbs" / f"{rows[0]['qhash']}.jpg").exists()
    assert (idx / "grid" / f"{rows[0]['qhash']}.jpg").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([f"p{i}.jpg" for i in range(6)] + ["junk.jpg"])   # nothing written into the shoot
    sharp = [r["sharp"] for r in rows]
    assert min(sharp[:4]) > max(sharp[4:])
    ids, M = db.load_embeds(conn); assert M.shape == (6, 512)
    s2 = index_folder(tmp_path, faces=True, workers=2)
    assert s2["skipped"] == 7 and s2["indexed"] == 0 and s2["errors"] == 0

def test_missing_then_restored(tmp_path):
    from conftest import make_image
    import os, shutil
    p = make_image(tmp_path, "a.jpg")
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    st = p.stat(); backup = tmp_path.parent / "a_backup.jpg"; shutil.copy2(p, backup); p.unlink()
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    conn = db.connect(tmp_path)
    assert conn.execute("SELECT status FROM photos WHERE rel='a.jpg'").fetchone()[0] == "missing"
    shutil.copy2(backup, p); os.utime(p, (st.st_atime, st.st_mtime))
    s = index_folder(tmp_path, faces=False, workers=1, embed=False)
    assert s["indexed"] == 1
    assert conn.execute("SELECT status FROM photos WHERE rel='a.jpg'").fetchone()[0] == "ok"

def test_no_faces_then_faces_reprocesses(tmp_path):
    from conftest import make_image
    for i in range(3):
        make_image(tmp_path, f"p{i}.jpg", seed=i)
    s1 = index_folder(tmp_path, faces=False, workers=1, embed=False)
    assert s1["indexed"] == 3
    conn = db.connect(tmp_path)
    assert [r[0] for r in conn.execute("SELECT n_faces FROM photos")] == [None, None, None]
    assert db.photos_without_faces(conn) == {"p0.jpg", "p1.jpg", "p2.jpg"}
    s2 = index_folder(tmp_path, faces=True, workers=1, embed=False)
    assert s2["indexed"] == 3 and s2["skipped"] == 0
    assert [r[0] for r in conn.execute("SELECT n_faces FROM photos")] == [0, 0, 0]
    s3 = index_folder(tmp_path, faces=True, workers=1, embed=False)
    assert s3["indexed"] == 0 and s3["skipped"] == 3

def test_retry_errors_reprocesses_error_rows(tmp_path):
    from conftest import make_image
    bad = tmp_path / "bad.jpg"; bad.write_bytes(b"nope")
    s1 = index_folder(tmp_path, faces=False, workers=1, embed=False)
    assert s1["errors"] == 1
    s2 = index_folder(tmp_path, faces=False, workers=1, embed=False)
    assert s2["errors"] == 0 and s2["skipped"] == 1          # error rows are not retried by default
    s3 = index_folder(tmp_path, faces=False, workers=1, embed=False, retry_errors=True)
    assert s3["errors"] == 1 and s3["skipped"] == 0          # still broken, but it was tried again
    st = bad.stat(); make_image(tmp_path, "bad.jpg", seed=9); import os; os.utime(bad, (st.st_atime, st.st_mtime + 5))
    s4 = index_folder(tmp_path, faces=False, workers=1, embed=False)
    assert s4["indexed"] == 1 and s4["errors"] == 0

def test_raw_and_std_run_in_separate_pools_with_one_counter(tmp_path):
    from conftest import make_image
    make_image(tmp_path, "a.jpg", seed=1); make_image(tmp_path, "b.jpg", seed=2)
    (tmp_path / "c.nef").write_bytes(b"not a raw file")
    seen = []
    s = index_folder(tmp_path, faces=False, workers=8, embed=False, progress=lambda d: seen.append(dict(d)))
    feats = [d for d in seen if d["stage"] == "features"]
    assert [d["done"] for d in feats] == [1, 2, 3] and all(d["total"] == 3 for d in feats)
    assert s["indexed"] == 2 and s["errors"] == 1

def test_faces_true_without_models_fails_fast(tmp_path, monkeypatch):
    import pytest
    from conftest import make_image
    make_image(tmp_path, "a.jpg")
    monkeypatch.setattr("photosort.index.YUNET_PATH", tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError):
        index_folder(tmp_path, faces=True, workers=1, embed=False)
