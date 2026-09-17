import numpy as np
from photosort.index import index_folder
from photosort import db

def test_index_then_incremental(tmp_path):
    from tests.conftest import make_image
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
