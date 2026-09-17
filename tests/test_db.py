import numpy as np
from photosort import db

def test_roundtrip(tmp_path):
    conn = db.connect(tmp_path)
    pid = db.upsert_photo(conn, dict(rel="a.jpg", size=1, mtime=1.0, qhash="h", sibling=None, width=10, height=10,
        taken_at=None, camera=None, phash="0"*16, sharp_tile=1.0, sharp_max=2.0, sharp_eye=None, sharp=1.0, n_faces=0, status="ok"))
    pid2 = db.upsert_photo(conn, dict(rel="a.jpg", size=2, mtime=2.0, qhash="h2", sibling=None, width=10, height=10,
        taken_at=None, camera=None, phash="0"*16, sharp_tile=1.0, sharp_max=2.0, sharp_eye=None, sharp=1.0, n_faces=0, status="ok"))
    assert pid == pid2
    assert db.known_files(conn) == {"a.jpg": (2, 2.0)}
    assert db.photos_missing_embed(conn) == [(pid, "a.jpg")]
    db.set_embed(conn, pid, np.ones(512, np.float32))
    ids, M = db.load_embeds(conn)
    assert ids.tolist() == [pid] and M.shape == (1, 512) and M.dtype == np.float32
    db.replace_faces(conn, pid, [dict(x=1,y=2,w=3,h=4,score=0.9,landmarks="[]",eye_sharp=5.0,embed=np.ones(128,np.float32).tobytes())])
    fids, pids, F = db.load_face_embeds(conn)
    assert F.shape == (1, 128) and pids.tolist() == [pid]
    db.mark_missing(conn, set())
    assert conn.execute("select status from photos").fetchone()[0] == "missing"
    assert db.known_files(conn) == {}
    d = db.index_dir(tmp_path)
    assert (d / "thumbs").is_dir() and (d / "grid").is_dir()
    assert not str(d.resolve()).startswith(str(tmp_path.resolve()))   # never inside the shoot
    assert not (tmp_path / ".photosort").exists()
