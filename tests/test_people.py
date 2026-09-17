import numpy as np
from photosort import db
from photosort.people import cluster_faces, name_person, list_people, export_people

def _fake_shoot(tmp_path, n_people=3, per=4):
    conn = db.connect(tmp_path); rng = np.random.default_rng(1)
    centers = rng.normal(size=(n_people, 128)); centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    for k in range(n_people):
        for j in range(per):
            (tmp_path / f"p{k}_{j}.jpg").write_bytes(b"x")
            pid = db.upsert_photo(conn, dict(rel=f"p{k}_{j}.jpg", qhash=f"q{k}{j}", n_faces=1, status="ok"))
            v = centers[k] + rng.normal(scale=0.05, size=128); v /= np.linalg.norm(v)
            db.replace_faces(conn, pid, [dict(x=0,y=0,w=10,h=10,score=0.9,landmarks="[]",eye_sharp=1.0,embed=v.astype(np.float32).tobytes())])
    return conn

def test_cluster_and_name(tmp_path):
    _fake_shoot(tmp_path)
    people = cluster_faces(tmp_path, eps=0.3)
    assert len(people) == 3 and all(p["n"] == 4 for p in people)
    name_person(tmp_path, people[0]["id"], "Arya")
    assert list_people(tmp_path)[0]["name"] == "Arya"
    out = export_people(tmp_path)
    assert (out / "people" / "Arya").is_dir() and len(list((out / "people" / "Arya").iterdir())) == 4
    assert (out / "solo").is_dir() and len(list((out / "solo").iterdir())) == 12
