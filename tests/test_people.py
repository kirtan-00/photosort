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

def test_name_survives_recluster(tmp_path):
    _fake_shoot(tmp_path)
    people = cluster_faces(tmp_path, eps=0.3)
    target = people[1]["id"]
    name_person(tmp_path, target, "Arya")
    target_faces = {r[0] for r in db.connect(tmp_path).execute("SELECT id FROM faces WHERE person_id=?", (target,))}
    people2 = cluster_faces(tmp_path, eps=0.35)
    named = [p for p in people2 if p["name"] == "Arya"]
    assert len(named) == 1
    conn = db.connect(tmp_path)
    assert {r[0] for r in conn.execute("SELECT id FROM faces WHERE person_id=?", (named[0]["id"],))} == target_faces
    assert sum(1 for p in people2 if p["name"]) == 1

def test_person_name_export_is_sanitised(tmp_path):
    _fake_shoot(tmp_path)
    people = cluster_faces(tmp_path, eps=0.3)
    name_person(tmp_path, people[0]["id"], "../../Ar/ya")
    out = export_people(tmp_path)
    names = sorted(p.name for p in (out / "people").iterdir())
    assert "_.._Ar_ya" in names and not any(n in (".", "..") for n in names)
    assert not (out.parent / "Ar").exists() and not (out / "people" / "Ar").exists()
    assert sorted(x.name for x in tmp_path.iterdir()) == sorted(f"p{k}_{j}.jpg" for k in range(3) for j in range(4))

def test_cover_falls_back_when_face_row_gone(tmp_path):
    conn = _fake_shoot(tmp_path)
    people = cluster_faces(tmp_path, eps=0.3)
    p0 = people[0]
    conn.execute("DELETE FROM faces WHERE id=?", (p0["cover_face_id"],)); conn.commit()
    again = [p for p in list_people(tmp_path) if p["id"] == p0["id"]][0]
    assert again["cover_face_id"] != p0["cover_face_id"] and again["cover_qhash"] is not None
    conn.execute("DELETE FROM faces WHERE person_id=?", (p0["id"],)); conn.commit()
    again = [p for p in list_people(tmp_path) if p["id"] == p0["id"]][0]
    assert again["cover_face_id"] is None and again["cover_qhash"] is None and again["cover_box"] == [0, 0, 0, 0]


# find by reference

def _p0_reference(conn, w=50, h=50):
    """A fake reference face: person 0's centre plus a little noise, normalised."""
    from photosort.faces import Face
    rows = conn.execute("SELECT f.embed FROM faces f JOIN photos p ON p.id=f.photo_id WHERE p.rel LIKE 'p0_%'").fetchall()
    c = np.stack([np.frombuffer(r[0], np.float32) for r in rows]).mean(axis=0)
    c = c + np.random.default_rng(7).normal(scale=0.02, size=128); c = (c / np.linalg.norm(c)).astype(np.float32)
    return Face(0, 0, w, h, 0.95, np.zeros((5, 2)), c, 1.0)

def _rel_of(conn, photo_id):
    return conn.execute("SELECT rel FROM photos WHERE id=?", (photo_id,)).fetchone()[0]

def test_find_by_reference_matches_person0(tmp_path, monkeypatch):
    from photosort import people
    conn = _fake_shoot(tmp_path, n_people=3, per=4)
    big = _p0_reference(conn)
    small = _p0_reference(conn, w=5, h=5)
    small.embed = -big.embed   # a smaller decoy face that must be ignored
    monkeypatch.setattr(people, "_reference_faces", lambda path: [small, big])
    out = people.find_by_reference(tmp_path, tmp_path / "p0_0.jpg")
    assert out["faces_in_reference"] == 2
    assert len(out["matches"]) == 4
    assert all(_rel_of(conn, m["photo_id"]).startswith("p0_") for m in out["matches"])
    sims = [m["sim"] for m in out["matches"]]
    assert sims == sorted(sims, reverse=True) and all(s >= 0.363 for s in sims)
    assert len({m["photo_id"] for m in out["matches"]}) == 4
    assert out["person_id"] is None   # not clustered yet
    assert people.find_by_reference(tmp_path, tmp_path / "p0_0.jpg", min_sim=0.99)["matches"] == []

def test_find_by_reference_no_face(tmp_path, monkeypatch):
    from photosort import people
    _fake_shoot(tmp_path)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [])
    out = people.find_by_reference(tmp_path, tmp_path / "p0_0.jpg")
    assert out == {"faces_in_reference": 0, "matches": [], "person_id": None}

def test_find_by_reference_reports_cluster(tmp_path, monkeypatch):
    from photosort import people
    conn = _fake_shoot(tmp_path)
    cluster_faces(tmp_path, eps=0.3)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_p0_reference(conn)])
    out = people.find_by_reference(tmp_path, tmp_path / "p0_0.jpg")
    expected = {r[0] for r in conn.execute(
        "SELECT DISTINCT f.person_id FROM faces f JOIN photos p ON p.id=f.photo_id WHERE p.rel LIKE 'p0_%'")}
    assert len(expected) == 1 and out["person_id"] == expected.pop()
