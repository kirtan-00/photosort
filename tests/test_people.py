import numpy as np
from photosort import db
from photosort.people import cluster_faces, name_person, list_people, export_people
from photosort.config import FACE_MATCH_MIN_SIM

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
    assert sims == sorted(sims, reverse=True) and all(s >= FACE_MATCH_MIN_SIM for s in sims)
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

def test_find_by_reference_rejects_tiny_face(tmp_path, monkeypatch):
    from photosort import people
    conn = _fake_shoot(tmp_path)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_p0_reference(conn, w=20, h=20)])
    out = people.find_by_reference(tmp_path, tmp_path / "p0_0.jpg")
    assert out["faces_in_reference"] == 1 and out["reference_face_too_small"] is True
    assert out["matches"] == [] and out["person_id"] is None

def test_find_by_reference_unreadable_reference_raises(tmp_path):
    from photosort import people
    _fake_shoot(tmp_path)   # p0_0.jpg is a one-byte stub, so the decoder rejects it
    import pytest
    with pytest.raises(people.ReferenceUnreadable):
        people.find_by_reference(tmp_path, tmp_path / "p0_0.jpg")


# named people from reference photos

def _person_reference(conn, k, w=50, h=50, seed=7):
    """A fake reference face for person k: that person's centre plus a little noise, normalised."""
    from photosort.faces import Face
    rows = conn.execute("SELECT f.embed FROM faces f JOIN photos p ON p.id=f.photo_id WHERE p.rel LIKE ?", (f"p{k}_%",)).fetchall()
    c = np.stack([np.frombuffer(r[0], np.float32) for r in rows]).mean(axis=0)
    c = c + np.random.default_rng(seed).normal(scale=0.02, size=128); c = (c / np.linalg.norm(c)).astype(np.float32)
    return Face(0, 0, w, h, 0.95, np.zeros((5, 2)), c, 1.0)

def _listing(tmp_path):
    import os
    return sorted(os.listdir(tmp_path))

def test_save_reference_stores_a_row_per_photo(tmp_path, monkeypatch):
    from photosort import people
    conn = _fake_shoot(tmp_path, n_people=3, per=4)
    before = _listing(tmp_path)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_person_reference(conn, 0)])
    out = people.save_reference(tmp_path, "Arya", tmp_path / "p0_0.jpg")
    assert isinstance(out["id"], int) and out["name"] == "Arya" and out["faces_in_reference"] == 1
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_person_reference(conn, 0, seed=8)])
    out2 = people.save_reference(tmp_path, "  Arya ", tmp_path / "p0_1.jpg")   # a second photo of the same person
    assert out2["name"] == "Arya" and out2["id"] != out["id"]
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_person_reference(conn, 1)])
    out3 = people.save_reference(tmp_path, "Priest", tmp_path / "p1_0.jpg")
    rows = db.list_references(db.connect(tmp_path))
    assert [(r["id"], r["name"], r["source"]) for r in rows] == [
        (out["id"], "Arya", str(tmp_path / "p0_0.jpg")), (out2["id"], "Arya", str(tmp_path / "p0_1.jpg")),
        (out3["id"], "Priest", str(tmp_path / "p1_0.jpg"))]
    assert all(r["created_at"] for r in rows)
    ids, names, R = db.load_reference_embeds(db.connect(tmp_path))
    assert ids.tolist() == [out["id"], out2["id"], out3["id"]] and names == ["Arya", "Arya", "Priest"]
    assert R.shape == (3, 128) and R.dtype == np.float32
    assert np.allclose(np.linalg.norm(R, axis=1), 1.0, atol=1e-3)
    assert _listing(tmp_path) == before

def test_save_reference_rejects_no_face_tiny_face_and_unreadable(tmp_path, monkeypatch):
    import pytest
    from photosort import people
    conn = _fake_shoot(tmp_path)
    before = _listing(tmp_path)
    with pytest.raises(people.ReferenceUnreadable):        # p0_0.jpg is a one-byte stub, real decode fails
        people.save_reference(tmp_path, "Arya", tmp_path / "p0_0.jpg")
    monkeypatch.setattr(people, "_reference_faces", lambda path: [])
    with pytest.raises(ValueError, match="no usable face"):
        people.save_reference(tmp_path, "Arya", tmp_path / "p0_0.jpg")
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_person_reference(conn, 0, w=20, h=20)])
    with pytest.raises(ValueError, match="too small"):
        people.save_reference(tmp_path, "Arya", tmp_path / "p0_0.jpg")
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_person_reference(conn, 0)])
    with pytest.raises(ValueError, match="name"):
        people.save_reference(tmp_path, "   ", tmp_path / "p0_0.jpg")
    assert db.list_references(db.connect(tmp_path)) == []
    assert _listing(tmp_path) == before

def _two_named_people(tmp_path, monkeypatch, conn):
    from photosort import people
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_person_reference(conn, 0)])
    people.save_reference(tmp_path, "Arya", tmp_path / "p0_0.jpg")
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_person_reference(conn, 0, seed=8)])
    people.save_reference(tmp_path, "Arya", tmp_path / "p0_1.jpg")
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_person_reference(conn, 1)])
    people.save_reference(tmp_path, "Priest", tmp_path / "p1_0.jpg")

def test_match_references_one_list_per_name(tmp_path, monkeypatch):
    from photosort import people
    conn = _fake_shoot(tmp_path, n_people=3, per=4)
    before = _listing(tmp_path)
    _two_named_people(tmp_path, monkeypatch, conn)
    out = people.match_references(tmp_path)
    assert set(out) == {"Arya", "Priest"}
    for name, k in [("Arya", 0), ("Priest", 1)]:
        assert len(out[name]) == 4 and len({m["photo_id"] for m in out[name]}) == 4
        assert all(_rel_of(conn, m["photo_id"]).startswith(f"p{k}_") for m in out[name])
        sims = [m["sim"] for m in out[name]]
        assert sims == sorted(sims, reverse=True) and all(s >= FACE_MATCH_MIN_SIM for s in sims)
    strict = people.match_references(tmp_path, min_sim=0.99)
    assert strict == {"Arya": [], "Priest": []}
    assert _listing(tmp_path) == before

def test_match_references_photo_with_two_people_is_under_both(tmp_path, monkeypatch):
    from photosort import people
    conn = _fake_shoot(tmp_path, n_people=3, per=4)
    # one extra frame holding a face of person 0 and a face of person 1
    (tmp_path / "both.jpg").write_bytes(b"x")
    pid = db.upsert_photo(conn, dict(rel="both.jpg", qhash="qboth", n_faces=2, status="ok"))
    faces = []
    for k in (0, 1):
        v = _person_reference(conn, k, seed=11 + k).embed
        faces.append(dict(x=0, y=0, w=10, h=10, score=0.9, landmarks="[]", eye_sharp=1.0, embed=v.tobytes()))
    db.replace_faces(conn, pid, faces)
    before = _listing(tmp_path)
    _two_named_people(tmp_path, monkeypatch, conn)
    out = people.match_references(tmp_path)
    assert pid in {m["photo_id"] for m in out["Arya"]} and pid in {m["photo_id"] for m in out["Priest"]}
    assert len(out["Arya"]) == 5 and len(out["Priest"]) == 5
    assert _listing(tmp_path) == before

def test_export_references_ids_maps_safe_folder_to_ids(tmp_path, monkeypatch):
    from photosort import people
    conn = _fake_shoot(tmp_path, n_people=3, per=4)
    before = _listing(tmp_path)
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_person_reference(conn, 0)])
    people.save_reference(tmp_path, "Ar/ya", tmp_path / "p0_0.jpg")
    monkeypatch.setattr(people, "_reference_faces", lambda path: [_person_reference(conn, 1)])
    people.save_reference(tmp_path, "Priest", tmp_path / "p1_0.jpg")
    out = people.export_references_ids(tmp_path, None, FACE_MATCH_MIN_SIM)
    assert set(out) == {"Ar_ya", "Priest"}
    assert sorted(_rel_of(conn, i) for i in out["Ar_ya"]) == [f"p0_{j}.jpg" for j in range(4)]
    assert sorted(_rel_of(conn, i) for i in out["Priest"]) == [f"p1_{j}.jpg" for j in range(4)]
    only = people.export_references_ids(tmp_path, ["Priest"], FACE_MATCH_MIN_SIM)
    assert set(only) == {"Priest"} and len(only["Priest"]) == 4
    assert people.export_references_ids(tmp_path, ["nobody"], FACE_MATCH_MIN_SIM) == {}
    assert people.export_references_ids(tmp_path, ["Priest"], 0.99) == {"Priest": []}
    assert _listing(tmp_path) == before

def test_export_references_writes_one_folder_per_person(tmp_path, tmp_path_factory, monkeypatch):
    import os
    from photosort import people
    conn = _fake_shoot(tmp_path, n_people=3, per=4)
    (tmp_path / "p0_0.ARW").write_bytes(b"raw")
    conn.execute("UPDATE photos SET sibling='p0_0.ARW' WHERE rel='p0_0.jpg'")
    conn.execute("UPDATE photos SET size=1"); conn.commit()   # the stubs are one byte; _fake_shoot leaves size NULL
    before = _listing(tmp_path)
    _two_named_people(tmp_path, monkeypatch, conn)
    disk = tmp_path_factory.mktemp("disk")
    seen = []
    out = people.export_references(tmp_path, None, "copy", True, base=disk, progress=seen.append)
    assert out == disk / tmp_path.resolve().name / "people"
    assert sorted(x.name for x in (out / "Arya").iterdir()) == ["p0_0.ARW", "p0_0.jpg", "p0_1.jpg", "p0_2.jpg", "p0_3.jpg"]
    assert sorted(x.name for x in (out / "Priest").iterdir()) == [f"p1_{j}.jpg" for j in range(4)]
    assert seen[-1] == {"done": 9, "total": 9, "failed": 0} and not (out / "failed.txt").exists()
    assert people.references_bytes(tmp_path, None, True) == 4 + 4 + 3 and people.references_bytes(tmp_path, ["Priest"], False) == 4
    out2 = people.export_references(tmp_path, ["Priest"], "symlink", False, base=disk)
    links = [x for x in (out2 / "Priest").iterdir() if x.is_symlink()]
    assert len(links) == 4 and all(x.name.endswith("_" + x.resolve().name) for x in links)   # {id}_{name}, next to the first run's copies
    assert len(os.listdir(out2 / "Priest")) == 8
    assert _listing(tmp_path) == before

def test_reference_delete_and_rename(tmp_path, monkeypatch):
    from photosort import people
    conn = _fake_shoot(tmp_path)
    _two_named_people(tmp_path, monkeypatch, conn)
    c = db.connect(tmp_path)
    assert db.rename_reference(c, "Arya", "Arya Mehta") == 2
    assert [r["name"] for r in db.list_references(c)] == ["Arya Mehta", "Arya Mehta", "Priest"]
    assert set(people.match_references(tmp_path)) == {"Arya Mehta", "Priest"}
    first = db.list_references(c)[0]["id"]
    assert db.delete_reference(c, first) is True and db.delete_reference(c, first) is False
    assert len(db.list_references(c)) == 2 and len(people.match_references(tmp_path)["Arya Mehta"]) == 4
