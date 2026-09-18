"""Zero-shot scene categories on top of the index. Never writes under the shoot root
unless apply_on_disk() is called explicitly (a same-volume move with an undo log)."""
from __future__ import annotations
import csv, os, shutil
from pathlib import Path
import numpy as np
from . import db
from .export import export_dir

# One category = several prompts; a photo's category score is the max cosine over its prompts.
CATEGORIES: dict[str, list[str]] = {
    "ocean": ["the open sea with waves", "a seascape with the horizon over the water", "boats on the sea",
              "waves crashing on rocks", "the ocean at sunset"],
    "beach": ["a sandy beach", "the seashore with sand and footprints", "beach umbrellas and sunbeds",
              "a beach with people walking on the sand", "a coastline seen from the beach"],
    "people": ["a portrait of a person", "a group of people posing for a photo", "a crowd of people",
               "a person standing and looking at the camera", "a selfie"],
    "building": ["a building facade", "an old fort or church", "a temple or monument", "a house or hotel",
                 "architecture of a town", "a lighthouse"],
    "road": ["a road with vehicles", "a street in a town", "a highway", "a road through the countryside",
             "a scooter on a road"],
    "birds-animals": ["a bird", "birds flying", "a dog", "a cow on the road", "a wild animal", "fish",
                       "seabirds flying low over the ocean", "birds over the water"],
}
# A pseudo-category, not one of CATEGORIES: it competes in the same softmax so things that look like
# nothing on the real list (food, night sky, screenshots, dark/blurry frames) pull probability mass
# away from whichever real category they happen to resemble most (a night sky is dark and blue, same
# as the ocean prompts, so without this "other" never fires on raw argmax). Never becomes a folder name
# of its own; a win here maps to FALLBACK. Kept separate from CATEGORIES so write_manifest's folder list
# (CATEGORIES keys + FALLBACK) doesn't grow a second "other" entry.
NEGATIVE_PROMPTS = ["a plate of food on a table", "a night sky full of stars", "a screenshot of a phone or computer screen",
                     "a blurry or badly lit photograph", "a page of text or a document"]
FALLBACK = "other"
TEMPERATURE = 100.0   # CLIP's logit scale; turns cosine similarity into a peaked softmax
MIN_PROB = 0.35        # best category must own at least this much of the softmax mass: "other"
MIN_PROB_MARGIN = 0.15  # best minus second-best probability; smaller means ambiguous: "other"
# T=100 amplifies even meaningless cosine gaps into a "confident" softmax: on the calibration set every
# correctly-classified real photo's winning raw cosine was >= 0.1497, while a random (non-photo) unit
# vector's best raw cosine was 0.0814 despite a deceptively "confident" softmax. This absolute floor
# catches that case; the two MIN_PROB* thresholds above then separate genuinely ambiguous real photos.
MIN_COSINE = 0.12

def _prompt_matrix(embedder) -> tuple[list[str], np.ndarray, list[int]]:
    """names includes CATEGORIES keys followed by one pseudo-category "__other__" owning
    NEGATIVE_PROMPTS, so callers that only want the real categories should slice names[:-1]."""
    names, texts, owner = [], [], []
    for i, (cat, prompts) in enumerate(CATEGORIES.items()):
        names.append(cat)
        for p in prompts:
            texts.append(p); owner.append(i)
    neg_idx = len(names)
    names.append("__other__")
    for p in NEGATIVE_PROMPTS:
        texts.append(p); owner.append(neg_idx)
    return names, embedder.encode_text(texts), owner

def _score(M: np.ndarray, T: np.ndarray, owner: np.ndarray, names: list[str]):
    """Per row of M: (category name or FALLBACK, softmax score, margin) after the confidence gates.
    One matrix pass, so photos, videos and segments are scored together on a stacked M."""
    S = M @ T.T                                    # (N, prompts)
    per_cat = np.stack([S[:, owner == i].max(axis=1) for i in range(len(names))], axis=1)
    logits = per_cat * TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)     # numerically stable softmax
    probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
    out = []
    for k in range(len(M)):
        order = np.argsort(-probs[k]); best, second = order[0], order[1]
        score, margin = float(probs[k, best]), float(probs[k, best] - probs[k, second])
        raw_cos = float(per_cat[k, best])
        name = names[best]
        cat = FALLBACK if name == "__other__" else name
        if cat != FALLBACK and (raw_cos < MIN_COSINE or score < MIN_PROB or margin < MIN_PROB_MARGIN):
            cat = FALLBACK
        out.append((cat, score, margin))
    return out

def _classify_all(root: Path, people_by_faces: bool = True) -> tuple[list[dict], list[dict]]:
    """(photo results, segment results). Photos and videos: rel, sibling, category, score, margin, n_faces.
    Segments (of ok videos): id, photo_id, category, score. Both come out of one pass over the stacked
    embedding matrix. A face-bearing photo is always "people"; segments carry no faces, so never."""
    from .embed import get_embedder
    root = Path(root); conn = db.connect(root)
    names, T, owner = _prompt_matrix(get_embedder())
    owner = np.array(owner)
    ids, M = db.load_embeds(conn)
    seg_ids, SM = db.load_segment_embeds(conn)
    rows = {r["id"]: r for r in conn.execute("SELECT id, rel, sibling, n_faces FROM photos WHERE status='ok'")}
    seg_photo = {r[0]: r[1] for r in conn.execute("SELECT id, photo_id FROM segments")}
    scored = _score(np.vstack([M, SM]), T, owner, names) if len(M) + len(SM) else []
    photos, segments = [], []
    for k, pid in enumerate(ids.tolist()):
        r = rows.get(pid)
        if r is None:
            continue
        cat, score, margin = scored[k]
        if people_by_faces and (r["n_faces"] or 0) >= 1:
            cat = "people"
        photos.append(dict(id=pid, rel=r["rel"], sibling=r["sibling"], category=cat,
                           score=round(score, 4), margin=round(margin, 4), n_faces=r["n_faces"]))
    for k, sid in enumerate(seg_ids.tolist()):
        cat, score, _ = scored[len(ids) + k]
        segments.append(dict(id=sid, photo_id=seg_photo.get(sid), category=cat, score=round(score, 4)))
    photos.sort(key=lambda d: (d["category"], -d["score"]))
    return photos, segments

def classify(root: Path, people_by_faces: bool = True) -> list[dict]:
    """Returns one dict per indexed photo or video: rel, sibling, category, score, margin, n_faces.
    score/margin are softmax probabilities (not raw cosine): score is how much of the probability
    mass the winning bucket (a real category, or the "other" pseudo-category) owns, margin is its
    lead over the runner-up. A face-bearing photo is always "people" regardless of these."""
    return _classify_all(root, people_by_faces=people_by_faces)[0]

def classify_and_store(root: Path, people_by_faces: bool = True) -> dict[str, int]:
    """Runs the stacked pass and persists category + category_score onto photos (and videos) and onto
    their segments. Returns counts per category over photos and videos, the same rows the
    Categories tab lists (segments are not counted)."""
    from collections import Counter
    root = Path(root)
    results, segs = _classify_all(root, people_by_faces=people_by_faces)
    conn = db.connect(root)
    conn.executemany("UPDATE photos SET category=?, category_score=? WHERE id=?",
                      [(r["category"], r["score"], r["id"]) for r in results])
    conn.executemany("UPDATE segments SET category=?, category_score=? WHERE id=?",
                      [(r["category"], r["score"], r["id"]) for r in segs])
    conn.commit()
    return dict(Counter(r["category"] for r in results))

def write_manifest(root: Path, results: list[dict]) -> Path:
    """categories.csv + one folder of symlinks per category under the Desktop export dir.
    Symlinks point at the files on the disk; nothing is copied, nothing is written under root.
    A shoot with per-day/per-location subfolders can have the same filename in several places, so
    each symlink is named after its full relative path ('/' -> '__') rather than the bare filename;
    that also shows at a glance where the photo came from. True collisions (same rel-derived name,
    which only happens if the shoot itself already used '__' in a folder name) fall back to an id prefix."""
    root = Path(root); base = export_dir(root, "categories"); base.mkdir(parents=True, exist_ok=True)
    for cat in list(CATEGORIES) + [FALLBACK]:
        d = base / cat
        if d.exists():
            for old in d.iterdir():
                if old.is_symlink(): old.unlink()
        d.mkdir(exist_ok=True)
    with open(base / "categories.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["category", "score", "margin", "faces", "jpeg", "raw"])
        for r in results:
            src = root / r["rel"]; raw = (root / r["sibling"]) if r["sibling"] else None
            w.writerow([r["category"], r["score"], r["margin"], r["n_faces"], str(src), str(raw) if raw else ""])
            for f, rel in ((src, r["rel"]), (raw, r["sibling"])):
                if f is None: continue
                name = rel.replace("/", "__")
                dst = base / r["category"] / name
                if dst.exists() or dst.is_symlink():
                    dst = base / r["category"] / f"{r['id']}_{name}"
                os.symlink(f, dst)
    return base

def apply_on_disk(root: Path, results: list[dict], dry_run: bool = True) -> Path:
    """EXPLICIT OPT-IN ONLY. Moves each JPEG and its RAW sibling into <root>/_sorted/<category>/
    (same-volume rename, no copy). Writes <export>/undo.csv (new_path,old_path) first so it can be reversed.
    With dry_run=True nothing on the disk changes; the plan is written to <export>/move-plan.csv."""
    root = Path(root); base = export_dir(root, "categories"); base.mkdir(parents=True, exist_ok=True)
    plan = []
    for r in results:
        for rel in (r["rel"], r["sibling"]):
            if not rel: continue
            src = root / rel
            if not src.exists() or "_sorted" in Path(rel).parts: continue
            plan.append((src, root / "_sorted" / r["category"] / src.name))
    with open(base / ("move-plan.csv" if dry_run else "undo.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["new_path", "old_path"])
        for src, dst in plan: w.writerow([str(dst), str(src)])
    if dry_run:
        return base / "move-plan.csv"
    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(f"{src.stat().st_ino}_{src.name}")
        shutil.move(str(src), str(dst))
    return root / "_sorted"

def undo_on_disk(undo_csv: Path) -> int:
    n = 0
    with open(undo_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            new, old = Path(row["new_path"]), Path(row["old_path"])
            if new.exists() and not old.exists():
                old.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(new), str(old)); n += 1
    return n
