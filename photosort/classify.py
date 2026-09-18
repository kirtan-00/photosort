"""Zero-shot scene categories on top of the index. Never writes under the shoot root
unless apply_on_disk() is called explicitly (a same-volume move with an undo log)."""
from __future__ import annotations
import csv, os, shutil
from pathlib import Path
import numpy as np
from . import db
from .export import export_dir
from .vocab import VOCAB

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
# Below this a match is "less sure": shown after a divider, sorted by confidence, exported only on request.
SURE_MIN = 0.5
# Discovered categories: k-means over the shoot's embeddings, each cluster named by the vocabulary label
# closest to its centroid. Deterministic (random_state=0), no LLM.
DISCOVER_MIN_PHOTOS = 16   # fewer embedded photos than this: nothing to discover
DISCOVER_MIN_SIZE = 8      # a smaller cluster is folded into its nearest neighbour
DISCOVER_MAX_K = 24

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
    """Per row of M: (category name or FALLBACK, softmax score, margin, guess, guess score) after the
    confidence gates. guess is the best REAL category and its probability whatever the gates decided,
    so a photo filed under "other" can still be shown under its guess as "less sure".
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
        real = [i for i in order if names[i] != "__other__"]
        guess, guess_score = names[real[0]], float(probs[k, real[0]])
        out.append((cat, score, margin, guess, guess_score))
    return out

def _classify_all(root: Path, people_by_faces: bool = True) -> tuple[list[dict], list[dict]]:
    """(photo results, segment results). Photos and videos: rel, sibling, category, score, margin, n_faces,
    guess, guess_score. Segments (of ok videos): id, photo_id, category, score. Both come out of one pass
    over the stacked embedding matrix. A face-bearing photo is always "people", with score 1.0: the face
    detector decided, not the softmax, so it is never "less sure"; segments carry no faces, so never."""
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
        cat, score, margin, guess, guess_score = scored[k]
        if people_by_faces and (r["n_faces"] or 0) >= 1:
            cat, score, guess, guess_score = "people", 1.0, "people", 1.0
        photos.append(dict(id=pid, rel=r["rel"], sibling=r["sibling"], category=cat,
                           score=round(score, 4), margin=round(margin, 4), n_faces=r["n_faces"],
                           guess=guess, guess_score=round(guess_score, 4)))
    for k, sid in enumerate(seg_ids.tolist()):
        cat, score, _, _, _ = scored[len(ids) + k]
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
    """Runs the stacked pass and persists category + category_score (and the best real guess with its
    probability) onto photos (and videos), and category + score onto their segments. Returns counts per
    category over photos and videos, the same rows the Categories tab lists (segments are not counted)."""
    from collections import Counter
    root = Path(root)
    results, segs = _classify_all(root, people_by_faces=people_by_faces)
    conn = db.connect(root)
    conn.executemany("UPDATE photos SET category=?, category_score=?, category_guess=?, category_guess_score=? WHERE id=?",
                      [(r["category"], r["score"], r["guess"], r["guess_score"], r["id"]) for r in results])
    conn.executemany("UPDATE segments SET category=?, category_score=? WHERE id=?",
                      [(r["category"], r["score"], r["id"]) for r in segs])
    conn.commit()
    return dict(Counter(r["category"] for r in results))

# ---------- discovered categories ----------

def discover_k(n: int) -> int:
    """Clusters for n embedded photos: sqrt(n / 25) clamped to 4..DISCOVER_MAX_K (3,677 photos -> 12)."""
    return min(DISCOVER_MAX_K, max(4, round((n / 25) ** 0.5)))

def _vocab_matrix(embedder) -> tuple[list[str], np.ndarray]:
    """(labels, unit text matrix) for VOCAB, encoded once per embedder and kept on it."""
    T = getattr(embedder, "_vocab_T", None)
    if T is None or len(T) != len(VOCAB):
        T = embedder.encode_text(list(VOCAB))
        embedder._vocab_T = T
    return list(VOCAB), T

def _fold_small(labels: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Merge every cluster under DISCOVER_MIN_SIZE into the cluster whose centroid is nearest (cosine),
    smallest first, recomputing centroids as it goes, until nothing small is left or one cluster remains.
    Labels come back renumbered 0..m-1 in order of first appearance."""
    labels = labels.copy()
    while True:
        names, sizes = np.unique(labels, return_counts=True)
        if len(names) <= 1:
            break
        small = [(int(s), int(c)) for s, c in zip(sizes, names) if s < DISCOVER_MIN_SIZE]
        if not small:
            break
        _, victim = min(small)
        cents = {int(c): M[labels == c].mean(axis=0) for c in names}
        for c in cents:
            cents[c] /= (np.linalg.norm(cents[c]) or 1.0)
        others = [int(c) for c in names if int(c) != victim]
        sims = [float(cents[victim] @ cents[c]) for c in others]
        labels[labels == victim] = others[int(np.argmax(sims))]
    order = {int(c): i for i, c in enumerate(dict.fromkeys(labels.tolist()))}
    return np.array([order[int(c)] for c in labels], np.int64)

def discover(root: Path, k: int | None = None) -> list[dict]:
    """Cluster the shoot's embeddings (photos and videos alike) with k-means and name every cluster from
    VOCAB by zero-shot scoring of its centroid. Returns [{id, name, size, score, photo_ids, photo_scores}]
    sorted by size desc: score is the centroid's cosine to the label, photo_scores are each member's
    cosine to the centroid rescaled to 0..1 across the cluster (the "less sure" half sits below 0.5).
    Two clusters with the same best label: the later (smaller) one takes its next-best unused label.
    Deterministic for a fixed index. Empty below DISCOVER_MIN_PHOTOS embedded photos."""
    from sklearn.cluster import KMeans
    root = Path(root); conn = db.connect(root)
    ids, M = db.load_embeds(conn)
    n = len(ids)
    if n < DISCOVER_MIN_PHOTOS:
        return []
    k = min(n, k if k is not None else discover_k(n))
    labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(M)
    labels = _fold_small(labels, M)
    from .embed import get_embedder
    vocab, T = _vocab_matrix(get_embedder())
    clusters = []
    for c in range(int(labels.max()) + 1):
        idx = np.where(labels == c)[0]
        cent = M[idx].mean(axis=0); cent /= (np.linalg.norm(cent) or 1.0)
        cos = M[idx] @ cent
        lo, hi = float(cos.min()), float(cos.max())
        rescaled = (cos - lo) / (hi - lo) if hi > lo else np.ones_like(cos)
        clusters.append(dict(idx=idx, cent=cent, size=len(idx), first=int(ids[idx].min()), photo_scores=rescaled))
    clusters.sort(key=lambda c: (-c["size"], c["first"]))
    used: set[str] = set()
    out = []
    for i, c in enumerate(clusters):
        scores = T @ c["cent"]
        for j in np.argsort(-scores):
            if vocab[j] not in used:
                name, score = vocab[j], float(scores[j]); break
        used.add(name)
        out.append(dict(id=i, name=name, size=c["size"], score=round(score, 4),
                        photo_ids=[int(p) for p in ids[c["idx"]]],
                        photo_scores=[round(float(v), 4) for v in c["photo_scores"]]))
    return out

def discover_and_store(root: Path, k: int | None = None) -> dict[str, int]:
    """Runs discover and persists cluster + cluster_score on photos. Returns {name: size}. A shoot too small
    to discover anything stores nothing and returns {}."""
    root = Path(root)
    found = discover(root, k=k)
    if not found:
        return {}
    conn = db.connect(root)
    conn.execute("UPDATE photos SET cluster=NULL, cluster_score=NULL")
    for c in found:
        conn.executemany("UPDATE photos SET cluster=?, cluster_score=? WHERE id=?",
                         [(c["name"], s, pid) for pid, s in zip(c["photo_ids"], c["photo_scores"])])
    conn.commit()
    return {c["name"]: c["size"] for c in found}

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
