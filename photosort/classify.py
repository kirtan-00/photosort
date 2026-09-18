"""Zero-shot scene categories on top of the index. Never writes under the shoot root
unless apply_on_disk() is called explicitly (a same-volume move with an undo log)."""
from __future__ import annotations
import csv, os, shutil
from pathlib import Path
import numpy as np
from . import db
from .export import export_dir

# One category = several prompts; a photo's category score is the max over its prompts.
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
    "birds-animals": ["a bird", "birds flying", "a dog", "a cow on the road", "a wild animal", "fish"],
}
FALLBACK = "other"
MIN_SCORE = 0.16      # below this cosine the best category is too weak: "other"
MIN_MARGIN = 0.004    # best minus second-best; smaller means ambiguous: "other"

def _prompt_matrix(embedder) -> tuple[list[str], np.ndarray, list[int]]:
    names, texts, owner = [], [], []
    for i, (cat, prompts) in enumerate(CATEGORIES.items()):
        names.append(cat)
        for p in prompts:
            texts.append(p); owner.append(i)
    return names, embedder.encode_text(texts), owner

def classify(root: Path, people_by_faces: bool = True) -> list[dict]:
    """Returns one dict per indexed photo: rel, sibling, category, score, margin, n_faces."""
    from .embed import get_embedder
    root = Path(root); conn = db.connect(root)
    names, T, owner = _prompt_matrix(get_embedder())
    owner = np.array(owner)
    ids, M = db.load_embeds(conn)
    rows = {r["id"]: r for r in conn.execute("SELECT id, rel, sibling, n_faces FROM photos WHERE status='ok'")}
    S = M @ T.T                                    # (N, prompts)
    per_cat = np.stack([S[:, owner == i].max(axis=1) for i in range(len(names))], axis=1)
    out = []
    for k, pid in enumerate(ids.tolist()):
        r = rows.get(pid)
        if r is None:
            continue
        order = np.argsort(-per_cat[k]); best, second = order[0], order[1]
        score, margin = float(per_cat[k, best]), float(per_cat[k, best] - per_cat[k, second])
        cat = names[best]
        if people_by_faces and (r["n_faces"] or 0) >= 1:
            cat = "people"
        elif score < MIN_SCORE or margin < MIN_MARGIN:
            cat = FALLBACK
        out.append(dict(id=pid, rel=r["rel"], sibling=r["sibling"], category=cat,
                        score=round(score, 4), margin=round(margin, 4), n_faces=r["n_faces"]))
    out.sort(key=lambda d: (d["category"], -d["score"]))
    return out

def write_manifest(root: Path, results: list[dict]) -> Path:
    """categories.csv + one folder of symlinks per category under the Desktop export dir.
    Symlinks point at the files on the disk; nothing is copied, nothing is written under root."""
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
            for f in (src, raw):
                if f is None: continue
                dst = base / r["category"] / f.name
                if dst.exists() or dst.is_symlink():
                    dst = base / r["category"] / f"{r['id']}_{f.name}"
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
