from __future__ import annotations
import csv, os, shutil
from pathlib import Path
from . import db
from .config import export_root

def export_ids(root: Path, ids: list[int], name: str, mode: str = "copy") -> Path:
    root = Path(root); out = export_root() / root.resolve().name / name; out.mkdir(parents=True, exist_ok=True)
    conn = db.connect(root)
    q = ",".join("?" * len(ids)) if ids else "NULL"
    rows = conn.execute(f"SELECT id, rel, sharp, n_faces, taken_at FROM photos WHERE id IN ({q}) ORDER BY id", ids).fetchall()
    if mode == "csv":
        with open(out / "photos.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["id", "path", "sharp", "n_faces", "taken_at"])
            for r in rows: w.writerow([r["id"], str(root / r["rel"]), r["sharp"], r["n_faces"], r["taken_at"]])
        return out
    for r in rows:
        src = root / r["rel"]; dst = out / Path(r["rel"]).name
        if dst.exists() or dst.is_symlink():
            dst = out / f"{r['id']}_{Path(r['rel']).name}"
        if mode == "copy": shutil.copy2(src, dst)
        else: os.symlink(src.resolve(), dst)
    return out
