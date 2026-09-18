from __future__ import annotations
import csv, os, re, shutil
from pathlib import Path
from . import db
from .config import export_root

def safe_segment(name: str) -> str:
    """One folder-name segment: no separators, no leading/trailing dots or spaces, never '.' or '..'."""
    if name in (".", ".."):
        raise ValueError(f"illegal export name segment: {name!r}")
    return re.sub(r"[\\/:]+", "_", name).strip(" .") or "export"

def safe_name(name: str) -> str:
    """Sanitise a possibly nested export name ('people/Arya') segment by segment.
    Absolute paths and empty segments are rejected outright."""
    if name == "":
        name = "export"
    if Path(name).is_absolute() or name.startswith(("/", "\\")):
        raise ValueError(f"export name must be relative: {name!r}")
    parts = [p for p in name.split("/")]
    if any(p == "" for p in parts):
        raise ValueError(f"export name has an empty segment: {name!r}")
    return "/".join(safe_segment(p) for p in parts)

def export_dir(root: Path, name: str) -> Path:
    """Resolve the export folder for a shoot and refuse anything outside export_root
    or inside the (read-only) shoot root. Does not create it."""
    root = Path(root)
    base = export_root().resolve()
    out = base / root.resolve().name / safe_name(name)
    res = out.resolve()
    if not res.is_relative_to(base):
        raise ValueError(f"export path escapes the export folder: {name!r}")
    if res.is_relative_to(root.resolve()):
        raise ValueError(f"export path would land inside the source folder: {name!r}")
    return out

def export_ids(root: Path, ids: list[int], name: str, mode: str = "copy") -> Path:
    root = Path(root); out = export_dir(root, name)
    conn = db.connect(root)
    rows = []
    for i in range(0, len(ids), 900):            # chunk: SQLite caps bound variables
        chunk = ids[i:i + 900]; q = ",".join("?" * len(chunk))
        rows += conn.execute(f"SELECT id, rel, sharp, n_faces, taken_at FROM photos WHERE id IN ({q}) AND status='ok' ORDER BY id", chunk).fetchall()
    out.mkdir(parents=True, exist_ok=True)
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
