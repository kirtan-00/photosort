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

def export_dir(root: Path, name: str, base: Path | None = None) -> Path:
    """Resolve the export folder for a shoot: <base>/<shoot>/<name>, base defaulting to export_root().
    Refuses a name that escapes the base, a base that is the (read-only) shoot root or inside it,
    and a name that would land inside the shoot root. Does not create anything."""
    root = Path(root); root_res = root.resolve()
    base = (Path(base) if base is not None else export_root()).resolve()
    if base == root_res or base.is_relative_to(root_res):
        raise ValueError("destination is inside the source folder")
    out = base / root_res.name / safe_name(name)
    res = out.resolve()
    if not res.is_relative_to(base):
        raise ValueError(f"export path escapes the export folder: {name!r}")
    if res.is_relative_to(root_res):
        raise ValueError(f"export path would land inside the source folder: {name!r}")
    return out

def export_bytes(root: Path, ids: list[int]) -> int:
    conn = db.connect(Path(root)); total = 0
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]; q = ",".join("?" * len(chunk))
        total += conn.execute(f"SELECT COALESCE(SUM(size), 0) FROM photos WHERE id IN ({q}) AND status='ok'", chunk).fetchone()[0]
    return int(total)

def transfer_files(root: Path, jobs: list[tuple[int, str, Path]], mode: str, failed_file: Path, progress=None) -> list[str]:
    """The per-file loop every export shares. jobs are (photo id, rel, destination folder); each file
    lands in its folder under its own name, or {id}_{name} when that name is already taken. mode is
    "copy" or "symlink". A per-file OSError is counted, not raised, and the list is written to
    failed_file at the end. progress (if given) sees {done, total, failed} after every file.
    Destination folders must already exist."""
    root = Path(root); notify = progress or (lambda d: None)
    failed: list[str] = []; total = len(jobs)
    for n, (pid, rel, dst_dir) in enumerate(jobs, 1):
        src = root / rel; name = Path(rel).name; dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            dst = dst_dir / f"{pid}_{name}"
        try:
            if not src.exists():                  # os.symlink would happily point at nothing
                raise FileNotFoundError(str(src))
            if mode == "copy": shutil.copy2(src, dst)
            else: os.symlink(src.resolve(), dst)
        except OSError as e:
            failed.append(f"{rel}\t{e}")
        notify({"done": n, "total": total, "failed": len(failed)})
    if failed:
        failed_file.write_text("\n".join(failed) + "\n")
    return failed

def export_ids(root: Path, ids: list[int], name: str, mode: str = "copy", progress=None, base: Path | None = None) -> Path:
    root = Path(root); out = export_dir(root, name, base)
    notify = progress or (lambda d: None)
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
        notify({"done": len(rows), "total": len(rows), "failed": 0})
        return out
    transfer_files(root, [(r["id"], r["rel"], out) for r in rows], mode, out / "failed.txt", progress)
    return out
