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

def category_rows(root: Path, categories: list[str] | None) -> list:
    """status='ok' rows (id, rel, sibling, size, category) in the given categories. None means every
    category that has a photo; "unclassified" (category NULL) only when named explicitly."""
    conn = db.connect(Path(root))
    if categories is None:
        return conn.execute("SELECT id, rel, sibling, size, category FROM photos WHERE status='ok' AND category IS NOT NULL ORDER BY category, id").fetchall()
    names = [c for c in categories if c != "unclassified"]
    rows = []
    if names:
        q = ",".join("?" * len(names))
        rows += conn.execute(f"SELECT id, rel, sibling, size, category FROM photos WHERE status='ok' AND category IN ({q}) ORDER BY category, id", names).fetchall()
    if "unclassified" in categories:
        rows += conn.execute("SELECT id, rel, sibling, size, 'unclassified' AS category FROM photos WHERE status='ok' AND category IS NULL ORDER BY id").fetchall()
    return rows

def categories_bytes(root: Path, categories: list[str] | None, include_raw: bool = False) -> int:
    """Bytes a copy of these categories needs: JPEG sizes from the DB, RAW siblings stat'ed on the
    disk (a sibling that fails to stat is skipped, the export will report it as failed)."""
    root = Path(root); total = 0
    for r in category_rows(root, categories):
        total += r["size"] or 0
        if include_raw and r["sibling"]:
            try: total += os.stat(root / r["sibling"]).st_size
            except OSError: pass
    return int(total)

def export_categories(root: Path, categories: list[str] | None, mode: str = "copy", include_raw: bool = False,
                      base: Path | None = None, progress=None) -> Path:
    """<base>/<shoot>/categories/<category>/<file> for every ok photo in the chosen categories, and its
    RAW sibling next to it when include_raw. Returns the categories folder."""
    if mode == "csv":
        raise ValueError("csv is not supported for a category export")
    root = Path(root); out = export_dir(root, "categories", base)
    rows = category_rows(root, categories)
    jobs: list[tuple[int, str, Path]] = []
    for r in rows:
        d = out / safe_segment(r["category"])
        jobs.append((r["id"], r["rel"], d))
        if include_raw and r["sibling"]:
            jobs.append((r["id"], r["sibling"], d))
    out.mkdir(parents=True, exist_ok=True)
    for d in {j[2] for j in jobs}:
        d.mkdir(parents=True, exist_ok=True)
    transfer_files(root, jobs, mode, out / "failed.txt", progress)
    return out
