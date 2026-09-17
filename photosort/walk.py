from __future__ import annotations
import hashlib, os
from dataclasses import dataclass
from pathlib import Path
from .config import IMAGE_EXTS, RAW_EXTS, INDEX_DIRNAME

@dataclass
class ImageFile:
    path: Path
    rel: str
    size: int
    mtime: float
    is_raw: bool
    sibling: str | None = None

def find_images(root: Path) -> list[ImageFile]:
    root = Path(root)
    found: dict[str, ImageFile] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "photosort-out"]
        for fn in filenames:
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            st = p.stat()
            rel = str(p.relative_to(root))
            found[rel] = ImageFile(p, rel, st.st_size, st.st_mtime, ext in RAW_EXTS)
    # pair RAW+JPEG by stem within the same directory: keep the JPEG
    by_stem: dict[tuple[str, str], list[ImageFile]] = {}
    for f in found.values():
        by_stem.setdefault((str(f.path.parent), f.path.stem.lower()), []).append(f)
    out: list[ImageFile] = []
    for group in by_stem.values():
        raws = [g for g in group if g.is_raw]
        std = [g for g in group if not g.is_raw]
        if raws and std:
            keep = sorted(std, key=lambda g: g.rel)[0]
            keep.sibling = raws[0].rel
            out.append(keep)
        else:
            out.extend(group)
    return sorted(out, key=lambda f: f.rel)

def quick_hash(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha1()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk:
            fh.seek(max(size - chunk, chunk))
            h.update(fh.read(chunk))
    return h.hexdigest()
