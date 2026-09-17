from __future__ import annotations
import argparse, sys, time
from pathlib import Path

def _progress(d):
    if d["total"]:
        sys.stderr.write(f"\r{d['stage']:>8} {d['done']}/{d['total']}   "); sys.stderr.flush()
    if d["stage"] == "done":
        sys.stderr.write("\n")

def cmd_index(a):
    from .index import index_folder
    s = index_folder(Path(a.folder), faces=not a.no_faces, workers=a.workers, progress=_progress)
    print(f"indexed {s['indexed']}  skipped {s['skipped']}  errors {s['errors']}  embedded {s['embedded']}  in {s['seconds']}s")

def cmd_bench(a):
    from .walk import find_images
    from .index import process_one, index_folder
    from .embed import get_embedder
    from . import db
    from PIL import Image
    root = Path(a.folder); files = find_images(root)[: a.n]
    if not files:
        print("no images"); return
    db.connect(root)
    t = time.time(); ok = 0
    for f in files:
        r = process_one((str(root), f.rel, True)); ok += r["error"] is None
    feat_ms = (time.time() - t) / len(files) * 1000
    idx = db.index_dir(root)
    from .walk import quick_hash
    ims = [Image.open(idx / "thumbs" / f"{quick_hash(f.path)}.jpg") for f in files if (idx / "thumbs" / f"{quick_hash(f.path)}.jpg").exists()]
    E = get_embedder(); E.encode_images(ims[:4])
    t = time.time(); E.encode_images(ims); emb_ms = (time.time() - t) / max(len(ims), 1) * 1000
    total = len(find_images(root))
    per = feat_ms / 4 + emb_ms   # 4 workers on features, embed is serial on the GPU
    print(f"files {len(files)} ok {ok}  features {feat_ms:.0f} ms/photo (1 core)  embed {emb_ms:.1f} ms/photo")
    print(f"projected @4 workers: {per:.0f} ms/photo  -> this folder ({total}) {total*per/60000:.1f} min, 10k photos {10000*per/60000:.1f} min")

def main(argv=None):
    p = argparse.ArgumentParser(prog="photosort")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index"); s.add_argument("folder"); s.add_argument("--no-faces", action="store_true"); s.add_argument("--workers", type=int); s.set_defaults(fn=cmd_index)
    s = sub.add_parser("bench"); s.add_argument("folder"); s.add_argument("--n", type=int, default=200); s.set_defaults(fn=cmd_bench)
    a = p.parse_args(argv); a.fn(a)

if __name__ == "__main__":
    main()
