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
    s = index_folder(Path(a.folder), faces=not a.no_faces, workers=a.workers, progress=_progress, retry_errors=a.retry_errors)
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

def cmd_find(a):
    from .search import Index, Filters
    from .export import export_ids
    ix = Index(Path(a.folder))
    f = Filters(sharp_min_pct=a.sharp, faces=a.faces)
    res = ix.search(text=a.query, filters=f, limit=a.limit)
    for r in res: print(f"{r['score']:.3f}  {r['sharp_pct']:5.1f}%  {'?' if r['n_faces'] is None else r['n_faces']}f  {r['rel']}")
    if a.out:
        print("exported to", export_ids(Path(a.folder), [r["id"] for r in res], a.out, a.mode))

def cmd_people(a):
    from .people import cluster_faces, export_people, name_person
    people = cluster_faces(Path(a.folder), eps=a.eps)
    for p in people: print(f"person_{p['id']:02d}  {p['n']} photos")
    if a.export: print("exported to", export_people(Path(a.folder), a.mode))

def free_port(start: int, tries: int = 10, host: str = "127.0.0.1") -> int:
    """First port in [start, start+tries) that binds; a stale server may still hold the default."""
    import socket
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"no free port in {start}-{start + tries - 1}")

def cmd_classify(a):
    from .classify import classify, classify_and_store, write_manifest, apply_on_disk, undo_on_disk
    from collections import Counter
    root = Path(a.folder)
    if a.undo:
        print("restored", undo_on_disk(Path(a.undo)), "files"); return
    classify_and_store(root)          # persists category + score in the index (UI reads these)
    res = classify(root)
    for cat, n in sorted(Counter(r["category"] for r in res).items(), key=lambda x: -x[1]):
        print(f"{cat:>14}  {n}")
    print("manifest + symlink folders:", write_manifest(root, res))
    if a.apply_on_disk:
        print("MOVING files on the disk into _sorted/ ...")
        print("moved into", apply_on_disk(root, res, dry_run=False), "; undo with --undo <export>/undo.csv")
    elif a.plan_on_disk:
        print("dry-run move plan:", apply_on_disk(root, res, dry_run=True))

def cmd_serve(a):
    import uvicorn, webbrowser, threading
    from .server import create_app
    folder = Path(a.folder) if a.folder else None
    app = create_app(folder)
    port = free_port(a.port)
    if port != a.port:
        sys.stderr.write(f"port {a.port} busy, using {port}\n")
    if a.open:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    where = str(folder) if folder else "(no folder open, pick one in the app)"
    print(f"photosort serving {where} at http://127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

def main(argv=None):
    p = argparse.ArgumentParser(prog="photosort")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index"); s.add_argument("folder"); s.add_argument("--no-faces", action="store_true"); s.add_argument("--workers", type=int)
    s.add_argument("--retry-errors", action="store_true", help="re-process photos that failed last time"); s.set_defaults(fn=cmd_index)
    s = sub.add_parser("bench"); s.add_argument("folder"); s.add_argument("--n", type=int, default=200); s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("find"); s.add_argument("folder"); s.add_argument("query", nargs="?")
    s.add_argument("--sharp", type=float, help="min sharpness percentile 0-100"); s.add_argument("--faces", choices=["none","one","two","group"])
    s.add_argument("--limit", type=int, default=50); s.add_argument("--out", help="export folder name (created under ~/Desktop/photosort-out/<shoot>/)"); s.add_argument("--mode", default="copy", choices=["copy","symlink","csv"])
    s.set_defaults(fn=cmd_find)
    s = sub.add_parser("people"); s.add_argument("folder"); s.add_argument("--eps", type=float, default=0.5)
    s.add_argument("--export", action="store_true"); s.add_argument("--mode", default="copy", choices=["copy","symlink"]); s.set_defaults(fn=cmd_people)
    s = sub.add_parser("classify"); s.add_argument("folder")
    s.add_argument("--plan-on-disk", action="store_true", help="write move-plan.csv only, touch nothing")
    s.add_argument("--apply-on-disk", action="store_true", help="MOVE files into <folder>/_sorted/<category>/ (same volume, undo.csv written first)")
    s.add_argument("--undo", help="path to undo.csv from a previous --apply-on-disk"); s.set_defaults(fn=cmd_classify)
    s = sub.add_parser("serve"); s.add_argument("folder", nargs="?", help="photo folder; omit to open the picker in the app")
    s.add_argument("--port", type=int, default=7777); s.add_argument("--open", action="store_true"); s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv); a.fn(a)

if __name__ == "__main__":
    main()
