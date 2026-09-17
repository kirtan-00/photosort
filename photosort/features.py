from __future__ import annotations
from pathlib import Path
import cv2, imagehash, numpy as np
from PIL import Image
from .config import SHARP_TILE_GRID

def to_gray(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.uint8)

def phash(im: Image.Image) -> str:
    return str(imagehash.phash(im))

def _lapvar(g: np.ndarray) -> float:
    if g.size < 16:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def sharpness_tiles(gray: np.ndarray, grid: int = SHARP_TILE_GRID) -> tuple[float, float]:
    h, w = gray.shape
    th, tw = max(h // grid, 8), max(w // grid, 8)
    vals = [_lapvar(gray[y:y + th, x:x + tw]) for y in range(0, h - th + 1, th) for x in range(0, w - tw + 1, tw)]
    if not vals:
        return 0.0, 0.0
    return float(np.percentile(vals, 90)), float(max(vals))

def eye_sharpness(gray: np.ndarray, landmarks: np.ndarray) -> float:
    re, le = landmarks[0], landmarks[1]
    cx, cy = (re + le) / 2
    d = max(float(np.linalg.norm(le - re)), 8.0)
    hw, hh = 0.9 * d, 0.45 * d
    h, w = gray.shape
    x0, x1 = int(max(cx - hw, 0)), int(min(cx + hw, w))
    y0, y1 = int(max(cy - hh, 0)), int(min(cy + hh, h))
    return _lapvar(gray[y0:y1, x0:x1])

def exif_info(path: Path) -> dict:
    out = {"taken_at": None, "camera": None, "width": None, "height": None}
    try:
        with Image.open(path) as im:
            out["width"], out["height"] = im.size
            ex = im.getexif()
            dt = ex.get(0x0132) or ex.get_ifd(0x8769).get(0x9003)
            if dt:
                d, t = str(dt).split(" ", 1)
                out["taken_at"] = d.replace(":", "-") + "T" + t
            make, model = ex.get(0x010F), ex.get(0x0110)
            if model:
                out["camera"] = (f"{make} {model}" if make and make not in model else model).strip()
    except Exception:
        try:
            import pyexiv2
            m = pyexiv2.Image(str(path)); e = m.read_exif(); m.close()
            dt = e.get("Exif.Photo.DateTimeOriginal") or e.get("Exif.Image.DateTime")
            if dt:
                d, t = dt.split(" ", 1); out["taken_at"] = d.replace(":", "-") + "T" + t
            out["camera"] = e.get("Exif.Image.Model")
            out["width"] = int(e.get("Exif.Photo.PixelXDimension", 0)) or None
            out["height"] = int(e.get("Exif.Photo.PixelYDimension", 0)) or None
        except Exception:
            pass
    return out
