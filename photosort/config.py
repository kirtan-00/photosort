import os
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

CLIP_MODEL = "MobileCLIP-S1"
CLIP_PRETRAINED = "datacompdr"
EMBED_DIM = 512

INDEX_DIRNAME = ".photosort"
DB_NAME = "index.db"
PREVIEW_EDGE = 1024
GRID_EDGE = 320
THUMB_QUALITY = 85

STD_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"}
RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2"}
IMAGE_EXTS = STD_EXTS | RAW_EXTS

JPEG_WORKERS = 4
RAW_WORKERS = 2
EMBED_BATCH = 32

FACE_SCORE_MIN = 0.7
FACE_CLUSTER_EPS = 0.5      # cosine distance; tune on real data
FACE_MIN_SAMPLES = 2
GROUP_MIN_FACES = 3
# OpenCV's published SFace cosine threshold is 0.363, but that is for verified crops. On a real
# event shoot with thousands of small faces, 0.5 and below is noise (a different bearded man at
# 0.5, 1,039 "matches" at 0.3); 0.55 keeps the same person across lighting and sunglasses.
FACE_MATCH_MIN_SIM = 0.55
FACE_REF_MIN_EDGE = 48      # a reference face smaller than this (preview px, 1024 decode) is not trusted

SOFT_PERCENTILE = 15        # bottom 15% of sharpness in a shoot = "soft"
SHARP_TILE_GRID = 8

def app_home() -> Path:
    return Path(os.environ.get("PHOTOSORT_HOME") or (Path.home() / "Library" / "Application Support" / "photosort"))

def export_root() -> Path:
    return Path(os.environ.get("PHOTOSORT_EXPORT_DIR") or (Path.home() / "Desktop" / "photosort-out"))

def settings_path() -> Path:
    return app_home() / "settings.json"

def shoot_slug(root: Path) -> str:
    r = Path(root).resolve()
    return f"{r.name or 'root'}-{hashlib.sha1(str(r).encode()).hexdigest()[:8]}"
