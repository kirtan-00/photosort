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
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mts", ".avi"}
# Folder names the walk never enters, wherever they sit: photosort-out is our own export folder when it
# sits inside a shoot (dot-dirs are skipped too, in walk.py).
SKIP_DIRS = {"photosort-out"}
# Folder names pruned only directly under a directory named M4ROOT (any case), the Sony card layout
# (PRIVATE/M4ROOT): one poster JPEG per clip under THMBNL (160 px: 12 of the 16 "other" photos on the
# first video shoot), proxy clips under SUB (C0001S03.MP4, duplicates of CLIP/; when a card carries proxies
# they could feed frame sampling later, for now they are duplicates and must not appear in the grid), and
# bookkeeping under TAKE and GENERAL. Scoped because SUB, TAKE and GENERAL are ordinary words a client's
# own folders may use.
SONY_CARD_DIRS = {"THMBNL", "SUB", "TAKE", "GENERAL"}
SONY_CARD_ROOT = "M4ROOT"

# Videos: no LLM, no faces. ffmpeg samples frames, CLIP embeds them, the clip's embedding is their mean.
VIDEO_FRAMES = 6            # evenly spaced between 5% and 95% of the duration
VIDEO_WORKERS = 3           # each worker runs its own ffmpeg, which is multi-threaded already
SCENE_THRESHOLD = 0.4       # ffmpeg scene score above which two keyframes are a cut
SCENE_MIN_DURATION_S = 8.0  # shorter clips skip the scene pass and are one segment
SCENE_MAX_DURATION_S = 300.0  # longer clips skip it too and get fixed windows: the keyframe pass is decode-bound
LONG_SEGMENT_S = 120.0      # the window on a long clip (widened evenly when MAX_SEGMENTS would be exceeded)
FFMPEG_HWACCEL = "videotoolbox"   # macOS hardware decode; retried without it once if a codec is not accelerated
MAX_SEGMENTS = 24           # longest segments kept when a clip has more cuts than this
MIN_SEGMENT_S = 1.0         # a cut that would leave a shorter segment is merged into the previous one

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

# Fixed categories: the bin for photos the confidence gates rejected, and the probability below which a
# match (a category score, a discovered cluster's rescaled cosine) is "less sure": listed after a divider,
# sorted by confidence, exported only on request.
CATEGORY_FALLBACK = "other"
SURE_MIN = 0.5

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
