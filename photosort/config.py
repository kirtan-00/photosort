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

SOFT_PERCENTILE = 15        # bottom 15% of sharpness in a shoot = "soft"
SHARP_TILE_GRID = 8
