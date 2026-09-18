"""Videos without an LLM: ffprobe for the facts, ffmpeg for a handful of frames and the scene cuts.
Only ever reads the source file. Every call goes through subprocess with a timeout, so a broken clip
costs a wait, never a hang."""
from __future__ import annotations
import io, json, os, platform, re, shutil, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image
from .config import PREVIEW_EDGE, VIDEO_FRAMES, SCENE_THRESHOLD, MAX_SEGMENTS, MIN_SEGMENT_S, SCENE_MIN_DURATION_S, FFMPEG_HWACCEL

class VideoUnreadable(RuntimeError):
    """ffmpeg/ffprobe is missing, or the file gave no usable frame."""

# Homebrew's bin is not on the PATH of an app started outside a terminal (launchd, Finder).
_FALLBACK_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
FRAME_TIMEOUT = 60          # seconds for one frame
SCENE_TIMEOUT = 900         # seconds for the one-pass scene scan of a whole clip (4K at 320 px wide)
DEDUP_S = 0.5               # a segment midpoint this close to an even sample reuses that frame

def _bin(name: str) -> str:
    if os.environ.get("PHOTOSORT_NO_FFMPEG"):        # tests: behave as if ffmpeg were not installed
        raise VideoUnreadable(f"{name} not found; install it with: brew install ffmpeg")
    found = shutil.which(name)
    if found:
        return found
    for d in _FALLBACK_DIRS:
        p = Path(d) / name
        if p.is_file():
            return str(p)
    raise VideoUnreadable(f"{name} not found; install it with: brew install ffmpeg")

def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)

# Hardware decode (videotoolbox) makes a 4K h264 pass several times faster on a Mac. Not every codec is
# accelerated, so a decode that fails with it is retried once without, and the process remembers.
_HWACCEL_OK = True

def _hwaccel_args() -> list[str]:
    if FFMPEG_HWACCEL and _HWACCEL_OK and platform.system() == "Darwin":
        return ["-hwaccel", FFMPEG_HWACCEL]
    return []

def _decode(pre: list[str], path: Path, post: list[str], timeout: float) -> subprocess.CompletedProcess:
    """ffmpeg -nostdin -v error [-hwaccel X] <pre> -i <path> <post>, retried once without the hwaccel
    when it was on and the command failed; hwaccel stays off for the process only when that retry works."""
    global _HWACCEL_OK
    ff = _bin("ffmpeg")
    hw = _hwaccel_args()
    out = _run([ff, "-nostdin", "-v", "error", *hw, *pre, "-i", str(path), *post], timeout)
    if out.returncode != 0 and hw:
        # Only a retry that succeeds proves the codec is not accelerated; a file that fails both
        # ways says nothing about hwaccel, so it must not switch it off for every later clip.
        retry = _run([ff, "-nostdin", "-v", "error", *pre, "-i", str(path), *post], timeout)
        if retry.returncode == 0:
            _HWACCEL_OK = False
        return retry
    return out

def _norm_time(s: str | None) -> str | None:
    """ffprobe's 2026-09-18T10:00:00.000000Z -> 2026-09-18T10:00:00, the shape exif_info produces."""
    if not s:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})", s)
    return f"{m.group(1)}T{m.group(2)}" if m else None

def probe(path: Path) -> dict:
    """duration (s), width, height of the first video stream, plus taken_at (the creation_time tag,
    normalised) and camera (make/model tags) when the container carries them, else None."""
    out = _run([_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], timeout=60)
    try:
        info = json.loads(out.stdout or b"{}")
    except ValueError:
        info = {}
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if out.returncode != 0 or not streams:
        raise VideoUnreadable(f"{path}: ffprobe found no video stream ({(out.stderr or b'').decode(errors='replace').strip()[:200]})")
    v = streams[0]; fmt = info.get("format", {})
    def _f(x):
        try: return float(x)
        except (TypeError, ValueError): return 0.0
    duration = _f(fmt.get("duration")) or _f(v.get("duration"))
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(_f(sd["rotation"])))
    if abs(rot) % 180 == 90:                          # ffmpeg autorotates the frames; report the upright size
        w, h = h, w
    tags = dict(fmt.get("tags", {}) or {}); tags.update(v.get("tags", {}) or {})
    low = {k.lower(): str(val) for k, val in tags.items()}
    model = next((low[k] for k in ("com.apple.quicktime.model", "model", "com.android.model") if low.get(k)), None)
    make = next((low[k] for k in ("com.apple.quicktime.make", "make", "com.android.manufacturer") if low.get(k)), None)
    camera = None
    if model:
        camera = (f"{make} {model}" if make and make not in model else model).strip()
    return dict(duration=duration, width=w, height=h, taken_at=_norm_time(low.get("creation_time")), camera=camera)

def sample_times(duration: float, n: int = VIDEO_FRAMES) -> list[float]:
    """n instants evenly spaced between 5% and 95% of the clip (the ends are often slates, black or a shaky start)."""
    if duration <= 0:
        return [0.0]
    if n <= 1:
        return [duration / 2]
    lo, hi = 0.05 * duration, 0.95 * duration
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

def frame_at(path: Path, t: float, edge: int = PREVIEW_EDGE) -> Image.Image | None:
    """One frame at t seconds, longest edge at most `edge`, as an RGB image; None when ffmpeg gives nothing.
    Input seeking (-ss before -i) lands on the nearest keyframe first and decodes forward, fast on h264."""
    try:
        out = _decode(["-ss", f"{max(t, 0.0):.3f}"], path,
                      ["-frames:v", "1", "-vf", f"scale='min({int(edge)},iw)':-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"],
                      timeout=FRAME_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    try:
        im = Image.open(io.BytesIO(out.stdout)); im.load()
        return im.convert("RGB")
    except Exception:
        return None

def scene_cuts(path: Path, threshold: float = SCENE_THRESHOLD) -> list[float]:
    """Instants (s) where ffmpeg's scene score jumps above threshold, one pass over the keyframes only
    (-skip_frame nokey, an input option) at 320 px wide. Cuts therefore land on keyframes, which is where
    a stream-copy export can cut anyway. A pass that fails or times out reports no cuts (the clip
    becomes one segment), never raises. showinfo prints on stderr, so -v error is overridden to info here."""
    try:
        out = _decode(["-skip_frame", "nokey"], path,
                      ["-v", "info", "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
                      timeout=SCENE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return []
    if out.returncode != 0:
        return []
    times = [float(m) for m in re.findall(rb"pts_time:\s*([0-9]+(?:\.[0-9]+)?)", out.stderr or b"")]
    return sorted(set(times))

def segments_from_cuts(cuts: list[float], duration: float, min_s: float = MIN_SEGMENT_S,
                       max_segments: int = MAX_SEGMENTS) -> list[tuple[float, float]]:
    """[0, c1), [c1, c2), ..., [cn, duration). A cut that would leave a segment shorter than min_s
    (against the previous kept cut, or against the end) is dropped, which merges it into the previous
    segment. Over max_segments, the shortest segment is folded into its shorter neighbour until the
    cap holds, so the longest stretches survive. A clip with no cuts is one segment."""
    duration = float(duration)
    if duration <= 0:
        return [(0.0, 0.0)]
    bounds = [0.0]
    for c in sorted(float(c) for c in cuts):
        if c - bounds[-1] < min_s or duration - c < min_s:
            continue
        bounds.append(c)
    bounds.append(duration)
    segs = [(a, b) for a, b in zip(bounds, bounds[1:])]
    while len(segs) > max(1, max_segments):
        i = min(range(len(segs)), key=lambda k: segs[k][1] - segs[k][0])
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = i - 1
        else:
            j = i - 1 if (segs[i - 1][1] - segs[i - 1][0]) <= (segs[i + 1][1] - segs[i + 1][0]) else i + 1
        a, b = min(i, j), max(i, j)
        segs[a:b + 1] = [(segs[a][0], segs[b][1])]
    return segs

def sample_frames(path: Path, duration: float) -> tuple[list[tuple[float, Image.Image]], list[tuple[float, float]]]:
    """The evenly spaced frames plus one frame at each segment midpoint (skipped when an even sample sits
    within DEDUP_S of it), sorted by time, and the segment list. A clip shorter than
    SCENE_MIN_DURATION_S skips the scene pass and is one segment. Raises VideoUnreadable when not one
    frame decodes."""
    _bin("ffmpeg")
    cuts = scene_cuts(path) if duration >= SCENE_MIN_DURATION_S else []
    segs = segments_from_cuts(cuts, duration)
    times = list(sample_times(duration))
    for a, b in segs:
        mid = (a + b) / 2
        if all(abs(mid - t) > DEDUP_S for t in times):
            times.append(mid)
    frames: list[tuple[float, Image.Image]] = []
    for t in sorted(times):
        im = frame_at(path, t)
        if im is not None:
            frames.append((t, im))
    if not frames:
        raise VideoUnreadable(f"{path}: no frame could be decoded")
    return frames, segs

# Log thumbnails, display only. Sony writes <stem>M01.XML next to each clip with the capture gamma; a clip
# shot in S-Log3 is converted to Rec.709 before its frames are saved, so the grid is not grey mush and the
# embedder sees a normal-contrast frame. Labels were identical either way on the first shoot, so this is a
# thumbnail improvement, not a classification claim. DJI D-Log M has no published curve: left alone.

SIDECAR_MAX_BYTES = 65536

def capture_gamma(path: Path) -> str | None:
    """The CaptureGammaEquation value ("s-log3-cine", "s-log3", ...) from Sony's <stem>M01.XML sidecar, read
    only, first SIDECAR_MAX_BYTES only; None when there is no sidecar or it does not parse."""
    path = Path(path)
    sidecar = path.with_name(path.stem + "M01.XML")
    try:
        with open(sidecar, "rb") as fh:
            head = fh.read(SIDECAR_MAX_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r'CaptureGammaEquation"\s+value="([^"]+)"', head)
    return m.group(1).strip().lower() if m else None

def is_slog3(gamma: str | None) -> bool:
    return bool(gamma) and gamma.lower().startswith("s-log3")

# Sony's published S-Log3 curve (10-bit code value cv, 18% grey at cv 420, black at cv 95) inverted to
# scene-linear, one entry per 8-bit level.
def _slog3_lut() -> np.ndarray:
    cv = np.arange(256, dtype=np.float64) / 255.0 * 1023.0
    knee = 171.2102946929
    lin = np.where(cv >= knee,
                   10.0 ** ((cv - 420.0) / 261.5) * (0.18 + 0.01) - 0.01,
                   (cv - 95.0) * 0.01125 / (knee - 95.0))
    return lin.astype(np.float32)

_SLOG3_LIN = _slog3_lut()
# S-Gamut3.Cine -> Rec.709 primaries (rows sum to 1, so grey stays grey).
_SGAMUT3CINE_TO_709 = np.array([[1.6269474, -0.5401385, -0.0868088],
                                [-0.1785155, 1.4179409, -0.2394254],
                                [-0.0273959, -0.0916826, 1.1190786]], np.float32)

def slog3_to_rec709(im: Image.Image) -> Image.Image:
    """An S-Log3 / S-Gamut3.Cine frame as a Rec.709 RGB image of the same size: LUT to linear per channel,
    the 3x3 matrix, clip to 0..1, then the Rec.709 OETF (4.5 L below 0.018, else 1.099 L^0.45 - 0.099).
    cv 420 (8-bit 105, 18% grey) lands near 0.41; black stays black."""
    arr = np.asarray(im.convert("RGB"))
    lin = _SLOG3_LIN[arr]                                  # (H, W, 3) scene-linear
    rgb = np.clip(lin @ _SGAMUT3CINE_TO_709.T, 0.0, 1.0)
    out = np.where(rgb < 0.018, 4.5 * rgb, 1.099 * np.power(rgb, 0.45) - 0.099)
    return Image.fromarray(np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8), "RGB")

def mtime_iso(mtime: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
