"""Videos without an LLM: ffprobe for the facts, ffmpeg for a handful of frames and the scene cuts.
Only ever reads the source file. Every call goes through subprocess with a timeout, so a broken clip
costs a wait, never a hang."""
from __future__ import annotations
import io, json, os, re, shutil, subprocess, time
from pathlib import Path
from PIL import Image
from .config import PREVIEW_EDGE, VIDEO_FRAMES, SCENE_THRESHOLD, MAX_SEGMENTS, MIN_SEGMENT_S

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
        ff = _bin("ffmpeg")
    except VideoUnreadable:
        raise
    cmd = [ff, "-nostdin", "-v", "error", "-ss", f"{max(t, 0.0):.3f}", "-i", str(path), "-frames:v", "1",
           "-vf", f"scale='min({int(edge)},iw)':-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"]
    try:
        out = _run(cmd, timeout=FRAME_TIMEOUT)
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
    """Instants (s) where ffmpeg's scene score jumps above threshold, one decode pass at 320 px wide.
    A pass that fails or times out reports no cuts (the clip becomes one segment), never raises."""
    ff = _bin("ffmpeg")
    cmd = [ff, "-nostdin", "-i", str(path), "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo",
           "-an", "-f", "null", "-"]
    try:
        out = _run(cmd, timeout=SCENE_TIMEOUT)
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
    within DEDUP_S of it), sorted by time, and the segment list. Raises VideoUnreadable when not one
    frame decodes."""
    _bin("ffmpeg")
    segs = segments_from_cuts(scene_cuts(path), duration)
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

def mtime_iso(mtime: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
