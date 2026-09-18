import os
import pytest
from PIL import Image
from conftest import make_video, needs_ffmpeg

pytestmark = needs_ffmpeg


@pytest.fixture
def two_scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    return make_video(d / "two.mp4", scenes=2, work=d / "work")


@pytest.fixture
def one_scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    return make_video(d / "one.mp4", scenes=1, work=d / "work")


def test_probe_reports_duration_and_size(one_scene, two_scene):
    from photosort.video import probe
    info = probe(one_scene)
    assert abs(info["duration"] - 3.0) < 0.2 and info["width"] == 320 and info["height"] == 240
    assert abs(probe(two_scene)["duration"] - 10.0) < 0.2


def test_sample_times_are_evenly_spaced_between_5_and_95_percent():
    from photosort.video import sample_times
    t = sample_times(10.0, n=6)
    assert len(t) == 6 and t[0] == pytest.approx(0.5) and t[-1] == pytest.approx(9.5)
    gaps = [b - a for a, b in zip(t, t[1:])]
    assert all(g == pytest.approx(gaps[0]) for g in gaps)
    assert sample_times(10.0, n=1) == [pytest.approx(5.0)]
    assert sample_times(0.0) == [0.0]


def test_frame_at_decodes_one_frame(one_scene, tmp_path):
    from photosort.video import frame_at
    im = frame_at(one_scene, 1.0)
    assert isinstance(im, Image.Image) and im.size == (320, 240) and im.mode == "RGB"
    small = frame_at(one_scene, 1.0, edge=160)
    assert max(small.size) == 160
    (tmp_path / "junk.mp4").write_bytes(b"not a video")
    assert frame_at(tmp_path / "junk.mp4", 0.5) is None


def test_scene_cuts_finds_the_one_hard_cut(two_scene, one_scene):
    from photosort.video import scene_cuts
    cuts = scene_cuts(two_scene)
    assert len(cuts) == 1 and abs(cuts[0] - 5.0) < 0.2
    assert scene_cuts(one_scene) == []


def test_segments_from_cuts_merges_short_ones_and_caps_the_count():
    from photosort.video import segments_from_cuts
    assert segments_from_cuts([], 4.0) == [(0.0, 4.0)]
    assert segments_from_cuts([2.0], 4.0) == [(0.0, 2.0), (2.0, 4.0)]
    # a cut 0.3 s after another makes a segment shorter than MIN_SEGMENT_S: merged into the previous one
    assert segments_from_cuts([2.0, 2.3, 3.0], 4.0) == [(0.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    # a cut too close to the end is dropped too
    assert segments_from_cuts([2.0, 3.8], 4.0) == [(0.0, 2.0), (2.0, 4.0)]
    # more than the cap: the shortest segments are folded into a neighbour until the cap holds
    cuts = [float(i) for i in range(1, 40)]
    segs = segments_from_cuts(cuts, 40.0, max_segments=24)
    assert len(segs) == 24 and segs[0][0] == 0.0 and segs[-1][1] == 40.0
    assert all(b[0] == a[1] for a, b in zip(segs, segs[1:]))          # contiguous, no gaps


def test_sample_frames_returns_even_frames_plus_segment_midpoints(two_scene):
    from photosort.video import sample_frames, sample_times, probe
    d = probe(two_scene)["duration"]
    frames, segs = sample_frames(two_scene, d)
    assert len(segs) == 2 and segs[0][0] == 0.0 and abs(segs[0][1] - 5.0) < 0.2 and abs(segs[1][1] - d) < 1e-6
    times = [t for t, _ in frames]
    assert times == sorted(times) and len(set(times)) == len(times)
    for t in sample_times(d):
        assert t in times
    # every segment midpoint has a frame within half a second (here the even samples already cover both)
    for s, e in segs:
        mid = (s + e) / 2
        assert min(abs(t - mid) for t in times) <= 0.5
    assert len(frames) == 6                                              # both midpoints deduplicated
    assert all(isinstance(im, Image.Image) for _, im in frames)


def test_sample_frames_adds_a_midpoint_frame_when_no_even_sample_is_near(two_scene, monkeypatch):
    import photosort.video as v
    monkeypatch.setattr(v, "sample_times", lambda d, n=6: [0.5, 9.5])   # nothing near the midpoints 2.5 and 7.5
    frames, segs = v.sample_frames(two_scene, 10.0)
    assert len(segs) == 2 and abs(segs[0][1] - 5.0) < 0.2
    times = [t for t, _ in frames]
    assert times[0] == 0.5 and times[-1] == 9.5 and len(times) == 4
    assert any(abs(t - 2.5) < 0.1 for t in times) and any(abs(t - 7.5) < 0.1 for t in times)


def test_short_clips_skip_the_scene_pass(one_scene, monkeypatch):
    import photosort.video as v
    def boom(path, threshold=0.4):
        raise AssertionError("scene pass ran on a clip shorter than SCENE_MIN_DURATION_S")
    monkeypatch.setattr(v, "scene_cuts", boom)
    frames, segs = v.sample_frames(one_scene, 3.0)
    assert segs == [(0.0, 3.0)] and len(frames) == 6


def test_hwaccel_is_dropped_after_one_failure_and_retried_without(one_scene, monkeypatch):
    import subprocess as sp
    import photosort.video as v
    monkeypatch.setattr(v.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(v, "FFMPEG_HWACCEL", "videotoolbox")
    monkeypatch.setattr(v, "_HWACCEL_OK", True)
    real_run = sp.run
    calls = []
    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        if "-hwaccel" in cmd:
            return sp.CompletedProcess(cmd, 1, b"", b"hwaccel init failed")
        return real_run(cmd, **kw)
    monkeypatch.setattr(v.subprocess, "run", fake_run)
    im = v.frame_at(one_scene, 1.0)
    assert im is not None and im.size == (320, 240)
    assert len(calls) == 2
    assert "-hwaccel" in calls[0] and calls[0].index("-hwaccel") < calls[0].index("-i") and calls[0][calls[0].index("-hwaccel") + 1] == "videotoolbox"
    assert "-hwaccel" not in calls[1]
    assert v._HWACCEL_OK is False
    assert v.frame_at(one_scene, 1.5) is not None
    assert len(calls) == 3 and "-hwaccel" not in calls[2]                 # remembered: no retry dance next time
    # off the Mac, or with the setting cleared, no hwaccel flag at all
    monkeypatch.setattr(v, "_HWACCEL_OK", True); monkeypatch.setattr(v, "FFMPEG_HWACCEL", "")
    assert v.frame_at(one_scene, 1.0) is not None and "-hwaccel" not in calls[-1]


def test_scene_pass_decodes_keyframes_only(two_scene, monkeypatch):
    import subprocess as sp
    import photosort.video as v
    seen = []
    real_run = sp.run
    def spy(cmd, **kw):
        seen.append(list(cmd)); return real_run(cmd, **kw)
    monkeypatch.setattr(v.subprocess, "run", spy)
    assert len(v.scene_cuts(two_scene)) == 1
    cmd = seen[-1]
    assert "-skip_frame" in cmd and cmd[cmd.index("-skip_frame") + 1] == "nokey" and cmd.index("-skip_frame") < cmd.index("-i")
    assert "scale=320:-2,select=" in cmd[cmd.index("-vf") + 1]


def test_unreadable_video_raises(tmp_path):
    from photosort.video import sample_frames, VideoUnreadable, probe
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"\x00" * 4096)
    with pytest.raises(VideoUnreadable):
        probe(bad)
    with pytest.raises(VideoUnreadable):
        sample_frames(bad, 3.0)


def test_missing_ffmpeg_is_a_clear_error_not_a_crash(one_scene, monkeypatch):
    import photosort.video as v
    monkeypatch.setattr(v, "_FALLBACK_DIRS", ())
    monkeypatch.setattr(v.shutil, "which", lambda name: None)
    with pytest.raises(v.VideoUnreadable, match="ffprobe not found"):
        v.probe(one_scene)
    with pytest.raises(v.VideoUnreadable, match="ffmpeg not found"):
        v.sample_frames(one_scene, 3.0)


def test_export_segments_writes_one_trimmed_clip_per_segment(tmp_path, tmp_path_factory):
    from photosort import db
    from photosort.index import index_folder
    from photosort.export import export_segments
    from photosort.video import probe
    make_video(tmp_path / "clip.mp4", scenes=2, work=tmp_path_factory.mktemp("work"))
    before = sorted(os.listdir(tmp_path))
    index_folder(tmp_path, faces=False, workers=1, embed=False)
    conn = db.connect(tmp_path)
    pid = conn.execute("SELECT id FROM photos WHERE rel='clip.mp4'").fetchone()[0]
    base = tmp_path_factory.mktemp("out")
    seen = []
    out = export_segments(tmp_path, [pid], None, "copy", base, seen.append)
    assert out == base / tmp_path.resolve().name / "segments"
    files = sorted(p.name for p in out.iterdir() if p.suffix == ".mp4")
    assert files == ["clip_00_0.0-5.0.mp4", "clip_01_5.0-10.0.mp4"]
    for f in files:
        assert abs(probe(out / f)["duration"] - 5.0) < 0.5                # stream copy: cut lands on a keyframe
    assert seen[-1] == {"done": 2, "total": 2, "failed": 0, "skipped": 0}
    # only the segments whose category matches
    conn.execute("UPDATE segments SET category='beach' WHERE photo_id=? AND idx=1", (pid,)); conn.commit()
    out2 = export_segments(tmp_path, [pid], "beach", "copy", tmp_path_factory.mktemp("out2"), None)
    assert [p.name for p in out2.iterdir() if p.suffix == ".mp4"] == ["clip_01_5.0-10.0.mp4"]
    # a re-run over the same folder skips what is there
    seen2 = []
    export_segments(tmp_path, [pid], None, "copy", base, seen2.append)
    assert seen2[-1] == {"done": 2, "total": 2, "failed": 0, "skipped": 2}
    assert sorted(os.listdir(tmp_path)) == before
