from pathlib import Path

from video2prompt.ingest import VideoInfo
from video2prompt.segment import _single_shot_fallback


def test_single_shot_fallback_spans_full_duration():
    video = VideoInfo(
        path=Path("fake.mp4"), duration=12.5, fps=30, width=1920, height=1080, has_audio=False
    )
    shots = _single_shot_fallback(video)
    assert len(shots) == 1
    assert shots[0].start_time == 0.0
    assert shots[0].end_time == 12.5
