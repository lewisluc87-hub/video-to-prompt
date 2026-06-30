"""Stage 3: per-shot keyframe sampling via OpenCV."""
from __future__ import annotations

from pathlib import Path

import cv2

from .ingest import VideoInfo
from .segment import Shot


def sample_keyframes(
    video: VideoInfo,
    shot: Shot,
    out_dir: Path,
    max_frames: int = 3,
) -> list[Path]:
    """Extract up to `max_frames` evenly spaced frames within [start, end).

    Always includes the first frame; adds middle/last as the shot gets longer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video.path}")

    fps = video.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(shot.start_time * fps)
    end_frame = max(start_frame + 1, int(shot.end_time * fps))

    n = max(1, min(max_frames, end_frame - start_frame))
    if n == 1:
        target_frames = [start_frame]
    else:
        step = (end_frame - start_frame - 1) / (n - 1)
        target_frames = [int(start_frame + i * step) for i in range(n)]

    paths: list[Path] = []
    for i, frame_idx in enumerate(target_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = out_dir / f"shot{shot.index}_f{i}.jpg"
        cv2.imwrite(str(out_path), frame)
        paths.append(out_path)

    cap.release()
    return paths
