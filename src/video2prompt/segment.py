"""Stage 2: shot/scene segmentation.

Wraps PySceneDetect's content-aware detector. Falls back to a single
whole-video "shot" if PySceneDetect isn't installed or finds no cuts,
so the rest of the pipeline always has at least one shot to work with.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ingest import VideoInfo


@dataclass
class Shot:
    index: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def detect_shots(video: VideoInfo, threshold: float = 27.0) -> list[Shot]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ImportError:
        return _single_shot_fallback(video)

    video_stream = open_video(str(video.path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video_stream)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        return _single_shot_fallback(video)

    shots = []
    for i, (start, end) in enumerate(scene_list):
        shots.append(Shot(index=i, start_time=start.get_seconds(), end_time=end.get_seconds()))
    return shots


def _single_shot_fallback(video: VideoInfo) -> list[Shot]:
    return [Shot(index=0, start_time=0.0, end_time=video.duration or 0.0)]
