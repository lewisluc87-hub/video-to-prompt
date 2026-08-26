"""Data models shared across the pipeline.

These are the contract between pipeline stages. See SPEC.md section 5 for the
canonical JSON shape this maps to.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Subject(BaseModel):
    label: str
    confidence: float
    bbox_norm: tuple[float, float, float, float]  # x0, y0, x1, y1, normalized 0-1
    track_id: int | None = None


class CameraMotion(BaseModel):
    type: Literal["static", "pan", "tilt", "zoom_in", "zoom_out", "tracking", "unknown"] = "unknown"
    direction: str | None = None
    magnitude: Literal["none", "slow", "medium", "fast"] = "none"


class ColorInfo(BaseModel):
    dominant_palette: list[str] = Field(default_factory=list)  # hex strings
    brightness: Literal["low-key", "mid", "high-key"] = "mid"
    contrast: Literal["low", "medium", "high"] = "medium"


class ShotRecord(BaseModel):
    shot_index: int
    start_time: float
    end_time: float
    duration: float
    resolution: str
    fps: float

    subjects: list[Subject] = Field(default_factory=list)
    shot_size_estimate: str = "unknown"
    composition: str = "unknown"
    camera_motion: CameraMotion = Field(default_factory=CameraMotion)
    color: ColorInfo = Field(default_factory=ColorInfo)
    pacing_note: str = ""

    transcript_segment: str | None = None
    on_screen_text: str | None = None

    keyframe_paths: list[str] = Field(default_factory=list)
    debug_overlay_paths: list[str] = Field(default_factory=list)

    generated_prompt: str | None = None

    # Scene-breakdown mode fields (see SPEC.md section on breakdown output)
    scene_role: str | None = None  # e.g. "Hook / Problem Statement"
    scene_description: str | None = None  # narrative prose for this scene


class VideoAnalysis(BaseModel):
    source_path: str
    duration: float
    fps: float
    resolution: str
    shots: list[ShotRecord] = Field(default_factory=list)
    master_prompt: str | None = None
    core_prompt: str | None = None  # one-sentence summary, breakdown mode
