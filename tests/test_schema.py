"""Tests for video2prompt.schema — the pipeline's shared data contract."""
import pytest
from pydantic import ValidationError

from video2prompt.schema import (
    Subject,
    CameraMotion,
    ColorInfo,
    ShotRecord,
    VideoAnalysis,
)


# ---------- Subject ----------

def test_subject_valid_construction():
    s = Subject(label="person", confidence=0.92, bbox_norm=(0.1, 0.2, 0.5, 0.8))
    assert s.label == "person"
    assert s.confidence == 0.92
    assert s.bbox_norm == (0.1, 0.2, 0.5, 0.8)
    assert s.track_id is None


def test_subject_missing_required_field_raises():
    with pytest.raises(ValidationError):
        Subject(confidence=0.9, bbox_norm=(0, 0, 1, 1))  # missing label


def test_subject_track_id_optional_defaults_none():
    s = Subject(label="car", confidence=0.5, bbox_norm=(0, 0, 1, 1))
    assert s.track_id is None
    s2 = Subject(label="car", confidence=0.5, bbox_norm=(0, 0, 1, 1), track_id=7)
    assert s2.track_id == 7


# ---------- CameraMotion ----------

def test_camera_motion_defaults():
    cm = CameraMotion()
    assert cm.type == "unknown"
    assert cm.direction is None
    assert cm.magnitude == "none"


def test_camera_motion_rejects_invalid_type():
    with pytest.raises(ValidationError):
        CameraMotion(type="dolly_zoom_supreme")  # not in the Literal set


def test_camera_motion_accepts_valid_literal():
    cm = CameraMotion(type="zoom_in", magnitude="fast")
    assert cm.type == "zoom_in"
    assert cm.magnitude == "fast"


# ---------- ColorInfo ----------

def test_color_info_defaults():
    c = ColorInfo()
    assert c.dominant_palette == []
    assert c.brightness == "mid"
    assert c.contrast == "medium"


def test_color_info_rejects_invalid_brightness():
    with pytest.raises(ValidationError):
        ColorInfo(brightness="super-bright")


# ---------- ShotRecord ----------

def _minimal_shot(**overrides):
    """Build a minimal valid ShotRecord, overriding fields as needed."""
    base = dict(
        shot_index=0,
        start_time=0.0,
        end_time=2.5,
        duration=2.5,
        resolution="1920x1080",
        fps=24.0,
    )
    base.update(overrides)
    return ShotRecord(**base)


def test_shot_record_minimal_construction_uses_defaults():
    shot = _minimal_shot()
    assert shot.subjects == []
    assert shot.shot_size_estimate == "unknown"
    assert shot.composition == "unknown"
    assert isinstance(shot.camera_motion, CameraMotion)
    assert isinstance(shot.color, ColorInfo)
    assert shot.pacing_note == ""
    assert shot.transcript_segment is None
    assert shot.on_screen_text is None
    assert shot.keyframe_paths == []
    assert shot.debug_overlay_paths == []
    assert shot.generated_prompt is None
    assert shot.scene_role is None
    assert shot.scene_description is None


def test_shot_record_missing_required_field_raises():
    with pytest.raises(ValidationError):
        ShotRecord(start_time=0.0, end_time=1.0, duration=1.0, resolution="1080p", fps=30)
        # missing shot_index


def test_shot_record_with_nested_subjects():
    subj = Subject(label="dog", confidence=0.8, bbox_norm=(0, 0, 0.3, 0.3))
    shot = _minimal_shot(subjects=[subj])
    assert len(shot.subjects) == 1
    assert shot.subjects[0].label == "dog"


def test_shot_record_round_trip_serialization():
    shot = _minimal_shot(generated_prompt="A wide shot of a city skyline at dusk.")
    data = shot.model_dump()
    rebuilt = ShotRecord(**data)
    assert rebuilt == shot


def test_shot_record_scene_breakdown_fields_optional():
    shot = _minimal_shot(scene_role="Hook / Problem Statement", scene_description="Opens on a cluttered desk.")
    assert shot.scene_role == "Hook / Problem Statement"
    assert shot.scene_description == "Opens on a cluttered desk."


# ---------- VideoAnalysis ----------

def test_video_analysis_minimal_construction():
    va = VideoAnalysis(source_path="clip.mp4", duration=10.0, fps=30.0, resolution="1280x720")
    assert va.shots == []
    assert va.master_prompt is None
    assert va.core_prompt is None


def test_video_analysis_with_shots_round_trip():
    shot = _minimal_shot()
    va = VideoAnalysis(
        source_path="clip.mp4",
        duration=2.5,
        fps=24.0,
        resolution="1920x1080",
        shots=[shot],
    )
    data = va.model_dump()
    rebuilt = VideoAnalysis(**data)
    assert len(rebuilt.shots) == 1
    assert rebuilt.shots[0].shot_index == 0


def test_video_analysis_missing_required_field_raises():
    with pytest.raises(ValidationError):
        VideoAnalysis(duration=10.0, fps=30.0, resolution="1280x720")  # missing source_path