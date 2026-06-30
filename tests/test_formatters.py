from video2prompt.formatters import format_shot
from video2prompt.schema import CameraMotion, ColorInfo, ShotRecord, Subject


def make_shot() -> ShotRecord:
    return ShotRecord(
        shot_index=0,
        start_time=0.0,
        end_time=3.0,
        duration=3.0,
        resolution="1920x1080",
        fps=30,
        subjects=[Subject(label="dog", confidence=0.9, bbox_norm=(0.3, 0.2, 0.6, 0.9))],
        shot_size_estimate="medium shot",
        composition="center-weighted, subject centered",
        camera_motion=CameraMotion(type="zoom_in", magnitude="medium"),
        color=ColorInfo(dominant_palette=["#ffffff", "#000000"], brightness="mid", contrast="medium"),
        generated_prompt="A dog runs across a sunlit field",
    )


def test_generic_passthrough():
    shot = make_shot()
    assert format_shot(shot, "generic") == shot.generated_prompt


def test_sora_adds_cinematic_prefix():
    shot = make_shot()
    out = format_shot(shot, "sora")
    assert out.startswith("A cinematic shot:")


def test_runway_is_comma_separated():
    shot = make_shot()
    out = format_shot(shot, "runway")
    assert "dog" in out
    assert "," in out


def test_veo_includes_camera_clause():
    shot = make_shot()
    out = format_shot(shot, "veo")
    assert "Camera:" in out
    assert "zoom in" in out


def test_unknown_target_raises():
    shot = make_shot()
    try:
        format_shot(shot, "nonexistent")
        assert False, "expected ValueError"
    except ValueError:
        pass
