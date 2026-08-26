from video2prompt.llm.template_fallback import TemplateProvider
from video2prompt.schema import CameraMotion, ColorInfo, ShotRecord, Subject


def make_shot(**overrides) -> ShotRecord:
    defaults = {
        "shot_index": 0,
        "start_time": 0.0,
        "end_time": 3.0,
        "duration": 3.0,
        "resolution": "1920x1080",
        "fps": 30,
        "subjects": [Subject(label="person", confidence=0.9, bbox_norm=(0.3, 0.2, 0.6, 0.9))],
        "shot_size_estimate": "medium close-up",
        "composition": "rule-of-thirds, subject left",
        "camera_motion": CameraMotion(type="pan", direction="left-to-right", magnitude="slow"),
        "color": ColorInfo(dominant_palette=["#112233", "#aabbcc"], brightness="low-key", contrast="high"),
    }    
    defaults.update(overrides)
    return ShotRecord(**defaults)


def test_template_provider_is_deterministic():
    provider = TemplateProvider()
    shot = make_shot()
    a = provider.generate_shot_prompt(shot, [])
    b = provider.generate_shot_prompt(shot, [])
    assert a == b
    assert "person" in a
    assert "pan" in a or "left-to-right" in a


def test_template_provider_handles_no_subjects():
    provider = TemplateProvider()
    shot = make_shot(subjects=[])
    prompt = provider.generate_shot_prompt(shot, [])
    assert "the scene" in prompt


def test_master_prompt_joins_shots():
    provider = TemplateProvider()
    merged = provider.generate_master_prompt(["A.", "B."], {})
    assert "A." in merged and "B." in merged
