from video2prompt.llm.template_fallback import TemplateProvider
from video2prompt.schema import CameraMotion, ColorInfo, ShotRecord


def make_shot(on_screen_text=None, index=0) -> ShotRecord:
    return ShotRecord(
        shot_index=index,
        start_time=0.0,
        end_time=3.0,
        duration=3.0,
        resolution="1920x1080",
        fps=30,
        camera_motion=CameraMotion(),
        color=ColorInfo(),
        on_screen_text=on_screen_text,
    )


def test_breakdown_extracts_on_screen_text_lines():
    provider = TemplateProvider()
    shot = make_shot(on_screen_text="My views didn't move.\nNot vidIQ's fault.")
    result = provider.generate_scene_breakdown(shot, [])
    assert result["on_screen_text"] == ["My views didn't move.", "Not vidIQ's fault."]
    assert "My views didn't move." in result["scene_description"]


def test_breakdown_detects_cta_role():
    provider = TemplateProvider()
    shot = make_shot(on_screen_text="COMMENT VIDIQ\nand I'll send you the link.", index=4)
    result = provider.generate_scene_breakdown(shot, [])
    assert result["scene_role"] == "CTA"


def test_breakdown_handles_no_text():
    provider = TemplateProvider()
    shot = make_shot(on_screen_text=None)
    result = provider.generate_scene_breakdown(shot, [])
    assert result["on_screen_text"] == []
    assert "No on-screen text" in result["scene_description"]


def test_core_prompt_chains_roles():
    provider = TemplateProvider()
    scenes = [{"scene_role": "Hook"}, {"scene_role": "CTA"}]
    core = provider.generate_core_prompt(scenes, {})
    assert "Hook" in core and "CTA" in core
