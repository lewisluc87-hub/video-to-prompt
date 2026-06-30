"""Zero-network, zero-cost fallback: fills a fixed template straight from
the structured shot JSON. Used when --no-llm is passed or no API key is
configured. Lower quality than the LLM path, but deterministic and free.
"""
from __future__ import annotations

from pathlib import Path

from ..schema import ShotRecord
from .base import LLMProvider


class TemplateProvider(LLMProvider):
    def generate_shot_prompt(self, shot: ShotRecord, keyframe_paths: list[Path]) -> str:
        subject_desc = self._subject_phrase(shot)
        motion = shot.camera_motion
        motion_phrase = self._motion_phrase(motion.type, motion.direction, motion.magnitude)
        palette = ", ".join(shot.color.dominant_palette[:3]) or "neutral tones"

        parts = [
            f"{shot.shot_size_estimate.capitalize()} of {subject_desc}",
            f"composition: {shot.composition}",
            f"camera: {motion_phrase}",
            f"lighting: {shot.color.brightness}, {shot.color.contrast} contrast",
            f"palette: {palette}",
        ]
        return ". ".join(parts) + "."

    def generate_master_prompt(self, shot_prompts: list[str], video_meta: dict) -> str:
        return " Then, ".join(shot_prompts)

    def generate_scene_breakdown(self, shot, keyframe_paths: list[Path]) -> dict:
        text_lines = [line for line in (shot.on_screen_text or "").splitlines() if line.strip()]
        if shot.shot_index == 0:
            role = "Opening"
        elif any("comment" in t.lower() or "subscribe" in t.lower() or "link in" in t.lower() for t in text_lines):
            role = "CTA"
        else:
            role = f"Scene {shot.shot_index + 1}"
        if text_lines:
            desc = f"On-screen content shows: {'; '.join(text_lines[:3])}."
        elif shot.subjects:
            labels = ", ".join(s.label for s in shot.subjects[:3])
            desc = f"Visual content shows: {labels}."
        else:
            desc = "No on-screen text or detected subjects; visual content only."
        return {"scene_role": role, "scene_description": desc, "on_screen_text": text_lines}

    def generate_core_prompt(self, scenes: list[dict], video_meta: dict) -> str:
        roles = " -> ".join(s.get("scene_role", "Scene") for s in scenes)
        return f"A video moving through: {roles}."

    @staticmethod
    def _subject_phrase(shot: ShotRecord) -> str:
        if not shot.subjects:
            return "the scene"
        labels = [s.label for s in shot.subjects[:3]]
        return ", ".join(labels)

    @staticmethod
    def _motion_phrase(motion_type: str, direction: str | None, magnitude: str) -> str:
        if motion_type == "static" or magnitude == "none":
            return "static shot, no camera movement"
        phrase = motion_type.replace("_", " ")
        if direction:
            phrase += f" {direction}"
        return f"{magnitude} {phrase}"
