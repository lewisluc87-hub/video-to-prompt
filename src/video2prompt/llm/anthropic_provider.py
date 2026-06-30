"""Anthropic-backed LLM provider. Requires ANTHROPIC_API_KEY in env."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from ..schema import ShotRecord
from .base import BREAKDOWN_SYSTEM_PROMPT, CORE_PROMPT_SYSTEM_PROMPT, SYSTEM_PROMPT, LLMProvider

_MODEL = "claude-sonnet-4-6"


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = _MODEL):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed. Run `pip install video2prompt[llm]`, "
                "or use --no-llm for template-only mode."
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it, or run with --no-llm "
                "for fully offline template mode."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate_shot_prompt(self, shot: ShotRecord, keyframe_paths: list[Path]) -> str:
        content = [{"type": "text", "text": self._shot_context_text(shot)}]
        for p in keyframe_paths[:3]:
            content.append(_image_block(p))

        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        return _extract_text(response).strip()

    def generate_master_prompt(self, shot_prompts: list[str], video_meta: dict) -> str:
        joined = "\n\n".join(f"Shot {i}: {p}" for i, p in enumerate(shot_prompts))
        prompt = (
            "Combine the following per-shot descriptions of a single source video "
            "into ONE cohesive prompt (under 120 words) for a video generation model "
            "that only accepts a single prompt. Preserve the overall narrative arc, "
            "dominant subject, setting, and visual style. Do not list shots separately "
            f"in your output -- write one continuous description.\n\nVideo metadata: "
            f"{json.dumps(video_meta)}\n\n{joined}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(response).strip()

    def generate_scene_breakdown(self, shot: ShotRecord, keyframe_paths: list[Path]) -> dict:
        content = [{"type": "text", "text": self._breakdown_context_text(shot)}]
        for p in keyframe_paths[:3]:
            content.append(_image_block(p))

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1000,
            system=BREAKDOWN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        text = _extract_text(response).strip()
        return _parse_breakdown_json(text)

    def generate_core_prompt(self, scenes: list[dict], video_meta: dict) -> str:
        lines = []
        for s in scenes:
            role = s.get("scene_role", "")
            desc = s.get("scene_description", "")
            texts = "; ".join(s.get("on_screen_text", []))
            lines.append(f"- {role}: {desc} (on-screen text: {texts})")
        prompt = (
            "Here are the per-scene descriptions of a screen-recorded demo video:\n\n"
            + "\n".join(lines)
            + f"\n\nVideo duration: {video_meta.get('duration', '?')}s"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=150,
            system=CORE_PROMPT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(response).strip()

    @staticmethod
    def _breakdown_context_text(shot: ShotRecord) -> str:
        text = (
            f"Shot {shot.shot_index}, {shot.start_time:.1f}s - {shot.end_time:.1f}s "
            f"(duration {shot.duration:.1f}s)."
        )
        if shot.on_screen_text:
            text += f"\n\nOCR-extracted on-screen text for this shot:\n{shot.on_screen_text}"
        else:
            text += "\n\nNo on-screen text detected for this shot."
        if shot.transcript_segment:
            text += (
                f"\n\nSpoken transcript overlapping this shot (context only, "
                f"do not quote): {shot.transcript_segment}"
            )
        return text

    @staticmethod
    def _shot_context_text(shot: ShotRecord) -> str:
        payload = shot.model_dump(exclude={"keyframe_paths", "debug_overlay_paths", "generated_prompt"})
        text = f"Structured shot data:\n{json.dumps(payload, indent=2)}"
        if shot.transcript_segment:
            text += (
                f"\n\nTranscript overlapping this shot (CONTEXT ONLY, do not quote): "
                f"{shot.transcript_segment}"
            )
        return text


def _image_block(path: Path) -> dict:
    media_type = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _extract_text(response) -> str:
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def _parse_breakdown_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    data = _try_parse_json(cleaned)
    if data is None:
        # Likely truncated mid-response (hit max_tokens). Try closing it off.
        data = _try_parse_json(_attempt_json_repair(cleaned))

    if data is None:
        return {
            "scene_role": "Unlabeled",
            "scene_description": "(LLM response was truncated or malformed for this shot; "
                                  "raw on-screen text is still available below.)",
            "on_screen_text": [],
        }

    return {
        "scene_role": data.get("scene_role", "Unlabeled"),
        "scene_description": data.get("scene_description", ""),
        "on_screen_text": data.get("on_screen_text", []) or [],
    }


def _try_parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _attempt_json_repair(text: str) -> str:
    """Best-effort repair for a JSON object truncated mid-string/mid-array
    (happens when a response hits max_tokens before finishing). Closes any
    open string, then pads with closing brackets/braces to match what's open.
    """
    repaired = text
    if repaired.count('"') % 2 == 1:
        repaired += '"'
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    repaired += "]" * max(0, open_brackets)
    repaired += "}" * max(0, open_braces)
    return repaired
