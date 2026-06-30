"""Provider interface for the LLM reasoning layer (spec section 8)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..schema import ShotRecord

SYSTEM_PROMPT = """\
You write prompts for AI video generation models (Sora, Runway, Veo) based on \
a structured analysis of one shot from a source video, plus sampled keyframes.

Rules:
- Describe the subject, action, setting, camera movement, lighting, and mood \
in natural, concrete visual language, in one dense paragraph (40-80 words).
- Use standard cinematography vocabulary for shot size and camera movement.
- If a transcript is provided, use it ONLY to understand intent, tone, and \
relationships between subjects/setting. Never quote, restate, or reference \
spoken dialogue in your output -- video-gen models don't take spoken text as \
a meaningful signal, only visual description.
- Do not invent details that aren't visible or reasonably inferable from the \
provided frames and structured data. Do not invent off-screen elements.
- Output only the prompt paragraph itself, nothing else.
"""

BREAKDOWN_SYSTEM_PROMPT = """\
You write scene-by-scene breakdowns of videos, based on structured shot data, \
sampled keyframes, and OCR-extracted on-screen text for one shot. This applies \
to any kind of video -- marketing/demo content, tutorials, vlogs, narrative \
footage, screen recordings, presentations, anything -- not just software demos.

For each shot you are given, output a JSON object with exactly these keys:
- "scene_role": a short label naming this scene's narrative function within \
the video, inferred from what's shown and any on-screen/spoken content. \
Examples across different video types: "Hook / Problem Statement", "CTA", \
"Establishing Shot", "Demonstration", "Result / Payoff", "Introduction", \
"Transition", "Conclusion" -- choose whatever label actually fits this shot, \
don't force it into a fixed template if the video isn't structured that way.
- "scene_description": one or two sentences in plain prose describing what \
happens in the shot. If it's screen/UI content, describe the interaction \
(clicks, typing, on-screen changes) rather than camera movement. If it's \
camera footage, describe the action and setting concretely.
- "on_screen_text": a list of the distinct text overlays/captions/UI labels \
shown in this shot, in the order they appear, copied as closely as the OCR \
data allows. Use the OCR text provided; lightly clean obvious OCR errors but \
don't invent text that isn't given to you. Empty list if none.

Rules:
- Use the OCR text as a primary source for on_screen_text and for \
understanding what the scene is about, when present.
- Use the transcript (if any) for extra context on tone/intent and spoken \
content; you may summarize what's said in scene_description but don't quote \
long passages verbatim.
- Do not invent UI elements, numbers, text, or actions not present in the \
provided data.
- Output ONLY the JSON object, nothing else -- no markdown fences, no preamble.
"""

CORE_PROMPT_SYSTEM_PROMPT = """\
You write a single-sentence summary of what a video is about overall, given a \
list of per-scene descriptions and on-screen text from the full video. This \
applies to any kind of video, not just marketing or demo content.

Output ONE sentence (under 50 words) capturing the video's subject, what \
happens across it, and its apparent purpose or outcome (a call-to-action if \
one is present, a narrative payoff if it's a story, a conclusion if it's a \
tutorial -- whatever fits). Output only the sentence, nothing else.
"""


class LLMProvider(ABC):
    @abstractmethod
    def generate_shot_prompt(self, shot: ShotRecord, keyframe_paths: list[Path]) -> str:
        ...

    @abstractmethod
    def generate_master_prompt(self, shot_prompts: list[str], video_meta: dict) -> str:
        ...

    @abstractmethod
    def generate_scene_breakdown(self, shot: ShotRecord, keyframe_paths: list[Path]) -> dict:
        """Returns {"scene_role": str, "scene_description": str, "on_screen_text": list[str]}."""
        ...

    @abstractmethod
    def generate_core_prompt(self, scenes: list[dict], video_meta: dict) -> str:
        ...
