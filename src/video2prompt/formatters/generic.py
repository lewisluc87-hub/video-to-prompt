"""Generic, model-agnostic formatter -- the default. Just passes the
LLM/template-generated prose through, since that's already written as a
descriptive paragraph that works reasonably with most video-gen tools.
"""
from __future__ import annotations

from ..schema import ShotRecord


def format_prompt(shot: ShotRecord) -> str:
    return shot.generated_prompt or ""
