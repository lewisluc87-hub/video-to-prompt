"""Sora-style formatter: cinematic, multi-sentence narrative description.

NOTE: vendor prompt conventions shift over time -- recheck this against
current OpenAI Sora documentation periodically (see SPEC.md section 9).
"""
from __future__ import annotations

from ..schema import ShotRecord


def format_prompt(shot: ShotRecord) -> str:
    base = (shot.generated_prompt or "").strip()
    if not base:
        return ""
    if not base.endswith((".", "!", "?")):
        base += "."
    return f"A cinematic shot: {base}"
