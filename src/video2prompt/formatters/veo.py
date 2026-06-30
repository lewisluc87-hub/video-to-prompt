"""Veo-style formatter: prose description plus an explicit camera-direction
clause, per Google's documented preference for stating camera movement
clearly and separately. Recheck against current Veo docs periodically.
"""
from __future__ import annotations

from ..schema import ShotRecord


def format_prompt(shot: ShotRecord) -> str:
    base = (shot.generated_prompt or "").strip()
    motion = shot.camera_motion
    if motion.type in ("static", "unknown") or motion.magnitude == "none":
        camera_clause = "Camera: static."
    else:
        direction = f" {motion.direction}" if motion.direction else ""
        camera_clause = f"Camera: {motion.magnitude} {motion.type.replace('_', ' ')}{direction}."

    if base and not base.endswith((".", "!", "?")):
        base += "."
    return f"{base} {camera_clause}".strip()
