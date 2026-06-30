"""Runway-style formatter: short, comma-separated descriptor list
(subject, action, camera move, style) matching Gen-3/4 prompting
conventions. Recheck against current Runway docs periodically.
"""
from __future__ import annotations

from ..schema import ShotRecord


def format_prompt(shot: ShotRecord) -> str:
    subjects = ", ".join(s.label for s in shot.subjects[:3]) or "scene"
    motion = shot.camera_motion
    camera = motion.type.replace("_", " ")
    if motion.direction:
        camera += f" {motion.direction}"

    parts = [
        subjects,
        shot.shot_size_estimate,
        f"camera {camera}" if camera != "unknown" else None,
        f"{shot.color.brightness} lighting",
        ", ".join(shot.color.dominant_palette[:2]) or None,
    ]
    return ", ".join(p for p in parts if p)
