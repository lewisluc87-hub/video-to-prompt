"""Shot-size and composition heuristics derived from detected subject bboxes."""
from __future__ import annotations

from ..schema import Subject


def estimate_shot_size(subjects: list[Subject]) -> str:
    if not subjects:
        return "wide / establishing (no dominant subject detected)"

    primary = max(subjects, key=lambda s: _bbox_area(s.bbox_norm))
    area = _bbox_area(primary.bbox_norm)

    if area > 0.55:
        return "close-up"
    if area > 0.25:
        return "medium close-up"
    if area > 0.08:
        return "medium shot"
    return "wide shot"


def estimate_composition(subjects: list[Subject]) -> str:
    if not subjects:
        return "centered / no dominant subject"

    primary = max(subjects, key=lambda s: _bbox_area(s.bbox_norm))
    x0, y0, x1, y1 = primary.bbox_norm
    cx = (x0 + x1) / 2

    if cx < 0.4:
        horiz = "subject left"
    elif cx > 0.6:
        horiz = "subject right"
    else:
        horiz = "subject centered"

    # rough rule-of-thirds check
    near_third = any(abs(cx - t) < 0.12 for t in (1 / 3, 2 / 3))
    style = "rule-of-thirds" if near_third else "center-weighted"

    return f"{style}, {horiz}"


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)
