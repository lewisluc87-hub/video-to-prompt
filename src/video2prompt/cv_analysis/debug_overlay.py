"""Draws bounding boxes, optical-flow vectors, and composition guides onto
a keyframe for `--debug` verification mode. See SPEC.md section 6.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..schema import CameraMotion, ColorInfo, Subject

_GREEN = (60, 220, 60)
_YELLOW = (40, 220, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def render_debug_overlay(
    frame_path: Path,
    out_path: Path,
    subjects: list[Subject],
    camera_motion: CameraMotion,
    color: ColorInfo,
    shot_size_estimate: str,
    flow: np.ndarray | None = None,
) -> Path:
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(frame_path)
    h, w = img.shape[:2]

    _draw_thirds_grid(img, w, h)
    for s in subjects:
        _draw_subject_box(img, s, w, h)
    if flow is not None:
        _draw_flow_vectors(img, flow, w, h)
    _draw_palette_swatch(img, color)
    _draw_caption(img, camera_motion, shot_size_estimate)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


def _draw_thirds_grid(img, w: int, h: int) -> None:
    for fx in (1 / 3, 2 / 3):
        x = int(fx * w)
        cv2.line(img, (x, 0), (x, h), (100, 100, 100), 1, cv2.LINE_AA)
    for fy in (1 / 3, 2 / 3):
        y = int(fy * h)
        cv2.line(img, (0, y), (w, y), (100, 100, 100), 1, cv2.LINE_AA)


def _draw_subject_box(img, subject: Subject, w: int, h: int) -> None:
    x0, y0, x1, y1 = subject.bbox_norm
    p0 = (int(x0 * w), int(y0 * h))
    p1 = (int(x1 * w), int(y1 * h))
    cv2.rectangle(img, p0, p1, _GREEN, 2)
    label = f"{subject.label} {subject.confidence:.2f}"
    cv2.putText(
        img, label, (p0[0], max(0, p0[1] - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN, 1, cv2.LINE_AA,
    )


def _draw_flow_vectors(img, flow: np.ndarray, w: int, h: int, step: int = 24) -> None:
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]
            if (fx ** 2 + fy ** 2) ** 0.5 < 0.6:
                continue
            end = (int(x + fx * 4), int(y + fy * 4))
            cv2.arrowedLine(img, (x, y), end, _YELLOW, 1, tipLength=0.4)


def _draw_palette_swatch(img, color: ColorInfo, swatch_size: int = 24) -> None:
    for i, hex_color in enumerate(color.dominant_palette[:5]):
        bgr = _hex_to_bgr(hex_color)
        x0 = 10 + i * (swatch_size + 4)
        y0 = 10
        cv2.rectangle(img, (x0, y0), (x0 + swatch_size, y0 + swatch_size), bgr, -1)
        cv2.rectangle(img, (x0, y0), (x0 + swatch_size, y0 + swatch_size), _WHITE, 1)


def _draw_caption(img, camera_motion: CameraMotion, shot_size_estimate: str) -> None:
    h = img.shape[0]
    text = f"{shot_size_estimate} | motion: {camera_motion.type}"
    if camera_motion.direction:
        text += f" ({camera_motion.direction})"
    text += f" [{camera_motion.magnitude}]"
    y = h - 12
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def write_debug_index(debug_dir: Path, shot_overlays: dict[int, list[Path]]) -> Path:
    """Writes an index.md linking all overlay thumbnails in shot order."""
    lines = ["# Debug Verification Index", ""]
    for shot_index in sorted(shot_overlays):
        lines.append(f"## Shot {shot_index}")
        for p in shot_overlays[shot_index]:
            rel = p.name
            lines.append(f"![shot {shot_index} frame]({rel})")
        lines.append("")
    index_path = debug_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path
