"""Subject/object detection.

Uses YOLOv8n via ultralytics if installed (extra: `video2prompt[objects]`).
Falls back to a simple frame-differencing/contour heuristic that just
flags "something is moving here" boxes -- much cruder, but keeps the
pipeline functional with zero extra ML deps for `--no-llm` / fully-local
runs.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..schema import Subject

_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def detect_subjects(frame_path: Path, conf_threshold: float = 0.5) -> list[Subject]:
    model = _get_yolo()
    if model is not None:
        return _detect_with_yolo(model, frame_path, conf_threshold)
    return _detect_with_contours(frame_path)


def _detect_with_yolo(model, frame_path: Path, conf_threshold: float) -> list[Subject]:
    results = model.predict(source=str(frame_path), verbose=False, conf=conf_threshold)
    if not results:
        return []
    result = results[0]
    h, w = result.orig_shape
    subjects = []
    for box in result.boxes:
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        label = model.names.get(cls_id, str(cls_id))
        conf = float(box.conf[0].item())
        subjects.append(
            Subject(
                label=label,
                confidence=round(conf, 3),
                bbox_norm=(x0 / w, y0 / h, x1 / w, y1 / h),
            )
        )
    return subjects


def _detect_with_contours(frame_path: Path) -> list[Subject]:
    """Crude fallback: largest contiguous bright/dark region vs. background.

    This is NOT object recognition -- it just finds the most visually
    prominent blob so there's *something* in `subjects` to reason about
    when ultralytics isn't installed. Label is always "subject".
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (w * h * 0.01):
        return []
    x, y, bw, bh = cv2.boundingRect(largest)
    return [
        Subject(
            label="subject",
            confidence=0.3,
            bbox_norm=(x / w, y / h, (x + bw) / w, (y + bh) / h),
        )
    ]
