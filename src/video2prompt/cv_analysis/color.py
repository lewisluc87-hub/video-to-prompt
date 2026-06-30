"""Dominant palette, brightness, and contrast estimation per keyframe."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..schema import ColorInfo


def analyze_color(frame_path: Path, n_colors: int = 3) -> ColorInfo:
    img = cv2.imread(str(frame_path))
    if img is None:
        return ColorInfo()

    small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_colors, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
    )
    counts = np.bincount(labels.flatten())
    order = np.argsort(-counts)
    palette = []
    for idx in order:
        b, g, r = centers[idx].astype(int)
        palette.append(f"#{r:02x}{g:02x}{b:02x}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))

    if mean_brightness < 85:
        brightness = "low-key"
    elif mean_brightness > 170:
        brightness = "high-key"
    else:
        brightness = "mid"

    contrast = "low" if std_contrast < 35 else "high" if std_contrast > 65 else "medium"

    return ColorInfo(dominant_palette=palette, brightness=brightness, contrast=contrast)
