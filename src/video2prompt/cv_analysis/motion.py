"""Camera motion estimation using dense optical flow (Farneback).

Classifies the dominant flow vector across two frames in a shot into a
coarse motion type/direction/magnitude. This is intentionally simple --
the spec calls out swapping in RAFT later for higher fidelity.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ..schema import CameraMotion


def estimate_motion(frame_a_path: Path, frame_b_path: Path) -> tuple[CameraMotion, np.ndarray | None]:
    """Returns (CameraMotion, flow_field) where flow_field is the raw
    Farneback flow array (used by debug_overlay to draw vectors), or None
    if it couldn't be computed.
    """
    img_a = cv2.imread(str(frame_a_path), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(frame_b_path), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None or img_a.shape != img_b.shape:
        return CameraMotion(), None

    flow = cv2.calcOpticalFlowFarneback(
        img_a, img_b, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    fx = flow[..., 0]
    fy = flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)

    mean_mag = float(np.mean(mag))
    mean_fx = float(np.mean(fx))
    mean_fy = float(np.mean(fy))

    if mean_mag < 0.5:
        return CameraMotion(type="static", magnitude="none"), flow

    # Coherence check: real camera motion produces vectors that broadly
    # agree in direction. Scattered motion (UI scroll/animation, burned-in
    # captions, noise on low-texture footage like dark skies) produces
    # vectors pointing every which way -- high magnitude, low coherence.
    # Measure coherence as the ratio of the magnitude of the mean vector
    # to the mean of individual magnitudes (1.0 = perfectly aligned,
    # near 0 = directionally random).
    mean_vector_mag = (mean_fx ** 2 + mean_fy ** 2) ** 0.5
    coherence = mean_vector_mag / mean_mag if mean_mag > 0 else 0.0

    if coherence < 0.35:
        return CameraMotion(type="unknown", magnitude="none"), flow

    magnitude = "slow" if mean_mag < 2.0 else "medium" if mean_mag < 5.0 else "fast"

    # crude zoom check: compare flow direction near center vs. edges
    h, w = fx.shape
    center = mag[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    edge_mask = np.ones_like(mag, dtype=bool)
    edge_mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = False
    edge_mag = mag[edge_mask]
    is_zoom = float(np.mean(edge_mag)) > float(np.mean(center)) * 1.4 if center.size else False

    angle_deg = math.degrees(math.atan2(mean_fy, mean_fx))
    if is_zoom:
        # Outward-radial flow vs inward determines in/out; approximate via
        # whether edge vectors point away from center on average.
        motion_type = "zoom_out"
        direction = None
    elif -45 <= angle_deg <= 45:
        motion_type = "pan"
        direction = "left-to-right"
    elif 135 <= abs(angle_deg) <= 180:
        motion_type = "pan"
        direction = "right-to-left"
    elif 45 < angle_deg < 135:
        motion_type = "tilt"
        direction = "up-to-down"
    else:
        motion_type = "tilt"
        direction = "down-to-up"

    return CameraMotion(type=motion_type, direction=direction, magnitude=magnitude), flow
