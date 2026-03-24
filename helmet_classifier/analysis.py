from __future__ import annotations

import math

import cv2
import numpy as np

from .config import AppConfig
from .constants import HEAD_KEYPOINTS, LEFT_SHOULDER, RIGHT_SHOULDER
from .schemas import Box, ColorDebugData, RoiMeta


def collect_valid_points(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    indices: tuple[int, ...],
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = keypoints_conf[list(indices)] >= min_conf
    valid_indices = np.array(indices)[mask]
    if valid_indices.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return keypoints_xy[valid_indices], keypoints_conf[valid_indices]


def estimate_head_roi(
    person_box: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    frame_shape: tuple[int, int, int],
    min_conf: float,
) -> tuple[Box | None, RoiMeta]:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in person_box]
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)

    head_points, head_weights = collect_valid_points(keypoints_xy, keypoints_conf, HEAD_KEYPOINTS, min_conf)
    shoulder_points, _ = collect_valid_points(
        keypoints_xy,
        keypoints_conf,
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        min_conf,
    )

    shoulder_span = 0.0
    if shoulder_points.shape[0] == 2:
        shoulder_span = float(np.linalg.norm(shoulder_points[0] - shoulder_points[1]))

    roi_source = "none"
    if head_points.shape[0] >= 2:
        roi_source = "head"
        center_x = float(np.average(head_points[:, 0], weights=head_weights))
        center_y = float(np.average(head_points[:, 1], weights=head_weights))
        roi_w = max(box_w * 0.22, shoulder_span * 0.85, 24.0)
        roi_h = max(box_h * 0.18, shoulder_span * 0.75, 24.0)
        center_y -= roi_h * 0.15
    elif shoulder_points.shape[0] == 2:
        roi_source = "shoulder"
        center_x = float(np.mean(shoulder_points[:, 0]))
        center_y = float(np.mean(shoulder_points[:, 1])) - max(shoulder_span * 0.85, box_h * 0.16, 20.0)
        roi_w = max(shoulder_span * 0.95, box_w * 0.22, 24.0)
        roi_h = max(shoulder_span * 0.90, box_h * 0.18, 24.0)
    else:
        return None, RoiMeta(
            roi_source=roi_source,
            head_point_count=int(head_points.shape[0]),
            shoulder_point_count=int(shoulder_points.shape[0]),
            shoulder_span=shoulder_span,
        )

    roi_x1 = int(round(center_x - roi_w * 0.50))
    roi_y1 = int(round(center_y - roi_h * 0.60))
    roi_x2 = int(round(center_x + roi_w * 0.50))
    roi_y2 = int(round(center_y + roi_h * 0.40))

    head_top_limit = int(math.floor(y1 - box_h * 0.05))
    head_bottom_limit = int(math.ceil(y1 + box_h * 0.33))

    roi_x1 = max(0, roi_x1)
    roi_y1 = max(0, max(roi_y1, head_top_limit))
    roi_x2 = min(frame_w, roi_x2)
    roi_y2 = min(frame_h, min(roi_y2, head_bottom_limit))

    if roi_x2 - roi_x1 < 12 or roi_y2 - roi_y1 < 12:
        return None, RoiMeta(
            roi_source=f"{roi_source}-too-small",
            head_point_count=int(head_points.shape[0]),
            shoulder_point_count=int(shoulder_points.shape[0]),
            shoulder_span=shoulder_span,
        )

    return (
        (roi_x1, roi_y1, roi_x2, roi_y2),
        RoiMeta(
            roi_source=roi_source,
            head_point_count=int(head_points.shape[0]),
            shoulder_point_count=int(shoulder_points.shape[0]),
            shoulder_span=shoulder_span,
        ),
    )


def classify_helmet_color(
    roi_bgr: np.ndarray,
    config: AppConfig,
) -> tuple[str, str, float, float, ColorDebugData]:
    if roi_bgr.size == 0:
        return "manager", "non-red", 0.0, 0.0, ColorDebugData.empty()

    height, width = roi_bgr.shape[:2]
    top_end = max(1, int(height * 0.75))
    x_margin = max(0, int(width * 0.10))
    analysis_x1 = x_margin if width - (2 * x_margin) >= 4 else 0
    analysis_x2 = width - x_margin if width - (2 * x_margin) >= 4 else width
    core = roi_bgr[:top_end, analysis_x1:analysis_x2]
    if core.size == 0:
        core = roi_bgr
        analysis_x1 = 0
        analysis_x2 = width
        top_end = height

    blurred = cv2.GaussianBlur(core, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    white_mask_core = cv2.inRange(hsv, (0, 0, config.white_v_min), (179, config.white_s_max, 255))
    red_mask_low = cv2.inRange(
        hsv,
        (0, config.red_s_min, config.red_v_min),
        (config.red_h_low_max, 255, 255),
    )
    red_mask_high = cv2.inRange(
        hsv,
        (config.red_h_high_min, config.red_s_min, config.red_v_min),
        (179, 255, 255),
    )
    red_mask_core = cv2.bitwise_or(red_mask_low, red_mask_high)

    kernel = np.ones((3, 3), dtype=np.uint8)
    white_mask_core = cv2.morphologyEx(white_mask_core, cv2.MORPH_OPEN, kernel)
    red_mask_core = cv2.morphologyEx(red_mask_core, cv2.MORPH_OPEN, kernel)

    area = float(core.shape[0] * core.shape[1])
    white_pixels = int(np.count_nonzero(white_mask_core))
    red_pixels = int(np.count_nonzero(red_mask_core))
    white_ratio = float(white_pixels / area)
    red_ratio = float(red_pixels / area)

    white_mask = np.zeros((height, width), dtype=np.uint8)
    red_mask = np.zeros((height, width), dtype=np.uint8)
    white_mask[:top_end, analysis_x1:analysis_x2] = white_mask_core
    red_mask[:top_end, analysis_x1:analysis_x2] = red_mask_core

    label = "worker" if red_ratio >= config.red_ratio_threshold else "manager"
    color_name = "red" if label == "worker" else "non-red"

    return label, color_name, white_ratio, red_ratio, ColorDebugData(
        roi_bgr=roi_bgr.copy(),
        analysis_bgr=core.copy(),
        white_mask=white_mask,
        red_mask=red_mask,
        analysis_x1=analysis_x1,
        analysis_y1=0,
        analysis_x2=analysis_x2,
        analysis_y2=top_end,
        analysis_area=int(area),
        white_pixels=white_pixels,
        red_pixels=red_pixels,
    )
