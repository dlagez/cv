from __future__ import annotations

import math

import cv2
import numpy as np

from .config import AppConfig
from .constants import (
    HEAD_KEYPOINTS,
    LEFT_HIP,
    LEFT_SHOULDER,
    MANAGER_HELMET_COLORS,
    MANAGER_VEST_COLOR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)
from .schemas import (
    Box,
    DecisionResult,
    HeadRoiMeta,
    HelmetColorDebugData,
    HelmetColorResult,
    TorsoRoiMeta,
    VestColorDebugData,
    VestColorResult,
)


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


def _clamp_torso_box(
    box: tuple[float, float, float, float],
    person_box: np.ndarray,
    frame_shape: tuple[int, int, int],
) -> Box:
    frame_h, frame_w = frame_shape[:2]
    box_x1, box_y1, box_x2, box_y2 = [float(v) for v in person_box]
    box_w = max(box_x2 - box_x1, 1.0)
    box_h = max(box_y2 - box_y1, 1.0)
    left_limit = box_x1 + box_w * 0.02
    right_limit = box_x2 - box_w * 0.02
    top_limit = box_y1 + box_h * 0.05
    bottom_limit = box_y2 - box_h * 0.06
    roi_x1 = max(0, int(round(max(box[0], left_limit))))
    roi_y1 = max(0, int(round(max(box[1], top_limit))))
    roi_x2 = min(frame_w, int(round(min(box[2], right_limit))))
    roi_y2 = min(frame_h, int(round(min(box[3], bottom_limit))))
    return (roi_x1, roi_y1, roi_x2, roi_y2)


def _torso_roi_size(box: Box) -> tuple[int, int, int]:
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return width, height, width * height


def _torso_roi_meets_min_constraints(box: Box, config: AppConfig) -> bool:
    width, height, area = _torso_roi_size(box)
    return (
        width >= config.torso_roi_min_width
        and height >= config.torso_roi_min_height
        and area >= config.torso_roi_min_area
    )


def _expand_small_torso_roi(
    box: Box,
    person_box: np.ndarray,
    frame_shape: tuple[int, int, int],
    config: AppConfig,
) -> Box:
    box_x1, box_y1, box_x2, box_y2 = [float(v) for v in person_box]
    box_w = max(box_x2 - box_x1, 1.0)
    box_h = max(box_y2 - box_y1, 1.0)
    width, height, _ = _torso_roi_size(box)
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    target_w = max(float(width), float(config.torso_roi_min_width), box_w * 0.26)
    target_h = max(float(height), float(config.torso_roi_min_height), box_h * 0.30)
    expanded = (
        center_x - target_w * 0.50,
        center_y - target_h * 0.42,
        center_x + target_w * 0.50,
        center_y + target_h * 0.58,
    )
    return _clamp_torso_box(expanded, person_box, frame_shape)


def _build_fallback_torso_roi(
    person_box: np.ndarray,
    frame_shape: tuple[int, int, int],
    config: AppConfig,
) -> Box | None:
    box_x1, box_y1, box_x2, box_y2 = [float(v) for v in person_box]
    box_w = max(box_x2 - box_x1, 1.0)
    box_h = max(box_y2 - box_y1, 1.0)
    fallback = (
        box_x1 + box_w * config.torso_fallback_x_margin_ratio,
        box_y1 + box_h * config.torso_fallback_top_ratio,
        box_x2 - box_w * config.torso_fallback_x_margin_ratio,
        box_y1 + box_h * config.torso_fallback_bottom_ratio,
    )
    clamped = _clamp_torso_box(fallback, person_box, frame_shape)
    if not _torso_roi_meets_min_constraints(clamped, config):
        return None
    return clamped


def estimate_head_roi(
    person_box: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    frame_shape: tuple[int, int, int],
    min_conf: float,
) -> tuple[Box | None, HeadRoiMeta]:
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
        roi_w = max(box_w * 0.16, shoulder_span * 0.56, 20.0)
        roi_h = max(box_h * 0.13, shoulder_span * 0.52, 20.0)
        center_y -= roi_h * 0.22
    elif shoulder_points.shape[0] == 2:
        roi_source = "shoulder"
        center_x = float(np.mean(shoulder_points[:, 0]))
        center_y = float(np.mean(shoulder_points[:, 1])) - max(shoulder_span * 0.92, box_h * 0.18, 20.0)
        roi_w = max(shoulder_span * 0.62, box_w * 0.18, 20.0)
        roi_h = max(shoulder_span * 0.58, box_h * 0.14, 20.0)
    else:
        return None, HeadRoiMeta(
            roi_source=roi_source,
            head_point_count=int(head_points.shape[0]),
            shoulder_point_count=int(shoulder_points.shape[0]),
            shoulder_span=shoulder_span,
        )

    roi_x1 = int(round(center_x - roi_w * 0.50))
    roi_y1 = int(round(center_y - roi_h * 0.62))
    roi_x2 = int(round(center_x + roi_w * 0.50))
    roi_y2 = int(round(center_y + roi_h * 0.30))

    head_top_limit = int(math.floor(y1 - box_h * 0.05))
    head_bottom_limit = int(math.ceil(y1 + box_h * 0.26))

    roi_x1 = max(0, roi_x1)
    roi_y1 = max(0, max(roi_y1, head_top_limit))
    roi_x2 = min(frame_w, roi_x2)
    roi_y2 = min(frame_h, min(roi_y2, head_bottom_limit))

    if roi_x2 - roi_x1 < 12 or roi_y2 - roi_y1 < 12:
        return None, HeadRoiMeta(
            roi_source=f"{roi_source}-too-small",
            head_point_count=int(head_points.shape[0]),
            shoulder_point_count=int(shoulder_points.shape[0]),
            shoulder_span=shoulder_span,
        )

    return (
        (roi_x1, roi_y1, roi_x2, roi_y2),
        HeadRoiMeta(
            roi_source=roi_source,
            head_point_count=int(head_points.shape[0]),
            shoulder_point_count=int(shoulder_points.shape[0]),
            shoulder_span=shoulder_span,
        ),
    )


def estimate_torso_roi(
    person_box: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    frame_shape: tuple[int, int, int],
    min_conf: float,
    config: AppConfig,
) -> tuple[Box | None, TorsoRoiMeta]:
    frame_h, frame_w = frame_shape[:2]
    box_x1, box_y1, box_x2, box_y2 = [float(v) for v in person_box]
    box_w = max(box_x2 - box_x1, 1.0)
    box_h = max(box_y2 - box_y1, 1.0)

    shoulder_points, _ = collect_valid_points(
        keypoints_xy,
        keypoints_conf,
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        min_conf,
    )
    hip_points, _ = collect_valid_points(
        keypoints_xy,
        keypoints_conf,
        (LEFT_HIP, RIGHT_HIP),
        min_conf,
    )

    shoulder_span = 0.0
    if shoulder_points.shape[0] == 2:
        shoulder_span = float(np.linalg.norm(shoulder_points[0] - shoulder_points[1]))

    hip_span = 0.0
    if hip_points.shape[0] == 2:
        hip_span = float(np.linalg.norm(hip_points[0] - hip_points[1]))

    if shoulder_points.shape[0] < 2:
        fallback_box = _build_fallback_torso_roi(person_box, frame_shape, config)
        if fallback_box is not None:
            return fallback_box, TorsoRoiMeta(
                roi_source="torso-person-fallback",
                shoulder_point_count=int(shoulder_points.shape[0]),
                hip_point_count=int(hip_points.shape[0]),
                shoulder_span=shoulder_span,
                hip_span=hip_span,
            )
        return None, TorsoRoiMeta(
            roi_source="torso-none",
            shoulder_point_count=int(shoulder_points.shape[0]),
            hip_point_count=int(hip_points.shape[0]),
            shoulder_span=shoulder_span,
            hip_span=hip_span,
        )

    shoulder_center = np.mean(shoulder_points, axis=0)
    shoulder_center_x = float(shoulder_center[0])
    shoulder_center_y = float(shoulder_center[1])

    if hip_points.shape[0] == 2:
        roi_source = "torso-4pt"
        hip_center = np.mean(hip_points, axis=0)
        hip_center_y = float(hip_center[1])
        center_x = float(np.mean(np.concatenate((shoulder_points[:, 0], hip_points[:, 0]))))
        body_width = max(shoulder_span, hip_span, box_w * 0.30, 28.0)
        roi_w = body_width * 0.92
        torso_height = max(hip_center_y - shoulder_center_y, box_h * 0.22, 26.0)
        roi_y1 = shoulder_center_y + max(2.0, torso_height * 0.04)
        roi_y2 = hip_center_y + max(2.0, torso_height * 0.10)
        roi_y2 = min(roi_y2, box_y1 + box_h * 0.78)
    else:
        roi_source = "torso-shoulder-box"
        center_x = shoulder_center_x
        body_width = max(shoulder_span, box_w * 0.32, 28.0)
        roi_w = body_width * 0.88
        roi_y1 = shoulder_center_y + max(2.0, box_h * 0.01, shoulder_span * 0.04)
        roi_y2 = max(box_y1 + box_h * 0.68, roi_y1 + max(28.0, shoulder_span * 1.05, box_h * 0.24))
        roi_y2 = min(roi_y2, box_y1 + box_h * 0.76)

    torso_box = _clamp_torso_box(
        (
            center_x - roi_w * 0.50,
            roi_y1,
            center_x + roi_w * 0.50,
            roi_y2,
        ),
        person_box,
        frame_shape,
    )

    if not _torso_roi_meets_min_constraints(torso_box, config):
        expanded_box = _expand_small_torso_roi(torso_box, person_box, frame_shape, config)
        if _torso_roi_meets_min_constraints(expanded_box, config):
            return expanded_box, TorsoRoiMeta(
                roi_source=f"{roi_source}-expanded",
                shoulder_point_count=int(shoulder_points.shape[0]),
                hip_point_count=int(hip_points.shape[0]),
                shoulder_span=shoulder_span,
                hip_span=hip_span,
            )

        fallback_box = _build_fallback_torso_roi(person_box, frame_shape, config)
        if fallback_box is not None:
            return fallback_box, TorsoRoiMeta(
                roi_source="torso-person-fallback",
                shoulder_point_count=int(shoulder_points.shape[0]),
                hip_point_count=int(hip_points.shape[0]),
                shoulder_span=shoulder_span,
                hip_span=hip_span,
            )

        return None, TorsoRoiMeta(
            roi_source="torso-none",
            shoulder_point_count=int(shoulder_points.shape[0]),
            hip_point_count=int(hip_points.shape[0]),
            shoulder_span=shoulder_span,
            hip_span=hip_span,
        )

    return (
        torso_box,
        TorsoRoiMeta(
            roi_source=roi_source,
            shoulder_point_count=int(shoulder_points.shape[0]),
            hip_point_count=int(hip_points.shape[0]),
            shoulder_span=shoulder_span,
            hip_span=hip_span,
        ),
    )


def _select_analysis_region(
    roi_bgr: np.ndarray,
    *,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
    side_margin_ratio: float,
) -> tuple[np.ndarray, int, int, int, int]:
    height, width = roi_bgr.shape[:2]
    x_margin = max(0, int(width * side_margin_ratio))
    top_margin = max(0, int(height * top_margin_ratio))
    bottom_margin = max(0, int(height * bottom_margin_ratio))
    analysis_x1 = x_margin if width - (2 * x_margin) >= 4 else 0
    analysis_x2 = width - x_margin if width - (2 * x_margin) >= 4 else width
    analysis_y1 = top_margin if height - top_margin - bottom_margin >= 4 else 0
    analysis_y2 = height - bottom_margin if height - top_margin - bottom_margin >= 4 else height
    core = roi_bgr[analysis_y1:analysis_y2, analysis_x1:analysis_x2]
    if core.size == 0:
        return roi_bgr, 0, 0, width, height
    return core, analysis_x1, analysis_y1, analysis_x2, analysis_y2


def classify_helmet_color(
    roi_bgr: np.ndarray,
    config: AppConfig,
) -> HelmetColorResult:
    if roi_bgr.size == 0:
        return HelmetColorResult(
            helmet_color="unknown",
            white_ratio=0.0,
            red_ratio=0.0,
            debug=HelmetColorDebugData.empty(),
        )

    height, width = roi_bgr.shape[:2]
    core, analysis_x1, analysis_y1, analysis_x2, analysis_y2 = _select_analysis_region(
        roi_bgr,
        top_margin_ratio=0.0,
        bottom_margin_ratio=0.25,
        side_margin_ratio=0.10,
    )

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

    area = float(max(1, core.shape[0] * core.shape[1]))
    white_pixels = int(np.count_nonzero(white_mask_core))
    red_pixels = int(np.count_nonzero(red_mask_core))
    white_ratio = float(white_pixels / area)
    red_ratio = float(red_pixels / area)

    if red_ratio >= config.helmet_red_ratio_threshold and red_ratio >= white_ratio:
        helmet_color = "red"
    elif white_ratio >= config.helmet_white_ratio_threshold:
        helmet_color = "white"
    else:
        helmet_color = "other"

    white_mask = np.zeros((height, width), dtype=np.uint8)
    red_mask = np.zeros((height, width), dtype=np.uint8)
    white_mask[analysis_y1:analysis_y2, analysis_x1:analysis_x2] = white_mask_core
    red_mask[analysis_y1:analysis_y2, analysis_x1:analysis_x2] = red_mask_core

    return HelmetColorResult(
        helmet_color=helmet_color,
        white_ratio=white_ratio,
        red_ratio=red_ratio,
        debug=HelmetColorDebugData(
            roi_bgr=roi_bgr.copy(),
            analysis_bgr=core.copy(),
            white_mask=white_mask,
            red_mask=red_mask,
            analysis_x1=analysis_x1,
            analysis_y1=analysis_y1,
            analysis_x2=analysis_x2,
            analysis_y2=analysis_y2,
            analysis_area=int(area),
            white_pixels=white_pixels,
            red_pixels=red_pixels,
        ),
    )


def classify_vest_color(
    roi_bgr: np.ndarray,
    config: AppConfig,
) -> VestColorResult:
    if roi_bgr.size == 0:
        return VestColorResult(
            vest_color="unknown",
            yellow_green_ratio=0.0,
            red_ratio=0.0,
            orange_ratio=0.0,
            white_ratio=0.0,
            debug=VestColorDebugData.empty(),
        )

    height, width = roi_bgr.shape[:2]
    core, analysis_x1, analysis_y1, analysis_x2, analysis_y2 = _select_analysis_region(
        roi_bgr,
        top_margin_ratio=0.03,
        bottom_margin_ratio=0.04,
        side_margin_ratio=0.08,
    )

    blurred = cv2.GaussianBlur(core, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    yellow_green_mask_core = cv2.inRange(
        hsv,
        (config.vest_yellow_green_h_min, config.vest_yellow_green_s_min, config.vest_yellow_green_v_min),
        (config.vest_yellow_green_h_max, 255, 255),
    )
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
    orange_mask_core = cv2.inRange(
        hsv,
        (config.vest_orange_h_min, config.vest_orange_s_min, config.vest_orange_v_min),
        (config.vest_orange_h_max, 255, 255),
    )
    white_mask_core = cv2.inRange(hsv, (0, 0, config.white_v_min), (179, config.white_s_max, 255))

    kernel = np.ones((3, 3), dtype=np.uint8)
    yellow_green_mask_core = cv2.morphologyEx(yellow_green_mask_core, cv2.MORPH_OPEN, kernel)
    red_mask_core = cv2.morphologyEx(red_mask_core, cv2.MORPH_OPEN, kernel)
    orange_mask_core = cv2.morphologyEx(orange_mask_core, cv2.MORPH_OPEN, kernel)
    white_mask_core = cv2.morphologyEx(white_mask_core, cv2.MORPH_OPEN, kernel)

    area = float(max(1, core.shape[0] * core.shape[1]))
    non_white_mask_core = cv2.bitwise_not(white_mask_core)
    yellow_green_mask_core = cv2.bitwise_and(yellow_green_mask_core, non_white_mask_core)
    red_mask_core = cv2.bitwise_and(red_mask_core, non_white_mask_core)
    orange_mask_core = cv2.bitwise_and(orange_mask_core, non_white_mask_core)
    yellow_green_pixels = int(np.count_nonzero(yellow_green_mask_core))
    red_pixels = int(np.count_nonzero(red_mask_core))
    orange_pixels = int(np.count_nonzero(orange_mask_core))
    white_pixels = int(np.count_nonzero(white_mask_core))
    non_white_pixels = max(int(area) - white_pixels, 0)

    effective_area = float(max(non_white_pixels, 1))
    yellow_green_ratio = float(yellow_green_pixels / effective_area)
    red_ratio = float(red_pixels / effective_area)
    orange_ratio = float(orange_pixels / effective_area)
    white_ratio = float(white_pixels / area)

    if (
        non_white_pixels < config.vest_non_white_min_pixels
        or (non_white_pixels / area) < config.vest_non_white_min_ratio
    ):
        vest_color = "unknown"
    else:
        candidates: list[tuple[str, float]] = []
        if yellow_green_ratio >= config.vest_yellow_green_ratio_threshold:
            candidates.append(("yellow_green_fluorescent", yellow_green_ratio))
        if red_ratio >= config.vest_red_ratio_threshold:
            candidates.append(("red", red_ratio))
        if orange_ratio >= config.vest_orange_ratio_threshold:
            candidates.append(("orange", orange_ratio))
        vest_color = max(candidates, key=lambda item: item[1])[0] if candidates else "other"

    yellow_green_mask = np.zeros((height, width), dtype=np.uint8)
    red_mask = np.zeros((height, width), dtype=np.uint8)
    orange_mask = np.zeros((height, width), dtype=np.uint8)
    white_mask = np.zeros((height, width), dtype=np.uint8)
    yellow_green_mask[analysis_y1:analysis_y2, analysis_x1:analysis_x2] = yellow_green_mask_core
    red_mask[analysis_y1:analysis_y2, analysis_x1:analysis_x2] = red_mask_core
    orange_mask[analysis_y1:analysis_y2, analysis_x1:analysis_x2] = orange_mask_core
    white_mask[analysis_y1:analysis_y2, analysis_x1:analysis_x2] = white_mask_core

    return VestColorResult(
        vest_color=vest_color,
        yellow_green_ratio=yellow_green_ratio,
        red_ratio=red_ratio,
        orange_ratio=orange_ratio,
        white_ratio=white_ratio,
        debug=VestColorDebugData(
            roi_bgr=roi_bgr.copy(),
            analysis_bgr=core.copy(),
            yellow_green_mask=yellow_green_mask,
            red_mask=red_mask,
            orange_mask=orange_mask,
            white_mask=white_mask,
            analysis_x1=analysis_x1,
            analysis_y1=analysis_y1,
            analysis_x2=analysis_x2,
            analysis_y2=analysis_y2,
            analysis_area=int(area),
            yellow_green_pixels=yellow_green_pixels,
            red_pixels=red_pixels,
            orange_pixels=orange_pixels,
            white_pixels=white_pixels,
        ),
    )


def decide_person_label(
    *,
    helmet_box: Box | None,
    torso_box: Box | None,
    helmet_result: HelmetColorResult,
    vest_result: VestColorResult,
    config: AppConfig,
) -> DecisionResult:
    helmet_match = helmet_box is not None and helmet_result.helmet_color in MANAGER_HELMET_COLORS
    vest_match = torso_box is not None and vest_result.vest_color == MANAGER_VEST_COLOR

    if not config.enable_joint_decision:
        is_legacy_worker = helmet_box is not None and helmet_result.helmet_color == "red"
        return DecisionResult(
            label="worker" if is_legacy_worker else "manager",
            helmet_match_manager_rule=helmet_match,
            vest_match_manager_rule=False,
            manager_rule_matched=False,
            final_decision_rule="legacy_head_only_red_rule" if is_legacy_worker else "legacy_head_only_non_red_rule",
        )

    manager_rule_matched = helmet_match and vest_match
    return DecisionResult(
        label="manager" if manager_rule_matched else "worker",
        helmet_match_manager_rule=helmet_match,
        vest_match_manager_rule=vest_match,
        manager_rule_matched=manager_rule_matched,
        final_decision_rule="命中管理人员联合规则" if manager_rule_matched else "未命中管理人员联合规则",
    )
