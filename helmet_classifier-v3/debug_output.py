from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .schemas import Box, DebugLayout, HelmetColorDebugData, VestColorDebugData


def ensure_debug_layout(debug_root: Path) -> DebugLayout:
    debug_root.mkdir(parents=True, exist_ok=True)
    frames_dir = debug_root / "frames"
    panels_dir = debug_root / "panels"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    return DebugLayout(
        root_dir=debug_root,
        frames_dir=frames_dir,
        panels_dir=panels_dir,
        csv_path=debug_root / "records.csv",
    )


def crop_with_margin(frame: np.ndarray, box: np.ndarray, margin_ratio: float = 0.25) -> tuple[np.ndarray, tuple[int, int]]:
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    x_margin = int(round(box_w * margin_ratio))
    y_margin = int(round(box_h * margin_ratio))
    crop_x1 = max(0, x1 - x_margin)
    crop_y1 = max(0, y1 - y_margin)
    crop_x2 = min(frame_w, x2 + x_margin)
    crop_y2 = min(frame_h, y2 + y_margin)
    return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy(), (crop_x1, crop_y1)


def fit_panel_image(image: np.ndarray, target_w: int = 320, target_h: int = 240) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.size == 0:
        return np.full((target_h, target_w, 3), 25, dtype=np.uint8)

    src_h, src_w = image.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 25, dtype=np.uint8)
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return canvas


def label_panel(panel: np.ndarray, title: str) -> np.ndarray:
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 26), (45, 45, 45), -1)
    cv2.putText(panel, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def save_debug_panel(
    panels_dir: Path,
    frame: np.ndarray,
    frame_index: int,
    detection_index: int,
    record: dict[str, object],
    person_box: np.ndarray,
    helmet_box: Box | None,
    torso_box: Box | None,
    helmet_debug: HelmetColorDebugData,
    vest_debug: VestColorDebugData,
) -> None:
    person_crop, (crop_x1, crop_y1) = crop_with_margin(frame, person_box)
    person_x1, person_y1, person_x2, person_y2 = [int(round(v)) for v in person_box]
    cv2.rectangle(
        person_crop,
        (person_x1 - crop_x1, person_y1 - crop_y1),
        (person_x2 - crop_x1, person_y2 - crop_y1),
        (0, 255, 255),
        2,
    )
    if helmet_box is not None:
        hx1, hy1, hx2, hy2 = helmet_box
        cv2.rectangle(
            person_crop,
            (hx1 - crop_x1, hy1 - crop_y1),
            (hx2 - crop_x1, hy2 - crop_y1),
            (0, 0, 255),
            2,
        )
    if torso_box is not None:
        tx1, ty1, tx2, ty2 = torso_box
        cv2.rectangle(
            person_crop,
            (tx1 - crop_x1, ty1 - crop_y1),
            (tx2 - crop_x1, ty2 - crop_y1),
            (0, 180, 0),
            2,
        )

    top_row = np.hstack(
        (
            label_panel(fit_panel_image(person_crop), "person_crop"),
            label_panel(fit_panel_image(helmet_debug.roi_bgr), "helmet_roi"),
            label_panel(fit_panel_image(vest_debug.roi_bgr), "torso_roi"),
            label_panel(fit_panel_image(vest_debug.analysis_bgr), "vest_analysis"),
        )
    )
    bottom_row = np.hstack(
        (
            label_panel(fit_panel_image(cv2.cvtColor(vest_debug.yellow_green_mask, cv2.COLOR_GRAY2BGR)), "yellow_green_mask"),
            label_panel(fit_panel_image(cv2.cvtColor(vest_debug.red_mask, cv2.COLOR_GRAY2BGR)), "red_mask"),
            label_panel(fit_panel_image(cv2.cvtColor(vest_debug.orange_mask, cv2.COLOR_GRAY2BGR)), "orange_mask"),
            label_panel(fit_panel_image(cv2.cvtColor(vest_debug.white_mask, cv2.COLOR_GRAY2BGR)), "white_mask"),
        )
    )
    footer = np.full((170, top_row.shape[1], 3), 18, dtype=np.uint8)
    lines = [
        f"frame={frame_index} det={detection_index} label={record['label']} decision={record['final_decision_rule']}",
        f"head_source={record['roi_source']} torso_source={record['torso_roi_source']} person_conf={record['person_confidence']}",
        f"head_pts={record['head_point_count']} shoulder_pts={record['shoulder_point_count']} hip_pts={record['hip_point_count']}",
        f"helmet={record['helmet_color']} W={record['white_ratio']} R={record['red_ratio']} torso_box=({record['torso_x1']},{record['torso_y1']},{record['torso_x2']},{record['torso_y2']})",
        f"vest={record['vest_color']} YG={record['vest_yellow_green_ratio']} R={record['vest_red_ratio']} O={record['vest_orange_ratio']} W={record['vest_white_ratio']}",
        f"helmet_box=({record['helmet_x1']},{record['helmet_y1']},{record['helmet_x2']},{record['helmet_y2']}) vest_analysis=({record['vest_analysis_x1']},{record['vest_analysis_y1']},{record['vest_analysis_x2']},{record['vest_analysis_y2']})",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            footer,
            line,
            (10, 24 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    panel = np.vstack((top_row, bottom_row, footer))
    panel_path = panels_dir / f"frame_{frame_index:06d}_det_{detection_index:02d}.jpg"
    cv2.imwrite(str(panel_path), panel)


def should_capture_debug(
    config: AppConfig,
    frame_index: int,
    label: str,
    sample_count: int,
) -> bool:
    if not config.save_debug_artifacts:
        return False
    if sample_count >= config.debug_max_samples:
        return False
    if label == "worker":
        return True
    if frame_index == 1:
        return True
    if config.debug_sample_every <= 0:
        return False
    return frame_index % config.debug_sample_every == 0
