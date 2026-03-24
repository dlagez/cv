"""这个脚本用于对视频中的人员安全帽颜色进行规则化识别：它先使用 YOLO 姿态模型逐帧检测人员和关键点，再根据头部或肩部关键点估计头盔所在区域，对该区域做 HSV 颜色分析，并按“红色像素占帽子分析区域 15% 及以上就判为工作人员，否则判为管理人员”的规则完成分类，最后把中文标注直接绘制回输出视频中。"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

HEAD_KEYPOINTS = (0, 1, 2, 3, 4)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
DISPLAY_LABELS = {
    "manager": "管理人员",
    "worker": "工作人员",
}
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify helmet colors in a video with YOLO pose keypoints and HSV rules."
    )
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="YOLO pose model path. Example: yolo11m-pose.pt",
    )
    parser.add_argument("--output", default="", help="Output video path. Defaults to outputs/helmet-classify/<name>.mp4")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--person-conf", type=float, default=0.35, help="Person detection confidence threshold.")
    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.35,
        help="Minimum keypoint confidence used to estimate the helmet area.",
    )
    parser.add_argument(
        "--white-ratio-threshold",
        type=float,
        default=0.18,
        help="Minimum white pixel ratio required to classify a white helmet.",
    )
    parser.add_argument(
        "--red-ratio-threshold",
        type=float,
        default=0.15,
        help="Minimum red pixel ratio required to classify a worker helmet.",
    )
    parser.add_argument("--white-s-max", type=int, default=55, help="Maximum HSV saturation for white.")
    parser.add_argument("--white-v-min", type=int, default=170, help="Minimum HSV value for white.")
    parser.add_argument("--red-h-low-max", type=int, default=18, help="Upper hue bound of the low red range.")
    parser.add_argument("--red-h-high-min", type=int, default=145, help="Lower hue bound of the high red range.")
    parser.add_argument("--red-s-min", type=int, default=75, help="Minimum HSV saturation for red.")
    parser.add_argument("--red-v-min", type=int, default=60, help="Minimum HSV value for red.")
    parser.add_argument("--device", default="", help="Ultralytics device. Empty uses the default device.")
    parser.add_argument("--codec", default="mp4v", help="Preferred output codec. Falls back to XVID if needed.")
    parser.add_argument("--font-path", default="", help="Optional path to a Chinese font file.")
    parser.add_argument("--font-size", type=int, default=22, help="Font size used for the Chinese overlay labels.")
    parser.add_argument(
        "--draw-helmet-box",
        action="store_true",
        help="Draw the estimated helmet ROI used for color analysis.",
    )
    parser.add_argument(
        "--debug-text",
        action="store_true",
        help="Append white/red ratios to the overlay labels.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Process only the first N frames for quick testing.")
    parser.add_argument(
        "--save-debug-artifacts",
        action="store_true",
        help="Save sampled debug panels, masks, and CSV records for manual inspection.",
    )
    parser.add_argument(
        "--debug-dir",
        default="",
        help="Root directory for sampled debug outputs. Each run is stored in a timestamped subdirectory.",
    )
    parser.add_argument(
        "--debug-sample-every",
        type=int,
        default=60,
        help="Save one sampled detection every N frames, plus all worker detections.",
    )
    parser.add_argument(
        "--debug-max-samples",
        type=int,
        default=80,
        help="Maximum number of sampled detections to save in debug mode.",
    )
    return parser.parse_args()


def default_output_path(source: Path) -> Path:
    return Path("outputs") / "helmet-classify" / f"{source.stem}.mp4"


def default_debug_dir(source: Path) -> Path:
    return Path("outputs") / "helmet-debug" / source.stem


def allocate_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base_dir / timestamp
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{timestamp}-{suffix:02d}"
        suffix += 1
    return candidate


def create_video_writer(output_path: Path, fps: float, width: int, height: int, codec: str) -> tuple[cv2.VideoWriter, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if writer.isOpened():
        return writer, output_path

    fallback_path = output_path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(fallback_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )
    if writer.isOpened():
        print(f"[warn] Failed to open {output_path} with codec {codec}. Falling back to {fallback_path}.")
        return writer, fallback_path

    raise RuntimeError(f"Could not create output writer for {output_path}")


def resolve_font_path(font_path: str) -> Path:
    if font_path:
        candidate = Path(font_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Chinese font not found: {candidate}")

    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No Chinese font found. Checked common Windows fonts.")


def load_label_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    resolved_font = resolve_font_path(font_path)
    return ImageFont.truetype(str(resolved_font), size=font_size)


def ensure_debug_layout(debug_dir: Path) -> tuple[Path, Path, Path]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = debug_dir / "frames"
    panels_dir = debug_dir / "panels"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir, frames_dir, panels_dir


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
) -> tuple[tuple[int, int, int, int] | None, dict[str, float | int | str]]:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in person_box]
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)

    head_points, head_weights = collect_valid_points(keypoints_xy, keypoints_conf, HEAD_KEYPOINTS, min_conf)
    shoulder_points, _ = collect_valid_points(
        keypoints_xy, keypoints_conf, (LEFT_SHOULDER, RIGHT_SHOULDER), min_conf
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
        return None, {
            "roi_source": roi_source,
            "head_point_count": int(head_points.shape[0]),
            "shoulder_point_count": int(shoulder_points.shape[0]),
            "shoulder_span": shoulder_span,
        }

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
        return None, {
            "roi_source": f"{roi_source}-too-small",
            "head_point_count": int(head_points.shape[0]),
            "shoulder_point_count": int(shoulder_points.shape[0]),
            "shoulder_span": shoulder_span,
        }
    return (
        (roi_x1, roi_y1, roi_x2, roi_y2),
        {
            "roi_source": roi_source,
            "head_point_count": int(head_points.shape[0]),
            "shoulder_point_count": int(shoulder_points.shape[0]),
            "shoulder_span": shoulder_span,
        },
    )


def classify_helmet_color(
    roi_bgr: np.ndarray,
    args: argparse.Namespace,
) -> tuple[str, str, float, float, dict[str, int | np.ndarray]]:
    if roi_bgr.size == 0:
        empty_mask = np.zeros((1, 1), dtype=np.uint8)
        empty_roi = np.zeros((1, 1, 3), dtype=np.uint8)
        return "manager", "non-red", 0.0, 0.0, {
            "roi_bgr": empty_roi,
            "analysis_bgr": empty_roi,
            "white_mask": empty_mask,
            "red_mask": empty_mask,
            "analysis_x1": 0,
            "analysis_y1": 0,
            "analysis_x2": 0,
            "analysis_y2": 0,
            "analysis_area": 0,
            "white_pixels": 0,
            "red_pixels": 0,
        }

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

    white_mask_core = cv2.inRange(hsv, (0, 0, args.white_v_min), (179, args.white_s_max, 255))
    red_mask_low = cv2.inRange(hsv, (0, args.red_s_min, args.red_v_min), (args.red_h_low_max, 255, 255))
    red_mask_high = cv2.inRange(hsv, (args.red_h_high_min, args.red_s_min, args.red_v_min), (179, 255, 255))
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

    if red_ratio >= args.red_ratio_threshold:
        label, color_name = "worker", "red"
    else:
        label, color_name = "manager", "non-red"

    return label, color_name, white_ratio, red_ratio, {
        "roi_bgr": roi_bgr.copy(),
        "analysis_bgr": core.copy(),
        "white_mask": white_mask,
        "red_mask": red_mask,
        "analysis_x1": analysis_x1,
        "analysis_y1": 0,
        "analysis_x2": analysis_x2,
        "analysis_y2": top_end,
        "analysis_area": int(area),
        "white_pixels": white_pixels,
        "red_pixels": red_pixels,
    }


def make_overlay_label(label: str, white_ratio: float, red_ratio: float, debug_text: bool) -> str:
    display_label = DISPLAY_LABELS.get(label, label)
    if debug_text:
        return f"{display_label} W:{white_ratio:.2f} R:{red_ratio:.2f}"
    return display_label


def draw_detection_boxes(
    frame: np.ndarray,
    person_box: np.ndarray,
    helmet_box: tuple[int, int, int, int] | None,
    label: str,
    args: argparse.Namespace,
) -> tuple[int, int, tuple[int, int, int], tuple[int, int, int]]:
    palette = {
        "manager": (255, 255, 255),
        "worker": (0, 0, 255),
    }
    draw_color = palette.get(label, (0, 255, 255))
    x1, y1, x2, y2 = [int(round(v)) for v in person_box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)

    text_y = max(22, y1 - 10)
    if args.draw_helmet_box and helmet_box is not None:
        hx1, hy1, hx2, hy2 = helmet_box
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 1)

    text_color = (0, 0, 0) if label == "manager" else (255, 255, 255)
    return x1, text_y, draw_color, text_color


def render_text_overlays(
    frame: np.ndarray,
    overlays: list[tuple[str, int, int, tuple[int, int, int], tuple[int, int, int]]],
    font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    if not overlays:
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    for text, x, y, bg_bgr, text_bgr in overlays:
        left = max(0, x)
        top = max(0, y - font.size - 10)
        text_bbox = draw.textbbox((left + 8, top + 5), text, font=font)
        rect_right = min(frame.shape[1] - 1, text_bbox[2] + 8)
        rect_bottom = min(frame.shape[0] - 1, text_bbox[3] + 5)
        draw.rectangle(
            [(left, top), (rect_right, rect_bottom)],
            fill=(bg_bgr[2], bg_bgr[1], bg_bgr[0]),
        )
        draw.text(
            (left + 8, top + 5),
            text,
            font=font,
            fill=(text_bgr[2], text_bgr[1], text_bgr[0]),
        )

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


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
    helmet_box: tuple[int, int, int, int] | None,
    color_debug: dict[str, int | np.ndarray],
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

    roi_bgr = color_debug.get("roi_bgr")
    white_mask = color_debug.get("white_mask")
    red_mask = color_debug.get("red_mask")
    if not isinstance(roi_bgr, np.ndarray):
        roi_bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    if not isinstance(white_mask, np.ndarray):
        white_mask = np.zeros((1, 1), dtype=np.uint8)
    if not isinstance(red_mask, np.ndarray):
        red_mask = np.zeros((1, 1), dtype=np.uint8)

    top_row = np.hstack(
        (
            label_panel(fit_panel_image(person_crop), "person_crop"),
            label_panel(fit_panel_image(roi_bgr), "helmet_roi"),
        )
    )
    bottom_row = np.hstack(
        (
            label_panel(fit_panel_image(cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)), "red_mask"),
            label_panel(fit_panel_image(cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)), "white_mask"),
        )
    )
    footer = np.full((130, top_row.shape[1], 3), 18, dtype=np.uint8)
    lines = [
        f"frame={frame_index} det={detection_index} label={record['label']} roi_source={record['roi_source']}",
        f"person_conf={record['person_confidence']} head_pts={record['head_point_count']} shoulder_pts={record['shoulder_point_count']}",
        f"person_box=({record['person_x1']},{record['person_y1']},{record['person_x2']},{record['person_y2']})",
        f"helmet_box=({record['helmet_x1']},{record['helmet_y1']},{record['helmet_x2']},{record['helmet_y2']}) analysis=({record['analysis_x1']},{record['analysis_y1']},{record['analysis_x2']},{record['analysis_y2']})",
        f"white_ratio={record['white_ratio']} red_ratio={record['red_ratio']} white_px={record['white_pixels']} red_px={record['red_pixels']}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            footer,
            line,
            (10, 24 + idx * 22),
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
    args: argparse.Namespace,
    frame_index: int,
    label: str,
    sample_count: int,
) -> bool:
    if not args.save_debug_artifacts:
        return False
    if sample_count >= args.debug_max_samples:
        return False
    if label == "worker":
        return True
    if frame_index == 1:
        return True
    return frame_index % args.debug_sample_every == 0


def process_video(args: argparse.Namespace) -> Path:
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    output_path = Path(args.output) if args.output else default_output_path(source)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer, actual_output_path = create_video_writer(output_path, fps, width, height, args.codec)
    font = load_label_font(args.font_path, args.font_size)
    debug_csv_file = None
    debug_writer = None
    debug_frames_dir = None
    debug_panels_dir = None
    debug_sample_count = 0
    debug_frame_requested = False

    if args.save_debug_artifacts:
        debug_base_dir = Path(args.debug_dir) if args.debug_dir else default_debug_dir(source)
        debug_root = allocate_run_dir(debug_base_dir)
        _, debug_frames_dir, debug_panels_dir = ensure_debug_layout(debug_root)
        debug_csv_path = debug_root / "records.csv"
        debug_csv_file = debug_csv_path.open("w", newline="", encoding="utf-8-sig")
        debug_writer = csv.DictWriter(
            debug_csv_file,
            fieldnames=[
                "frame_index",
                "detection_index",
                "label",
                "color_name",
                "person_confidence",
                "roi_source",
                "head_point_count",
                "shoulder_point_count",
                "shoulder_span",
                "has_helmet_roi",
                "person_x1",
                "person_y1",
                "person_x2",
                "person_y2",
                "helmet_x1",
                "helmet_y1",
                "helmet_x2",
                "helmet_y2",
                "analysis_x1",
                "analysis_y1",
                "analysis_x2",
                "analysis_y2",
                "analysis_area",
                "white_pixels",
                "red_pixels",
                "white_ratio",
                "red_ratio",
            ],
        )
        debug_writer.writeheader()

    model = YOLO(args.model)
    inference_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.person_conf,
        "verbose": False,
    }
    if args.device:
        inference_kwargs["device"] = args.device

    frame_index = 0
    summary = {"manager": 0, "worker": 0}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        if args.max_frames and frame_index > args.max_frames:
            break

        results = model(frame, **inference_kwargs)
        result = results[0]
        text_overlays: list[tuple[str, int, int, tuple[int, int, int], tuple[int, int, int]]] = []
        debug_frame_requested = False

        if result.boxes is not None and result.keypoints is not None and result.boxes.xyxy is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()

            for detection_index, (person_box, person_conf, person_kpt_xy, person_kpt_conf) in enumerate(
                zip(boxes_xyxy, boxes_conf, keypoints_xy, keypoints_conf),
                start=1,
            ):
                helmet_box, roi_meta = estimate_head_roi(person_box, person_kpt_xy, person_kpt_conf, frame.shape, args.keypoint_conf)
                label = "manager"
                color_name = "non-red"
                white_ratio = 0.0
                red_ratio = 0.0
                color_debug = {
                    "roi_bgr": np.zeros((1, 1, 3), dtype=np.uint8),
                    "white_mask": np.zeros((1, 1), dtype=np.uint8),
                    "red_mask": np.zeros((1, 1), dtype=np.uint8),
                    "analysis_x1": 0,
                    "analysis_y1": 0,
                    "analysis_x2": 0,
                    "analysis_y2": 0,
                    "analysis_area": 0,
                    "white_pixels": 0,
                    "red_pixels": 0,
                }

                if helmet_box is not None:
                    hx1, hy1, hx2, hy2 = helmet_box
                    roi = frame[hy1:hy2, hx1:hx2]
                    label, color_name, white_ratio, red_ratio, color_debug = classify_helmet_color(roi, args)

                summary[label] += 1
                text_x, text_y, bg_color, text_color = draw_detection_boxes(frame, person_box, helmet_box, label, args)
                overlay_label = make_overlay_label(label, white_ratio, red_ratio, args.debug_text)
                text_overlays.append((overlay_label, text_x, text_y, bg_color, text_color))

                if debug_writer is not None and debug_panels_dir is not None and should_capture_debug(
                    args, frame_index, label, debug_sample_count
                ):
                    helmet_x1, helmet_y1, helmet_x2, helmet_y2 = helmet_box if helmet_box is not None else (0, 0, 0, 0)
                    record = {
                        "frame_index": frame_index,
                        "detection_index": detection_index,
                        "label": label,
                        "color_name": color_name,
                        "person_confidence": f"{float(person_conf):.4f}",
                        "roi_source": roi_meta["roi_source"],
                        "head_point_count": roi_meta["head_point_count"],
                        "shoulder_point_count": roi_meta["shoulder_point_count"],
                        "shoulder_span": f"{float(roi_meta['shoulder_span']):.2f}",
                        "has_helmet_roi": int(helmet_box is not None),
                        "person_x1": int(round(person_box[0])),
                        "person_y1": int(round(person_box[1])),
                        "person_x2": int(round(person_box[2])),
                        "person_y2": int(round(person_box[3])),
                        "helmet_x1": helmet_x1,
                        "helmet_y1": helmet_y1,
                        "helmet_x2": helmet_x2,
                        "helmet_y2": helmet_y2,
                        "analysis_x1": color_debug["analysis_x1"],
                        "analysis_y1": color_debug["analysis_y1"],
                        "analysis_x2": color_debug["analysis_x2"],
                        "analysis_y2": color_debug["analysis_y2"],
                        "analysis_area": color_debug["analysis_area"],
                        "white_pixels": color_debug["white_pixels"],
                        "red_pixels": color_debug["red_pixels"],
                        "white_ratio": f"{white_ratio:.4f}",
                        "red_ratio": f"{red_ratio:.4f}",
                    }
                    debug_writer.writerow(record)
                    save_debug_panel(debug_panels_dir, frame, frame_index, detection_index, record, person_box, helmet_box, color_debug)
                    debug_sample_count += 1
                    debug_frame_requested = True

        frame = render_text_overlays(frame, text_overlays, font)

        if debug_frame_requested and debug_frames_dir is not None:
            frame_path = debug_frames_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

        writer.write(frame)

        if frame_index == 1 or frame_index % 30 == 0:
            suffix = f"/{total_frames}" if total_frames else ""
            print(f"[info] Processed frame {frame_index}{suffix}")

    cap.release()
    writer.release()
    if debug_csv_file is not None:
        debug_csv_file.close()

    print(f"[done] Output saved to {actual_output_path.resolve()}")
    if args.save_debug_artifacts:
        print(f"[done] Debug outputs saved to {debug_root.resolve()}")
        print(f"[done] Debug samples saved: {debug_sample_count}")
    print(
        "[done] 标签统计:"
        f" 管理人员={summary['manager']}, 工作人员={summary['worker']}"
    )
    return actual_output_path.resolve()


def main() -> None:
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
