"""这个脚本用于对视频中的人员安全帽颜色进行规则化识别：它先使用 YOLO 姿态模型逐帧检测人员和关键点，再根据头部或肩部关键点估计头盔所在区域，对该区域做 HSV 颜色分析，并将红帽映射为“工作人员”、其余情况统一映射为“管理人员”，最后把中文标注直接绘制回输出视频中。"""

from __future__ import annotations

import argparse
import math
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
        default=0.06,
        help="Minimum red pixel ratio required to classify a red helmet.",
    )
    parser.add_argument(
        "--dominance-ratio",
        type=float,
        default=1.25,
        help="Winning color must exceed the other color by this factor.",
    )
    parser.add_argument("--white-s-max", type=int, default=55, help="Maximum HSV saturation for white.")
    parser.add_argument("--white-v-min", type=int, default=170, help="Minimum HSV value for white.")
    parser.add_argument("--red-s-min", type=int, default=120, help="Minimum HSV saturation for red.")
    parser.add_argument("--red-v-min", type=int, default=90, help="Minimum HSV value for red.")
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
    return parser.parse_args()


def default_output_path(source: Path) -> Path:
    return Path("outputs") / "helmet-classify" / f"{source.stem}.mp4"


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
) -> tuple[int, int, int, int] | None:
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

    if head_points.shape[0] >= 2:
        center_x = float(np.average(head_points[:, 0], weights=head_weights))
        center_y = float(np.average(head_points[:, 1], weights=head_weights))
        roi_w = max(box_w * 0.22, shoulder_span * 0.85, 24.0)
        roi_h = max(box_h * 0.18, shoulder_span * 0.75, 24.0)
        center_y -= roi_h * 0.15
    elif shoulder_points.shape[0] == 2:
        center_x = float(np.mean(shoulder_points[:, 0]))
        center_y = float(np.mean(shoulder_points[:, 1])) - max(shoulder_span * 0.85, box_h * 0.16, 20.0)
        roi_w = max(shoulder_span * 0.95, box_w * 0.22, 24.0)
        roi_h = max(shoulder_span * 0.90, box_h * 0.18, 24.0)
    else:
        return None

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
        return None
    return roi_x1, roi_y1, roi_x2, roi_y2


def classify_helmet_color(roi_bgr: np.ndarray, args: argparse.Namespace) -> tuple[str, str, float, float]:
    if roi_bgr.size == 0:
        return "manager", "non-red", 0.0, 0.0

    height, width = roi_bgr.shape[:2]
    top_end = max(1, int(height * 0.75))
    x_margin = max(0, int(width * 0.10))
    core = roi_bgr[:top_end, x_margin : width - x_margin] if width - (2 * x_margin) >= 4 else roi_bgr[:top_end, :]
    if core.size == 0:
        core = roi_bgr

    blurred = cv2.GaussianBlur(core, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, (0, 0, args.white_v_min), (179, args.white_s_max, 255))
    red_mask_low = cv2.inRange(hsv, (0, args.red_s_min, args.red_v_min), (10, 255, 255))
    red_mask_high = cv2.inRange(hsv, (160, args.red_s_min, args.red_v_min), (179, 255, 255))
    red_mask = cv2.bitwise_or(red_mask_low, red_mask_high)

    kernel = np.ones((3, 3), dtype=np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    area = float(core.shape[0] * core.shape[1])
    white_ratio = float(np.count_nonzero(white_mask) / area)
    red_ratio = float(np.count_nonzero(red_mask) / area)

    if red_ratio >= args.red_ratio_threshold and red_ratio > white_ratio * args.dominance_ratio:
        return "worker", "red", white_ratio, red_ratio
    return "manager", "non-red", white_ratio, red_ratio


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

        if result.boxes is not None and result.keypoints is not None and result.boxes.xyxy is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()

            for person_box, person_kpt_xy, person_kpt_conf in zip(boxes_xyxy, keypoints_xy, keypoints_conf):
                helmet_box = estimate_head_roi(person_box, person_kpt_xy, person_kpt_conf, frame.shape, args.keypoint_conf)
                label = "manager"
                color_name = "non-red"
                white_ratio = 0.0
                red_ratio = 0.0

                if helmet_box is not None:
                    hx1, hy1, hx2, hy2 = helmet_box
                    roi = frame[hy1:hy2, hx1:hx2]
                    label, color_name, white_ratio, red_ratio = classify_helmet_color(roi, args)

                summary[label] += 1
                text_x, text_y, bg_color, text_color = draw_detection_boxes(frame, person_box, helmet_box, label, args)
                overlay_label = make_overlay_label(label, white_ratio, red_ratio, args.debug_text)
                text_overlays.append((overlay_label, text_x, text_y, bg_color, text_color))

        frame = render_text_overlays(frame, text_overlays, font)

        writer.write(frame)

        if frame_index == 1 or frame_index % 30 == 0:
            suffix = f"/{total_frames}" if total_frames else ""
            print(f"[info] Processed frame {frame_index}{suffix}")

    cap.release()
    writer.release()

    print(f"[done] Output saved to {actual_output_path.resolve()}")
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
