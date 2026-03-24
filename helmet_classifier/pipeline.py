from __future__ import annotations

import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from .analysis import classify_helmet_color, estimate_head_roi
from .config import AppConfig
from .debug_output import ensure_debug_layout, save_debug_panel, should_capture_debug
from .paths import allocate_run_dir, default_debug_dir, default_output_path
from .render import build_overlay, draw_detection_boxes, load_label_font, render_text_overlays
from .schemas import ColorDebugData, OverlayText


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


def process_video(config: AppConfig) -> Path:
    source = Path(config.source)
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    output_path = Path(config.output) if config.output else default_output_path(source)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer, actual_output_path = create_video_writer(output_path, fps, width, height, config.codec)
    font = load_label_font(config.font_path, config.font_size)
    debug_layout = None
    debug_csv_file = None
    debug_writer = None
    debug_sample_count = 0

    if config.save_debug_artifacts:
        debug_base_dir = Path(config.debug_dir) if config.debug_dir else default_debug_dir(source)
        debug_layout = ensure_debug_layout(allocate_run_dir(debug_base_dir))
        debug_csv_file = debug_layout.csv_path.open("w", newline="", encoding="utf-8-sig")
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

    model = YOLO(config.model)
    inference_kwargs = {
        "imgsz": config.imgsz,
        "conf": config.person_conf,
        "verbose": False,
    }
    if config.device:
        inference_kwargs["device"] = config.device

    frame_index = 0
    summary = {"manager": 0, "worker": 0}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if config.max_frames and frame_index > config.max_frames:
                break

            results = model(frame, **inference_kwargs)
            result = results[0]
            text_overlays: list[OverlayText] = []
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
                    helmet_box, roi_meta = estimate_head_roi(
                        person_box,
                        person_kpt_xy,
                        person_kpt_conf,
                        frame.shape,
                        config.keypoint_conf,
                    )
                    label = "manager"
                    color_name = "non-red"
                    white_ratio = 0.0
                    red_ratio = 0.0
                    color_debug = ColorDebugData.empty()

                    if helmet_box is not None:
                        hx1, hy1, hx2, hy2 = helmet_box
                        roi = frame[hy1:hy2, hx1:hx2]
                        label, color_name, white_ratio, red_ratio, color_debug = classify_helmet_color(roi, config)

                    summary[label] += 1
                    text_x, text_y, bg_color, text_color = draw_detection_boxes(
                        frame,
                        person_box,
                        helmet_box,
                        label,
                        config,
                    )
                    text_overlays.append(
                        build_overlay(
                            label=label,
                            white_ratio=white_ratio,
                            red_ratio=red_ratio,
                            debug_text=config.debug_text,
                            text_x=text_x,
                            text_y=text_y,
                            bg_color=bg_color,
                            text_color=text_color,
                        )
                    )

                    if debug_writer is not None and debug_layout is not None and should_capture_debug(
                        config,
                        frame_index,
                        label,
                        debug_sample_count,
                    ):
                        helmet_x1, helmet_y1, helmet_x2, helmet_y2 = helmet_box if helmet_box is not None else (0, 0, 0, 0)
                        record = {
                            "frame_index": frame_index,
                            "detection_index": detection_index,
                            "label": label,
                            "color_name": color_name,
                            "person_confidence": f"{float(person_conf):.4f}",
                            "roi_source": roi_meta.roi_source,
                            "head_point_count": roi_meta.head_point_count,
                            "shoulder_point_count": roi_meta.shoulder_point_count,
                            "shoulder_span": f"{roi_meta.shoulder_span:.2f}",
                            "has_helmet_roi": int(helmet_box is not None),
                            "person_x1": int(round(person_box[0])),
                            "person_y1": int(round(person_box[1])),
                            "person_x2": int(round(person_box[2])),
                            "person_y2": int(round(person_box[3])),
                            "helmet_x1": helmet_x1,
                            "helmet_y1": helmet_y1,
                            "helmet_x2": helmet_x2,
                            "helmet_y2": helmet_y2,
                            "analysis_x1": color_debug.analysis_x1,
                            "analysis_y1": color_debug.analysis_y1,
                            "analysis_x2": color_debug.analysis_x2,
                            "analysis_y2": color_debug.analysis_y2,
                            "analysis_area": color_debug.analysis_area,
                            "white_pixels": color_debug.white_pixels,
                            "red_pixels": color_debug.red_pixels,
                            "white_ratio": f"{white_ratio:.4f}",
                            "red_ratio": f"{red_ratio:.4f}",
                        }
                        debug_writer.writerow(record)
                        save_debug_panel(
                            debug_layout.panels_dir,
                            frame,
                            frame_index,
                            detection_index,
                            record,
                            person_box,
                            helmet_box,
                            color_debug,
                        )
                        debug_sample_count += 1
                        debug_frame_requested = True

            frame = render_text_overlays(frame, text_overlays, font)

            if debug_frame_requested and debug_layout is not None:
                frame_path = debug_layout.frames_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)

            writer.write(frame)

            if frame_index == 1 or frame_index % 30 == 0:
                suffix = f"/{total_frames}" if total_frames else ""
                print(f"[info] Processed frame {frame_index}{suffix}")
    finally:
        cap.release()
        writer.release()
        if debug_csv_file is not None:
            debug_csv_file.close()

    print(f"[done] Output saved to {actual_output_path.resolve()}")
    if debug_layout is not None:
        print(f"[done] Debug outputs saved to {debug_layout.root_dir.resolve()}")
        print(f"[done] Debug samples saved: {debug_sample_count}")
    print(f"[done] 标签统计: 管理人员={summary['manager']}, 工作人员={summary['worker']}")
    return actual_output_path.resolve()
