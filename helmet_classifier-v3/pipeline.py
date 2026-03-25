from __future__ import annotations

import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from .analysis import (
    classify_helmet_color,
    classify_vest_color,
    decide_person_label,
    estimate_head_roi,
    estimate_torso_roi,
)
from .config import AppConfig
from .debug_output import ensure_debug_layout, save_debug_panel, should_capture_debug
from .paths import allocate_run_dir, default_debug_dir, default_output_path
from .render import build_overlay, draw_detection_boxes, load_label_font, render_text_overlays
from .schemas import (
    HelmetColorDebugData,
    HelmetColorResult,
    OverlayText,
    VestColorDebugData,
    VestColorResult,
)


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


def _empty_helmet_result() -> HelmetColorResult:
    return HelmetColorResult(
        helmet_color="unknown",
        white_ratio=0.0,
        red_ratio=0.0,
        debug=HelmetColorDebugData.empty(),
    )


def _empty_vest_result() -> VestColorResult:
    return VestColorResult(
        vest_color="unknown",
        yellow_green_ratio=0.0,
        red_ratio=0.0,
        orange_ratio=0.0,
        white_ratio=0.0,
        debug=VestColorDebugData.empty(),
    )


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
                "helmet_color",
                "vest_color",
                "person_confidence",
                "roi_source",
                "torso_roi_source",
                "head_point_count",
                "shoulder_point_count",
                "hip_point_count",
                "shoulder_span",
                "hip_span",
                "has_helmet_roi",
                "has_torso_roi",
                "person_x1",
                "person_y1",
                "person_x2",
                "person_y2",
                "helmet_x1",
                "helmet_y1",
                "helmet_x2",
                "helmet_y2",
                "torso_x1",
                "torso_y1",
                "torso_x2",
                "torso_y2",
                "analysis_x1",
                "analysis_y1",
                "analysis_x2",
                "analysis_y2",
                "analysis_area",
                "vest_analysis_x1",
                "vest_analysis_y1",
                "vest_analysis_x2",
                "vest_analysis_y2",
                "vest_analysis_area",
                "white_pixels",
                "red_pixels",
                "white_ratio",
                "red_ratio",
                "vest_yellow_green_pixels",
                "vest_red_pixels",
                "vest_orange_pixels",
                "vest_white_pixels",
                "vest_yellow_green_ratio",
                "vest_red_ratio",
                "vest_orange_ratio",
                "vest_white_ratio",
                "helmet_match_manager_rule",
                "vest_match_manager_rule",
                "manager_rule_matched",
                "final_decision_rule",
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
                    helmet_box, head_meta = estimate_head_roi(
                        person_box,
                        person_kpt_xy,
                        person_kpt_conf,
                        frame.shape,
                        config.keypoint_conf,
                    )
                    torso_box, torso_meta = estimate_torso_roi(
                        person_box,
                        person_kpt_xy,
                        person_kpt_conf,
                        frame.shape,
                        config.torso_keypoint_conf,
                    )

                    helmet_result = _empty_helmet_result()
                    if helmet_box is not None:
                        hx1, hy1, hx2, hy2 = helmet_box
                        helmet_roi = frame[hy1:hy2, hx1:hx2]
                        helmet_result = classify_helmet_color(helmet_roi, config)

                    vest_result = _empty_vest_result()
                    if torso_box is not None:
                        tx1, ty1, tx2, ty2 = torso_box
                        torso_roi = frame[ty1:ty2, tx1:tx2]
                        vest_result = classify_vest_color(torso_roi, config)

                    decision = decide_person_label(
                        helmet_box=helmet_box,
                        torso_box=torso_box,
                        helmet_result=helmet_result,
                        vest_result=vest_result,
                        config=config,
                    )

                    summary[decision.label] += 1
                    text_x, text_y, bg_color, text_color = draw_detection_boxes(
                        frame,
                        person_box,
                        helmet_box,
                        torso_box,
                        decision.label,
                        config,
                    )
                    text_overlays.append(
                        build_overlay(
                            label=decision.label,
                            helmet_color=helmet_result.helmet_color,
                            vest_color=vest_result.vest_color,
                            debug_text=config.debug_text,
                            enable_joint_decision=config.enable_joint_decision,
                            text_x=text_x,
                            text_y=text_y,
                            bg_color=bg_color,
                            text_color=text_color,
                        )
                    )

                    if debug_writer is not None and debug_layout is not None and should_capture_debug(
                        config,
                        frame_index,
                        decision.label,
                        debug_sample_count,
                    ):
                        helmet_x1, helmet_y1, helmet_x2, helmet_y2 = helmet_box if helmet_box is not None else (0, 0, 0, 0)
                        torso_x1, torso_y1, torso_x2, torso_y2 = torso_box if torso_box is not None else (0, 0, 0, 0)
                        record = {
                            "frame_index": frame_index,
                            "detection_index": detection_index,
                            "label": decision.label,
                            "color_name": helmet_result.helmet_color,
                            "helmet_color": helmet_result.helmet_color,
                            "vest_color": vest_result.vest_color,
                            "person_confidence": f"{float(person_conf):.4f}",
                            "roi_source": head_meta.roi_source,
                            "torso_roi_source": torso_meta.roi_source,
                            "head_point_count": head_meta.head_point_count,
                            "shoulder_point_count": torso_meta.shoulder_point_count,
                            "hip_point_count": torso_meta.hip_point_count,
                            "shoulder_span": f"{torso_meta.shoulder_span:.2f}",
                            "hip_span": f"{torso_meta.hip_span:.2f}",
                            "has_helmet_roi": int(helmet_box is not None),
                            "has_torso_roi": int(torso_box is not None),
                            "person_x1": int(round(person_box[0])),
                            "person_y1": int(round(person_box[1])),
                            "person_x2": int(round(person_box[2])),
                            "person_y2": int(round(person_box[3])),
                            "helmet_x1": helmet_x1,
                            "helmet_y1": helmet_y1,
                            "helmet_x2": helmet_x2,
                            "helmet_y2": helmet_y2,
                            "torso_x1": torso_x1,
                            "torso_y1": torso_y1,
                            "torso_x2": torso_x2,
                            "torso_y2": torso_y2,
                            "analysis_x1": helmet_result.debug.analysis_x1,
                            "analysis_y1": helmet_result.debug.analysis_y1,
                            "analysis_x2": helmet_result.debug.analysis_x2,
                            "analysis_y2": helmet_result.debug.analysis_y2,
                            "analysis_area": helmet_result.debug.analysis_area,
                            "vest_analysis_x1": vest_result.debug.analysis_x1,
                            "vest_analysis_y1": vest_result.debug.analysis_y1,
                            "vest_analysis_x2": vest_result.debug.analysis_x2,
                            "vest_analysis_y2": vest_result.debug.analysis_y2,
                            "vest_analysis_area": vest_result.debug.analysis_area,
                            "white_pixels": helmet_result.debug.white_pixels,
                            "red_pixels": helmet_result.debug.red_pixels,
                            "white_ratio": f"{helmet_result.white_ratio:.4f}",
                            "red_ratio": f"{helmet_result.red_ratio:.4f}",
                            "vest_yellow_green_pixels": vest_result.debug.yellow_green_pixels,
                            "vest_red_pixels": vest_result.debug.red_pixels,
                            "vest_orange_pixels": vest_result.debug.orange_pixels,
                            "vest_white_pixels": vest_result.debug.white_pixels,
                            "vest_yellow_green_ratio": f"{vest_result.yellow_green_ratio:.4f}",
                            "vest_red_ratio": f"{vest_result.red_ratio:.4f}",
                            "vest_orange_ratio": f"{vest_result.orange_ratio:.4f}",
                            "vest_white_ratio": f"{vest_result.white_ratio:.4f}",
                            "helmet_match_manager_rule": int(decision.helmet_match_manager_rule),
                            "vest_match_manager_rule": int(decision.vest_match_manager_rule),
                            "manager_rule_matched": int(decision.manager_rule_matched),
                            "final_decision_rule": decision.final_decision_rule,
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
                            torso_box,
                            helmet_result.debug,
                            vest_result.debug,
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
    print(f"[done] Labels summary: manager={summary['manager']}, worker={summary['worker']}")
    return actual_output_path.resolve()
