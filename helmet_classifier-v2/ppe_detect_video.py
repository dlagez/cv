"""Run PPE detection on a video and keep only Person and Hardhat boxes."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

TARGET_LABELS = {
    "Hardhat": "\u5b89\u5168\u5e3d",
    "Person": "\u4eba\u5458",
}

FONT_CANDIDATES = (
    "msyh.ttc",
    "msyhbd.ttc",
    "simhei.ttf",
    "simsun.ttc",
    "Arial.Unicode.ttf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect only person and hardhat in a video.")
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument("--model", required=True, help="YOLO model path.")
    parser.add_argument("--output", default="", help="Output video path.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Base confidence threshold used for model inference.")
    parser.add_argument("--person-conf", type=float, default=0.45, help="Final confidence threshold for Person boxes.")
    parser.add_argument("--hardhat-conf", type=float, default=0.25, help="Final confidence threshold for Hardhat boxes.")
    parser.add_argument("--device", default="", help="Ultralytics device. Empty uses default device.")
    parser.add_argument("--line-width", type=int, default=1, help="Bounding box line width.")
    parser.add_argument("--font-size", type=float, default=20, help="Label font size. Values below 4 are treated as scale-like input and normalized.")
    parser.add_argument("--max-frames", type=int, default=0, help="Process only the first N frames for testing.")
    return parser.parse_args()


def create_writer(output_path: Path, fps: float, width: int, height: int) -> tuple[cv2.VideoWriter, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
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
        return writer, fallback_path

    raise RuntimeError(f"Could not create output writer for {output_path}")


def default_output_path(source: Path) -> Path:
    return Path("helmet_classifier-v2") / "outputs" / f"{source.stem}-person-hardhat.mp4"


def resolve_target_classes(model: YOLO) -> dict[int, str]:
    targets: dict[int, str] = {}
    for cls_id, name in model.names.items():
        if name in TARGET_LABELS:
            targets[int(cls_id)] = TARGET_LABELS[name]
    missing = set(TARGET_LABELS) - {model.names[cls_id] for cls_id in targets}
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise RuntimeError(f"Model does not contain required classes: {missing_list}")
    return targets


def resolve_font_name() -> str:
    windows_fonts = Path(r"C:\Windows\Fonts")
    for name in FONT_CANDIDATES:
        if (windows_fonts / name).exists():
            return name
    return "Arial.Unicode.ttf"


def resolve_font_size(raw_value: float) -> int:
    if raw_value < 4:
        return 20
    return max(12, int(round(raw_value)))


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    model = YOLO(args.model)
    target_classes = resolve_target_classes(model)

    output_path = Path(args.output) if args.output else default_output_path(source)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    writer, actual_output_path = create_writer(output_path, fps, width, height)

    inference_kwargs = {
        "imgsz": args.imgsz,
        "conf": min(args.conf, args.person_conf, args.hardhat_conf),
        "verbose": False,
    }
    if args.device:
        inference_kwargs["device"] = args.device

    frame_index = 0
    font_name = resolve_font_name()
    font_size = resolve_font_size(args.font_size)
    kept_counts = {
        "\u4eba\u5458": 0,
        "\u5b89\u5168\u5e3d": 0,
    }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if args.max_frames and frame_index > args.max_frames:
                break
            result = model(frame, **inference_kwargs)[0]
            annotator = Annotator(
                frame.copy(),
                line_width=args.line_width,
                font_size=font_size,
                font=font_name,
                pil=True,
                example="\u4eba\u5458\u5b89\u5168\u5e3d",
            )

            if result.boxes is not None and result.boxes.xyxy is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, classes):
                    if cls_id not in target_classes:
                        continue
                    raw_name = model.names[int(cls_id)]
                    threshold = args.person_conf if raw_name == "Person" else args.hardhat_conf
                    if conf < threshold:
                        continue
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    label_name = target_classes[cls_id]
                    label = f"{label_name} {conf:.2f}"
                    annotator.box_label((x1, y1, x2, y2), label, color=colors(cls_id, True))
                    kept_counts[label_name] += 1

            writer.write(annotator.result())
            if frame_index == 1 or frame_index % 30 == 0:
                suffix = f"/{total_frames}" if total_frames else ""
                print(f"[info] Processed frame {frame_index}{suffix}")
    finally:
        cap.release()
        writer.release()

    print(f"[done] Output saved to {actual_output_path.resolve()}")
    print(
        f"[done] \u68c0\u6d4b\u7edf\u8ba1: "
        f"\u4eba\u5458={kept_counts['\u4eba\u5458']}, "
        f"\u5b89\u5168\u5e3d={kept_counts['\u5b89\u5168\u5e3d']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
