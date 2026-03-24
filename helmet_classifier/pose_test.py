from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
from ultralytics import YOLO

POSE_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

POSE_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass(slots=True)
class PoseTestConfig:
    sources: list[str]
    model: str = "yolo11n-pose.pt"
    output_dir: str = "outputs/pose-test"
    imgsz: int = 960
    conf: float = 0.25
    keypoint_conf: float = 0.25
    device: str = ""


@dataclass(slots=True)
class DetectionSummary:
    detection_index: int
    confidence: float
    bbox_xyxy: list[int]
    visible_keypoint_count: int
    keypoints: list[dict[str, float | int | str]]


class PoseImageTester:
    def __init__(self, config: PoseTestConfig) -> None:
        self.config = config
        self.model = YOLO(config.model)

    def run(self) -> list[dict[str, object]]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries: list[dict[str, object]] = []

        for source in self.config.sources:
            summaries.append(self._process_one(Path(source), output_dir))

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        return summaries

    def _process_one(self, image_path: Path, output_dir: Path) -> dict[str, object]:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(
            source=str(image_path),
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            device=self.config.device or None,
            verbose=False,
        )
        if not results:
            raise RuntimeError(f"No prediction results returned for {image_path}")

        result = results[0]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        box_data = result.boxes
        keypoint_data = result.keypoints

        detections: list[DetectionSummary] = []
        if box_data is not None and keypoint_data is not None and box_data.xyxy is not None:
            boxes_xyxy = box_data.xyxy.cpu().numpy()
            boxes_conf = box_data.conf.cpu().numpy()
            kpt_xy = keypoint_data.xy.cpu().numpy()
            kpt_conf = keypoint_data.conf.cpu().numpy()

            for index, (bbox, confidence, points_xy, points_conf) in enumerate(
                zip(boxes_xyxy, boxes_conf, kpt_xy, kpt_conf),
                start=1,
            ):
                summary = self._draw_detection(image, index, bbox, float(confidence), points_xy, points_conf)
                detections.append(summary)

        annotated_path = output_dir / f"{image_path.stem}_pose.jpg"
        cv2.imwrite(str(annotated_path), image)

        image_summary = {
            "image": str(image_path),
            "annotated_image": str(annotated_path),
            "model": self.config.model,
            "imgsz": self.config.imgsz,
            "conf": self.config.conf,
            "keypoint_conf": self.config.keypoint_conf,
            "detection_count": len(detections),
            "detections": [asdict(item) for item in detections],
        }

        image_summary_path = output_dir / f"{image_path.stem}_summary.json"
        image_summary_path.write_text(json.dumps(image_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return image_summary

    def _draw_detection(
        self,
        image: cv2.Mat,
        detection_index: int,
        bbox: Sequence[float],
        confidence: float,
        points_xy: Sequence[Sequence[float]],
        points_conf: Sequence[float],
    ) -> DetectionSummary:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        visible_count = 0
        keypoints: list[dict[str, float | int | str]] = []

        for point_index, (xy, point_conf) in enumerate(zip(points_xy, points_conf)):
            x, y = float(xy[0]), float(xy[1])
            point_conf = float(point_conf)
            is_visible = point_conf >= self.config.keypoint_conf
            if is_visible:
                visible_count += 1
                cv2.circle(image, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
                cv2.putText(
                    image,
                    str(point_index),
                    (int(round(x)) + 5, int(round(y)) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            keypoints.append(
                {
                    "index": point_index,
                    "name": POSE_KEYPOINT_NAMES[point_index],
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "confidence": round(point_conf, 4),
                    "visible": int(is_visible),
                }
            )

        for start, end in POSE_SKELETON:
            if points_conf[start] < self.config.keypoint_conf or points_conf[end] < self.config.keypoint_conf:
                continue
            pt1 = (int(round(points_xy[start][0])), int(round(points_xy[start][1])))
            pt2 = (int(round(points_xy[end][0])), int(round(points_xy[end][1])))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)

        label = f"person#{detection_index} conf={confidence:.2f} kp={visible_count}"
        label_y = max(20, y1 - 8)
        cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        return DetectionSummary(
            detection_index=detection_index,
            confidence=round(confidence, 4),
            bbox_xyxy=[x1, y1, x2, y2],
            visible_keypoint_count=visible_count,
            keypoints=keypoints,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO pose detection on one or more images and save debug overlays.")
    parser.add_argument("--source", action="append", required=True, help="Input image path. Repeat the flag for multiple images.")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="YOLO pose model path.")
    parser.add_argument("--output-dir", default="outputs/pose-test", help="Directory used to save annotated images and JSON summaries.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Person detection confidence threshold.")
    parser.add_argument("--keypoint-conf", type=float, default=0.25, help="Minimum confidence required to draw a keypoint.")
    parser.add_argument("--device", default="", help="Ultralytics device. Empty uses the default device.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PoseTestConfig:
    namespace = build_parser().parse_args(argv)
    return PoseTestConfig(
        sources=list(namespace.source),
        model=namespace.model,
        output_dir=namespace.output_dir,
        imgsz=namespace.imgsz,
        conf=namespace.conf,
        keypoint_conf=namespace.keypoint_conf,
        device=namespace.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    tester = PoseImageTester(config)
    summaries = tester.run()
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
