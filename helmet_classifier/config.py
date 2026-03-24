from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class AppConfig:
    source: str
    model: str = "yolo11n-pose.pt"
    output: str = ""
    imgsz: int = 960
    person_conf: float = 0.35
    keypoint_conf: float = 0.35
    white_ratio_threshold: float = 0.18
    red_ratio_threshold: float = 0.15
    white_s_max: int = 55
    white_v_min: int = 170
    red_h_low_max: int = 18
    red_h_high_min: int = 145
    red_s_min: int = 75
    red_v_min: int = 60
    device: str = ""
    codec: str = "mp4v"
    font_path: str = ""
    font_size: int = 22
    draw_helmet_box: bool = False
    debug_text: bool = False
    max_frames: int = 0
    save_debug_artifacts: bool = False
    debug_dir: str = ""
    debug_sample_every: int = 60
    debug_max_samples: int = 80


def build_parser() -> argparse.ArgumentParser:
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
        help="Legacy white ratio threshold kept for compatibility.",
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AppConfig:
    namespace = build_parser().parse_args(argv)
    return AppConfig(**vars(namespace))
