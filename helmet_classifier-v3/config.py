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
    torso_keypoint_conf: float = 0.35
    helmet_white_ratio_threshold: float = 0.18
    helmet_red_ratio_threshold: float = 0.15
    vest_yellow_green_ratio_threshold: float = 0.28
    vest_red_ratio_threshold: float = 0.20
    vest_orange_ratio_threshold: float = 0.20
    vest_white_ratio_threshold: float = 0.18
    vest_non_white_min_ratio: float = 0.10
    vest_non_white_min_pixels: int = 24
    white_s_max: int = 55
    white_v_min: int = 170
    red_h_low_max: int = 18
    red_h_high_min: int = 145
    red_s_min: int = 75
    red_v_min: int = 60
    vest_yellow_green_h_min: int = 28
    vest_yellow_green_h_max: int = 95
    vest_yellow_green_s_min: int = 70
    vest_yellow_green_v_min: int = 110
    vest_orange_h_min: int = 8
    vest_orange_h_max: int = 28
    vest_orange_s_min: int = 80
    vest_orange_v_min: int = 80
    torso_roi_min_width: int = 32
    torso_roi_min_height: int = 48
    torso_roi_min_area: int = 1400
    torso_fallback_x_margin_ratio: float = 0.16
    torso_fallback_top_ratio: float = 0.18
    torso_fallback_bottom_ratio: float = 0.76
    device: str = ""
    codec: str = "mp4v"
    font_path: str = ""
    font_size: int = 22
    draw_helmet_box: bool = False
    draw_torso_box: bool = False
    debug_text: bool = False
    max_frames: int = 0
    save_debug_artifacts: bool = False
    debug_dir: str = ""
    debug_sample_every: int = 60
    debug_max_samples: int = 80
    enable_joint_decision: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify helmet and reflective vest colors in a video with YOLO pose keypoints and HSV rules."
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
        "--torso-keypoint-conf",
        type=float,
        default=0.35,
        help="Minimum keypoint confidence used to estimate the torso area.",
    )
    parser.add_argument(
        "--helmet-white-ratio-threshold",
        type=float,
        default=0.18,
        help="Minimum white pixel ratio required to classify a helmet as white.",
    )
    parser.add_argument(
        "--helmet-red-ratio-threshold",
        type=float,
        default=0.15,
        help="Minimum red pixel ratio required to classify a helmet as red.",
    )
    parser.add_argument(
        "--vest-yellow-green-ratio-threshold",
        type=float,
        default=0.28,
        help="Minimum yellow-green pixel ratio required to classify a vest as fluorescent yellow-green.",
    )
    parser.add_argument(
        "--vest-red-ratio-threshold",
        type=float,
        default=0.20,
        help="Minimum red pixel ratio required to classify a vest as red.",
    )
    parser.add_argument(
        "--vest-orange-ratio-threshold",
        type=float,
        default=0.20,
        help="Minimum orange pixel ratio required to classify a vest as orange.",
    )
    parser.add_argument(
        "--vest-white-ratio-threshold",
        type=float,
        default=0.18,
        help="Reference threshold for white reflective-strip statistics. Not used as a final vest color class.",
    )
    parser.add_argument(
        "--vest-non-white-min-ratio",
        type=float,
        default=0.10,
        help="Minimum non-white area ratio required before making a final vest-color decision.",
    )
    parser.add_argument(
        "--vest-non-white-min-pixels",
        type=int,
        default=24,
        help="Minimum non-white pixel count required before making a final vest-color decision.",
    )
    parser.add_argument(
        "--white-ratio-threshold",
        dest="helmet_white_ratio_threshold",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--red-ratio-threshold",
        dest="helmet_red_ratio_threshold",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--white-s-max", type=int, default=55, help="Maximum HSV saturation for white.")
    parser.add_argument("--white-v-min", type=int, default=170, help="Minimum HSV value for white.")
    parser.add_argument("--red-h-low-max", type=int, default=18, help="Upper hue bound of the low red range.")
    parser.add_argument("--red-h-high-min", type=int, default=145, help="Lower hue bound of the high red range.")
    parser.add_argument("--red-s-min", type=int, default=75, help="Minimum HSV saturation for red.")
    parser.add_argument("--red-v-min", type=int, default=60, help="Minimum HSV value for red.")
    parser.add_argument(
        "--vest-yellow-green-h-min",
        type=int,
        default=28,
        help="Lower hue bound of the fluorescent yellow-green vest range.",
    )
    parser.add_argument(
        "--vest-yellow-green-h-max",
        type=int,
        default=95,
        help="Upper hue bound of the fluorescent yellow-green vest range.",
    )
    parser.add_argument(
        "--vest-yellow-green-s-min",
        type=int,
        default=70,
        help="Minimum HSV saturation for fluorescent yellow-green vest pixels.",
    )
    parser.add_argument(
        "--vest-yellow-green-v-min",
        type=int,
        default=110,
        help="Minimum HSV value for fluorescent yellow-green vest pixels.",
    )
    parser.add_argument(
        "--vest-orange-h-min",
        type=int,
        default=8,
        help="Lower hue bound of the orange vest range.",
    )
    parser.add_argument(
        "--vest-orange-h-max",
        type=int,
        default=28,
        help="Upper hue bound of the orange vest range.",
    )
    parser.add_argument(
        "--vest-orange-s-min",
        type=int,
        default=80,
        help="Minimum HSV saturation for orange vest pixels.",
    )
    parser.add_argument(
        "--vest-orange-v-min",
        type=int,
        default=80,
        help="Minimum HSV value for orange vest pixels.",
    )
    parser.add_argument(
        "--torso-roi-min-width",
        type=int,
        default=32,
        help="Minimum torso ROI width. Smaller ROIs are expanded or fallback-adjusted.",
    )
    parser.add_argument(
        "--torso-roi-min-height",
        type=int,
        default=48,
        help="Minimum torso ROI height. Smaller ROIs are expanded or fallback-adjusted.",
    )
    parser.add_argument(
        "--torso-roi-min-area",
        type=int,
        default=1400,
        help="Minimum torso ROI area. Smaller ROIs are expanded or fallback-adjusted.",
    )
    parser.add_argument(
        "--torso-fallback-x-margin-ratio",
        type=float,
        default=0.16,
        help="Horizontal margin ratio used by the conservative person-box fallback torso ROI.",
    )
    parser.add_argument(
        "--torso-fallback-top-ratio",
        type=float,
        default=0.18,
        help="Top ratio used by the conservative person-box fallback torso ROI.",
    )
    parser.add_argument(
        "--torso-fallback-bottom-ratio",
        type=float,
        default=0.76,
        help="Bottom ratio used by the conservative person-box fallback torso ROI.",
    )
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
        "--draw-torso-box",
        action="store_true",
        help="Draw the estimated torso ROI used for vest color analysis.",
    )
    parser.add_argument(
        "--debug-text",
        action="store_true",
        help="Append helmet and vest color summaries to the overlay labels.",
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
    parser.add_argument(
        "--legacy-helmet-only",
        action="store_false",
        dest="enable_joint_decision",
        help="Disable the new helmet+vest joint decision and keep the legacy helmet-only decision rule.",
    )
    parser.set_defaults(enable_joint_decision=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AppConfig:
    namespace = build_parser().parse_args(argv)
    return AppConfig(**vars(namespace))
