from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class HeadRoiMeta:
    roi_source: str
    head_point_count: int
    shoulder_point_count: int
    shoulder_span: float


@dataclass(frozen=True, slots=True)
class TorsoRoiMeta:
    roi_source: str
    shoulder_point_count: int
    hip_point_count: int
    shoulder_span: float
    hip_span: float


@dataclass(slots=True)
class HelmetColorDebugData:
    roi_bgr: np.ndarray
    analysis_bgr: np.ndarray
    white_mask: np.ndarray
    red_mask: np.ndarray
    analysis_x1: int
    analysis_y1: int
    analysis_x2: int
    analysis_y2: int
    analysis_area: int
    white_pixels: int
    red_pixels: int

    @classmethod
    def empty(cls) -> "HelmetColorDebugData":
        empty_mask = np.zeros((1, 1), dtype=np.uint8)
        empty_roi = np.zeros((1, 1, 3), dtype=np.uint8)
        return cls(
            roi_bgr=empty_roi,
            analysis_bgr=empty_roi.copy(),
            white_mask=empty_mask,
            red_mask=empty_mask.copy(),
            analysis_x1=0,
            analysis_y1=0,
            analysis_x2=0,
            analysis_y2=0,
            analysis_area=0,
            white_pixels=0,
            red_pixels=0,
        )


@dataclass(slots=True)
class VestColorDebugData:
    roi_bgr: np.ndarray
    analysis_bgr: np.ndarray
    yellow_mask: np.ndarray
    green_mask: np.ndarray
    yellow_green_mask: np.ndarray
    red_mask: np.ndarray
    orange_mask: np.ndarray
    white_mask: np.ndarray
    analysis_x1: int
    analysis_y1: int
    analysis_x2: int
    analysis_y2: int
    analysis_area: int
    yellow_pixels: int
    green_pixels: int
    yellow_green_pixels: int
    red_pixels: int
    orange_pixels: int
    white_pixels: int

    @classmethod
    def empty(cls) -> "VestColorDebugData":
        empty_mask = np.zeros((1, 1), dtype=np.uint8)
        empty_roi = np.zeros((1, 1, 3), dtype=np.uint8)
        return cls(
            roi_bgr=empty_roi,
            analysis_bgr=empty_roi.copy(),
            yellow_mask=empty_mask,
            green_mask=empty_mask.copy(),
            yellow_green_mask=empty_mask,
            red_mask=empty_mask.copy(),
            orange_mask=empty_mask.copy(),
            white_mask=empty_mask.copy(),
            analysis_x1=0,
            analysis_y1=0,
            analysis_x2=0,
            analysis_y2=0,
            analysis_area=0,
            yellow_pixels=0,
            green_pixels=0,
            yellow_green_pixels=0,
            red_pixels=0,
            orange_pixels=0,
            white_pixels=0,
        )


@dataclass(slots=True)
class HelmetColorResult:
    helmet_color: str
    white_ratio: float
    red_ratio: float
    debug: HelmetColorDebugData


@dataclass(slots=True)
class VestColorResult:
    vest_color: str
    yellow_ratio: float
    green_ratio: float
    yellow_green_ratio: float
    red_ratio: float
    orange_ratio: float
    white_ratio: float
    debug: VestColorDebugData


@dataclass(frozen=True, slots=True)
class DecisionResult:
    label: str
    helmet_match_manager_rule: bool
    vest_match_manager_rule: bool
    manager_rule_matched: bool
    final_decision_rule: str


@dataclass(frozen=True, slots=True)
class OverlayText:
    text: str
    x: int
    y: int
    bg_bgr: tuple[int, int, int]
    text_bgr: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class DebugLayout:
    root_dir: Path
    frames_dir: Path
    panels_dir: Path
    csv_path: Path
