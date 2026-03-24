from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class RoiMeta:
    roi_source: str
    head_point_count: int
    shoulder_point_count: int
    shoulder_span: float


@dataclass(slots=True)
class ColorDebugData:
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
    def empty(cls) -> "ColorDebugData":
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
