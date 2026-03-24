from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import AppConfig
from .constants import DISPLAY_LABELS, FONT_CANDIDATES
from .schemas import Box, OverlayText


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


def make_overlay_label(label: str, white_ratio: float, red_ratio: float, debug_text: bool) -> str:
    display_label = DISPLAY_LABELS.get(label, label)
    if debug_text:
        return f"{display_label} W:{white_ratio:.2f} R:{red_ratio:.2f}"
    return display_label


def draw_detection_boxes(
    frame: np.ndarray,
    person_box: np.ndarray,
    helmet_box: Box | None,
    label: str,
    config: AppConfig,
) -> tuple[int, int, tuple[int, int, int], tuple[int, int, int]]:
    palette = {
        "manager": (255, 255, 255),
        "worker": (0, 0, 255),
    }
    draw_color = palette.get(label, (0, 255, 255))
    x1, y1, x2, y2 = [int(round(v)) for v in person_box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)

    text_y = max(22, y1 - 10)
    if config.draw_helmet_box and helmet_box is not None:
        hx1, hy1, hx2, hy2 = helmet_box
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 1)

    text_color = (0, 0, 0) if label == "manager" else (255, 255, 255)
    return x1, text_y, draw_color, text_color


def build_overlay(
    label: str,
    white_ratio: float,
    red_ratio: float,
    debug_text: bool,
    text_x: int,
    text_y: int,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> OverlayText:
    return OverlayText(
        text=make_overlay_label(label, white_ratio, red_ratio, debug_text),
        x=text_x,
        y=text_y,
        bg_bgr=bg_color,
        text_bgr=text_color,
    )


def render_text_overlays(
    frame: np.ndarray,
    overlays: list[OverlayText],
    font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    if not overlays:
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    for overlay in overlays:
        left = max(0, overlay.x)
        top = max(0, overlay.y - font.size - 10)
        text_bbox = draw.textbbox((left + 8, top + 5), overlay.text, font=font)
        rect_right = min(frame.shape[1] - 1, text_bbox[2] + 8)
        rect_bottom = min(frame.shape[0] - 1, text_bbox[3] + 5)
        draw.rectangle(
            [(left, top), (rect_right, rect_bottom)],
            fill=(overlay.bg_bgr[2], overlay.bg_bgr[1], overlay.bg_bgr[0]),
        )
        draw.text(
            (left + 8, top + 5),
            overlay.text,
            font=font,
            fill=(overlay.text_bgr[2], overlay.text_bgr[1], overlay.text_bgr[0]),
        )

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
