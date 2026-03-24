from __future__ import annotations

from datetime import datetime
from pathlib import Path


def default_output_path(source: Path) -> Path:
    return Path("outputs") / "helmet-classify" / f"{source.stem}.mp4"


def default_debug_dir(source: Path) -> Path:
    return Path("outputs") / "helmet-debug" / source.stem


def allocate_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base_dir / timestamp
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{timestamp}-{suffix:02d}"
        suffix += 1
    return candidate
