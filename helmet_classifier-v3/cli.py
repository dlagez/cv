from __future__ import annotations

from typing import Sequence

from .config import parse_args
from .pipeline import process_video


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    process_video(config)
    return 0
