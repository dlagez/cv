from __future__ import annotations

from pathlib import Path

HEAD_KEYPOINTS = (0, 1, 2, 3, 4)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

MANAGER_HELMET_COLORS = frozenset({"red", "white"})
MANAGER_VEST_COLOR = "yellow_green_fluorescent"

DISPLAY_LABELS = {
    "manager": "管理人员",
    "worker": "工作人员",
}

FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
)
