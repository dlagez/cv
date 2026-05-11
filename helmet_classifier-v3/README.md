# helmet_classifier-v3 使用说明

基于 YOLO 姿态估计 + HSV 颜色规则的安全帽与反光衣分类管线。

- **管理人员**：白色/红色安全帽 + 黄绿色荧光反光衣
- **工作人员**：其他颜色安全帽 + 红色/橙色反光衣（或不符合管理人员条件的所有情况）

## 环境要求

Python 3.12，依赖 ultralytics、opencv-python、numpy。使用项目根目录的 `.venv`：

```bash
.venv/bin/pip install -r requirements.txt
```

## 命令行调用

```bash
# 最基本用法
python helmet_classifier-v3 --source /path/to/video.mp4

# 使用更强的姿态模型（暗光或小目标场景推荐）
python helmet_classifier-v3 --source /path/to/video.mp4 --model yolo11m-pose.pt

# 指定输出路径
python helmet_classifier-v3 --source /path/to/video.mp4 --output /path/to/output.mp4

# 只处理前 N 帧（快速测试）
python helmet_classifier-v3 --source /path/to/video.mp4 --max-frames 100

# 绘制调试信息（头盔/反光衣框 + 颜色详情文字）
python helmet_classifier-v3 --source /path/to/video.mp4 --draw-helmet-box --draw-torso-box --debug-text

# 保存调试产物（CSV、面板截图）供人工排查
python helmet_classifier-v3 --source /path/to/video.mp4 --save-debug-artifacts

# 使用旧版仅头盔判断逻辑（关闭头盔+反光衣联合判断）
python helmet_classifier-v3 --source /path/to/video.mp4 --legacy-helmet-only

# 指定设备
python helmet_classifier-v3 --source /path/to/video.mp4 --device cuda:0
# python helmet_classifier-v3 --source /path/to/video.mp4 --device cpu
```

## Python 调用

> 注意：包目录名为 `helmet_classifier-v3`（含连字符），无法直接用 `import` 导入。如需在 Python 代码中调用，有两种方式：

**方式一：重命名目录（推荐）**

```bash
mv helmet_classifier-v3 helmet_classifier_v3
```

然后正常导入：

```python
from helmet_classifier_v3.cli import main
from helmet_classifier_v3.config import AppConfig
from helmet_classifier_v3.pipeline import process_video

# 通过参数列表调用
main(["--source", "/path/to/video.mp4", "--max-frames", "100"])

# 或直接用配置类
config = AppConfig(source="/path/to/video.mp4", max_frames=100)
output_path = process_video(config)
```

**方式二：通过子进程调用（不改名）**

```python
import subprocess

subprocess.run([
    ".venv/bin/python", "helmet_classifier-v3",
    "--source", "/path/to/video.mp4",
    "--max-frames", "100",
])
```

## 常用参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | 必填 | 输入视频路径 |
| `--model` | `yolo11n-pose.pt` | YOLO 姿态模型路径 |
| `--output` | 自动生成 | 输出视频路径（默认 `output/videos/<name>_<时间戳>.mp4`） |
| `--imgsz` | `960` | 推理图像尺寸 |
| `--person-conf` | `0.35` | 人员检测置信度阈值 |
| `--keypoint-conf` | `0.35` | 头部区域关键点置信度阈值 |
| `--torso-keypoint-conf` | `0.35` | 躯干区域关键点置信度阈值 |
| `--max-frames` | `0`（全部） | 仅处理前 N 帧 |
| `--device` | 自动 | 推理设备（`cuda:0` / `cpu`） |
| `--codec` | `mp4v` | 输出编码（失败时回退 XVID） |
| `--font-path` | 自动查找 | 中文字体路径（Linux 需指定） |
| `--font-size` | `22` | 标签字体大小 |
| `--draw-helmet-box` | 否 | 绘制头部 ROI 框 |
| `--draw-torso-box` | 否 | 绘制躯干 ROI 框 |
| `--debug-text` | 否 | 标签追加颜色详情文字 |
| `--save-debug-artifacts` | 否 | 保存调试产物 |
| `--legacy-helmet-only` | 否 | 仅用头盔颜色判断，关闭联合判断 |
| `--helmet-white-ratio-threshold` | `0.18` | 白色安全帽判定比例 |
| `--helmet-red-ratio-threshold` | `0.15` | 红色安全帽判定比例 |
| `--vest-yellow-green-ratio-threshold` | `0.08` | 黄绿荧光反光衣判定比例 |
| `--vest-red-ratio-threshold` | `0.20` | 红色反光衣判定比例 |
| `--vest-orange-ratio-threshold` | `0.20` | 橙色反光衣判定比例 |
