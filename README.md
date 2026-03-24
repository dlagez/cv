# YOLO Environment

This workspace contains a Python 3.12 virtual environment with Ultralytics installed.

## Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

## Project Layout

```text
helmet_classifier/
  __main__.py        # python -m helmet_classifier
  cli.py             # CLI entry
  config.py          # parameters and argument parsing
  analysis.py        # pose ROI estimation and HSV color rules
  render.py          # Chinese overlays and box drawing
  debug_output.py    # debug CSV, frames, and panel generation
  pipeline.py        # video processing pipeline
  paths.py           # output/debug path helpers
  schemas.py         # shared data structures
helmet_color_classify.py  # compatibility entry
run-helmet-classify.ps1   # PowerShell wrapper
run-person-detect.ps1     # person detection wrapper
doc/                      # notes and parameter docs
data/                     # input videos
outputs/                  # generated videos and debug artifacts
```

## Detect persons in a video

```powershell
.\run-person-detect.ps1 -Source ".\your_video.mp4"
```

## Track persons across frames

```powershell
.\run-person-detect.ps1 -Mode track -Source ".\your_video.mp4"
```

## Classify helmet colors with pose + HSV rules

This pipeline uses a YOLO pose model to estimate each person's head area and classifies the helmet color with HSV thresholds:

- red helmet ratio above threshold -> `工作人员`
- all other cases -> `管理人员`

```powershell
.\run-helmet-classify.ps1 -Source ".\your_video.mp4"
```

Use a stronger pose model if the scene is dark or the helmets are small:

```powershell
.\run-helmet-classify.ps1 -Source ".\your_video.mp4" -Model "yolo11m-pose.pt"
```

Useful tuning flags:

```powershell
.\run-helmet-classify.ps1 -Source ".\your_video.mp4" -DrawHelmetBox -DebugText
.\run-helmet-classify.ps1 -Source ".\your_video.mp4" -WhiteRatioThreshold 0.14 -RedRatioThreshold 0.07
```

## Use a custom model file

```powershell
.\run-person-detect.ps1 -Source ".\your_video.mp4" -Model ".\yolo26n.pt"
```

## Raw CLI examples

```powershell
.\.venv\Scripts\yolo.exe predict model=yolo11n.pt source="your_video.mp4" classes=0 conf=0.30 save=True
.\.venv\Scripts\yolo.exe track model=yolo11n.pt source="your_video.mp4" classes=0 conf=0.30 save=True
```

## Reinstall from scratch

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```
