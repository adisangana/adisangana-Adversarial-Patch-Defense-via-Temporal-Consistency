# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project studying adversarial patch attacks on YOLOv8 object detection across temporal video sequences. The goal is to understand how adversarial patches affect detection consistency frame-by-frame.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.11. Note `numpy<2` is pinned to avoid breaking API changes.

## Pipeline

The project runs as a sequential pipeline — each step must complete before the next:

```
RAF Files → PNG Frames → Adversarial Patch → YOLO Detection → Visualization
```

**Step 1 — Convert RAW images:**
```bash
python convert_raw.py [--data-dir data]
```

**Step 2 — Generate and apply adversarial patch:**
```bash
python apply_patch.py [--frames START END] [--patch-size 200] [--patch-x 100] [--patch-y 100] [--steps 50] [--lr 0.01]
# To reuse an existing patch:
python apply_patch.py --patch-file patch.png
```

**Step 3 — Run YOLO detection on clean and patched frames:**
```bash
python detect.py [--model yolov8n.pt] [--conf 0.25] [--data-dir data]
```

**Step 4 — Visualize temporal results:**
```bash
python visualize.py [--class-name person] [--out results.png] [--data-dir data]
```

## Architecture

| File | Role |
|------|------|
| `convert_raw.py` | Fujifilm RAF → PNG via `rawpy`, skips already-converted files |
| `apply_patch.py` | Gradient ascent (Adam, 50 steps) to maximize YOLOv8 objectness score; applies patch to selected frames |
| `detect.py` | YOLOv8 inference on both `frames/` and `patched/`; outputs per-frame JSON + annotated PNG |
| `visualize.py` | 3-panel matplotlib figure: detection counts, max confidence, and frame thumbnails |

### Key Design Details

- **Adversarial objective:** Maximizes `raw[..., 4]` (objectness logit) — a universal attack, not class-specific.
- **Patch placement:** Fixed coordinates (default `x=100, y=100`), clamped to image bounds.
- **Patched frame tracking:** `data/patch_meta.json` records position, size, and which frames were patched — used by `visualize.py` to shade the attack region on plots.
- **Detection JSON schema:** `{"frame": "DSCF0117.png", "detections": [{"class_id": int, "class_name": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]}`

### Data Layout

```
data/
  *.RAF                   # Raw input images
  frames/                 # Converted PNGs (clean)
  patched/                # Patched frame PNGs
  patch.png               # Learned adversarial patch
  patch_meta.json         # Patch position/size + patched frame list
  detections/
    clean/                # JSON + annotated PNGs for clean frames
    patched/              # JSON + annotated PNGs for patched frames
```
