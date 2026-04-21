# CSE-5619-Temporal-Consistency

This repository contains project utilities for temporal consistency experiments and a YOLOv5-based evaluation script.

## Setup

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <your-repo-url>
   cd CSE-5619-Temporal-Consistency
   ```

2. **Set up a Python virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## First-Run Download Behavior

On first use, if the default YOLOv5 assets are not already available, the pipeline will:
1. Load the official YOLOv5 code through `torch.hub`.
2. Download the default `yolov5s` weights if needed.
3. Cache YOLOv5 files under `.cache/torch`.

If `yolov5s.pt` already exists in the repository root, the script will reuse it.

## Temporal Consistency Pipeline

The end-to-end defense entry point is [`pipeline.py`](./pipeline.py).

It loads a dataset directory of matching `*.png` and `*.json` frames, applies the temporal consistency defense, prints per-degree and overall metrics, and optionally saves plots plus a JSON report.

### Expected Dataset Layout

`pipeline.py` expects image and detection files in the same folder, for example:

```text
data/
  brightness130_degree00_distance00.png
  brightness130_degree00_distance00.json
  brightness130_degree00_distance01.png
  brightness130_degree00_distance01.json
```

The JSON files should follow the schema written by [`detect.py`](./detect.py). Frames missing their sibling PNG image are skipped.

### Run the Pipeline

From the repository root, run:

```bash
python3 pipeline.py --dataset data
```

This loads all degree sequences in the dataset, evaluates each sequence from the maximum distance toward distance `0`, runs the temporal defense using the previous 5 evaluated frames by default, prints a summary table, and saves plots under `results/` by default.

### Understanding the Plots

Typical generated files:

- `results/sequence_degree00.png` (Sequence analysis plot showing attacks and defended labels)
- `results/brightness_degree00.png` (Brightness analysis plot showing scene stability gates)
- `results/summary.png` (Overall performance summary across all sequences)

**Plot Keys:**
*   **`sequence_degree*.png`:**
    *   **Raw Labels:** The original classification from YOLOv5.
    *   **Defended Labels:** The final classification after applying the temporal consistency defense.
    *   **Flagged as Attack:** Indicates frames where the defense identified an inconsistency between the raw label and the temporal consensus.
    *   **Consensus:** The temporal majority vote over the preceding frames (window size).
*   **`brightness_degree*.png`:**
    *   **Luminance (blue line):** Mean image brightness for the current frame.
    *   **Window Average (orange line):** The average mean brightness of the preceding frames in the temporal window.
    *   **Delta Gate (red shaded area):** Frames where the absolute difference between current luminance and window average exceeds the `brightness-thresh`. These frames are treated as scene changes (e.g., abrupt lighting shifts) and temporal consistency checks are bypassed.

### Pipeline Options

Show all CLI options:

```bash
python3 pipeline.py --help
```

Common flags:

- `--dataset`: dataset directory containing paired `*.png` and `*.json` files. Required.
- `--window-size`: number of previous frames used for temporal consensus. Increasing this makes the defense more robust to long-duration attacks but slower to adapt to genuine scene changes. Default: 5.
- `--brightness-thresh`: luminance-delta threshold used to suppress scene-change false positives. Lower values make the defense more sensitive to lighting changes (bypassing the temporal check more often). Default: 10.0.
- `--ground-truth`: expected correct class label. Used for computing metrics. Default `stop sign`.
- `--out-dir`: output directory for plots and JSON report. Default `results/`.
- `--degrees`: restrict evaluation to specific degree values (e.g., `--degrees 0 10`). Useful for debugging specific angles.
- `--no-plots`: skip plot generation entirely. Faster execution if only console metrics are needed.
- `--show-plots`: display plots interactively instead of only saving them.
- `--report-json`: write the computed metrics to a JSON file at the specified path.

### Regenerate YOLO JSONs Before Defense

If you want the pipeline to rerun YOLOv5 and overwrite existing detection JSON files before applying the defense:

```bash
python3 pipeline.py \
  --dataset data \
  --run-yolo \
  --yolo-model yolov5s \
  --yolo-conf 0.25 \
  --report-json results/report.json
```

Relevant YOLO flags:

- `--run-yolo`: run YOLOv5 on the dataset before defense. This ensures detections are up-to-date with your current YOLO configuration.
- `--yolo-model`: YOLOv5 model variant to load (e.g., `yolov5s`, `yolov5m`).
- `--yolo-conf`: confidence threshold passed to YOLOv5. Lowering this increases recall but might introduce false positives.

## Core Modules

- [`pipeline.py`](./pipeline.py): The main entry point for running the temporal consistency defense and generating evaluation metrics.
- [`detect.py`](./detect.py): A utility to independently run YOLOv5 inference on a directory of images and generate the `.json` files required by the pipeline.
- [`tc_defense/`](./tc_defense/): The package containing the core logic for the temporal defense, dataset loading, and visualization.

## Notes

- The scripts suppress the noisy `urllib3` LibreSSL warning and the YOLOv5 `amp.autocast` deprecation warning during inference.
- Large model weights and virtual environments usually should not be committed to Git. In most cases, commit the source files and let users install dependencies locally.