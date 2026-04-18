# CSE-5619-Temporal-Consistency

This repository contains project utilities for temporal consistency experiments and a YOLOv5-based stop-sign evaluation script.

## Stop Sign Evaluation with YOLOv5

The main evaluation script is [`evaluate_stop_signs.py`](./evaluate_stop_signs.py).

It runs YOLOv5 on a folder of images and exports a CSV report with:

- `brightness`
- `degree`
- `distance`
- `top_classified_object`
- `top_classified_object_accuracy`
- `stopsign_accuracy`

The script extracts `brightness`, `degree`, and `distance` from filenames such as `brightness130_degree00_distance07.png`.

If YOLOv5 does not detect anything in an image, the object field is blank and both confidence values are `0.000000`.

## Setup

```bash
git clone <your-repo-url>
cd CSE-5619-Temporal-Consistency

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x evaluate_stop_signs.py
```

## Run the Evaluator

From the repository root, run:

```bash
./evaluate_stop_signs.py --images-dir 130 --device cpu
```

If your image set is stored somewhere else on disk, pass that absolute or relative path through `--images-dir`.

Typical console output:

```text
YOLOv5 🚀 ...
Fusing layers...
YOLOv5s summary: ...
Adding AutoShape...
Processed 30 image(s). Stop sign detected in 11 image(s).
```

If `--output-csv` is omitted, the script writes the report inside the image folder using this pattern:

```text
<images-dir>/<folder-name>_yolov5_stop_sign_report.csv
```

## CSV Example

```csv
brightness,degree,distance,top_classified_object,top_classified_object_accuracy,stopsign_accuracy
130,00,00,person,0.675126,0.000000
130,00,01,person,0.620032,0.000000
130,00,12,stop sign,0.884211,0.884211
```

## Common Options

Show all CLI options:

```bash
./evaluate_stop_signs.py --help
```

Common flags:

- `--images-dir`: folder containing input images
- `--output-csv`: optional path for the CSV report; if omitted, the report is written inside `--images-dir`
- `--weights`: YOLOv5 model name such as `yolov5s`, or a path to a custom `.pt` model
- `--device`: `cpu`, `mps`, `0`, `0,1`, etc.
- `--batch-size`: number of images per inference batch
- `--imgsz`: inference image size
- `--conf-thres`: confidence threshold
- `--iou-thres`: IoU threshold for NMS
- `--stop-label`: class label to treat as the stop-sign class
- `--recursive`: scan subdirectories under the image directory

By default, the script only scans image files directly inside `--images-dir`. Use `--recursive` if you want to include subdirectories.

## Using a Custom Model

If you have your own trained YOLOv5 weights:

```bash
./evaluate_stop_signs.py \
  --images-dir /path/to/images \
  --output-csv /path/to/results.csv \
  --device cpu \
  --weights /path/to/best.pt \
  --stop-label "stop sign"
```

If your model uses a different class name such as `stopsign`, pass that value through `--stop-label`.

## Supported Image Formats

The script scans these file extensions:

- `.bmp`
- `.jpeg`
- `.jpg`
- `.png`
- `.tif`
- `.tiff`
- `.webp`

## First-Run Download Behavior

On first use, if the default YOLOv5 assets are not already available, the script will:

1. load the official YOLOv5 code through `torch.hub`
2. download the default `yolov5s` weights if needed
3. cache YOLOv5 files under `.cache/torch`

If `yolov5s.pt` already exists in the repository root, the script will reuse it.

## Temporal Consistency Pipeline

The end-to-end defense entry point is [`pipeline.py`](./pipeline.py).

It loads a dataset directory of matching `*.png` and `*.json` frames, applies the temporal consistency defense, prints per-degree and overall metrics, and optionally saves plots plus a JSON report.

### Expected Dataset Layout

`pipeline.py` expects image and detection files in the same folder, for example:

```text
130/
  brightness130_degree00_distance00.png
  brightness130_degree00_distance00.json
  brightness130_degree00_distance01.png
  brightness130_degree00_distance01.json
```

The JSON files should follow the schema written by [`detect.py`](./detect.py). Frames missing their sibling PNG image are skipped.

### Run the Pipeline

From the repository root, run:

```bash
python3 pipeline.py --dataset 130
```

This loads all degree sequences in the dataset, evaluates each sequence from the maximum distance toward distance `0`, runs the temporal defense using the previous 5 evaluated frames by default, prints a summary table, and saves plots under `results/` by default.

Typical generated files:

- `results/sequence_degree00.png`
- `results/brightness_degree00.png`
- `results/summary.png`

### Pipeline Options

Show all CLI options:

```bash
python3 pipeline.py --help
```

Common flags:

- `--dataset`: dataset directory containing paired `*.png` and `*.json` files
- `--window-size`: number of previous frames used for temporal consensus
- `--brightness-thresh`: luminance-delta threshold used to suppress scene-change false positives
- `--ground-truth`: expected correct class label, default `stop sign`
- `--out-dir`: output directory for plots, default `results/`
- `--degrees`: restrict evaluation to specific degree values
- `--no-plots`: skip plot generation entirely
- `--show-plots`: display plots interactively instead of only saving them
- `--report-json`: write the computed metrics to a JSON file

### Regenerate YOLO JSONs Before Defense

If you want the pipeline to rerun YOLOv5 and overwrite existing detection JSON files before applying the defense:

```bash
python3 pipeline.py \
  --dataset 130 \
  --run-yolo \
  --yolo-model yolov5s \
  --yolo-conf 0.25 \
  --report-json results/report.json
```

Relevant YOLO flags:

- `--run-yolo`: run YOLOv5 on the dataset before defense
- `--yolo-model`: YOLOv5 model variant to load
- `--yolo-conf`: confidence threshold passed to YOLOv5

## Other Project Scripts

- [`convert_raw.py`](./convert_raw.py)
- [`detect.py`](./detect.py)
- [`pipeline.py`](./pipeline.py)
- [`visualize.py`](./visualize.py)

## Notes

- If `./.venv/bin/python` exists, `evaluate_stop_signs.py` automatically re-runs itself with that interpreter.
- The script suppresses the noisy `urllib3` LibreSSL warning and the YOLOv5 `amp.autocast` deprecation warning during inference.
- Large model weights and virtual environments usually should not be committed to Git. In most cases, commit the source files and let users install dependencies locally.
