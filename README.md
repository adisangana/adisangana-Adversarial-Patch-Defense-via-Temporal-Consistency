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

## Other Project Scripts

- [`convert_raw.py`](./convert_raw.py)
- [`detect.py`](./detect.py)
- [`visualize.py`](./visualize.py)

## Notes

- If `./.venv/bin/python` exists, `evaluate_stop_signs.py` automatically re-runs itself with that interpreter.
- The script suppresses the noisy `urllib3` LibreSSL warning and the YOLOv5 `amp.autocast` deprecation warning during inference.
- Large model weights and virtual environments usually should not be committed to Git. In most cases, commit the source files and let users install dependencies locally.
