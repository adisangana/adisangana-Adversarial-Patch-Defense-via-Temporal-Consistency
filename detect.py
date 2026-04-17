#!/usr/bin/env python3
"""Run YOLOv5 inference on dataset images and write detection JSONs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


STOP_SIGN_CLASS = "stop sign"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLOv5 on dataset images and write JSONs.")
    p.add_argument("--dataset", required=True, type=Path,
                   help="Dataset directory containing *.png files")
    p.add_argument("--model", default="yolov5s",
                   help="YOLOv5 model variant (default: yolov5s)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing JSON files")
    return p.parse_args()


def _parse_filename(stem: str) -> dict | None:
    """Extract brightness/degree/direction/distance from filename stem."""
    m = re.match(
        r"brightness(\d+)_degree(\d+)_distance(\d+)",
        stem,
    )
    if not m:
        return None
    brightness, degree, distance = int(m.group(1)), int(m.group(2)), int(m.group(3))
    # direction convention: degree <= 17 → forward, else backward
    direction = "forward" if degree <= 17 else "backward"
    return {"brightness": brightness, "degree": degree,
            "distance": distance, "direction": direction}


def run_detection(dataset_dir: Path, model_name: str, conf_thresh: float, overwrite: bool) -> None:
    png_files = sorted(dataset_dir.glob("*.png"))
    if not png_files:
        print(f"ERROR: no PNG files found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading YOLOv5 ({model_name}) ...")
    model = torch.hub.load("ultralytics/yolov5", model_name, verbose=False)
    model.conf = conf_thresh

    to_process = []
    for png in png_files:
        meta = _parse_filename(png.stem)
        if meta is None:
            print(f"WARNING: unrecognised filename {png.name}, skipping", file=sys.stderr)
            continue
        json_path = png.with_suffix(".json")
        if json_path.exists() and not overwrite:
            continue
        to_process.append((png, json_path, meta))

    if not to_process:
        print("All JSONs already exist. Use --overwrite to re-run.")
        return

    print(f"Running inference on {len(to_process)} image(s) ...")
    for i, (png, json_path, meta) in enumerate(to_process, 1):
        results = model(str(png))
        df = results.pandas().xyxy[0]

        stop_rows = df[df["name"] == STOP_SIGN_CLASS]
        stop_sign_conf = float(stop_rows["confidence"].max()) if not stop_rows.empty else 0.0

        if df.empty:
            top1_label = None
            top1_conf = 0.0
        else:
            top_row = df.loc[df["confidence"].idxmax()]
            top1_label = str(top_row["name"])
            top1_conf = float(top_row["confidence"])

        out = {
            "brightness": meta["brightness"],
            "degree": meta["degree"],
            "direction": meta["direction"],
            "distance": meta["distance"],
            "yolov5": {
                "stop_sign_conf": stop_sign_conf,
                "top1_label": top1_label,
                "top1_conf": top1_conf,
            },
        }

        with json_path.open("w") as fh:
            json.dump(out, fh, indent=2)

        if i % 50 == 0 or i == len(to_process):
            print(f"  {i}/{len(to_process)} done")

    print("Detection complete.")


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        print(f"ERROR: dataset directory not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)
    run_detection(args.dataset, args.model, args.conf, args.overwrite)


if __name__ == "__main__":
    main()
