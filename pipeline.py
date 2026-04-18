#!/usr/bin/env python3
"""Temporal consistency adversarial patch detection pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tc_defense import (
    load_dataset,
    DefenseConfig,
    run_defense_all,
    evaluate_all,
    print_report,
)
from tc_defense.visualize import plot_sequence, plot_summary, plot_brightness_gate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Temporal consistency defense against adversarial patches."
    )
    p.add_argument("--dataset", required=True, type=Path,
                   help="Dataset directory containing *.json and *.png files")
    p.add_argument("--window-size", type=int, default=5,
                   help="Number of previous frames used for temporal consensus (default: 5)")
    p.add_argument("--brightness-thresh", type=float, default=10.0,
                   help="Mean luminance delta threshold for scene change (default: 10.0)")
    p.add_argument("--ground-truth", type=str, default="stop sign",
                   help='Correct label for all frames (default: "stop sign")')
    p.add_argument("--out-dir", type=Path, default=Path("results"),
                   help="Output directory for plots and report (default: results/)")
    p.add_argument("--degrees", type=int, nargs="+",
                   help="Only process these degree values (default: all)")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip all visualization")
    p.add_argument("--show-plots", action="store_true",
                   help="Display plots interactively")
    p.add_argument("--report-json", type=Path,
                   help="Write metrics as JSON to this path")

    yolo = p.add_argument_group("YOLOv5 detection (optional)")
    yolo.add_argument("--run-yolo", action="store_true",
                      help="Run YOLOv5 on images before defense, overwriting existing JSONs")
    yolo.add_argument("--yolo-model", default="yolov5s",
                      help="YOLOv5 model variant (default: yolov5s)")
    yolo.add_argument("--yolo-conf", type=float, default=0.25,
                      help="YOLOv5 confidence threshold (default: 0.25)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset directory not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    if args.run_yolo:
        from detect import run_detection
        run_detection(args.dataset, args.yolo_model, args.yolo_conf, overwrite=True)

    print(f"Loading dataset from {args.dataset} ...")
    try:
        sequences = load_dataset(args.dataset)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.degrees:
        sequences = [s for s in sequences if s.degree in args.degrees]
        if not sequences:
            print(f"ERROR: no sequences found for degrees {args.degrees}", file=sys.stderr)
            sys.exit(1)

    print(f"Loaded {len(sequences)} sequence(s): degrees {[s.degree for s in sequences]}")
    print(f"Total frames with images: {sum(len(s.frames) for s in sequences)}")

    config = DefenseConfig(
        window_size=args.window_size,
        brightness_threshold=args.brightness_thresh,
        ground_truth_label=args.ground_truth,
    )
    print(f"\nRunning defense (history={config.window_size}, "
          f"brightness_thresh={config.brightness_threshold}) ...")

    all_results = run_defense_all(sequences, config)
    metrics = evaluate_all(all_results, ground_truth_label=args.ground_truth)

    print()
    print_report(metrics)

    if args.report_json:
        _write_json_report(metrics, args.report_json)
        print(f"JSON report written to {args.report_json}")

    if not args.no_plots:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        save = not args.show_plots

        for seq in sequences:
            deg = seq.degree
            seq_metrics = next(m for m in metrics.per_sequence if m.degree == deg)
            results = all_results[deg]

            plot_sequence(
                deg, results, seq_metrics,
                ground_truth_label=args.ground_truth,
                out_path=args.out_dir / f"sequence_degree{deg:02d}.png" if save else None,
                show=args.show_plots,
            )
            plot_brightness_gate(
                deg, results, config.brightness_threshold,
                out_path=args.out_dir / f"brightness_degree{deg:02d}.png" if save else None,
                show=args.show_plots,
            )

        plot_summary(
            metrics,
            ground_truth_label=args.ground_truth,
            out_path=args.out_dir / "summary.png" if save else None,
            show=args.show_plots,
        )

        if save:
            print(f"\nPlots saved to {args.out_dir}/")


def _write_json_report(metrics, path: Path) -> None:
    import math
    import dataclasses

    def _clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    def _serialize(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        if isinstance(obj, float):
            return _clean(obj)
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(_serialize(metrics), fh, indent=2)


if __name__ == "__main__":
    main()
