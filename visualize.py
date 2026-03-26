"""
Visualize YOLO detection results over the temporal sequence.

Plots:
  1. Detection count per frame (clean vs patched overlay)
  2. Max confidence per frame for the most-detected class
  3. Side-by-side annotated thumbnails for a patched frame vs its clean version

Usage:
    python visualize.py
    python visualize.py --class-name person --out results.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_detections(det_dir: Path) -> dict[str, list[dict]]:
    results = {}
    for jp in sorted(det_dir.glob("*.json")):
        with open(jp) as f:
            data = json.load(f)
        results[data["frame"]] = data["detections"]
    return results


def most_common_class(clean_results: dict) -> str:
    counts: dict[str, int] = {}
    for dets in clean_results.values():
        for d in dets:
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    if not counts:
        return "person"
    return max(counts, key=counts.__getitem__)


def main(data_dir: Path, class_name: str | None, out_path: Path | None) -> None:
    det_dir = data_dir / "detections"
    clean_dir = det_dir / "clean"
    patched_dir = det_dir / "patched"

    if not clean_dir.exists():
        raise FileNotFoundError(f"No clean detections at {clean_dir}. Run detect.py first.")

    clean = load_detections(clean_dir)
    patched = load_detections(patched_dir) if patched_dir.exists() else {}

    # Load patch metadata if available
    meta_path = data_dir / "patch_meta.json"
    patched_frame_names: set[str] = set()
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        patched_frame_names = set(meta["patched_frames"])

    frame_names = sorted(clean.keys())
    n = len(frame_names)
    indices = list(range(n))

    # Pick target class
    target_class = class_name or most_common_class(clean)
    print(f"Visualising class: '{target_class}'")

    # --- Build series ---
    def count_class(dets: list[dict], cls: str) -> int:
        return sum(1 for d in dets if d["class_name"] == cls)

    def max_conf_class(dets: list[dict], cls: str) -> float:
        confs = [d["confidence"] for d in dets if d["class_name"] == cls]
        return max(confs) if confs else 0.0

    clean_counts = [count_class(clean[fn], target_class) for fn in frame_names]
    clean_confs = [max_conf_class(clean[fn], target_class) for fn in frame_names]

    # For patched frames use patched results, else fall back to clean
    merged_counts = []
    merged_confs = []
    for fn in frame_names:
        if fn in patched_frame_names and fn in patched:
            merged_counts.append(count_class(patched[fn], target_class))
            merged_confs.append(max_conf_class(patched[fn], target_class))
        else:
            merged_counts.append(clean_counts[frame_names.index(fn)])
            merged_confs.append(clean_confs[frame_names.index(fn)])

    patch_mask = [fn in patched_frame_names for fn in frame_names]
    short_names = [fn.replace(".png", "") for fn in frame_names]

    # --- Find a patched frame for thumbnail ---
    example_patched_frame = next((fn for fn in frame_names if fn in patched_frame_names), None)

    nrows = 3 if example_patched_frame else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 5 * nrows))
    fig.suptitle("Temporal Detection Analysis — Adversarial Patch", fontsize=14, fontweight="bold")

    # Shade patched regions
    def shade_patched(ax: plt.Axes) -> None:
        in_block = False
        start = 0
        for i, is_p in enumerate(patch_mask):
            if is_p and not in_block:
                start = i
                in_block = True
            elif not is_p and in_block:
                ax.axvspan(start - 0.5, i - 0.5, alpha=0.15, color="red", label="_patch region")
                in_block = False
        if in_block:
            ax.axvspan(start - 0.5, len(patch_mask) - 0.5, alpha=0.15, color="red")

    # Plot 1: detection counts
    ax = axes[0]
    ax.plot(indices, clean_counts, "o-", color="steelblue", linewidth=2, markersize=6, label=f"clean — {target_class} count")
    if patched:
        ax.plot(indices, merged_counts, "s--", color="tomato", linewidth=2, markersize=6, label=f"patched — {target_class} count")
    shade_patched(ax)
    ax.set_xticks(indices)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Detection count")
    ax.set_title(f"Detection Count per Frame  (class: {target_class})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: max confidence
    ax = axes[1]
    ax.plot(indices, clean_confs, "o-", color="steelblue", linewidth=2, markersize=6, label=f"clean — max conf")
    if patched:
        ax.plot(indices, merged_confs, "s--", color="tomato", linewidth=2, markersize=6, label=f"patched — max conf")
    shade_patched(ax)
    ax.set_xticks(indices)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Max confidence")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Max Confidence per Frame  (class: {target_class})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add red shading legend entry manually
    red_patch = mpatches.Patch(color="red", alpha=0.3, label="patched frames")
    for ax in axes[:2]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [red_patch])

    # Plot 3: thumbnails (clean vs patched)
    if example_patched_frame and nrows == 3:
        ax = axes[2]
        ax.axis("off")

        clean_img_path = clean_dir / example_patched_frame
        patched_img_path = patched_dir / example_patched_frame

        if clean_img_path.exists() and patched_img_path.exists():
            clean_img = Image.open(clean_img_path).convert("RGB")
            patched_img = Image.open(patched_img_path).convert("RGB")

            # Downsample for display
            thumb_w = 600
            ratio = thumb_w / clean_img.width
            thumb_h = int(clean_img.height * ratio)
            clean_thumb = clean_img.resize((thumb_w, thumb_h))
            patched_thumb = patched_img.resize((thumb_w, thumb_h))

            combined = Image.new("RGB", (thumb_w * 2 + 20, thumb_h), (240, 240, 240))
            combined.paste(clean_thumb, (0, 0))
            combined.paste(patched_thumb, (thumb_w + 20, 0))

            ax.imshow(np.array(combined))
            ax.set_title(f"Clean (left) vs Patched (right) — {example_patched_frame}", fontsize=10)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize detection results over frames")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--class-name", default=None, help="Object class to track (default: most common detected class)")
    parser.add_argument("--out", type=Path, default=None, help="Save plot to file instead of displaying")
    args = parser.parse_args()
    main(args.data_dir, args.class_name, args.out)
