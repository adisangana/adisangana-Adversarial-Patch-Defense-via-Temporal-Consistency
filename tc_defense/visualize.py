from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .defense import FrameResult
from .evaluate import OverallMetrics, SequenceMetrics


_GREEN = "#2ecc71"
_RED = "#e74c3c"
_BLUE = "#3498db"
_ORANGE = "#e67e22"
_GREY = "#95a5a6"


def _label_color(label: Optional[str], ground_truth: str) -> str:
    if label == ground_truth:
        return _GREEN
    return _RED if label is not None else _GREY


def plot_sequence(
    degree: int,
    results: list[FrameResult],
    metrics: SequenceMetrics,
    ground_truth_label: str = "stop sign",
    out_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    distances = [r.frame.distance for r in results]
    confs = [r.frame.stop_sign_conf for r in results]
    brightness_deltas = [r.brightness_delta for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        f"Degree {degree} — Raw {metrics.raw_accuracy:.1%} → Defended {metrics.defended_accuracy:.1%}",
        fontsize=13,
    )

    # Panel 1: raw vs defended labels
    ax = axes[0]
    raw_colors = [_label_color(r.raw_label, ground_truth_label) for r in results]
    def_colors = [_label_color(r.defended_label, ground_truth_label) for r in results]

    ax.scatter(distances, [1.1] * len(results), c=raw_colors, s=80, marker="o", label="Raw")
    ax.scatter(distances, [0.9] * len(results), c=def_colors, s=80, marker="s", label="Defended")

    for r in results:
        if r.flagged_as_attack:
            ax.axvline(r.frame.distance, color=_ORANGE, alpha=0.25, linewidth=8)

    ax.set_yticks([0.9, 1.1])
    ax.set_yticklabels(["Defended", "Raw"])
    ax.set_ylim(0.7, 1.3)
    ax.set_ylabel("Label")

    correct_patch = mpatches.Patch(color=_GREEN, label="Correct")
    wrong_patch = mpatches.Patch(color=_RED, label="Wrong")
    none_patch = mpatches.Patch(color=_GREY, label="No detection")
    flagged_patch = mpatches.Patch(color=_ORANGE, alpha=0.5, label="Flagged by defense")
    ax.legend(handles=[correct_patch, wrong_patch, none_patch, flagged_patch],
              loc="upper right", fontsize=8)

    # Panel 2: stop_sign_conf + brightness_delta
    ax2 = axes[1]
    ax2.plot(distances, confs, color=_BLUE, marker="o", markersize=4, label="stop_sign_conf")
    ax2.set_ylabel("Confidence", color=_BLUE)
    ax2.tick_params(axis="y", labelcolor=_BLUE)
    ax2.set_ylim(-0.05, 1.05)

    ax3 = ax2.twinx()
    ax3.plot(distances, brightness_deltas, color=_ORANGE, marker="x",
             markersize=4, linestyle="--", label="brightness delta")
    ax3.set_ylabel("Brightness delta", color=_ORANGE)
    ax3.tick_params(axis="y", labelcolor=_ORANGE)

    ax2.set_xlabel("Distance")
    ax2.legend(loc="upper left", fontsize=8)
    ax3.legend(loc="upper right", fontsize=8)
    ax2.invert_xaxis()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_summary(
    metrics: OverallMetrics,
    ground_truth_label: str = "stop sign",
    out_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    seqs = metrics.per_sequence
    degrees = [str(m.degree) for m in seqs]
    x = np.arange(len(degrees))
    width = 0.35

    def _vals(attr: str) -> list[float]:
        return [getattr(m, attr) if not math.isnan(getattr(m, attr)) else 0.0 for m in seqs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Summary — Raw {metrics.raw_accuracy:.1%} → Defended {metrics.defended_accuracy:.1%}"
        f"  |  ADR {metrics.attack_detection_rate:.1%}  |  FPR {metrics.false_positive_rate:.1%}",
        fontsize=13,
    )

    panels = [
        (axes[0, 0], "raw_accuracy", "Raw Accuracy per Degree"),
        (axes[0, 1], "defended_accuracy", "Defended Accuracy per Degree"),
        (axes[1, 0], "attack_detection_rate", "Attack Detection Rate per Degree"),
        (axes[1, 1], "false_positive_rate", "False Positive Rate per Degree"),
    ]

    for ax, attr, title in panels:
        raw = _vals("raw_accuracy") if attr == "defended_accuracy" else None
        vals = _vals(attr)
        if raw is not None:
            ax.bar(x - width / 2, raw, width, color=_RED, alpha=0.7, label="Raw")
            ax.bar(x + width / 2, vals, width, color=_GREEN, alpha=0.7, label="Defended")
            ax.legend(fontsize=8)
        else:
            color = _BLUE if "detection" in attr else _ORANGE
            ax.bar(x, vals, color=color, alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(degrees, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Rate")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_brightness_gate(
    degree: int,
    results: list[FrameResult],
    config_threshold: float,
    out_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    distances = [r.frame.distance for r in results]
    deltas = [r.brightness_delta for r in results]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f"Degree {degree} — Brightness Delta per Frame")
    bar_colors = [_RED if d >= config_threshold else _BLUE for d in deltas]
    ax.bar(distances, deltas, color=bar_colors, alpha=0.8)
    ax.axhline(config_threshold, color=_ORANGE, linestyle="--", label=f"threshold={config_threshold}")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Mean luminance delta")
    ax.legend()
    ax.invert_xaxis()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
