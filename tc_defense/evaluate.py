from __future__ import annotations

import math
from dataclasses import dataclass

from .defense import DefenseConfig, FrameResult


@dataclass
class SequenceMetrics:
    degree: int
    total_frames: int
    raw_correct: int
    defended_correct: int
    attacked_frames: int
    correctly_flagged: int
    false_positives: int
    raw_accuracy: float
    defended_accuracy: float
    attack_detection_rate: float
    false_positive_rate: float


@dataclass
class OverallMetrics:
    total_frames: int
    raw_accuracy: float
    defended_accuracy: float
    attack_detection_rate: float
    false_positive_rate: float
    per_sequence: list[SequenceMetrics]


def evaluate_sequence(
    degree: int,
    results: list[FrameResult],
    ground_truth_label: str = "stop sign",
) -> SequenceMetrics:
    total = len(results)
    raw_correct = sum(1 for r in results if r.raw_label == ground_truth_label)
    defended_correct = sum(1 for r in results if r.defended_label == ground_truth_label)
    attacked = sum(1 for r in results if r.raw_label != ground_truth_label)
    correctly_flagged = sum(
        1 for r in results
        if r.raw_label != ground_truth_label and r.flagged_as_attack
    )
    false_positives = sum(
        1 for r in results
        if r.raw_label == ground_truth_label and r.flagged_as_attack
    )

    clean = total - attacked
    return SequenceMetrics(
        degree=degree,
        total_frames=total,
        raw_correct=raw_correct,
        defended_correct=defended_correct,
        attacked_frames=attacked,
        correctly_flagged=correctly_flagged,
        false_positives=false_positives,
        raw_accuracy=raw_correct / total if total else math.nan,
        defended_accuracy=defended_correct / total if total else math.nan,
        attack_detection_rate=correctly_flagged / attacked if attacked else math.nan,
        false_positive_rate=false_positives / clean if clean else math.nan,
    )


def evaluate_all(
    all_results: dict[int, list[FrameResult]],
    ground_truth_label: str = "stop sign",
) -> OverallMetrics:
    per_seq = [
        evaluate_sequence(deg, results, ground_truth_label)
        for deg, results in sorted(all_results.items())
    ]

    total = sum(m.total_frames for m in per_seq)
    raw_correct = sum(m.raw_correct for m in per_seq)
    defended_correct = sum(m.defended_correct for m in per_seq)
    attacked = sum(m.attacked_frames for m in per_seq)
    correctly_flagged = sum(m.correctly_flagged for m in per_seq)
    clean = total - attacked
    false_positives = sum(m.false_positives for m in per_seq)

    return OverallMetrics(
        total_frames=total,
        raw_accuracy=raw_correct / total if total else math.nan,
        defended_accuracy=defended_correct / total if total else math.nan,
        attack_detection_rate=correctly_flagged / attacked if attacked else math.nan,
        false_positive_rate=false_positives / clean if clean else math.nan,
        per_sequence=per_seq,
    )


def print_report(metrics: OverallMetrics) -> None:
    def _fmt(v: float) -> str:
        return "N/A" if math.isnan(v) else f"{v:.1%}"

    header = f"{'Degree':>7} {'Frames':>7} {'Raw Acc':>8} {'Def Acc':>8} {'ADR':>7} {'FPR':>7}"
    print(header)
    print("-" * len(header))
    for m in metrics.per_sequence:
        print(
            f"{m.degree:>7d} {m.total_frames:>7d} {_fmt(m.raw_accuracy):>8}"
            f" {_fmt(m.defended_accuracy):>8} {_fmt(m.attack_detection_rate):>7}"
            f" {_fmt(m.false_positive_rate):>7}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':>7} {metrics.total_frames:>7d} {_fmt(metrics.raw_accuracy):>8}"
        f" {_fmt(metrics.defended_accuracy):>8} {_fmt(metrics.attack_detection_rate):>7}"
        f" {_fmt(metrics.false_positive_rate):>7}"
    )
    print()
    print("ADR = Attack Detection Rate  |  FPR = False Positive Rate")
