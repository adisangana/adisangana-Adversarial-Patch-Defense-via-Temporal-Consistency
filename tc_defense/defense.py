from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .loader import Frame, Sequence


@dataclass
class DefenseConfig:
    window_size: int = 5
    brightness_threshold: float = 10.0
    ground_truth_label: str = "stop sign"


@dataclass
class FrameResult:
    frame: Frame
    raw_label: Optional[str]
    defended_label: Optional[str]
    flagged_as_attack: bool
    brightness_delta: float
    window_consensus: Optional[str]


@functools.lru_cache(maxsize=None)
def _mean_luminance(image_path: str) -> float:
    img = Image.open(image_path).convert("L")
    return float(np.asarray(img).mean())


def _consensus(frames: list[Frame]) -> Optional[str]:
    counts = Counter(f.raw_label for f in frames if f.raw_label is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def run_defense(sequence: Sequence, config: DefenseConfig) -> list[FrameResult]:
    frames = sequence.frames
    results = []

    for i, frame in enumerate(frames):
        # Use only prior frames as temporal context for online evaluation.
        lo = max(0, i - config.window_size)
        neighbors = frames[lo:i]

        consensus = _consensus(neighbors)

        current_lum = _mean_luminance(str(frame.image_path))
        neighbor_lums = [_mean_luminance(str(f.image_path)) for f in neighbors]
        window_avg_lum = float(np.mean(neighbor_lums)) if neighbor_lums else current_lum
        brightness_delta = abs(current_lum - window_avg_lum)

        scene_stable = brightness_delta < config.brightness_threshold
        flagged = scene_stable and consensus is not None and frame.raw_label != consensus

        results.append(FrameResult(
            frame=frame,
            raw_label=frame.raw_label,
            defended_label=consensus if flagged else frame.raw_label,
            flagged_as_attack=flagged,
            brightness_delta=brightness_delta,
            window_consensus=consensus,
        ))

    return results


def run_defense_all(
    sequences: list[Sequence],
    config: DefenseConfig,
) -> dict[int, list[FrameResult]]:
    return {seq.degree: run_defense(seq, config) for seq in sequences}
