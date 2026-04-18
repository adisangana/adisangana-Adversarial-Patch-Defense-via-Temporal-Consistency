from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass
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


def _weighted_consensus(
    current_frame: Frame, 
    neighbors: list[Frame], 
    current_lum: float, 
    neighbor_lums: list[float]
) -> Optional[str]:
    if not neighbors:
        return None

    label_scores = defaultdict(float)
    
    # neighbors are ordered oldest to newest.
    # neighbors[-1] is distance 1 (nearest history)
    # neighbors[0] is distance len(neighbors) (furthest history)
    for idx, (neighbor, n_lum) in enumerate(zip(neighbors, neighbor_lums)):
        if neighbor.raw_label is None:
            continue
            
        # 1. Temporal Weight: nearest photo heavily weighted, furthest least important
        distance = len(neighbors) - idx
        time_weight = 1.0 / distance 
        
        # 2. YOLO Detection Confidence Weight: defaults to 1.0 if not present in the Frame object
        conf_weight = getattr(neighbor, 'confidence', 1.0)
        
        # 3. Brightness Weight: inversely proportional to the brightness delta
        b_delta = abs(current_lum - n_lum)
        brightness_weight = 1.0 / (1.0 + b_delta)
        
        total_weight = time_weight * conf_weight * brightness_weight
        label_scores[neighbor.raw_label] += total_weight
        
    if not label_scores:
        return None
        
    # Return the label with the highest accumulated score
    return max(label_scores.items(), key=lambda x: x[1])[0]


def run_defense(sequence: Sequence, config: DefenseConfig) -> list[FrameResult]:
    frames = sequence.frames
    results = []

    for i, frame in enumerate(frames):
        # Use only prior frames as temporal context for online evaluation.
        lo = max(0, i - config.window_size)
        neighbors = frames[lo:i]

        current_lum = _mean_luminance(str(frame.image_path))
        neighbor_lums = [_mean_luminance(str(f.image_path)) for f in neighbors]
        
        consensus = _weighted_consensus(frame, neighbors, current_lum, neighbor_lums)

        window_avg_lum = float(np.mean(neighbor_lums)) if neighbor_lums else current_lum
        brightness_delta = abs(current_lum - window_avg_lum)

        scene_stable = brightness_delta < config.brightness_threshold
        
        # Parameterize use of defense by current frame's detection status
        has_detection = frame.raw_label is not None
        flagged = scene_stable and consensus is not None and has_detection and frame.raw_label != consensus

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