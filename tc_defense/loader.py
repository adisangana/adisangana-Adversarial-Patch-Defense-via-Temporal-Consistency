from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Frame:
    degree: int
    distance: int
    direction: str
    brightness: int
    raw_label: Optional[str]
    stop_sign_conf: float
    top1_conf: float
    image_path: Path


@dataclass
class Sequence:
    degree: int
    frames: list[Frame]  # sorted by distance ascending, all have images


def _parse_json(path: Path) -> Optional[Frame]:
    """Parse one detection JSON. Returns None if sibling PNG is missing."""
    image_path = path.with_suffix(".png")
    if not image_path.exists():
        return None

    with path.open() as fh:
        d = json.load(fh)

    yolo = d["yolov5"]
    return Frame(
        degree=d["degree"],
        distance=d["distance"],
        direction=d["direction"],
        brightness=d["brightness"],
        raw_label=yolo["top1_label"],
        stop_sign_conf=yolo["stop_sign_conf"],
        top1_conf=yolo["top1_conf"],
        image_path=image_path,
    )


def load_dataset(dataset_dir: Path) -> list[Sequence]:
    """
    Load all complete (JSON + PNG) frames from dataset_dir, grouped into
    Sequences by degree, sorted by distance. Frames missing their PNG are
    skipped with a warning. Degrees where no frames have images are omitted.
    """
    json_files = sorted(dataset_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {dataset_dir}")

    by_degree: dict[int, list[Frame]] = {}
    skipped = 0

    for path in json_files:
        frame = _parse_json(path)
        if frame is None:
            print(f"WARNING: no image for {path.name}, skipping frame", file=sys.stderr)
            skipped += 1
            continue
        by_degree.setdefault(frame.degree, []).append(frame)

    if skipped:
        print(f"WARNING: skipped {skipped} frame(s) with missing PNG images", file=sys.stderr)

    sequences = [
        Sequence(degree=deg, frames=sorted(frames, key=lambda f: f.distance))
        for deg, frames in sorted(by_degree.items())
    ]

    if not sequences:
        raise RuntimeError("No complete sequences found (all frames missing PNG images)")

    return sequences
