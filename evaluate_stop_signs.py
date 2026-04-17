#!/usr/bin/env python3
"""Run YOLOv5 on a directory of images and export detection confidence metrics."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


DEFAULT_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FILENAME_PATTERN = re.compile(
    r"brightness(?P<brightness>\d+)_degree(?P<degree>\d+)_distance(?P<distance>\d+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a directory of images with YOLOv5 and write a CSV with the "
            "top detected object, its confidence, and the stop-sign confidence."
        )
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional path for the generated CSV report. If omitted, the report "
            "is written inside --images-dir."
        ),
    )
    parser.add_argument(
        "--weights",
        default="yolov5s",
        help=(
            "Built-in YOLOv5 model name such as 'yolov5s' or a path to a custom "
            ".pt weights file."
        ),
    )
    parser.add_argument(
        "--yolov5-repo",
        type=Path,
        default=None,
        help=(
            "Optional path to a local clone of the official YOLOv5 repository. "
            "If omitted, torch.hub will use ultralytics/yolov5."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images to run per inference batch.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="YOLOv5 confidence threshold.",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="YOLOv5 IoU threshold for NMS.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Torch device, for example 'cpu', 'mps', '0', or '0,1'.",
    )
    parser.add_argument(
        "--stop-label",
        default="stop sign",
        help="Class label to treat as the stop-sign class.",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Scan subdirectories under --images-dir.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top level of --images-dir. This is the default.",
    )
    parser.set_defaults(recursive=False)
    return parser.parse_args()


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ").replace("-", " ")


def extract_image_metadata(image_path: Path) -> dict[str, str]:
    match = FILENAME_PATTERN.search(image_path.stem)
    if not match:
        return {"brightness": "", "degree": "", "distance": ""}
    return {
        "brightness": match.group("brightness"),
        "degree": match.group("degree"),
        "distance": match.group("distance"),
    }


def collect_images(images_dir: Path, recursive: bool) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory for --images-dir: {images_dir}")

    walker: Iterable[Path]
    walker = images_dir.rglob("*") if recursive else images_dir.glob("*")
    images = sorted(
        path
        for path in walker
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(
            f"No images with supported extensions were found in {images_dir}"
        )
    return images


def chunked(items: Sequence[Path], size: int) -> Iterator[Sequence[Path]]:
    if size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    for index in range(0, len(items), size):
        yield items[index : index + size]


def resolve_builtin_model_name(weights: str) -> str:
    weights_path = Path(weights)
    if weights_path.exists():
        return weights
    if weights_path.suffix == ".pt" and weights_path.parent == Path("."):
        return weights_path.stem
    return weights


def resolve_local_weights_path(weights: str) -> Path | None:
    candidate = Path(weights).expanduser()
    if candidate.exists():
        return candidate.resolve()

    project_root = Path(__file__).resolve().parent
    named_candidate = project_root / candidate.name
    if named_candidate.exists():
        return named_candidate.resolve()

    if candidate.suffix != ".pt":
        builtin_candidate = project_root / f"{weights}.pt"
        if builtin_candidate.exists():
            return builtin_candidate.resolve()

    return None


def configure_runtime_dirs() -> None:
    cache_root = Path(__file__).resolve().parent / ".cache"
    torch_home = cache_root / "torch"
    matplotlib_dir = cache_root / "matplotlib"
    torch_home.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))


def resolve_output_csv(images_dir: Path, output_csv: Path | None) -> Path:
    if output_csv is not None:
        return output_csv.expanduser()
    return images_dir / f"{images_dir.name}_yolov5_stop_sign_report.csv"


def configure_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\..*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*LibreSSL.*",
        category=Warning,
    )


def maybe_reexec_into_project_venv() -> None:
    project_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if not project_python.exists():
        return
    if Path(sys.executable).resolve() == project_python.resolve():
        return
    os.execv(
        str(project_python),
        [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


def load_yolov5_model(args: argparse.Namespace):
    configure_runtime_dirs()
    configure_warning_filters()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the active environment. "
            "Install YOLOv5 dependencies first."
        ) from exc

    weights_path = resolve_local_weights_path(args.weights)
    use_local_repo = args.yolov5_repo is not None
    repo_or_dir = (
        str(args.yolov5_repo.expanduser().resolve())
        if use_local_repo
        else "ultralytics/yolov5"
    )
    source = "local" if use_local_repo else "github"

    if weights_path is not None:
        model = torch.hub.load(
            repo_or_dir,
            "custom",
            path=str(weights_path),
            source=source,
            trust_repo=True,
            verbose=False,
        )
    else:
        model_name = resolve_builtin_model_name(args.weights)
        model = torch.hub.load(
            repo_or_dir,
            model_name,
            source=source,
            trust_repo=True,
            verbose=False,
        )

    model.conf = args.conf_thres
    model.iou = args.iou_thres
    if args.device:
        model.to(args.device)
    return model


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def evaluate_images(args: argparse.Namespace) -> list[dict[str, object]]:
    model = load_yolov5_model(args)
    names = getattr(model, "names", {})
    stop_label = normalize_label(args.stop_label)
    image_paths = collect_images(args.images_dir.expanduser(), args.recursive)
    rows: list[dict[str, object]] = []

    for batch in chunked(image_paths, args.batch_size):
        results = model([str(path) for path in batch], size=args.imgsz)
        for image_path, detections in zip(batch, results.xyxy):
            top_class = ""
            top_confidence = 0.0
            stop_sign_confidence = 0.0
            metadata = extract_image_metadata(image_path)

            if detections is not None and len(detections) > 0:
                confidences = detections[:, 4]
                top_index = int(confidences.argmax().item())
                top_detection = detections[top_index]
                top_class_id = int(top_detection[5].item())
                top_class = class_name(names, top_class_id)
                top_confidence = float(top_detection[4].item())

                for detection in detections:
                    detected_class = class_name(names, int(detection[5].item()))
                    if normalize_label(detected_class) == stop_label:
                        stop_sign_confidence = max(
                            stop_sign_confidence,
                            float(detection[4].item()),
                        )

            rows.append(
                {
                    "brightness": metadata["brightness"],
                    "degree": metadata["degree"],
                    "distance": metadata["distance"],
                    "top_classified_object": top_class,
                    "top_classified_object_accuracy": f"{top_confidence:.6f}",
                    "stopsign_accuracy": f"{stop_sign_confidence:.6f}",
                }
            )

    return rows


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "brightness",
                "degree",
                "distance",
                "top_classified_object",
                "top_classified_object_accuracy",
                "stopsign_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    images_dir = args.images_dir.expanduser()
    output_csv = resolve_output_csv(images_dir, args.output_csv)
    rows = evaluate_images(args)
    write_csv(rows, output_csv)

    stop_sign_hits = sum(
        1 for row in rows if float(str(row["stopsign_accuracy"])) > 0.0
    )
    print(
        f"Processed {len(rows)} image(s). "
        f"Stop sign detected in {stop_sign_hits} image(s)."
    )
    return 0


if __name__ == "__main__":
    maybe_reexec_into_project_venv()
    raise SystemExit(main())
