"""
Run YOLOv8 on clean and patched frames, saving per-frame detection JSONs
and annotated images.

Output layout:
    data/detections/
        clean/
            DSCF0117.json
            DSCF0117.png   (annotated)
            ...
        patched/
            DSCF0121.json
            DSCF0121.png
            ...

Usage:
    python detect.py
    python detect.py --model yolov8s.pt --conf 0.25
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 198),
]


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS[det["class_id"] % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.text((x1 + 4, y1 + 2), label, fill=color, font=font)

    return img


def run_detection(
    model: YOLO,
    frame_paths: list[Path],
    out_dir: Path,
    conf: float,
) -> dict[str, list[dict]]:
    """Run YOLO on each frame, save annotated image + JSON, return results dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for fp in frame_paths:
        yolo_results = model(str(fp), conf=conf, verbose=False)
        r = yolo_results[0]

        detections = []
        boxes = r.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append({
                    "class_id": int(boxes.cls[i].item()),
                    "class_name": model.names[int(boxes.cls[i].item())],
                    "confidence": round(float(boxes.conf[i].item()), 4),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                })

        # Save JSON
        json_path = out_dir / fp.with_suffix(".json").name
        with open(json_path, "w") as f:
            json.dump({"frame": fp.name, "detections": detections}, f, indent=2)

        # Save annotated image
        img = Image.open(fp).convert("RGB")
        annotated = draw_detections(img, detections)
        annotated.save(out_dir / fp.name)

        n = len(detections)
        print(f"  {fp.name}: {n} detection{'s' if n != 1 else ''}")
        results[fp.name] = detections

    return results


def main(data_dir: Path, model_name: str, conf: float) -> None:
    frames_dir = data_dir / "frames"
    patched_dir = data_dir / "patched"
    det_dir = data_dir / "detections"

    clean_frames = sorted(frames_dir.glob("*.png"))
    if not clean_frames:
        raise FileNotFoundError(f"No PNG frames in {frames_dir}. Run convert_raw.py first.")

    model = YOLO(model_name)
    print(f"Model: {model_name}  |  conf threshold: {conf}")

    # --- Clean frames ---
    print(f"\n[Clean] {len(clean_frames)} frames")
    clean_results = run_detection(model, clean_frames, det_dir / "clean", conf)

    # --- Patched frames (if they exist) ---
    patched_frames = sorted(patched_dir.glob("*.png")) if patched_dir.exists() else []
    if patched_frames:
        print(f"\n[Patched] {len(patched_frames)} frames")
        run_detection(model, patched_frames, det_dir / "patched", conf)
    else:
        print("\nNo patched frames found — skipping patched detection.")

    # --- Summary comparison ---
    meta_path = data_dir / "patch_meta.json"
    if meta_path.exists() and patched_frames:
        with open(meta_path) as f:
            meta = json.load(f)
        patched_names = set(meta["patched_frames"])

        print("\n--- Detection count comparison ---")
        print(f"{'Frame':<25} {'Clean':>8} {'Patched':>10}")
        print("-" * 45)
        for fp in clean_frames:
            c_count = len(clean_results.get(fp.name, []))
            if fp.name in patched_names:
                patched_json = det_dir / "patched" / fp.with_suffix(".json").name
                if patched_json.exists():
                    with open(patched_json) as f:
                        p_data = json.load(f)
                    p_count = len(p_data["detections"])
                    marker = " *" if c_count != p_count else ""
                    print(f"{fp.name:<25} {c_count:>8} {p_count:>10}{marker}")
                else:
                    print(f"{fp.name:<25} {c_count:>8} {'N/A':>10}")
            else:
                print(f"{fp.name:<25} {c_count:>8} {'(clean)':>10}")

    print("\nDone. Results saved to", det_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 on clean and patched frames")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()
    main(args.data_dir, args.model, args.conf)
