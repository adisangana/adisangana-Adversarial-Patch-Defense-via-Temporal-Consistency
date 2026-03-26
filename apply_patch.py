"""
Generate an adversarial patch and apply it to a subset of frames.

Patch generation: gradient ascent on YOLOv8 objectness loss so the model
fails to detect objects in the patched region.

Usage:
    # Apply patch to frames 5-12 (0-indexed) with a 200x200 patch
    python apply_patch.py --frames 5 12 --patch-size 200

    # Use a pre-existing patch image instead of generating one
    python apply_patch.py --frames 5 12 --patch-file patch.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

def generate_patch(
    model: YOLO,
    frame_paths: list[Path],
    patch_size: int,
    patch_x: int,
    patch_y: int,
    steps: int = 50,
    lr: float = 0.01,
) -> np.ndarray:
    """
    Learn a patch that maximises YOLOv8's objectness loss (makes detections
    disappear) via gradient ascent.

    Returns an (H, W, 3) uint8 numpy array.
    """
    device = next(model.model.parameters()).device

    # Initialise patch in [0, 1] range
    patch = torch.rand(1, 3, patch_size, patch_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=lr)

    # Use a random subset of frames for patch training (max 5 to keep it fast)
    train_frames = frame_paths[: min(5, len(frame_paths))]

    print(f"Training patch ({patch_size}x{patch_size}) for {steps} steps...")

    for step in range(steps):
        step_loss = torch.tensor(0.0, device=device)

        for fp in train_frames:
            img = Image.open(fp).convert("RGB")
            w, h = img.size

            # Clamp patch position to image bounds
            x = min(patch_x, w - patch_size)
            y = min(patch_y, h - patch_size)

            # img -> tensor [1, 3, H, W] in [0, 1]
            img_t = torch.from_numpy(np.array(img)).float() / 255.0
            img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(device)

            # Paste patch onto image
            patched = img_t.clone()
            patched[:, :, y : y + patch_size, x : x + patch_size] = patch.clamp(0, 1)

            # Resize to YOLO's expected 640x640 input
            patched_resized = F.interpolate(patched, size=(640, 640), mode="bilinear", align_corners=False)

            # Forward pass through YOLO backbone + head to get raw predictions
            preds = model.model(patched_resized)

            # preds[0]: [batch, anchors, 5+num_classes] — first output tensor
            raw = preds[0] if isinstance(preds, (list, tuple)) else preds

            # Objectness scores are at index 4; maximise to confuse the model
            # (we want high objectness on random noise, not real objects)
            obj_scores = raw[..., 4]
            loss = -obj_scores.mean()  # gradient ascent = minimise negative

            step_loss = step_loss + loss

        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"  step {step+1}/{steps}  loss={step_loss.item():.4f}")

    patch_np = (patch.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return patch_np


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_patch_to_frame(
    src: Path,
    dst: Path,
    patch: np.ndarray,
    patch_x: int,
    patch_y: int,
) -> None:
    img = Image.open(src).convert("RGB")
    img_np = np.array(img)

    h, w = img_np.shape[:2]
    ph, pw = patch.shape[:2]

    x = min(patch_x, w - pw)
    y = min(patch_y, h - ph)

    img_np[y : y + ph, x : x + pw] = patch
    Image.fromarray(img_np).save(dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_dir: Path,
    frame_range: tuple[int, int],
    patch_size: int,
    patch_file: Path | None,
    patch_x: int,
    patch_y: int,
    steps: int,
    lr: float,
) -> None:
    frames_dir = data_dir / "frames"
    patched_dir = data_dir / "patched"
    patched_dir.mkdir(exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}. Run convert_raw.py first.")

    start, end = frame_range
    frames_to_patch = frame_paths[start : end + 1]
    print(f"Frames to patch: {[p.name for p in frames_to_patch]}")

    # --- Get or generate patch ---
    if patch_file and patch_file.exists():
        print(f"Loading patch from {patch_file}")
        patch = np.array(Image.open(patch_file).convert("RGB").resize((patch_size, patch_size)))
    else:
        model = YOLO("yolov8n.pt")
        model.model.eval()
        patch = generate_patch(
            model=model,
            frame_paths=frame_paths,
            patch_size=patch_size,
            patch_x=patch_x,
            patch_y=patch_y,
            steps=steps,
            lr=lr,
        )
        patch_out = data_dir / "patch.png"
        Image.fromarray(patch).save(patch_out)
        print(f"Patch saved to {patch_out}")

    # --- Apply patch to selected frames ---
    meta = {
        "patched_frames": [],
        "patch_size": patch_size,
        "patch_x": patch_x,
        "patch_y": patch_y,
    }

    for fp in frames_to_patch:
        dst = patched_dir / fp.name
        apply_patch_to_frame(fp, dst, patch, patch_x, patch_y)
        meta["patched_frames"].append(fp.name)
        print(f"  Patched {fp.name} -> {dst}")

    # Save metadata so detect.py knows which frames were patched
    meta_path = data_dir / "patch_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply adversarial patch to frames")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--frames",
        type=int,
        nargs=2,
        default=[4, 11],
        metavar=("START", "END"),
        help="0-indexed frame range to patch, inclusive (default: 4 11)",
    )
    parser.add_argument("--patch-size", type=int, default=200, help="Patch side length in pixels (default: 200)")
    parser.add_argument("--patch-file", type=Path, default=None, help="Path to a pre-existing patch PNG (skips generation)")
    parser.add_argument("--patch-x", type=int, default=100, help="Patch top-left x position in the image (default: 100)")
    parser.add_argument("--patch-y", type=int, default=100, help="Patch top-left y position in the image (default: 100)")
    parser.add_argument("--steps", type=int, default=50, help="Gradient ascent steps (default: 50)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        frame_range=tuple(args.frames),
        patch_size=args.patch_size,
        patch_file=args.patch_file,
        patch_x=args.patch_x,
        patch_y=args.patch_y,
        steps=args.steps,
        lr=args.lr,
    )
