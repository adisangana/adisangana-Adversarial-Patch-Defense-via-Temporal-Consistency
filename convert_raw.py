"""Convert Fujifilm RAF raw files in data/ to PNG in data/frames/."""

import argparse
import os
from pathlib import Path

import imageio
import rawpy


def convert_raf_to_png(src: Path, dst: Path) -> None:
    with rawpy.imread(str(src)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=False,
            output_bps=8,
        )
    imageio.imsave(str(dst), rgb)
    print(f"  {src.name} -> {dst.name}  ({rgb.shape[1]}x{rgb.shape[0]})")


def main(data_dir: Path) -> None:
    out_dir = data_dir / "frames"
    out_dir.mkdir(exist_ok=True)

    raf_files = sorted(data_dir.glob("*.RAF")) + sorted(data_dir.glob("*.raf"))
    if not raf_files:
        print(f"No RAF files found in {data_dir}")
        return

    print(f"Converting {len(raf_files)} RAF files -> {out_dir}")
    for raf in raf_files:
        dst = out_dir / raf.with_suffix(".png").name
        if dst.exists():
            print(f"  Skipping {raf.name} (already converted)")
            continue
        convert_raf_to_png(raf, dst)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RAF files to PNG")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing .RAF files (default: data/)",
    )
    args = parser.parse_args()
    main(args.data_dir)
