#!/usr/bin/env python3
"""Crop a stamp template from a photo, so remove_watermark.py can find and remove it.

Workflow:
  1. Run without --box to see the image dimensions:
         python extract_stamps.py photos/84198914.jpeg
  2. Crop the stamp (pixels, x1,y1 = top-left, x2,y2 = bottom-right):
         python extract_stamps.py photos/84198914.jpeg --box 14,114,136,236 --out stamps/badge.png
     Or use fractions of width/height with --rel:
         python extract_stamps.py photos/84198914.jpeg --rel --box 0.02,0.20,0.17,0.43 --out stamps/badge.png
  3. Re-run run.bat — every photo matching the template will be inpainted.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def cut_background(img: Image.Image, tol: int = 20) -> Image.Image:
    """Make the near-uniform, border-connected background transparent.

    Flood-fills inwards from the four corners and turns every reached pixel
    transparent. The result is an RGBA PNG where only the stamp artwork is
    opaque — such templates match the same stamp on *other* photos far more
    reliably, because background pixels no longer vote in the comparison.
    """
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    for seed in ((1, 1), (w - 2, 1), (1, h - 2), (w - 2, h - 2)):
        cv2.floodFill(
            arr.copy(), ff_mask, seed, 0,
            loDiff=(tol, tol, tol), upDiff=(tol, tol, tol), flags=flags,
        )
    bg = ff_mask[1:-1, 1:-1] > 0
    # Dilate the background by 1px to also remove background-colored fringe on the stamp edge.
    bg = cv2.dilate(bg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([arr, alpha]), "RGBA")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crop a stamp template for remove_watermark.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("image", help="Photo containing the stamp.")
    p.add_argument(
        "--box",
        help="Crop rectangle x1,y1,x2,y2 (pixels; fractions 0-1 with --rel). "
        "Omit to only print the image size.",
    )
    p.add_argument("--rel", action="store_true", help="Interpret --box as fractions (0-1).")
    p.add_argument(
        "--cut-bg",
        action="store_true",
        help="Remove the uniform background around the stamp (flood fill from "
        "the borders) and save a transparent-background PNG. Recommended: such "
        "templates match the same stamp across other photos much more reliably.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output path, e.g. stamps/badge.png (default: stamps/<image>_stamp.png).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.image)
    if not src.is_file():
        print(f"Error: file not found: {src}", file=sys.stderr)
        return 1

    img = ImageOps.exif_transpose(Image.open(src))
    print(f"Image: {src}  {img.width}x{img.height}px  mode={img.mode}")

    if not args.box:
        print("No --box given — nothing to do. Re-run with --box x1,y1,x2,y2 to crop a stamp.")
        return 0

    vals = [float(v) for v in args.box.split(",")]
    if len(vals) != 4:
        print(f"Error: --box needs 4 comma-separated numbers, got: {args.box}", file=sys.stderr)
        return 1
    x1, y1, x2, y2 = vals
    if args.rel:
        x1, x2 = x1 * img.width, x2 * img.width
        y1, y2 = y1 * img.height, y2 * img.height
    box = (round(x1), round(y1), round(x2), round(y2))
    if box[2] <= box[0] or box[3] <= box[1]:
        print(f"Error: invalid box {box} (need x2>x1 and y2>y1)", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else Path("stamps") / f"{src.stem}_stamp.png"
    crop = img.crop(box)
    if args.cut_bg:
        crop = cut_background(crop)
        if out.suffix.lower() != ".png":
            print(f"Note: --cut-bg needs alpha, saving as {out.with_suffix('.png')} instead.")
            out = out.with_suffix(".png")
        opaque = 100 * (np.array(crop)[..., 3] > 0).mean()
        print(f"Background removed — {opaque:.0f}% of the template is opaque stamp.")

    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out)
    print(f"Saved {out}  {crop.width}x{crop.height}px  box={box}")
    print("Tip: keep the crop tight around the stamp — it will be matched at multiple scales.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
