#!/usr/bin/env python3
"""Recursively detect and remove watermarks / stamps from images.

Pipeline:
  1. YOLO11x watermark detector finds watermark / stamp regions.
  2. If nothing is detected, the file is copied byte-for-byte (untouched).
  3. Otherwise the detected regions are inpainted with LaMa (big-lama).
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from ultralytics import YOLO

DETECTOR_REPO = "corzent/yolo11x_watermark_detection"
DETECTOR_FILE = "best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recursively remove watermarks/stamps from images."
    )
    p.add_argument("--input", default="photos", help="Input folder (scanned recursively).")
    p.add_argument("--output", default="photos_cleaned", help="Output folder (mirrors input structure).")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold (0-1).")
    p.add_argument("--padding", type=int, default=8, help="Extra pixels added around each detected box.")
    p.add_argument("--dilate", type=int, default=12, help="Mask dilation kernel size in pixels.")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device: cuda | cpu (default: auto).",
    )
    return p.parse_args()


def load_detector(device: str) -> YOLO:
    print(f"Loading watermark detector ({DETECTOR_REPO}) on {device} ...")
    ckpt = hf_hub_download(repo_id=DETECTOR_REPO, filename=DETECTOR_FILE)
    model = YOLO(ckpt)
    model.to(device)
    return model


def detect_boxes(model: YOLO, image_path: Path, conf: float, device: str) -> list[list[float]]:
    results = model.predict(str(image_path), conf=conf, device=device, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    return boxes.xyxy.cpu().numpy().round().astype(int).tolist()


def build_mask(width: int, height: int, boxes: list[list[float]], padding: int, dilate: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(width - 1, x2 + padding), min(height - 1, y2 + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask


def save_image(img: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(dest, quality=95, subsampling=0)
    else:
        img.save(dest)


def process_image(
    model: YOLO,
    lama,
    src: Path,
    dest: Path,
    args: argparse.Namespace,
) -> str:
    pil_img = Image.open(src)
    pil_img = ImageOps.exif_transpose(pil_img)  # honor EXIF orientation once

    boxes = detect_boxes(model, src, args.conf, args.device)
    if not boxes:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)  # untouched, byte-for-byte
        return "clean"

    has_alpha = pil_img.mode in ("RGBA", "LA") or "transparency" in pil_img.info
    alpha = pil_img.getchannel("A") if has_alpha else None
    rgb_img = pil_img.convert("RGB")

    mask = build_mask(rgb_img.width, rgb_img.height, boxes, args.padding, args.dilate)
    mask_img = Image.fromarray(mask)
    result = lama(rgb_img, mask_img)
    # LaMa pads to multiples of 8; crop back to the original dimensions.
    if result.size != rgb_img.size:
        result = result.crop((0, 0, rgb_img.width, rgb_img.height))

    if alpha is not None:
        result = result.convert("RGBA")
        result.putalpha(alpha)

    save_image(result, dest)
    return f"watermark x{len(boxes)}"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"Error: input folder not found: {input_dir}", file=sys.stderr)
        return 1

    images = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found under {input_dir}")
        return 0

    print(f"Found {len(images)} image(s). Device: {args.device}")

    model = load_detector(args.device)

    print("Loading LaMa inpainting model ...")
    from simple_lama_inpainting import SimpleLama

    lama = SimpleLama(device=torch.device(args.device))

    cleaned, untouched, failed = 0, 0, 0
    t0 = time.time()

    for i, src in enumerate(images, 1):
        rel = src.relative_to(input_dir)
        dest = output_dir / rel
        try:
            status = process_image(model, lama, src, dest, args)
        except Exception as e:  # keep batch going, report at the end
            failed += 1
            print(f"[{i}/{len(images)}] FAIL {rel}: {e}")
            continue
        if status == "clean":
            untouched += 1
        else:
            cleaned += 1
        print(f"[{i}/{len(images)}] {status:<16} {rel}")

    dt = time.time() - t0
    print("-" * 50)
    print(f"Done in {dt:.1f}s  |  cleaned: {cleaned}  untouched: {untouched}  failed: {failed}")
    print(f"Output: {output_dir.resolve()}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
