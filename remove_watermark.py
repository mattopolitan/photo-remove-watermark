#!/usr/bin/env python3
"""Recursively detect and remove watermarks / stamps from images.

Pipeline:
  1. YOLO11x watermark detector finds watermark / stamp regions.
  2. Multi-scale template matching finds user-provided stamps (see --stamps).
  3. If nothing is detected, the file is copied byte-for-byte (untouched).
  4. Otherwise all detected regions are inpainted with LaMa (big-lama).
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
STAMP_EXTS = IMAGE_EXTS
STAMP_SCALES = tuple(np.linspace(0.5, 1.6, 12).round(3))
NMS_IOU = 0.3
MAX_STAMP_HITS = 30


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
    p.add_argument(
        "--stamps",
        default="stamps",
        help="Folder with stamp template images. Every template is matched against "
        "each photo at multiple scales; matches are inpainted like detector hits. "
        "Folder may be missing or empty (template matching is skipped).",
    )
    p.add_argument(
        "--stamp-threshold",
        type=float,
        default=0.72,
        help="Template-match similarity threshold (0-1). Lower = catch more, "
        "higher = fewer false positives.",
    )
    return p.parse_args()


def load_detector(device: str) -> YOLO:
    print(f"Loading watermark detector ({DETECTOR_REPO}) on {device} ...")
    ckpt = hf_hub_download(repo_id=DETECTOR_REPO, filename=DETECTOR_FILE)
    model = YOLO(ckpt)
    model.to(device)
    return model


def detect_boxes(model: YOLO, bgr_img: np.ndarray, conf: float, device: str) -> list[list[int]]:
    results = model.predict(bgr_img, conf=conf, device=device, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    return boxes.xyxy.cpu().numpy().round().astype(int).tolist()


# ---------------------------------------------------------------------------
# Stamp template matching
# ---------------------------------------------------------------------------

def load_stamp_templates(stamps_dir: Path) -> list[dict]:
    """Load stamp templates as grayscale images (plus optional alpha mask)."""
    templates = []
    if not stamps_dir.is_dir():
        return templates
    for p in sorted(stamps_dir.rglob("*")):
        if p.suffix.lower() not in STAMP_EXTS:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3:
            continue
        gray = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
        mask = None
        if img.shape[2] == 4 and (img[..., 3] < 250).any():  # meaningful transparency
            mask = img[..., 3]
        templates.append({"name": p.name, "gray": gray, "mask": mask})
    return templates


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, area_a + area_b - inter)


def _nms(cands: list[dict]) -> list[dict]:
    keep: list[dict] = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if len(keep) >= MAX_STAMP_HITS:
            break
        if all(_iou(c["box"], k["box"]) < NMS_IOU for k in keep):
            keep.append(c)
    return keep


def match_stamps(rgb: np.ndarray, templates: list[dict], threshold: float) -> list[dict]:
    """Multi-scale template matching. Returns [{name, score, box=(x1,y1,x2,y2)}]."""
    if not templates:
        return []
    img_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h_img, w_img = img_gray.shape
    cands: list[dict] = []
    peak_kernel = np.ones((5, 5), np.uint8)
    for tpl in templates:
        th, tw = tpl["gray"].shape
        for scale in STAMP_SCALES:
            w, h = int(tw * scale), int(th * scale)
            if w < 12 or h < 12 or w > w_img - 2 or h > h_img - 2:
                continue
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            g = cv2.resize(tpl["gray"], (w, h), interpolation=interp)
            if tpl["mask"] is not None:
                m = cv2.resize(tpl["mask"], (w, h), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(img_gray, g, cv2.TM_CCORR_NORMED, mask=m)
            else:
                res = cv2.matchTemplate(img_gray, g, cv2.TM_CCOEFF_NORMED)
            dil = cv2.dilate(res, peak_kernel)
            ys, xs = np.where((res >= threshold) & (res >= dil - 1e-6))
            for y, x in zip(ys, xs):
                cands.append(
                    {
                        "name": tpl["name"],
                        "score": float(res[y, x]),
                        "box": (int(x), int(y), int(x + w), int(y + h)),
                    }
                )
    return _nms(cands)


# ---------------------------------------------------------------------------

def build_mask(width: int, height: int, boxes: list, padding: int, dilate: int) -> np.ndarray:
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
    stamp_templates: list[dict],
) -> str:
    pil_img = Image.open(src)
    pil_img = ImageOps.exif_transpose(pil_img)  # honor EXIF orientation once

    rgb_img = pil_img.convert("RGB")
    rgb = np.array(rgb_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    detections: list[tuple[str, list, float]] = []
    detections += [("watermark", b, 1.0) for b in detect_boxes(model, bgr, args.conf, args.device)]
    detections += [
        (d["name"], list(d["box"]), d["score"])
        for d in match_stamps(rgb, stamp_templates, args.stamp_threshold)
    ]

    if not detections:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)  # untouched, byte-for-byte
        return "clean"

    has_alpha = pil_img.mode in ("RGBA", "LA") or "transparency" in pil_img.info
    alpha = pil_img.getchannel("A") if has_alpha else None

    mask = build_mask(rgb_img.width, rgb_img.height, [d[1] for d in detections], args.padding, args.dilate)
    mask_img = Image.fromarray(mask)
    result = lama(rgb_img, mask_img)
    # LaMa pads to multiples of 8; crop back to the original dimensions.
    if result.size != rgb_img.size:
        result = result.crop((0, 0, rgb_img.width, rgb_img.height))

    if alpha is not None:
        result = result.convert("RGBA")
        result.putalpha(alpha)

    save_image(result, dest)

    parts = []
    n_wm = sum(1 for d in detections if d[0] == "watermark")
    if n_wm:
        parts.append(f"watermark x{n_wm}")
    for name in sorted({d[0] for d in detections if d[0] != "watermark"}):
        n = sum(1 for d in detections if d[0] == name)
        parts.append(f"stamp:{name} x{n}")
    return " ".join(parts)


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

    stamp_templates = load_stamp_templates(Path(args.stamps))
    if stamp_templates:
        print(f"Loaded {len(stamp_templates)} stamp template(s) from '{args.stamps}' "
              f"(threshold {args.stamp_threshold})")
    else:
        print(f"No stamp templates in '{args.stamps}' — YOLO detection only.")

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
            status = process_image(model, lama, src, dest, args, stamp_templates)
        except Exception as e:  # keep batch going, report at the end
            failed += 1
            print(f"[{i}/{len(images)}] FAIL {rel}: {e}")
            continue
        if status == "clean":
            untouched += 1
        else:
            cleaned += 1
        print(f"[{i}/{len(images)}] {status:<40} {rel}")

    dt = time.time() - t0
    print("-" * 50)
    print(f"Done in {dt:.1f}s  |  cleaned: {cleaned}  untouched: {untouched}  failed: {failed}")
    print(f"Output: {output_dir.resolve()}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
