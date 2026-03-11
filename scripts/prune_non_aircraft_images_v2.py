#!/usr/bin/env python3
"""
prune_non_aircraft_images_v2.py

Smarter pruning for Tamiya aircraft dataset.

Adds:
- placeholder / blank-image rejection
- banner / collage rejection
- dominant airplane requirement
- suspicious filename checks
- review bucket for uncertain cases

Example:
    python prune_non_aircraft_images_v2.py \
      --data-root ./data/tamiya_aircraft \
      --model yolov8n.pt \
      --conf 0.20
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SUSPECT_NAME_KEYWORDS = {
    "banner", "header", "logo", "placeholder", "dummy", "thumb",
    "spacer", "icon", "sample", "index", "top", "menu", "title"
}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_airplane_class(model: YOLO) -> int:
    for k, v in model.names.items():
        if str(v).lower() == "airplane":
            return int(k)
    raise RuntimeError("Could not find 'airplane' class")


def collect_images(data_root: Path) -> List[Path]:
    out = []
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if "rejected" not in p.parts and "review" not in p.parts:
                out.append(p)
    return sorted(out)


def move_to_bucket(img_path: Path, bucket: str) -> Path:
    model_dir = img_path.parent.parent if img_path.parent.name == "images" else img_path.parent
    dst_dir = model_dir / bucket
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / img_path.name

    if dst.exists():
        stem, suffix = dst.stem, dst.suffix
        i = 1
        while True:
            cand = dst_dir / f"{stem}_{i}{suffix}"
            if not cand.exists():
                dst = cand
                break
            i += 1

    shutil.move(str(img_path), str(dst))
    return dst


def suspicious_filename(img_path: Path) -> bool:
    name = img_path.stem.lower()
    return any(k in name for k in SUSPECT_NAME_KEYWORDS)


def image_stats(img: Image.Image) -> Dict:
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    gray = np.array(img.convert("L"), dtype=np.float32)

    std = float(gray.std())
    mean = float(gray.mean())

    # edge density via simple gradients
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_map = np.zeros_like(gray)
    edge_map[:, 1:] += gx
    edge_map[1:, :] += gy
    edge_density = float((edge_map > 20).mean())

    # coarse unique colors after quantization
    q = (arr // 16).astype(np.uint8)
    flat = q.reshape(-1, 3)
    unique_colors = int(np.unique(flat, axis=0).shape[0])

    # dominant coarse color ratio
    # convert RGB bins to single integer
    key = (flat[:, 0].astype(np.int32) << 8) + (flat[:, 1].astype(np.int32) << 4) + flat[:, 2].astype(np.int32)
    _, counts = np.unique(key, return_counts=True)
    dominant_ratio = float(counts.max() / flat.shape[0])

    # near grayscale ratio
    rg = np.abs(arr[:, :, 0].astype(np.int16) - arr[:, :, 1].astype(np.int16))
    gb = np.abs(arr[:, :, 1].astype(np.int16) - arr[:, :, 2].astype(np.int16))
    rb = np.abs(arr[:, :, 0].astype(np.int16) - arr[:, :, 2].astype(np.int16))
    near_gray_ratio = float(((rg < 8) & (gb < 8) & (rb < 8)).mean())

    return {
        "width": w,
        "height": h,
        "aspect_ratio": float(w / max(h, 1)),
        "gray_mean": mean,
        "gray_std": std,
        "edge_density": edge_density,
        "unique_colors_q16": unique_colors,
        "dominant_color_ratio_q16": dominant_ratio,
        "near_gray_ratio": near_gray_ratio,
    }


def obvious_placeholder_or_blank(stats: Dict) -> Tuple[bool, str]:
    w = stats["width"]
    h = stats["height"]
    ar = stats["aspect_ratio"]
    std = stats["gray_std"]
    edge_density = stats["edge_density"]
    unique_colors = stats["unique_colors_q16"]
    dom = stats["dominant_color_ratio_q16"]
    near_gray = stats["near_gray_ratio"]

    # small / thumbnail-ish junk
    if w < 180 or h < 180:
        return True, "too_small"

    # banner-like
    if ar > 1.8 or ar < 0.55:
        return True, "extreme_aspect_ratio"

    # almost blank / placeholder / flat tile
    if std < 18 and edge_density < 0.03 and dom > 0.55:
        return True, "low_information_placeholder"

    # extremely uniform grayscale-ish image
    if near_gray > 0.92 and std < 22 and unique_colors < 80:
        return True, "mostly_gray_low_variance"

    # huge single background color
    if dom > 0.72 and edge_density < 0.05:
        return True, "dominant_flat_background"

    return False, ""


def analyze_yolo(result, airplane_cls: int, w: int, h: int, conf_threshold: float) -> Dict:
    info = {
        "num_objects": 0,
        "distinct_classes": [],
        "best_airplane_conf": None,
        "best_airplane_bbox": None,
        "best_airplane_area_ratio": 0.0,
    }

    if result.boxes is None or len(result.boxes) == 0:
        return info

    classes = []
    best_conf = -1.0
    best_bbox = None
    best_area_ratio = 0.0

    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i].item())
        conf = float(result.boxes.conf[i].item())
        x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().tolist()
        area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(max(w * h, 1))

        if conf >= conf_threshold:
            classes.append(cls_id)

        if cls_id == airplane_cls and conf >= conf_threshold and conf > best_conf:
            best_conf = conf
            best_bbox = [float(x1), float(y1), float(x2), float(y2)]
            best_area_ratio = float(area_ratio)

    info["num_objects"] = len(classes)
    info["distinct_classes"] = sorted(set(classes))
    info["best_airplane_conf"] = None if best_conf < 0 else best_conf
    info["best_airplane_bbox"] = best_bbox
    info["best_airplane_area_ratio"] = best_area_ratio
    return info


def decision_from_stats_and_yolo(img_path: Path, stats: Dict, yolo_info: Dict) -> Tuple[str, str]:
    # Always keep primary image if you want safer behavior.
    # Comment this out if you want stricter pruning.
    if img_path.name == "001.jpg":
        return "keep", "primary_image_kept"

    if suspicious_filename(img_path):
        return "review", "suspicious_filename"

    if yolo_info["best_airplane_conf"] is None:
        return "reject", "no_airplane_detected"

    if yolo_info["best_airplane_area_ratio"] < 0.18:
        return "reject", "airplane_too_small"

    if yolo_info["num_objects"] > 3:
        return "reject", "too_many_objects"

    if len(yolo_info["distinct_classes"]) > 2:
        return "reject", "too_many_distinct_classes"

    return "keep", "dominant_airplane_detected"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default="prune_report_v2.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    device = get_device()

    print(f"[INFO] device={device}")
    print(f"[INFO] loading model={args.model}")
    model = YOLO(args.model)
    airplane_cls = find_airplane_class(model)

    images = collect_images(data_root)
    print(f"[INFO] found {len(images)} images")

    report = {"kept": [], "rejected": [], "review": [], "errors": []}

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path}")
        try:
            img = Image.open(img_path).convert("RGB")
            stats = image_stats(img)

            hard_reject, hard_reason = obvious_placeholder_or_blank(stats)
            if hard_reject:
                entry = {"image": str(img_path), "reason": hard_reason, "stats": stats}
                if args.dry_run:
                    report["rejected"].append(entry)
                else:
                    new_path = move_to_bucket(img_path, "rejected")
                    entry["moved_to"] = str(new_path)
                    report["rejected"].append(entry)
                continue

            results = model.predict(
                source=img,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )
            yolo_info = analyze_yolo(results[0], airplane_cls, stats["width"], stats["height"], args.conf)

            decision, reason = decision_from_stats_and_yolo(img_path, stats, yolo_info)
            entry = {
                "image": str(img_path),
                "reason": reason,
                "stats": stats,
                "yolo": yolo_info,
            }

            if decision == "keep":
                report["kept"].append(entry)
            elif decision == "review":
                if args.dry_run:
                    report["review"].append(entry)
                else:
                    new_path = move_to_bucket(img_path, "review")
                    entry["moved_to"] = str(new_path)
                    report["review"].append(entry)
            else:
                if args.dry_run:
                    report["rejected"].append(entry)
                else:
                    new_path = move_to_bucket(img_path, "rejected")
                    entry["moved_to"] = str(new_path)
                    report["rejected"].append(entry)

        except Exception as e:
            report["errors"].append({"image": str(img_path), "error": str(e)})

    report_path = data_root / args.report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[DONE] report={report_path}")
    print(
        f"[DONE] kept={len(report['kept'])} "
        f"review={len(report['review'])} "
        f"rejected={len(report['rejected'])} "
        f"errors={len(report['errors'])}"
    )


if __name__ == "__main__":
    main()