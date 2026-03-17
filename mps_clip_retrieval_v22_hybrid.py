#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import open_clip

# Optional YOLO
try:
    from ultralytics import YOLO
except:
    YOLO = None


# ----------------------------
# Device helper
# ----------------------------
def get_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif device_str == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------
# Load CLIP
# ----------------------------
def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model.to(device)
    model.eval()

    return model, preprocess, tokenizer


# ----------------------------
# Embed image
# ----------------------------
def embed_image(model, preprocess, image_path, device):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(image)
        emb = F.normalize(emb, dim=-1)

    return emb


# ----------------------------
# YOLO crop (optional)
# ----------------------------
def yolo_crop(image_path, weights, device):
    if YOLO is None:
        return None, {"detected": False, "reason": "yolo_not_available"}

    model = YOLO(weights)
    results = model(image_path, device=device)

    if len(results[0].boxes) == 0:
        return None, {"detected": False, "reason": "no_detection"}

    box = results[0].boxes[0]
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    conf = float(box.conf[0])

    img = Image.open(image_path).convert("RGB")
    crop = img.crop(xyxy)

    meta = {
        "detected": True,
        "confidence": conf,
        "bbox_xyxy": xyxy.tolist(),
    }

    return crop, meta


# ----------------------------
# Crop quality gate
# ----------------------------
def crop_quality_ok(meta, img_w, img_h):
    if not meta["detected"]:
        return False, ["no_detection"]

    x1, y1, x2, y2 = meta["bbox_xyxy"]

    area_ratio = (x2 - x1) * (y2 - y1) / (img_w * img_h)

    reasons = []

    if meta["confidence"] < 0.5:
        reasons.append("low_conf")

    if area_ratio > 0.75:
        reasons.append("too_large")

    return len(reasons) == 0, reasons


# ----------------------------
# Search
# ----------------------------
def search(args):
    device = get_device(args.device)
    print(f"[INFO] device={device}")

    model, preprocess, tokenizer = load_clip(device)

    print("[INFO] loading query image...")
    img = Image.open(args.query_image).convert("RGB")

    crop_meta = {"detected": False, "reason": "cropping_not_requested"}
    use_crop = False

    if args.yolo_weights:
        print("[INFO] running YOLO crop...")
        crop, crop_meta = yolo_crop(args.query_image, args.yolo_weights, device)

        ok, reasons = crop_quality_ok(crop_meta, img.width, img.height)

        if ok:
            img = crop
            use_crop = True
        else:
            print("[INFO] crop rejected:", reasons)

    print("[INFO] embedding query image...")
    image_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = model.encode_image(image_tensor)
        query_emb = F.normalize(query_emb, dim=-1)

    print("[INFO] loading embeddings...")
    emb = torch.load(
        Path(args.artifacts_dir) / "product_hybrid_embeddings.pt",
        weights_only=True,
        map_location=device,
    )

    with open(Path(args.artifacts_dir) / "product_hybrid_metadata.json") as f:
        meta = json.load(f)

    emb = emb.to(device)

    scores = (query_emb @ emb.T).squeeze(0)

    topk = min(args.topk, len(scores))
    vals, idxs = torch.topk(scores, topk)

    top_score = float(vals[0])

    # ----------------------------
    # NEW: no-match gate
    # ----------------------------
    if top_score < args.min_score:
        return {
            "match": False,
            "reason": f"low_confidence<{args.min_score}",
            "top_score": top_score,
            "device_used": device,
        }

    results = []
    for v, i in zip(vals, idxs):
        m = meta[i]

        results.append(
            {
                "score": float(v),
                "item_no": m.get("item_no"),
                "title": m["title"],
                "model_dir": m["model_dir"],
                "selected_images": m["images"],
                "num_selected_images": len(m["images"]),
            }
        )

    return {
        "match": True,
        "device_used": device,
        "query_image": args.query_image,
        "crop_meta": crop_meta,
        "query_embedding_source": "crop" if use_crop else "original",
        "top_product_hits": results,
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="cmd")

    s = sub.add_parser("search")
    s.add_argument("--artifacts-dir", required=True)
    s.add_argument("--query-image", required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--yolo-weights", default=None)
    s.add_argument("--topk", type=int, default=2)
    s.add_argument("--min-score", type=float, default=0.55)

    args = parser.parse_args()

    if args.cmd == "search":
        result = search(args)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()