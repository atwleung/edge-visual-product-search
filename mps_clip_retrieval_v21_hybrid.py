#!/usr/bin/env python3
"""
mps_clip_retrieval_v21_hybrid.py

Hybrid product-level retrieval:
- image embedding from representative product image(s)
- text embedding from product title
- combined into one product vector

Designed for:
- Mac M1 / M2 / M3 with MPS now
- easy migration to CUDA later

Key features:
- product-level embedding
- strict representative image selection (prefer 001/002/003 only)
- optional YOLO cropping at query time
- YOLO crop quality gate
- automatic fallback to original image if crop is poor
- default topk = 2

Usage:

1) Build hybrid product catalog:
python mps_clip_retrieval_v21_hybrid.py build \
  --data-root ./data/tamiya_aircraft \
  --out-dir ./artifacts_v21_hybrid \
  --device mps \
  --max-images-per-product 1 \
  --image-weight 0.6 \
  --text-weight 0.4

2) Search without YOLO:
python mps_clip_retrieval_v21_hybrid.py search \
  --artifacts-dir ./artifacts_v21_hybrid \
  --query-image ./queries/f14.jpg \
  --device mps

3) Search with YOLO, but fallback automatically if crop is poor:
python mps_clip_retrieval_v21_hybrid.py search \
  --artifacts-dir ./artifacts_v21_hybrid \
  --query-image ./queries/f14.jpg \
  --yolo-weights yolov8n.pt \
  --device mps \
  --save-crop ./query_crop.jpg
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ============================================================
# Device selection
# ============================================================

def get_best_device(prefer: str = "auto") -> str:
    """
    Main switch point for MPS vs CUDA.

    Mac:
      --device mps

    Future RTX 6000 / Linux:
      --device cuda

    auto:
      cuda > mps > cpu
    """
    prefer = prefer.lower()

    if prefer == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA requested but not available")

    if prefer == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("MPS requested but not available")

    if prefer == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================
# Helpers
# ============================================================

def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def l2_normalize_tensor(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def extract_numeric_prefix(filename: str) -> int:
    m = re.match(r"^(\d+)", filename)
    if m:
        return int(m.group(1))
    return 999999


def build_title_prompt(title: str) -> str:
    return f"a scale model kit product named {title}"


def bbox_area_ratio(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> float:
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    img_area = max(1, img_w * img_h)
    return float(box_area / img_area)


def count_touched_borders(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    margin: int = 4,
) -> int:
    touched = 0
    if x1 <= margin:
        touched += 1
    if y1 <= margin:
        touched += 1
    if x2 >= img_w - margin:
        touched += 1
    if y2 >= img_h - margin:
        touched += 1
    return touched


def crop_passes_quality_gate(crop_meta: Dict, img_w: int, img_h: int) -> Tuple[bool, Dict]:
    """
    Decide whether the YOLO crop is trustworthy enough to use.

    Returns:
        (use_crop: bool, quality_info: dict)
    """
    if not crop_meta.get("detected"):
        return False, {
            "use_crop": False,
            "reasons": ["no_detection"],
        }

    x1, y1, x2, y2 = crop_meta["bbox_xyxy"]
    conf = float(crop_meta.get("confidence", 0.0))

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    ar = float(w / h)
    area_ratio = bbox_area_ratio(x1, y1, x2, y2, img_w, img_h)
    touched_borders = count_touched_borders(x1, y1, x2, y2, img_w, img_h, margin=4)

    # Tunable thresholds
    min_conf = 0.50
    min_area_ratio = 0.10
    max_area_ratio = 0.75
    min_crop_ar = 0.50
    max_crop_ar = 2.20
    max_touched_borders = 1

    use_crop = True
    reasons = []

    if conf < min_conf:
        use_crop = False
        reasons.append(f"low_confidence<{min_conf}")

    if area_ratio < min_area_ratio:
        use_crop = False
        reasons.append(f"crop_too_small<{min_area_ratio}")

    if area_ratio > max_area_ratio:
        use_crop = False
        reasons.append(f"crop_too_large>{max_area_ratio}")

    if ar < min_crop_ar or ar > max_crop_ar:
        use_crop = False
        reasons.append(f"bad_crop_aspect_ratio={ar:.3f}")

    if touched_borders > max_touched_borders:
        use_crop = False
        reasons.append(f"touches_too_many_borders={touched_borders}")

    if not reasons:
        reasons.append("crop_accepted")

    return use_crop, {
        "use_crop": use_crop,
        "confidence": conf,
        "bbox_area_ratio": area_ratio,
        "crop_aspect_ratio": ar,
        "touched_borders": touched_borders,
        "reasons": reasons,
        "thresholds": {
            "min_conf": min_conf,
            "min_area_ratio": min_area_ratio,
            "max_area_ratio": max_area_ratio,
            "min_crop_ar": min_crop_ar,
            "max_crop_ar": max_crop_ar,
            "max_touched_borders": max_touched_borders,
        },
    }


# ============================================================
# OpenCLIP embedder
# ============================================================

class OpenCLIPEmbedder:
    def __init__(
        self,
        device: str,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def embed_pil(self, image: Image.Image) -> torch.Tensor:
        """
        Returns CPU tensor [D], normalized.
        """
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = l2_normalize_tensor(feat, dim=-1)
        return feat[0].detach().cpu()

    @torch.no_grad()
    def embed_paths(self, image_paths: List[Path], batch_size: int = 16) -> torch.Tensor:
        """
        Returns CPU tensor [N, D], normalized.
        """
        all_feats = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(self.preprocess(img))

            x = torch.stack(imgs, dim=0).to(self.device)
            feats = self.model.encode_image(x)
            feats = l2_normalize_tensor(feats, dim=-1)
            all_feats.append(feats.detach().cpu())

        return torch.cat(all_feats, dim=0)

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Returns CPU tensor [N, D], normalized.
        """
        all_feats = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = l2_normalize_tensor(feats, dim=-1)
            all_feats.append(feats.detach().cpu())

        return torch.cat(all_feats, dim=0)


# ============================================================
# Dataset collection
# ============================================================

def collect_products(data_root: Path, max_images_per_product: int = 1) -> List[Dict]:
    """
    Prefer exact primary images:
    001, 002, 003

    If more images are requested, fill remaining slots using the next
    numerically sorted images that were not already selected.

    This keeps primary images first, while still allowing more coverage
    when max_images_per_product > 3.
    """
    products = []

    preferred_names = [
        "001.jpg", "001.jpeg", "001.png", "001.webp",
        "002.jpg", "002.jpeg", "002.png", "002.webp",
        "003.jpg", "003.jpeg", "003.png", "003.webp",
    ]

    for model_dir in sorted(data_root.iterdir()):
        if not model_dir.is_dir():
            continue

        metadata_path = model_dir / "metadata.json"
        images_dir = model_dir / "images"

        if not metadata_path.exists() or not images_dir.exists():
            continue

        meta = load_json(metadata_path)
        title = meta.get("title", model_dir.name)
        item_no = meta.get("item_no")

        all_images = [p for p in images_dir.iterdir() if is_image(p)]
        if not all_images:
            continue

        all_images_sorted = sorted(all_images, key=lambda x: extract_numeric_prefix(x.name))
        name_to_path = {p.name.lower(): p for p in all_images_sorted}

        selected = []

        # 1) Prefer 001 / 002 / 003 first
        for name in preferred_names:
            p = name_to_path.get(name.lower())
            if p is not None and p not in selected:
                selected.append(p)
            if len(selected) >= max_images_per_product:
                break

        # 2) If caller wants more, fill from remaining sorted images
        if len(selected) < max_images_per_product:
            for p in all_images_sorted:
                if p not in selected:
                    selected.append(p)
                if len(selected) >= max_images_per_product:
                    break

        if not selected:
            continue

        products.append({
            "item_no": item_no,
            "title": title,
            "title_prompt": build_title_prompt(title),
            "model_dir": str(model_dir),
            "selected_images": [str(p) for p in selected],
        })

    return products



# ============================================================
# Build hybrid catalog
# ============================================================

def build_catalog(
    data_root: Path,
    out_dir: Path,
    device: str,
    batch_size: int = 16,
    max_images_per_product: int = 1,
    image_weight: float = 0.6,
    text_weight: float = 0.4,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if image_weight <= 0 or text_weight < 0:
        raise ValueError("image_weight must be > 0 and text_weight must be >= 0")
    if image_weight + text_weight <= 0:
        raise ValueError("sum of image_weight and text_weight must be > 0")

    print(f"[INFO] device={device}")
    print("[INFO] collecting products...")

    products = collect_products(
        data_root=data_root,
        max_images_per_product=max_images_per_product,
    )

    if not products:
        raise RuntimeError(f"No products found under {data_root}")

    print(f"[INFO] total products: {len(products)}")

    embedder = OpenCLIPEmbedder(
        device=device,
        model_name=model_name,
        pretrained=pretrained,
    )

    all_prompts = [p["title_prompt"] for p in products]
    print("[INFO] computing title text embeddings...")
    text_embs_cpu = embedder.embed_texts(all_prompts, batch_size=32)

    hybrid_vectors = []
    product_records = []

    for idx, product in enumerate(products, start=1):
        title = product["title"]
        item_no = product["item_no"]
        model_dir = product["model_dir"]
        selected_images = [Path(p) for p in product["selected_images"]]
        title_prompt = product["title_prompt"]

        print(f"[{idx}/{len(products)}] {title} ({len(selected_images)} images)")

        img_embs_cpu = embedder.embed_paths(selected_images, batch_size=batch_size)
        image_emb_cpu = img_embs_cpu.mean(dim=0, keepdim=True)
        image_emb_cpu = l2_normalize_tensor(image_emb_cpu, dim=-1)[0]

        text_emb_cpu = text_embs_cpu[idx - 1]

        hybrid = (image_weight * image_emb_cpu) + (text_weight * text_emb_cpu)
        hybrid = l2_normalize_tensor(hybrid.unsqueeze(0), dim=-1)[0]

        hybrid_vectors.append(hybrid)
        product_records.append({
            "item_no": item_no,
            "title": title,
            "title_prompt": title_prompt,
            "model_dir": model_dir,
            "selected_images": [str(p) for p in selected_images],
            "num_selected_images": len(selected_images),
        })

    embeddings_cpu = torch.stack(hybrid_vectors, dim=0)

    # Save CPU tensors for portability:
    # - build on MPS now
    # - load on CUDA later
    torch.save(embeddings_cpu, out_dir / "product_hybrid_embeddings.pt")

    build_meta = {
        "device_used_for_build": device,
        "model_name": model_name,
        "pretrained": pretrained,
        "num_products": len(product_records),
        "embedding_dim": int(embeddings_cpu.shape[1]),
        "max_images_per_product": max_images_per_product,
        "image_weight": image_weight,
        "text_weight": text_weight,
        "records": product_records,
    }
    save_json(build_meta, out_dir / "product_hybrid_metadata.json")

    print(f"[DONE] saved hybrid embeddings to {out_dir / 'product_hybrid_embeddings.pt'}")
    print(f"[DONE] saved metadata to {out_dir / 'product_hybrid_metadata.json'}")


# ============================================================
# YOLO cropper
# ============================================================

class YOLOCropper:
    def __init__(self, weights: str, device: str):
        self.weights = weights
        self.device = device
        self.model = YOLO(weights)

    def detect_best_crop(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        expand_ratio: float = 0.10,
    ) -> Tuple[Image.Image, Dict]:
        """
        Same code path works for:
        - MPS today
        - CUDA later
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return image, {
                "detected": False,
                "reason": "no_boxes",
            }

        best_idx = None
        best_area = -1.0

        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = result.boxes.xyxy[i].detach().cpu().tolist()
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best_idx = i

        if best_idx is None:
            return image, {
                "detected": False,
                "reason": "no_best_box",
            }

        x1, y1, x2, y2 = result.boxes.xyxy[best_idx].detach().cpu().tolist()
        cls_id = int(result.boxes.cls[best_idx].detach().cpu().item())
        conf = float(result.boxes.conf[best_idx].detach().cpu().item())

        w, h = image.size
        bw = x2 - x1
        bh = y2 - y1
        mx = bw * expand_ratio
        my = bh * expand_ratio

        x1 = max(0, int(x1 - mx))
        y1 = max(0, int(y1 - my))
        x2 = min(w, int(x2 + mx))
        y2 = min(h, int(y2 + my))

        crop = image.crop((x1, y1, x2, y2))

        return crop, {
            "detected": True,
            "class_id": cls_id,
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
            "device_used_for_yolo": self.device,
        }


# ============================================================
# Hybrid retrieval
# ============================================================

class TorchHybridProductSearch:
    def __init__(self, artifacts_dir: Path, device: str):
        self.artifacts_dir = artifacts_dir
        self.device = device

        self.meta = load_json(artifacts_dir / "product_hybrid_metadata.json")
        self.records = self.meta["records"]

        embeddings_cpu = torch.load(
            artifacts_dir / "product_hybrid_embeddings.pt",
            map_location="cpu",
        )

        # Main FAISS replacement tensor placement:
        # MPS now, CUDA later
        self.catalog = embeddings_cpu.to(self.device)
        self.catalog = l2_normalize_tensor(self.catalog, dim=-1)

    @torch.no_grad()
    def search(self, query_embedding_cpu: torch.Tensor, topk: int = 2) -> List[Dict]:
        q = query_embedding_cpu.unsqueeze(0).to(self.device)
        q = l2_normalize_tensor(q, dim=-1)

        scores = torch.matmul(self.catalog, q.T).squeeze(1)

        topk = min(topk, scores.shape[0])
        vals, idxs = torch.topk(scores, k=topk, dim=0)

        vals = vals.detach().cpu().tolist()
        idxs = idxs.detach().cpu().tolist()

        out = []
        for score, idx in zip(vals, idxs):
            rec = self.records[idx]
            out.append({
                "score": float(score),
                "item_no": rec.get("item_no"),
                "title": rec.get("title"),
                "title_prompt": rec.get("title_prompt"),
                "model_dir": rec.get("model_dir"),
                "selected_images": rec.get("selected_images"),
                "num_selected_images": rec.get("num_selected_images"),
            })

        return out


# ============================================================
# Search flow
# ============================================================

def run_search(
    artifacts_dir: Path,
    query_image: Path,
    device: str,
    yolo_weights: str | None,
    topk: int,
    save_crop: str | None,
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
) -> None:
    print(f"[INFO] device={device}")
    print("[INFO] loading query image...")
    img = Image.open(query_image).convert("RGB")
    img_w, img_h = img.size

    crop_meta = {
        "detected": False,
        "reason": "cropping_not_requested",
    }
    crop_quality = {
        "use_crop": False,
        "reasons": ["cropping_not_requested"],
    }
    query_for_embedding = img

    if yolo_weights:
        print(f"[INFO] running YOLO crop with {yolo_weights}...")
        cropper = YOLOCropper(weights=yolo_weights, device=device)
        crop, crop_meta = cropper.detect_best_crop(img)

        use_crop, crop_quality = crop_passes_quality_gate(crop_meta, img_w, img_h)

        if save_crop:
            crop.save(save_crop)
            print(f"[INFO] saved crop to {save_crop}")

        if use_crop:
            print("[INFO] YOLO crop accepted by quality gate; using crop for embedding.")
            query_for_embedding = crop
        else:
            print("[INFO] YOLO crop rejected by quality gate; falling back to original image.")
            query_for_embedding = img

    print("[INFO] embedding query image...")
    embedder = OpenCLIPEmbedder(
        device=device,
        model_name=clip_model_name,
        pretrained=clip_pretrained,
    )
    query_emb_cpu = embedder.embed_pil(query_for_embedding)

    print("[INFO] loading hybrid product catalog and searching...")
    searcher = TorchHybridProductSearch(artifacts_dir=artifacts_dir, device=device)
    hits = searcher.search(query_embedding_cpu=query_emb_cpu, topk=topk)

    output = {
        "device_used": device,
        "query_image": str(query_image),
        "crop_meta": crop_meta,
        "crop_quality": crop_quality,
        "query_embedding_source": "crop" if crop_quality.get("use_crop") else "original_image",
        "top_product_hits": hits,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--data-root", required=True)
    p_build.add_argument("--out-dir", required=True)
    p_build.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p_build.add_argument("--batch-size", type=int, default=16)
    p_build.add_argument("--max-images-per-product", type=int, default=1)
    p_build.add_argument("--image-weight", type=float, default=0.6)
    p_build.add_argument("--text-weight", type=float, default=0.4)
    p_build.add_argument("--clip-model-name", default="ViT-B-32")
    p_build.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")

    p_search = sub.add_parser("search")
    p_search.add_argument("--artifacts-dir", required=True)
    p_search.add_argument("--query-image", required=True)
    p_search.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p_search.add_argument("--yolo-weights", default=None)
    p_search.add_argument("--topk", type=int, default=2)
    p_search.add_argument("--save-crop", default=None)
    p_search.add_argument("--clip-model-name", default="ViT-B-32")
    p_search.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")

    args = parser.parse_args()
    device = get_best_device(args.device)

    if args.command == "build":
        build_catalog(
            data_root=Path(args.data_root),
            out_dir=Path(args.out_dir),
            device=device,
            batch_size=args.batch_size,
            max_images_per_product=args.max_images_per_product,
            image_weight=args.image_weight,
            text_weight=args.text_weight,
            model_name=args.clip_model_name,
            pretrained=args.clip_pretrained,
        )
    elif args.command == "search":
        run_search(
            artifacts_dir=Path(args.artifacts_dir),
            query_image=Path(args.query_image),
            device=device,
            yolo_weights=args.yolo_weights,
            topk=args.topk,
            save_crop=args.save_crop,
            clip_model_name=args.clip_model_name,
            clip_pretrained=args.clip_pretrained,
        )


if __name__ == "__main__":
    main()
