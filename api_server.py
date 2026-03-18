import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================

APP_TITLE = "YOLO + CLIP Retrieval API"
APP_VERSION = "1.0.0"

BASE_DIR = Path(__file__).resolve().parent

YOLO_MODEL_PATH = BASE_DIR / "best.pt"
FAISS_INDEX_PATH = BASE_DIR / "index.faiss"
METADATA_PATH = BASE_DIR / "metadata.json"

# Hugging Face CLIP model
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Retrieval params
TOP_K = 5
FINAL_TOP_K = 3

# Confidence gate
MIN_TOP1_SCORE = 0.52
MIN_TOP1_TOP2_MARGIN = 0.015

# Domain gating
ENABLE_DOMAIN_GATING = True

# Detection
ENABLE_CROPPING = True
YOLO_CONF_THRESHOLD = 0.25

# Input image limits
MAX_IMAGE_BYTES = 12 * 1024 * 1024  # 12 MB

# Allowed upload content-types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

# Optional domain label mapping:
# metadata item may contain "domain"
# e.g. {"title":"...", "item_no":"...", "domain":"aircraft"}
ALLOWED_QUERY_TO_RESULT_DOMAINS = {
    # detected_query_domain: set of allowed candidate domains
    "aircraft": {"aircraft"},
    "tank": {"tank", "afv", "military_vehicle"},
    "warship": {"warship", "naval"},
    "car": {"car", "vehicle"},
    "train": {"train", "rail"},
}

# If YOLO class names map to business domains, put them here
YOLO_CLASSNAME_TO_DOMAIN = {
    "airplane": "aircraft",
    "plane": "aircraft",
    "tank": "tank",
    "ship": "warship",
    "boat": "warship",
    "car": "car",
    "train": "train",
}

# If you have a fixed allowed title keyword-based heuristic fallback:
TITLE_KEYWORD_TO_DOMAIN = {
    "Bf": "aircraft",
    "F-": "aircraft",
    "A6M": "aircraft",
    "Spitfire": "aircraft",
    "Tiger": "tank",
    "Panzer": "tank",
    "Yamato": "warship",
    "Destroyer": "warship",
}


# ============================================================
# Utilities
# ============================================================

def now_ms() -> float:
    return time.perf_counter() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    return round(now_ms() - start_ms, 3)


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def pil_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def normalize_embedding(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def infer_domain_from_title(title: str) -> Optional[str]:
    for keyword, domain in TITLE_KEYWORD_TO_DOMAIN.items():
        if keyword.lower() in title.lower():
            return domain
    return None


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# ============================================================
# Response models
# ============================================================

class SearchResponse(BaseModel):
    matched: bool
    device_used: str
    crop_meta: Dict[str, Any]
    crop_quality: Dict[str, Any]
    query_embedding_source: str
    query_domain: Optional[str]
    domain_gate: Dict[str, Any]
    confidence_gate: Dict[str, Any]
    top_product_hits: List[Dict[str, Any]]
    timings_ms: Dict[str, float]


# ============================================================
# App / global state
# ============================================================

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model: Optional[CLIPModel] = None
clip_processor: Optional[CLIPProcessor] = None
yolo_model: Optional[YOLO] = None
faiss_index = None
metadata: List[Dict[str, Any]] = []


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event() -> None:
    global clip_model, clip_processor, yolo_model, faiss_index, metadata

    print(f"[startup] device={device}")

    # Load CLIP
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_model.to(device)
    clip_model.eval()

    # Optional FP16 on CUDA
    if device == "cuda":
        clip_model = clip_model.half()

    # Load YOLO
    if YOLO_MODEL_PATH.exists():
        yolo_model = YOLO(str(YOLO_MODEL_PATH))
    else:
        print(f"[startup] WARNING: YOLO model not found at {YOLO_MODEL_PATH}")
        yolo_model = None

    # Load FAISS
    if not FAISS_INDEX_PATH.exists():
        raise RuntimeError(f"FAISS index not found: {FAISS_INDEX_PATH}")
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

    # Load metadata
    if not METADATA_PATH.exists():
        raise RuntimeError(f"Metadata not found: {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[startup] metadata_count={len(metadata)}")
    print("[startup] ready")


# ============================================================
# Core logic
# ============================================================

def load_image_from_upload(upload: UploadFile, raw_bytes: bytes) -> Image.Image:
    if upload.content_type and upload.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {upload.content_type}"
        )

    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw_bytes)} bytes). Max allowed is {MAX_IMAGE_BYTES} bytes."
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image = pil_to_rgb(image)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def detect_and_crop_best(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any], Optional[str]]:
    """
    Returns:
        chosen_image,
        crop_meta,
        crop_quality,
        query_domain
    """
    if not ENABLE_CROPPING:
        return (
            image,
            {"detected": False, "reason": "cropping_disabled"},
            {"use_crop": False, "reasons": ["cropping_disabled"]},
            None,
        )

    if yolo_model is None:
        return (
            image,
            {"detected": False, "reason": "yolo_model_not_loaded"},
            {"use_crop": False, "reasons": ["yolo_model_not_loaded"]},
            None,
        )

    results = yolo_model.predict(
        source=np.array(image),
        conf=YOLO_CONF_THRESHOLD,
        verbose=False,
        device=0 if device == "cuda" else "cpu",
    )

    if not results:
        return (
            image,
            {"detected": False, "reason": "no_result"},
            {"use_crop": False, "reasons": ["no_result"]},
            None,
        )

    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return (
            image,
            {"detected": False, "reason": "no_detection"},
            {"use_crop": False, "reasons": ["no_detection"]},
            None,
        )

    best_idx = None
    best_conf = -1.0
    best_cls_name = None
    best_xyxy = None

    names = r.names if hasattr(r, "names") else {}

    for i, box in enumerate(boxes):
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        cls_name = names.get(cls_id, str(cls_id))
        xyxy = box.xyxy[0].tolist()

        if conf > best_conf:
            best_conf = conf
            best_idx = i
            best_cls_name = cls_name
            best_xyxy = xyxy

    if best_idx is None or best_xyxy is None:
        return (
            image,
            {"detected": False, "reason": "best_box_not_found"},
            {"use_crop": False, "reasons": ["best_box_not_found"]},
            None,
        )

    x1, y1, x2, y2 = [int(v) for v in best_xyxy]
    width, height = image.size

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    crop = image.crop((x1, y1, x2, y2))
    crop_w, crop_h = crop.size
    crop_area = crop_w * crop_h
    full_area = width * height
    crop_ratio = crop_area / full_area if full_area > 0 else 0.0

    reasons = []
    use_crop = True

    if crop_ratio < 0.03:
        use_crop = False
        reasons.append("crop_too_small")

    if crop_w < 32 or crop_h < 32:
        use_crop = False
        reasons.append("crop_dimensions_too_small")

    crop_meta = {
        "detected": True,
        "confidence": round(best_conf, 6),
        "bbox_xyxy": [x1, y1, x2, y2],
        "class_name": best_cls_name,
        "crop_ratio": round(crop_ratio, 6),
        "crop_size": [crop_w, crop_h],
    }

    crop_quality = {
        "use_crop": use_crop,
        "reasons": reasons if reasons else ["passed_crop_quality_gate"],
    }

    query_domain = YOLO_CLASSNAME_TO_DOMAIN.get(str(best_cls_name).lower())

    return (crop if use_crop else image, crop_meta, crop_quality, query_domain)


@torch.no_grad()
def embed_image(image: Image.Image) -> np.ndarray:
    assert clip_model is not None
    assert clip_processor is not None

    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    if device == "cuda":
        pixel_values = pixel_values.half()

    image_features = clip_model.get_image_features(pixel_values=pixel_values)
    vec = image_features[0].detach().float().cpu().numpy().astype(np.float32)
    vec = normalize_embedding(vec)
    return vec


def search_faiss(query_vec: np.ndarray, top_k: int = TOP_K) -> Tuple[np.ndarray, np.ndarray]:
    assert faiss_index is not None
    query = np.expand_dims(query_vec.astype(np.float32), axis=0)
    scores, indices = faiss_index.search(query, top_k)
    return scores[0], indices[0]


def build_hits(scores: np.ndarray, indices: np.ndarray) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(metadata):
            continue

        item = metadata[idx]
        hit = {
            "score": safe_float(score),
            "item_no": item.get("item_no"),
            "title": item.get("title"),
            "domain": item.get("domain"),
            "metadata_index": int(idx),
        }
        hits.append(hit)
    return hits


def apply_domain_gating(
    hits: List[Dict[str, Any]],
    query_domain: Optional[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not ENABLE_DOMAIN_GATING:
        return hits, {
            "enabled": False,
            "query_domain": query_domain,
            "allowed_domains": None,
            "reasons": ["domain_gating_disabled"],
        }

    if not query_domain:
        return hits, {
            "enabled": True,
            "query_domain": None,
            "allowed_domains": None,
            "reasons": ["query_domain_unknown_no_filter_applied"],
        }

    allowed_domains = ALLOWED_QUERY_TO_RESULT_DOMAINS.get(query_domain)
    if not allowed_domains:
        return hits, {
            "enabled": True,
            "query_domain": query_domain,
            "allowed_domains": None,
            "reasons": ["no_mapping_for_query_domain_no_filter_applied"],
        }

    filtered: List[Dict[str, Any]] = []
    rejected_count = 0

    for hit in hits:
        hit_domain = hit.get("domain")

        # fallback heuristic from title if metadata domain missing
        if not hit_domain and hit.get("title"):
            hit_domain = infer_domain_from_title(hit["title"])

        if hit_domain in allowed_domains:
            filtered.append(hit)
        else:
            rejected_count += 1

    # fallback: if everything filtered out, keep original hits
    if not filtered:
        return hits, {
            "enabled": True,
            "query_domain": query_domain,
            "allowed_domains": sorted(list(allowed_domains)),
            "reasons": ["all_hits_filtered_out_fallback_to_original_hits"],
            "rejected_count": rejected_count,
        }

    return filtered, {
        "enabled": True,
        "query_domain": query_domain,
        "allowed_domains": sorted(list(allowed_domains)),
        "reasons": ["domain_filter_applied"],
        "rejected_count": rejected_count,
    }


def apply_confidence_gate(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not hits:
        return {
            "matched": False,
            "top1_score": None,
            "top2_score": None,
            "score_margin": None,
            "min_top1_score": MIN_TOP1_SCORE,
            "min_top1_top2_margin": MIN_TOP1_TOP2_MARGIN,
            "reasons": ["no_hits"],
        }

    top1 = safe_float(hits[0].get("score"))
    top2 = safe_float(hits[1].get("score")) if len(hits) > 1 else None
    margin = None if (top1 is None or top2 is None) else (top1 - top2)

    reasons = []
    matched = True

    if top1 is None:
        matched = False
        reasons.append("top1_missing")
    else:
        if top1 < MIN_TOP1_SCORE:
            matched = False
            reasons.append("top1_below_threshold")

    if top2 is not None and margin is not None:
        if margin < MIN_TOP1_TOP2_MARGIN:
            matched = False
            reasons.append("top1_top2_margin_too_small")

    if matched:
        reasons.append("passed_confidence_gate")

    return {
        "matched": matched,
        "top1_score": round(top1, 6) if top1 is not None else None,
        "top2_score": round(top2, 6) if top2 is not None else None,
        "score_margin": round(margin, 6) if margin is not None else None,
        "min_top1_score": MIN_TOP1_SCORE,
        "min_top1_top2_margin": MIN_TOP1_TOP2_MARGIN,
        "reasons": reasons,
    }


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "title": APP_TITLE,
        "version": APP_VERSION,
        "device": device,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "metadata_count": len(metadata),
        "yolo_loaded": yolo_model is not None,
        "clip_loaded": clip_model is not None,
        "faiss_loaded": faiss_index is not None,
    }


@app.post("/search", response_model=SearchResponse)
async def search_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    t0 = now_ms()
    timings: Dict[str, float] = {}

    # --------------------------------------------------------
    # Read upload
    # --------------------------------------------------------
    t = now_ms()
    raw = await image.read()
    timings["read_upload"] = elapsed_ms(t)

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------
    t = now_ms()
    pil_image = load_image_from_upload(image, raw)
    timings["decode_image"] = elapsed_ms(t)

    # --------------------------------------------------------
    # Detect / crop
    # --------------------------------------------------------
    t = now_ms()
    chosen_image, crop_meta, crop_quality, query_domain = detect_and_crop_best(pil_image)
    timings["detect_and_crop"] = elapsed_ms(t)

    # --------------------------------------------------------
    # Embed query
    # --------------------------------------------------------
    t = now_ms()
    query_vec = embed_image(chosen_image)
    timings["clip_embedding"] = elapsed_ms(t)

    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------
    t = now_ms()
    scores, indices = search_faiss(query_vec, TOP_K)
    timings["faiss_search"] = elapsed_ms(t)

    # --------------------------------------------------------
    # Build hits
    # --------------------------------------------------------
    t = now_ms()
    hits = build_hits(scores, indices)
    timings["build_hits"] = elapsed_ms(t)

    # --------------------------------------------------------
    # Domain gate
    # --------------------------------------------------------
    t = now_ms()
    gated_hits, domain_gate = apply_domain_gating(hits, query_domain)
    timings["domain_gating"] = elapsed_ms(t)

    # --------------------------------------------------------
    # Confidence gate
    # --------------------------------------------------------
    t = now_ms()
    confidence_gate = apply_confidence_gate(gated_hits)
    timings["confidence_gating"] = elapsed_ms(t)

    timings["total"] = elapsed_ms(t0)

    response_hits = gated_hits[:FINAL_TOP_K]

    return {
        "matched": confidence_gate["matched"],
        "device_used": device,
        "crop_meta": crop_meta,
        "crop_quality": crop_quality,
        "query_embedding_source": "cropped_detection" if crop_quality.get("use_crop") else "original_image",
        "query_domain": query_domain,
        "domain_gate": domain_gate,
        "confidence_gate": confidence_gate,
        "top_product_hits": response_hits,
        "timings_ms": timings,
    }