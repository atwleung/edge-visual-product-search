import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================

APP_TITLE = "Edge Visual Product Search"
APP_VERSION = "2.2.0"

BASE_DIR = Path(__file__).resolve().parent

# ---- artifacts ----
ARTIFACT_DIR = Path("/root/Desktop/image_search/artifacts_v21_hybrid")
EMBEDDINGS_PATH = ARTIFACT_DIR / "product_hybrid_embeddings.pt"
METADATA_PATH = ARTIFACT_DIR / "product_hybrid_metadata.json"

# ---- optional detector ----
YOLO_MODEL_PATH = BASE_DIR / "best.pt"

# ---- CLIP ----
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ---- retrieval / gating ----
TOP_K = 5
FINAL_TOP_K = 3
MIN_TOP1_SCORE = 0.52
MIN_TOP1_TOP2_MARGIN = 0.015

ENABLE_DOMAIN_GATING = True
ENABLE_CROPPING = True
YOLO_CONF_THRESHOLD = 0.25

# ---- input limits ----
MAX_IMAGE_BYTES = 12 * 1024 * 1024  # 12 MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

# ============================================================
# Domain mapping
# ============================================================

ALLOWED_QUERY_TO_RESULT_DOMAINS = {
    "aircraft": {"aircraft"},
    "tank": {"tank", "afv", "military_vehicle"},
    "warship": {"warship", "naval"},
    "car": {"car", "vehicle"},
    "train": {"train", "rail"},
}

YOLO_CLASSNAME_TO_DOMAIN = {
    "airplane": "aircraft",
    "plane": "aircraft",
    "aircraft": "aircraft",
    "tank": "tank",
    "ship": "warship",
    "boat": "warship",
    "warship": "warship",
    "car": "car",
    "vehicle": "car",
    "train": "train",
}

TITLE_KEYWORD_TO_DOMAIN = {
    "bf": "aircraft",
    "f-": "aircraft",
    "a6m": "aircraft",
    "spitfire": "aircraft",
    "nakajima": "aircraft",
    "panzer": "tank",
    "tiger": "tank",
    "yamato": "warship",
    "destroyer": "warship",
}

# ============================================================
# Device selection
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_NAME = "mps"
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "cpu"

USE_FP16 = DEVICE_NAME == "cuda"

# ============================================================
# App / globals
# ============================================================

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

clip_model: Optional[CLIPModel] = None
clip_processor: Optional[CLIPProcessor] = None
yolo_model: Optional[YOLO] = None

metadata: List[Dict[str, Any]] = []
db_embeddings: Optional[torch.Tensor] = None  # [N, D], normalized, on DEVICE

# ============================================================
# Response schema
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
# Utility helpers
# ============================================================

def now_ms() -> float:
    return time.perf_counter() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    return round(now_ms() - start_ms, 3)


def maybe_sync_device() -> None:
    if DEVICE_NAME == "cuda":
        torch.cuda.synchronize()


def pil_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def infer_domain_from_title(title: str) -> Optional[str]:
    t = title.lower()
    for keyword, domain in TITLE_KEYWORD_TO_DOMAIN.items():
        if keyword in t:
            return domain
    return None


def normalize_torch_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)


def normalize_torch_vec(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=0)


# ============================================================
# Loading helpers
# ============================================================

def load_embeddings_pt(path: Path) -> torch.Tensor:
    emb_obj = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(emb_obj, torch.Tensor):
        emb = emb_obj
    elif isinstance(emb_obj, dict):
        if "embeddings" in emb_obj and isinstance(emb_obj["embeddings"], torch.Tensor):
            emb = emb_obj["embeddings"]
        else:
            raise RuntimeError(
                f"Unsupported dict structure in {path}. "
                f"Expected key 'embeddings' containing a Tensor."
            )
    else:
        raise RuntimeError(f"Unsupported embedding format in {path}: {type(emb_obj)}")

    if emb.ndim != 2:
        raise RuntimeError(f"Embeddings must be 2D, got shape={tuple(emb.shape)}")

    return emb.float()


def load_metadata_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        metadata_obj = json.load(f)

    if isinstance(metadata_obj, list):
        records = metadata_obj
        print("[startup] metadata loaded as top-level list")
    elif isinstance(metadata_obj, dict):
        for key in ["records", "items", "products", "metadata", "data"]:
            value = metadata_obj.get(key)
            if isinstance(value, list):
                records = value
                print(f"[startup] metadata list found under key '{key}'")
                break
        else:
            raise RuntimeError(
                f"Metadata JSON is a dict but no known list field was found. "
                f"Top-level keys: {list(metadata_obj.keys())}"
            )
    else:
        raise RuntimeError(
            f"Unsupported metadata JSON type: {type(metadata_obj).__name__}"
        )

    if not all(isinstance(x, dict) for x in records):
        raise RuntimeError("Metadata records must all be JSON objects")

    return records


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event() -> None:
    global clip_model, clip_processor, yolo_model, metadata, db_embeddings

    print(f"[startup] device={DEVICE_NAME}")
    print(f"[startup] embeddings_path={EMBEDDINGS_PATH}")
    print(f"[startup] metadata_path={METADATA_PATH}")
    print(f"[startup] yolo_path={YOLO_MODEL_PATH}")

    # ---- CLIP ----
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_model.to(DEVICE)
    clip_model.eval()

    if USE_FP16:
        clip_model = clip_model.half()

    # ---- YOLO (optional) ----
    if YOLO_MODEL_PATH.exists():
        yolo_model = YOLO(str(YOLO_MODEL_PATH))
        print(f"[startup] YOLO loaded from {YOLO_MODEL_PATH}")
    else:
        yolo_model = None
        print(f"[startup] WARNING: YOLO model not found at {YOLO_MODEL_PATH}")

    # ---- metadata ----
    if not METADATA_PATH.exists():
        raise RuntimeError(f"Metadata not found: {METADATA_PATH}")

    metadata = load_metadata_records(METADATA_PATH)

    # ---- embeddings ----
    if not EMBEDDINGS_PATH.exists():
        raise RuntimeError(f"Embeddings not found: {EMBEDDINGS_PATH}")

    emb = load_embeddings_pt(EMBEDDINGS_PATH)

    if emb.shape[0] != len(metadata):
        raise RuntimeError(
            f"Embeddings rows ({emb.shape[0]}) do not match metadata length ({len(metadata)})"
        )

    emb = normalize_torch_rows(emb)
    db_embeddings = emb.to(DEVICE)

    print(f"[startup] metadata_count={len(metadata)}")
    print(f"[startup] embeddings_shape={tuple(db_embeddings.shape)}")
    print("[startup] ready")

# ============================================================
# Image helpers
# ============================================================

def load_image_from_upload(upload: UploadFile, raw_bytes: bytes) -> Image.Image:
    if upload.content_type and upload.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {upload.content_type}",
        )

    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw_bytes)} bytes). Max allowed is {MAX_IMAGE_BYTES} bytes.",
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        return pil_to_rgb(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def detect_and_crop_best(
    image: Image.Image,
) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any], Optional[str]]:
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

    if DEVICE_NAME == "cuda":
        device_arg: Any = 0
    elif DEVICE_NAME == "mps":
        device_arg = "mps"
    else:
        device_arg = "cpu"

    results = yolo_model.predict(
        source=image,
        conf=YOLO_CONF_THRESHOLD,
        verbose=False,
        device=device_arg,
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

    names = r.names if hasattr(r, "names") else {}

    best_conf = -1.0
    best_cls_name = None
    best_xyxy = None

    for box in boxes:
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        cls_name = names.get(cls_id, str(cls_id))
        xyxy = box.xyxy[0].tolist()

        if conf > best_conf:
            best_conf = conf
            best_cls_name = cls_name
            best_xyxy = xyxy

    if best_xyxy is None:
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

# ============================================================
# Embedding / retrieval
# ============================================================

@torch.no_grad()
def embed_image(image: Image.Image) -> torch.Tensor:
    assert clip_model is not None
    assert clip_processor is not None

    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    if USE_FP16:
        pixel_values = pixel_values.half()

    maybe_sync_device()
    image_features = clip_model.get_image_features(pixel_values=pixel_values)
    maybe_sync_device()

    vec = image_features[0].detach().float()
    vec = normalize_torch_vec(vec)
    return vec


@torch.no_grad()
def search_torch(query_vec: torch.Tensor, top_k: int = TOP_K) -> Tuple[List[float], List[int]]:
    assert db_embeddings is not None

    maybe_sync_device()
    scores = torch.matmul(db_embeddings, query_vec)
    k = min(top_k, int(scores.shape[0]))
    topk_scores, topk_indices = torch.topk(scores, k=k)
    maybe_sync_device()

    return (
        topk_scores.detach().float().cpu().tolist(),
        topk_indices.detach().cpu().tolist(),
    )


def build_hits(scores: List[float], indices: List[int]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []

    for score, idx in zip(scores, indices):
        idx = int(idx)
        if idx < 0 or idx >= len(metadata):
            continue

        item = metadata[idx]
        hit = {
            "score": safe_float(score),
            "item_no": item.get("item_no"),
            "title": item.get("title"),
            "domain": item.get("domain"),
            "metadata_index": idx,
            "filename": item.get("filename"),
            "path": item.get("path"),
        }
        hits.append(hit)

    return hits

# ============================================================
# Gating
# ============================================================

def apply_domain_gating(
    hits: List[Dict[str, Any]],
    query_domain: Optional[str],
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

        if not hit_domain and hit.get("title"):
            hit_domain = infer_domain_from_title(hit["title"])

        if hit_domain in allowed_domains:
            filtered.append(hit)
        else:
            rejected_count += 1

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

    matched = True
    reasons = []

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
        "device": DEVICE_NAME,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    emb_shape = tuple(db_embeddings.shape) if db_embeddings is not None else None
    return {
        "ok": True,
        "device": DEVICE_NAME,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "metadata_count": len(metadata),
        "embeddings_shape": emb_shape,
        "yolo_loaded": yolo_model is not None,
        "clip_loaded": clip_model is not None,
    }


@app.post("/search", response_model=SearchResponse)
async def search_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    t0 = now_ms()
    timings: Dict[str, float] = {}

    t = now_ms()
    raw = await image.read()
    timings["read_upload"] = elapsed_ms(t)

    t = now_ms()
    pil_image = load_image_from_upload(image, raw)
    timings["decode_image"] = elapsed_ms(t)

    t = now_ms()
    chosen_image, crop_meta, crop_quality, query_domain = detect_and_crop_best(pil_image)
    maybe_sync_device()
    timings["detect_and_crop"] = elapsed_ms(t)

    t = now_ms()
    query_vec = embed_image(chosen_image)
    timings["clip_embedding"] = elapsed_ms(t)

    t = now_ms()
    scores, indices = search_torch(query_vec, top_k=TOP_K)
    timings["torch_search"] = elapsed_ms(t)

    t = now_ms()
    hits = build_hits(scores, indices)
    timings["build_hits"] = elapsed_ms(t)

    t = now_ms()
    gated_hits, domain_gate = apply_domain_gating(hits, query_domain)
    timings["domain_gating"] = elapsed_ms(t)

    t = now_ms()
    confidence_gate = apply_confidence_gate(gated_hits)
    timings["confidence_gating"] = elapsed_ms(t)

    timings["total"] = elapsed_ms(t0)

    response_hits = gated_hits[:FINAL_TOP_K]

    print(
        f"[timing] total={timings['total']} ms, "
        f"detect={timings['detect_and_crop']} ms, "
        f"clip={timings['clip_embedding']} ms, "
        f"search={timings['torch_search']} ms, "
        f"device={DEVICE_NAME}, matched={confidence_gate['matched']}"
    )

    return {
        "matched": confidence_gate["matched"],
        "device_used": DEVICE_NAME,
        "crop_meta": crop_meta,
        "crop_quality": crop_quality,
        "query_embedding_source": "cropped_detection" if crop_quality.get("use_crop") else "original_image",
        "query_domain": query_domain,
        "domain_gate": domain_gate,
        "confidence_gate": confidence_gate,
        "top_product_hits": response_hits,
        "timings_ms": timings,
    }