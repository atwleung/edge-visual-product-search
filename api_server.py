import io
import json
import asyncio
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import open_clip


# ============================================================
# Config
# ============================================================

ARTIFACTS_DIR = Path("/root/Desktop/image_search/artifacts_v21_hybrid")
DEFAULT_DEVICE = "cuda"
DEFAULT_TOPK = 2
DEFAULT_MIN_TOP1_SCORE = 0.58
DEFAULT_MIN_TOP1_TOP2_MARGIN = 0.03
DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"


# ============================================================
# Helpers
# ============================================================

def get_best_device(prefer: str = "auto") -> str:
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


def l2_normalize_tensor(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim)


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
    }


def evaluate_match_confidence(
    hits: List[Dict],
    min_top1_score: float = DEFAULT_MIN_TOP1_SCORE,
    min_top1_top2_margin: float = DEFAULT_MIN_TOP1_TOP2_MARGIN,
) -> Tuple[bool, Dict]:
    if not hits:
        return False, {
            "matched": False,
            "reasons": ["no_hits"],
            "min_top1_score": min_top1_score,
            "min_top1_top2_margin": min_top1_top2_margin,
        }

    top1_score = float(hits[0]["score"])
    top2_score = float(hits[1]["score"]) if len(hits) > 1 else None
    score_margin = (top1_score - top2_score) if top2_score is not None else None

    matched = True
    reasons = []

    high_confidence_override = 0.72

    if top1_score < min_top1_score:
        matched = False
        reasons.append("top1_below_threshold")

    if matched and top1_score < high_confidence_override:
        if top2_score is not None and score_margin < min_top1_top2_margin:
            matched = False
            reasons.append("top1_top2_margin_below_threshold")
    elif matched and top1_score >= high_confidence_override:
        reasons.append("high_top1_score_override")
    
    if matched:
        reasons.append("passed_confidence_gate")

    return matched, {
        "matched": matched,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "score_margin": score_margin,
        "min_top1_score": min_top1_score,
        "min_top1_top2_margin": min_top1_top2_margin,
        "reasons": reasons,
    }


def cuda_sync_if_needed(device: Optional[str]) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    def __init__(self, sync_device: Optional[str] = None):
        self.sync_device = sync_device
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.stages = OrderedDict()

    def mark(self, name: str, sync: bool = False) -> None:
        if sync:
            cuda_sync_if_needed(self.sync_device)
        now = time.perf_counter()
        self.stages[name] = round((now - self._last) * 1000.0, 3)
        self._last = now

    def total_ms(self, sync: bool = False) -> float:
        if sync:
            cuda_sync_if_needed(self.sync_device)
        return round((time.perf_counter() - self._t0) * 1000.0, 3)

    def as_dict(self, sync_total: bool = False) -> Dict[str, float]:
        d = dict(self.stages)
        d["total"] = self.total_ms(sync=sync_total)
        return d


# ============================================================
# Models
# ============================================================

class OpenCLIPEmbedder:
    def __init__(
        self,
        device: str,
        model_name: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
    ):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )
        self.model.eval().to(self.device)

    @torch.no_grad()
    def embed_pil(self, image: Image.Image) -> torch.Tensor:
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = l2_normalize_tensor(feat, dim=-1)
        return feat[0].detach().cpu()


class YOLOCropper:
    def __init__(self, weights: str, device: str):
        self.device = device
        self.model = YOLO(weights)

    def detect_best_crop(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        expand_ratio: float = 0.10,
    ) -> Tuple[Image.Image, Dict]:
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


class TorchHybridProductSearch:
    def __init__(self, artifacts_dir: Path, device: str):
        self.device = device

        with open(artifacts_dir / "product_hybrid_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.records = meta["records"]

        embeddings_cpu = torch.load(
            artifacts_dir / "product_hybrid_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )

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

        hits = []
        for score, idx in zip(vals, idxs):
            rec = self.records[int(idx)]
            hits.append({
                "score": float(score),
                "item_no": rec.get("item_no"),
                "title": rec.get("title"),
                "title_prompt": rec.get("title_prompt"),
                "model_dir": rec.get("model_dir"),
                "selected_images": rec.get("selected_images"),
                "num_selected_images": rec.get("num_selected_images"),
            })
        return hits


# ============================================================
# App state
# ============================================================

app = FastAPI(title="edge-visual-product-search API", version="0.1.0")

gpu_lock = asyncio.Lock()

STATE = {
    "device": None,
    "embedder": None,
    "cropper": None,
    "searcher": None,
}


@app.on_event("startup")
def startup_event():
    device = get_best_device(DEFAULT_DEVICE)

    STATE["device"] = device
    STATE["embedder"] = OpenCLIPEmbedder(device=device)
    STATE["searcher"] = TorchHybridProductSearch(
        artifacts_dir=ARTIFACTS_DIR,
        device=device,
    )

    STATE["cropper"] = YOLOCropper(
        weights=DEFAULT_YOLO_WEIGHTS,
        device=device,
    )

    print(f"[STARTUP] device={device}")
    print(f"[STARTUP] artifacts_dir={ARTIFACTS_DIR}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": STATE["device"],
    }


@app.post("/search")
async def search_image(
    image: UploadFile = File(...),
    use_yolo: bool = Form(False),
    topk: int = Form(DEFAULT_TOPK),
    min_top1_score: float = Form(DEFAULT_MIN_TOP1_SCORE),
    min_top1_top2_margin: float = Form(DEFAULT_MIN_TOP1_TOP2_MARGIN),
):
    timer = StageTimer(sync_device=STATE["device"])

    if gpu_lock.locked():
        raise HTTPException(status_code=429, detail="GPU busy, try again shortly.")

    raw = await image.read()
    timer.mark("read_upload")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    timer.mark("decode_image")

    async with gpu_lock:
        crop_meta = {
            "detected": False,
            "reason": "cropping_not_requested",
        }
        crop_quality = {
            "use_crop": False,
            "reasons": ["cropping_not_requested"],
        }
        query_for_embedding = pil_img

        if use_yolo:
            crop, crop_meta = STATE["cropper"].detect_best_crop(pil_img)
            timer.mark("yolo_detect", sync=True)

            use_crop_ok, crop_quality = crop_passes_quality_gate(
                crop_meta,
                pil_img.size[0],
                pil_img.size[1],
            )
            timer.mark("crop_quality_gate")

            if use_crop_ok:
                query_for_embedding = crop
        else:
            timer.mark("yolo_detect")
            timer.mark("crop_quality_gate")

        query_emb_cpu = STATE["embedder"].embed_pil(query_for_embedding)
        timer.mark("embed_image", sync=True)

        hits = STATE["searcher"].search(
            query_embedding_cpu=query_emb_cpu,
            topk=topk,
        )
        timer.mark("vector_search", sync=True)

        matched, confidence_gate = evaluate_match_confidence(
            hits=hits,
            min_top1_score=min_top1_score,
            min_top1_top2_margin=min_top1_top2_margin,
        )
        timer.mark("confidence_gate")

        result = {
            "matched": matched,
            "device_used": STATE["device"],
            "crop_meta": crop_meta,
            "crop_quality": crop_quality,
            "query_embedding_source": "crop" if crop_quality.get("use_crop") else "original_image",
            "confidence_gate": confidence_gate,
            "top_product_hits": hits if matched else [],
            "candidate_hits_debug": hits,
        }

        timer.mark("build_response")
        result["timing_ms"] = timer.as_dict(sync_total=True)

        headers = {
            "Server-Timing": ", ".join(
                f"{k};dur={v}" for k, v in result["timing_ms"].items()
            )
        }

        return JSONResponse(result, headers=headers)
