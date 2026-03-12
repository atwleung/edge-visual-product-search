# edge-visual-product-search

A lightweight **visual product search pipeline** using:

- **YOLOv8** for optional object detection
- **OpenCLIP** for image + text embeddings
- **Hybrid product embeddings (image + title)**
- **Mac M1 / M2 / M3 GPU acceleration via MPS**
- Easy migration to **CUDA edge GPUs (RTX 4000 / RTX 6000)**

This repository demonstrates a practical architecture for **eCommerce visual search** and **edge inference workloads**.

---

# Architecture

![edge-visual-product-search architecture](docs/edge_visual_product_search_architecture.png)

Pipeline:
Query Image
│
▼
Optional YOLO Detection
│
▼
Crop Quality Gate
│
▼
CLIP Image Embedding
│
▼
Vector Similarity Search
│
▼
Top-K Product Matches


Catalog products are embedded using a **hybrid vector**:

product_vector = 0.6 × image_embedding + 0.4 × title_embedding





This improves semantic grouping (for example distinguishing **F-14 vs F-4 vs Space Shuttle**).

---

# Key Design Ideas

## Hybrid Product Embeddings

Each product embedding combines:

- **visual signal** from the product image
- **semantic signal** from the product title

This helps differentiate visually similar products.

Example:

F-14 Tomcat vs F-4 Phantom look similar visually, but title embeddings help separate them.

---

## Example Queries

| Query | Description |
|------|-------------|
| ![](queries/f14.jpg) | F-14 Tomcat aircraft |
| ![](queries/shuttle2.jpg) | Space Shuttle Atlantis |

## YOLO Crop Quality Gate

YOLO detection is **optional** and guarded by a quality gate.

Loose bounding boxes can degrade retrieval performance.

The crop is rejected if:

- bounding box covers too much of the image
- bounding box touches image borders
- aspect ratio looks unreasonable
- detection confidence is low

If rejected, the system **falls back to the original image**.

---

## Edge-Friendly Architecture

This pipeline works well on **edge GPUs** because:

- YOLO inference is lightweight
- CLIP embedding models are small
- vector search is fast
- no large language model required

Additional technical discussion is available in the documentation:

- [Why This Architecture Works Well for Edge Inference](docs/edge_inference_architecture.md)

The same code supports:

Mac GPU:

```text
--device mps
```

NVIDIA GPU:

```text
--device cuda
```

---

# Installation

Requires **Python 3.9+**

Install dependencies:


pip install -r requirements.txt


Example requirements:


torch
torchvision
ultralytics
open_clip_torch
Pillow
numpy

---

# Build Product Embeddings

This project builds hybrid product embeddings for visual retrieval.

Each product is represented by a vector:

product_vector = 0.6 × image_embedding + 0.4 × title_embedding

Image embeddings come from OpenCLIP, and title embeddings help separate visually similar products.

The process involves three steps:

Download product images

Clean the dataset using YOLO

Generate hybrid embeddings

## Step 1 — Download Example Dataset

For demonstration purposes we use Tamiya military aircraft model kits.

Run the downloader:

python scripts/download_tamiya_aircraft.py \
  --out-dir data/tamiya_aircraft \
  --max-models 100

Example directory structure:

data/
  tamiya_aircraft/
    GRUMMAN_F-14A_TOMCAT__ITEM_61029/
      metadata.json
      images/
        001.jpg
        002.jpg
        003.jpg

Each product directory contains:

product title

item number

multiple product images

## Step 2 — Prune Non-Aircraft Images

The Tamiya website includes many images that are not useful for retrieval, such as:

paint color charts

decals

parts sprues

banner images

catalog graphics

We use YOLOv8 to remove images that are not aircraft.

Run:

python scripts/prune_non_aircraft_images_v2.py \
  --data-root data/tamiya_aircraft \
  --yolo-weights yolov8n.pt \
  --device mps

This script:

detects objects in each image

keeps images containing airplanes

removes images containing unrelated objects

rejects banner-style images

This improves retrieval quality significantly.

## Step 3 — Build Hybrid Product Embeddings

Once the dataset is cleaned, generate product embeddings.

python mps_clip_retrieval_v21_hybrid.py build \
  --data-root ./data/tamiya_aircraft \
  --out-dir ./artifacts_v21_hybrid \
  --device mps \
  --max-images-per-product 1 \
  --image-weight 0.6 \
  --text-weight 0.4

Parameters:

Parameter	Description
--data-root	product catalog directory
--out-dir	embedding artifacts
--device	mps (Mac GPU) or cuda (NVIDIA GPU)
--max-images-per-product	number of representative images
--image-weight	weight of image embedding
--text-weight	weight of title embedding

Example output:

artifacts_v21_hybrid/
  product_vectors.npy
  catalog_metadata.json

Each product is represented by a single vector used for retrieval.

## Why Use Product-Level Embeddings?

Instead of indexing every product image separately, this project creates one embedding per product.

Advantages:

smaller vector index

faster search

better semantic grouping

easier catalog updates

When a new SKU is added:

add product images

generate embedding

append to vector index

No retraining is required.

---

# Search

### Without YOLO

python mps_clip_retrieval_v21_hybrid.py search --artifacts-dir ./artifacts_v21_hybrid --query-image ./queries/f14.jpg --device mps


[INFO] device=mps
[INFO] loading query image...
[INFO] embedding query image...
[INFO] loading hybrid product catalog and searching...
{
  "device_used": "mps",
  "query_image": "queries/f14.jpg",
  "crop_meta": {
    "detected": false,
    "reason": "cropping_not_requested"
  },
  "crop_quality": {
    "use_crop": false,
    "reasons": [
      "cropping_not_requested"
    ]
  },
  "query_embedding_source": "original_image",
  "top_product_hits": [
    {
      "score": 0.7104616761207581,
      "item_no": "12693",
      "title": "GRUMMAN F-14A TOMCAT (LATE MODEL) CARRIER LAUNCH SET",
      "title_prompt": "a scale model kit product named GRUMMAN F-14A TOMCAT (LATE MODEL) CARRIER LAUNCH SET",
      "model_dir": "data/tamiya_aircraft/GRUMMAN_F-14A_TOMCAT_LATE_MODEL_CARRIER_LAUNCH_SET__ITEM_12693",
      "selected_images": [
        "data/tamiya_aircraft/GRUMMAN_F-14A_TOMCAT_LATE_MODEL_CARRIER_LAUNCH_SET__ITEM_12693/images/001.jpg"
      ],
      "num_selected_images": 1
    },
    {
      "score": 0.66544508934021,
      "item_no": "61029",
      "title": "GRUMMAN F-14A TOMCAT",
      "title_prompt": "a scale model kit product named GRUMMAN F-14A TOMCAT",
      "model_dir": "data/tamiya_aircraft/GRUMMAN_F-14A_TOMCAT__ITEM_61029",
      "selected_images": [
        "data/tamiya_aircraft/GRUMMAN_F-14A_TOMCAT__ITEM_61029/images/001.jpg"
      ],
      "num_selected_images": 1
    }
  ]
}

### With YOLO crop


python mps_clip_retrieval_v21_hybrid.py search
--artifacts-dir ./artifacts_v21_hybrid
--query-image ./queries/f14.jpg
--yolo-weights yolov8n.pt
--device mps


Default output returns **Top-2 product matches**.

---

# Example Results

python mps_clip_retrieval_v21_hybrid.py search --artifacts-dir ./artifacts_v21_hybrid --query-image ./queries/shuttle2.jpg --device mps                         
[INFO] device=mps
[INFO] loading query image...
[INFO] embedding query image...
[INFO] loading hybrid product catalog and searching...
{
  "device_used": "mps",
  "query_image": "queries/shuttle2.jpg",
  "crop_meta": {
    "detected": false,
    "reason": "cropping_not_requested"
  },
  "crop_quality": {
    "use_crop": false,
    "reasons": [
      "cropping_not_requested"
    ]
  },
  "query_embedding_source": "original_image",
  "top_product_hits": [
    {
      "score": 0.4864608645439148,
      "item_no": null,
      "title": "SPACE SHUTTLE ATLANTIS",
      "title_prompt": "a scale model kit product named SPACE SHUTTLE ATLANTIS",
      "model_dir": "data/tamiya_aircraft/SPACE_SHUTTLE_ATLANTIS",
      "selected_images": [
        "data/tamiya_aircraft/SPACE_SHUTTLE_ATLANTIS/images/001.jpg"
      ],
      "num_selected_images": 1
    }

which belongs to this Tamiya model: https://www.tamiya.com/english/products/60402/index.html

---

# Dataset Example

Example catalog structure:


data/
tamiya_aircraft/
MODEL_NAME/
metadata.json
images/
001.jpg
002.jpg


Only **primary product images** are used for embeddings.

---

# Privacy Note

Query images included in this repository have had:

- EXIF metadata removed
- GPS location data stripped

---

# Future Improvements

Possible extensions:

- FAISS / Qdrant vector database
- reranking stage
- richer text prompts
- larger product catalogs
- deployment on edge GPU infrastructure

---

# License

MIT