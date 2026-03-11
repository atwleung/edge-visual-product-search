#!/usr/bin/env python3
"""
download_tamiya_aircraft.py

Seed a small local catalog of Tamiya military aircraft kits for embedding / FAISS experiments.

Example:
    python download_tamiya_aircraft.py \
        --out-dir ./data/tamiya_aircraft \
        --max-models 100 \
        --sleep 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE_LIST_URL = (
    "https://www.tamiya.com/english/products/list.html"
    "?field_sort=d&cmdarticlesearch=1&genre_item=e_1040&absolutepage={page}"
)

SITE_ROOT = "https://www.tamiya.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    )
}

# Typical product page pattern:
# https://www.tamiya.com/english/products/60795/index.html
PRODUCT_HREF_RE = re.compile(r"/english/products/\d+/index\.html?$", re.I)

# Image extensions worth keeping
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def safe_get(session: requests.Session, url: str, timeout: int = 30) -> requests.Response:
    r = session.get(url, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        # Tamiya pages sometimes need help on encoding detection
        r.encoding = r.apparent_encoding
    return r


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip()
    text = text.replace("/", "_")
    text = re.sub(r"[®™]", "", text)
    text = re.sub(r"[^A-Za-z0-9._ -]+", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    return text[:max_len] or "unknown_model"


def clean_title(raw: str) -> str:
    raw = re.sub(r"\s+", " ", raw).strip()
    # Remove leading scale prefixes if you want shorter folder names
    # e.g. "1/72 SCALE GRUMMAN F-14D TOMCAT" -> "GRUMMAN F-14D TOMCAT"
    raw = re.sub(r"^\d+/\d+\s+SCALE\s+", "", raw, flags=re.I)
    return raw


def guess_filename_from_url(url: str, fallback_index: int) -> str:
    path = urlparse(url).path
    name = os.path.basename(path)
    if "." in name:
        return name
    return f"{fallback_index:03d}.jpg"


def collect_product_links(list_html: str) -> List[str]:
    soup = BeautifulSoup(list_html, "html.parser")
    links: List[str] = []
    seen: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if PRODUCT_HREF_RE.search(href):
            full = urljoin(SITE_ROOT, href)
            if full not in seen:
                seen.add(full)
                links.append(full)

    return links


def extract_title(soup: BeautifulSoup) -> str:
    # Try H1 first
    h1 = soup.find("h1")
    if h1:
        title = clean_title(h1.get_text(" ", strip=True))
        if title:
            return title

    # Fallback to og:title
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return clean_title(og["content"])

    # Fallback to HTML title
    if soup.title and soup.title.string:
        return clean_title(soup.title.string)

    return "unknown_model"


def extract_item_no(soup: BeautifulSoup) -> Optional[str]:
    text = soup.get_text(" ", strip=True)
    m = re.search(r"\bITEM\s+(\d{5})\b", text, re.I)
    if m:
        return m.group(1)
    return None


def extract_image_urls(product_url: str, soup: BeautifulSoup) -> List[str]:
    """
    Tries multiple strategies:
    - og:image
    - img tags
    - anchor hrefs pointing to image files
    """
    urls: List[str] = []
    seen: Set[str] = set()

    # 1) og:image
    for meta in soup.find_all("meta"):
        prop = (meta.get("property") or "").lower()
        content = meta.get("content")
        if prop == "og:image" and content:
            full = urljoin(product_url, content)
            if full not in seen:
                seen.add(full)
                urls.append(full)

    # 2) img tags
    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        full = urljoin(product_url, src)

        # keep likely product images, skip tiny icons
        alt = (img.get("alt") or "").lower()
        width = img.get("width")
        height = img.get("height")
        path_lower = urlparse(full).path.lower()

        ext = os.path.splitext(path_lower)[1]
        if ext and ext not in IMAGE_EXTS:
            continue

        likely_product = (
            "/products/" in path_lower
            or "item" in path_lower
            or "product" in path_lower
            or "large" in path_lower
            or "photo" in path_lower
            or "image" in path_lower
            or alt
        )

        too_small = False
        try:
            if width and height and int(width) < 120 and int(height) < 120:
                too_small = True
        except Exception:
            pass

        if likely_product and not too_small and full not in seen:
            seen.add(full)
            urls.append(full)

    # 3) anchor hrefs directly to image files
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(product_url, href)
        path_lower = urlparse(full).path.lower()
        ext = os.path.splitext(path_lower)[1]
        if ext in IMAGE_EXTS and full not in seen:
            seen.add(full)
            urls.append(full)

    # De-dup while preserving order
    cleaned: List[str] = []
    for u in urls:
        if u not in cleaned:
            cleaned.append(u)

    return cleaned


def download_file(session: requests.Session, url: str, out_path: Path) -> bool:
    try:
        with session.get(url, stream=True, timeout=30, headers=HEADERS) as r:
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"[WARN] Failed download: {url} -> {out_path} ({e})")
        return False


def scrape_product(session: requests.Session, product_url: str, out_dir: Path, sleep_s: float) -> Optional[Dict]:
    print(f"[INFO] Product: {product_url}")
    try:
        r = safe_get(session, product_url)
    except Exception as e:
        print(f"[WARN] Could not fetch product page: {product_url} ({e})")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    title = extract_title(soup)
    item_no = extract_item_no(soup)
    image_urls = extract_image_urls(product_url, soup)

    folder_name = slugify(title)
    if item_no:
        folder_name = f"{folder_name}__ITEM_{item_no}"

    product_dir = out_dir / folder_name
    images_dir = product_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for idx, img_url in enumerate(image_urls, start=1):
        raw_name = guess_filename_from_url(img_url, idx)
        ext = os.path.splitext(raw_name)[1].lower() or ".jpg"
        save_name = f"{idx:03d}{ext}"
        save_path = images_dir / save_name

        ok = download_file(session, img_url, save_path)
        if ok:
            downloaded.append(
                {
                    "image_url": img_url,
                    "local_path": str(save_path),
                }
            )
        time.sleep(sleep_s)

    metadata = {
        "title": title,
        "item_no": item_no,
        "product_url": product_url,
        "num_images": len(downloaded),
        "images": downloaded,
    }

    with open(product_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./data/tamiya_aircraft", help="Output directory")
    ap.add_argument("--max-models", type=int, default=100, help="Maximum number of models to download")
    ap.add_argument("--max-pages", type=int, default=20, help="Maximum listing pages to scan")
    ap.add_argument("--sleep", type=float, default=1.0, help="Sleep between requests in seconds")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    product_urls: List[str] = []
    seen: Set[str] = set()

    print("[INFO] Collecting product links from listing pages...")
    for page in range(1, args.max_pages + 1):
        list_url = BASE_LIST_URL.format(page=page)
        print(f"[INFO] Listing page {page}: {list_url}")

        try:
            r = safe_get(session, list_url)
        except Exception as e:
            print(f"[WARN] Could not fetch listing page {page}: {e}")
            break

        links = collect_product_links(r.text)
        print(f"[INFO] Found {len(links)} product links on page {page}")

        if not links:
            break

        new_count = 0
        for link in links:
            if link not in seen:
                seen.add(link)
                product_urls.append(link)
                new_count += 1
                if len(product_urls) >= args.max_models:
                    break

        print(f"[INFO] Added {new_count} new products; total={len(product_urls)}")
        if len(product_urls) >= args.max_models:
            break

        time.sleep(args.sleep)

    print(f"[INFO] Total products queued: {len(product_urls)}")

    manifest = []
    for i, product_url in enumerate(product_urls, start=1):
        print(f"[INFO] Scraping product {i}/{len(product_urls)}")
        meta = scrape_product(session, product_url, out_dir, args.sleep)
        if meta:
            manifest.append(meta)

    manifest_path = out_dir / "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Downloaded {len(manifest)} products into: {out_dir}")


if __name__ == "__main__":
    main()