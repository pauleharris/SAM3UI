"""
utils.py — helper functions for SAM 3 Gradio App
──────────────────────────────────────────────────
Sections:
  1. S3 helpers         — list / download images from any S3-compatible store
  2. SAM 3 inference    — run text-prompted segmentation, normalise results
  3. Visualisation      — colored overlay masks, dot-center view
  4. Output helpers     — save per-image folder, build results.json
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import boto3
from botocore.config import Config
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)

# 20 visually distinct RGB colors for detected objects
_PALETTE: List[Tuple[int, int, int]] = [
    (255,  59,  48), (255, 149,   0), (255, 204,   0), ( 52, 199,  89),
    (  0, 199, 190), ( 48, 176, 199), ( 50, 173, 230), (  0, 122, 255),
    ( 88,  86, 214), (175,  82, 222), (255,  45,  85), (162, 132,  94),
    (142, 142, 147), ( 99, 230, 226), (255, 230, 100), (150, 255, 150),
    (255, 150, 200), (200, 150, 255), (100, 200, 255), (255, 200, 100),
]


def get_color(idx: int) -> Tuple[int, int, int]:
    """Return a repeating palette color for detection index *idx*."""
    return _PALETTE[idx % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. S3 helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_s3_client(
    access_key: str,
    secret_key: str,
    endpoint_url: Optional[str] = None,
    region: str = "us-east-1",
) -> boto3.client:
    """
    Create a boto3 S3 client.

    Works with:
      - AWS S3 (leave endpoint_url blank)
      - MinIO, Cloudflare R2, Backblaze B2, etc. (provide endpoint_url)

    Credentials are passed explicitly so they never touch env variables or
    shared config files.
    """
    kwargs: Dict[str, Any] = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "region_name": region,
        # Disable checksum validation that some S3-compatible services reject
        "config": Config(signature_version="s3v4"),
    }
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url.rstrip("/")

    return boto3.client("s3", **kwargs)


def list_images_in_bucket(
    s3_client,
    bucket: str,
    prefix: str = "",
    max_images: int = 1000,
) -> List[str]:
    """
    Return a sorted list of S3 object keys that look like images.

    Args:
        s3_client:  A boto3 S3 client.
        bucket:     Bucket name.
        prefix:     Optional key prefix / virtual folder.
        max_images: Hard cap on the number of keys returned.

    Returns:
        List of object key strings.
    """
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")

    params: Dict[str, Any] = {"Bucket": bucket}
    if prefix:
        # Ensure the prefix ends with "/" so we don't accidentally match
        # keys that merely start with the same letters.
        params["Prefix"] = prefix.rstrip("/") + "/"

    for page in paginator.paginate(**params):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if Path(key).suffix.lower() in IMAGE_EXTENSIONS:
                keys.append(key)
            if len(keys) >= max_images:
                return sorted(keys)

    return sorted(keys)


def download_image_from_s3(
    s3_client,
    bucket: str,
    key: str,
) -> Image.Image:
    """
    Stream an image from S3 into memory and return a PIL Image (RGB).

    Raises:
        botocore.exceptions.ClientError on permission / not-found errors.
    """
    buffer = io.BytesIO()
    s3_client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAM 3 inference
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(value: Any) -> Any:
    """Convert a torch.Tensor (or list) to a plain numpy array / Python scalar."""
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value) if not isinstance(value, (float, int)) else value


def _normalise_raw(raw: Any) -> Dict[str, Any]:
    """
    Normalise the return value of Sam3Processor.predict() into a plain dict
    with keys: masks, scores, boxes, labels.

    Handles both dict returns and object-with-attributes returns.
    """
    if isinstance(raw, dict):
        return raw
    return {
        "masks":  getattr(raw, "masks",  []),
        "scores": getattr(raw, "scores", []),
        "boxes":  getattr(raw, "boxes",  []),
        "labels": getattr(raw, "labels", []),
    }


def run_sam3_inference(
    processor,
    image: Image.Image,
    text_prompt: str,
    confidence_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Run SAM 3 text-prompted segmentation on a PIL Image.

    Args:
        processor:             A Sam3Processor instance wrapping the loaded model.
        image:                 Input image (RGB PIL Image).
        text_prompt:           Natural-language description, e.g. "yellow school bus".
        confidence_threshold:  Detections below this score are discarded.

    Returns:
        List of detection dicts::

            {
                "label":  str,           # detected class / prompt token
                "score":  float,         # confidence 0–1
                "bbox":   [x1,y1,x2,y2], # pixel coords (float)
                "center": [cx, cy],      # mask centroid (int pixel coords)
                "mask":   np.ndarray,    # bool H×W segmentation mask
            }
    """
    raw = processor.predict(image=image, text=text_prompt)
    data = _normalise_raw(raw)

    masks  = data.get("masks",  [])
    scores = data.get("scores", [])
    boxes  = data.get("boxes",  [])
    labels = data.get("labels", [])

    detections: List[Dict[str, Any]] = []

    for i, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        # Convert tensors → numpy
        score = float(_to_numpy(score)) if not isinstance(score, float) else score
        mask  = _to_numpy(mask)
        box   = _to_numpy(box).tolist() if hasattr(_to_numpy(box), "tolist") else list(box)

        if score < confidence_threshold:
            continue

        # Ensure binary 2-D mask (SAM may return (1, H, W) logits)
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask > 0.5  # bool H×W

        # Compute centroid; fall back to bbox centre if mask is empty
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

        # Resolve label string
        if i < len(labels):
            label = str(labels[i].decode() if isinstance(labels[i], bytes) else labels[i])
        else:
            label = text_prompt.split(",")[0].strip()

        detections.append({
            "label":  label,
            "score":  round(score, 4),
            "bbox":   [round(float(v), 1) for v in box],
            "center": [cx, cy],
            "mask":   mask,
        })

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_overlay_image(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    mask_alpha: float = 0.45,
) -> Image.Image:
    """
    Return a copy of *image* with semi-transparent colored segmentation masks,
    bounding-box rectangles, and score labels drawn on top.

    Args:
        image:       Original RGB PIL Image.
        detections:  List of detection dicts from run_sam3_inference().
        mask_alpha:  Opacity of mask fill (0 = transparent, 1 = opaque).
    """
    base = image.convert("RGBA")
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    for idx, det in enumerate(detections):
        color = get_color(idx)
        mask_arr = det["mask"]  # bool H×W

        # Build a per-detection RGBA layer coloured wherever the mask is True
        colored = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
        colored[mask_arr] = (*color, int(255 * mask_alpha))
        mask_layer = Image.alpha_composite(mask_layer, Image.fromarray(colored, "RGBA"))

    # Flatten mask layer onto base
    composite = Image.alpha_composite(base, mask_layer).convert("RGB")
    draw = ImageDraw.Draw(composite)

    for idx, det in enumerate(detections):
        color = get_color(idx)
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label_text = f"{det['label']} {det['score']:.2f}"
        # White shadow for legibility on any background
        draw.text((x1 + 5, y1 + 3), label_text, fill=(255, 255, 255))
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)

    return composite


def create_dots_image(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    dot_radius: int = 14,
) -> Image.Image:
    """
    Return a clean "dots view": a faded copy of *image* with one colored dot
    placed at each detection's mask centroid.

    This gives a simple, at-a-glance spatial overview of all findings.

    Args:
        image:       Original RGB PIL Image.
        detections:  List of detection dicts from run_sam3_inference().
        dot_radius:  Radius of each dot in pixels.
    """
    # Fade the original so the dots stand out clearly
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    faded = Image.blend(image.convert("RGB"), white_bg, alpha=0.6)
    draw = ImageDraw.Draw(faded)

    for idx, det in enumerate(detections):
        color = get_color(idx)
        cx, cy = int(det["center"][0]), int(det["center"][1])
        r = dot_radius

        # White border ring for contrast on any background
        draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3],
                     fill=(255, 255, 255))
        # Filled dot in detection color
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        # Label to the right of the dot
        label_text = f"{det['label']} {det['score']:.2f}"
        draw.text((cx + r + 6, cy - 8), label_text, fill=(40, 40, 40))

    return faded


# ─────────────────────────────────────────────────────────────────────────────
# 4. Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_results_json(
    image_key: str,
    text_prompt: str,
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the JSON-serialisable results dict for one processed image.
    The "mask" key is intentionally excluded (it's a numpy array; saved separately).
    """
    clean_dets = [
        {
            "label":  det["label"],
            "score":  det["score"],
            "bbox":   det["bbox"],
            "center": det["center"],
        }
        for det in detections
    ]
    return {
        "source_key":     image_key,
        "prompt":         text_prompt,
        "num_detections": len(clean_dets),
        "detections":     clean_dets,
    }


def save_outputs(
    output_dir: str,
    image_key: str,
    original: Image.Image,
    overlay: Image.Image,
    dots: Image.Image,
    results: Dict[str, Any],
    detections: List[Dict[str, Any]],
    save_masks: bool = True,
) -> Dict[str, str]:
    """
    Save all per-image outputs inside ``output_dir/<image_stem>/``.

    Files written:
      - original.jpg
      - overlay.png
      - dots.png
      - results.json
      - mask_00.png, mask_01.png, … (if save_masks is True)

    Returns:
        Mapping of output name → absolute local path.
    """
    stem = Path(image_key).stem
    folder = Path(output_dir) / stem
    folder.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    orig_path = folder / "original.jpg"
    original.save(str(orig_path), "JPEG", quality=95)
    paths["original"] = str(orig_path)

    overlay_path = folder / "overlay.png"
    overlay.save(str(overlay_path), "PNG")
    paths["overlay"] = str(overlay_path)

    dots_path = folder / "dots.png"
    dots.save(str(dots_path), "PNG")
    paths["dots"] = str(dots_path)

    json_path = folder / "results.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    paths["results"] = str(json_path)

    if save_masks:
        for idx, det in enumerate(detections):
            mask_arr = det["mask"].astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_arr)
            mask_path = folder / f"mask_{idx:02d}.png"
            mask_img.save(str(mask_path), "PNG")
            paths[f"mask_{idx:02d}"] = str(mask_path)

    return paths
