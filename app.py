"""
app.py — Grounded SAM Gradio Application
─────────────────────────────────────────
Runs text-prompted image segmentation (Grounding DINO → SAM) on images stored
in any S3-compatible object store (AWS S3, MinIO, Cloudflare R2, Backblaze B2).

Usage:
    python app.py

The app will be available at http://0.0.0.0:7860 (or the machine's public IP
when deployed on VAST.AI or similar GPU cloud services).

Environment variables (optional overrides):
    GDINO_MODEL   HuggingFace model ID for Grounding DINO text detector.
                  Default: IDEA-Research/grounding-dino-base
    SAM_MODEL     HuggingFace model ID for SAM segmentation model.
                  Default: facebook/sam-vit-large
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Generator

import gradio as gr
import torch

from utils import (
    build_results_json,
    create_dots_image,
    create_overlay_image,
    create_s3_client,
    download_image_from_s3,
    list_images_in_bucket,
    run_sam3_inference,
    save_outputs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Running on device: %s", DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# Model — loaded once at startup
# ─────────────────────────────────────────────────────────────────────────────

# Global holders so the model is built only once and reused across all requests.
_SAM3_PROCESSOR = None
_MODEL_LOAD_ERROR: str | None = None  # human-readable error if loading failed


class GroundedSamProcessor:
    """
    Wraps Grounding DINO (text → boxes) + SAM (boxes → masks) into a single
    processor that matches the ``processor.predict(image, text)`` interface
    expected by ``run_sam3_inference`` in utils.py.
    """

    def __init__(self, gdino_model, gdino_proc, sam_model, sam_proc, device: str):
        self.gdino_model = gdino_model
        self.gdino_proc  = gdino_proc
        self.sam_model   = sam_model
        self.sam_proc    = sam_proc
        self.device      = device

    def predict(
        self,
        image,
        text: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> dict:
        import torch

        # ── Step 1: Grounding DINO — text → bounding boxes ──────────────────
        gdino_inputs = self.gdino_proc(
            images=image,
            text=text,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            gdino_outputs = self.gdino_model(**gdino_inputs)

        w, h = image.size
        results = self.gdino_proc.post_process_grounded_object_detection(
            gdino_outputs,
            gdino_inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes  = results["boxes"]   # tensor [N, 4] xyxy pixel coords
        scores = results["scores"]  # tensor [N]
        labels = results["labels"]  # list of str

        if boxes.shape[0] == 0:
            return {"masks": [], "scores": [], "boxes": [], "labels": []}

        # ── Step 2: SAM — boxes → segmentation masks ────────────────────────
        boxes_list = boxes.cpu().numpy().tolist()
        sam_inputs = self.sam_proc(
            images=image,
            input_boxes=[[boxes_list]],   # shape: [batch=1, N_boxes, 4]
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        # post_process_masks → list of (N_boxes, 3, H, W) tensors
        masks = self.sam_proc.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (N_boxes, 3, H, W)

        # Pick the highest-IoU mask candidate per box
        iou_scores = sam_outputs.iou_scores.cpu()  # (1, N_boxes, 3)
        best_idx   = iou_scores[0].argmax(dim=-1)  # (N_boxes,)
        best_masks = torch.stack(
            [masks[i, best_idx[i]] for i in range(masks.shape[0])]
        )  # (N_boxes, H, W) bool

        return {
            "masks":  best_masks.numpy(),
            "scores": scores.cpu().numpy().tolist(),
            "boxes":  boxes.cpu().numpy().tolist(),
            "labels": labels,
        }


def load_sam3_model() -> None:
    """
    Load Grounding DINO and SAM from HuggingFace Hub, then wrap them in a
    GroundedSamProcessor.  Sets _SAM3_PROCESSOR on success or _MODEL_LOAD_ERROR
    on failure.
    """
    global _SAM3_PROCESSOR, _MODEL_LOAD_ERROR

    gdino_model_id = os.environ.get("GDINO_MODEL", "IDEA-Research/grounding-dino-base")
    sam_model_id   = os.environ.get("SAM_MODEL",   "facebook/sam-vit-large")

    logger.info("Loading Grounding DINO: %s", gdino_model_id)
    logger.info("Loading SAM: %s", sam_model_id)

    try:
        from transformers import (  # type: ignore[import]
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )
    except ImportError:
        _MODEL_LOAD_ERROR = (
            "transformers is not installed.  Run:\n"
            "  pip install transformers>=4.45.0 accelerate\n"
        )
        logger.error(_MODEL_LOAD_ERROR)
        return

    try:
        gdino_proc  = AutoProcessor.from_pretrained(gdino_model_id)
        gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_model_id
        ).to(DEVICE).eval()
        logger.info("Grounding DINO loaded ✓")
    except Exception as exc:
        _MODEL_LOAD_ERROR = f"Failed to load Grounding DINO ({gdino_model_id}): {exc}"
        logger.exception(_MODEL_LOAD_ERROR)
        return

    try:
        sam_proc  = SamProcessor.from_pretrained(sam_model_id)
        sam_model = SamModel.from_pretrained(sam_model_id).to(DEVICE).eval()
        logger.info("SAM loaded ✓")
    except Exception as exc:
        _MODEL_LOAD_ERROR = f"Failed to load SAM ({sam_model_id}): {exc}"
        logger.exception(_MODEL_LOAD_ERROR)
        return

    _SAM3_PROCESSOR = GroundedSamProcessor(
        gdino_model, gdino_proc, sam_model, sam_proc, DEVICE
    )
    logger.info("Models ready ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Processing pipeline (generator — yields incremental UI updates)
# ─────────────────────────────────────────────────────────────────────────────

def process_images(
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    prefix: str,
    text_prompt: str,
    confidence_threshold: float,
    progress: gr.Progress = gr.Progress(),
) -> Generator:
    """
    Main pipeline.  This is a **generator** function; Gradio streams each
    ``yield`` to the UI so the user sees live progress.

    Yields a 5-tuple on every update:
        (overlay_gallery, dots_gallery, detections_markdown, zip_file_update, status_markdown)
    """

    # ── Guard: model must be loaded ──────────────────────────────────────────
    if _MODEL_LOAD_ERROR:
        msg = f"**Model load error:**\n```\n{_MODEL_LOAD_ERROR}\n```"
        yield [], [], msg, gr.update(visible=False), "Model not loaded."
        return

    if _SAM3_PROCESSOR is None:
        yield [], [], "**Model is still loading — please wait.**", gr.update(visible=False), "Waiting for model…"
        return

    # ── Validate user inputs ─────────────────────────────────────────────────
    if not access_key.strip() or not secret_key.strip():
        yield [], [], "**Error:** Access Key and Secret Key are required.", gr.update(visible=False), "Missing credentials."
        return

    if not bucket.strip():
        yield [], [], "**Error:** Bucket name is required.", gr.update(visible=False), "Missing bucket."
        return

    if not text_prompt.strip():
        yield [], [], "**Error:** Please enter a text prompt.", gr.update(visible=False), "No prompt."
        return

    # ── Connect to S3 ────────────────────────────────────────────────────────
    try:
        s3 = create_s3_client(
            access_key=access_key.strip(),
            secret_key=secret_key.strip(),
            endpoint_url=endpoint.strip() or None,
        )
    except Exception as exc:
        msg = f"**Could not create S3 client:** {exc}"
        yield [], [], msg, gr.update(visible=False), "S3 connection failed."
        return

    # ── List images ──────────────────────────────────────────────────────────
    yield [], [], "Scanning S3 bucket for images…", gr.update(visible=False), "Scanning S3…"

    try:
        image_keys = list_images_in_bucket(s3, bucket.strip(), prefix.strip())
    except Exception as exc:
        msg = f"**Could not list images:** {exc}"
        yield [], [], msg, gr.update(visible=False), "Failed to list images."
        return

    if not image_keys:
        msg = f"No images found under `s3://{bucket.strip()}/{prefix.strip()}`."
        yield [], [], msg, gr.update(visible=False), "No images found."
        return

    total = len(image_keys)
    yield [], [], f"Found **{total}** image(s).  Starting…", gr.update(visible=False), f"Found {total} images."

    # ── Temp directory for all outputs ───────────────────────────────────────
    output_dir = tempfile.mkdtemp(prefix="sam3_results_")
    logger.info("Writing outputs to %s", output_dir)

    overlay_gallery: list = []   # (path, caption) for the Overlay Gallery tab
    dots_gallery:    list = []   # (path, caption) for the Dots View tab
    det_md_blocks:   list = []   # per-image Markdown for the Detections tab

    # ── Process each image ───────────────────────────────────────────────────
    for i, key in enumerate(progress.tqdm(image_keys, desc="Segmenting")):
        image_name = Path(key).name
        status = f"Processing {i + 1} / {total}: `{image_name}`"
        yield overlay_gallery, dots_gallery, "\n\n---\n\n".join(det_md_blocks) or "_Processing…_", gr.update(visible=False), status

        try:
            # 1. Download from S3
            image = download_image_from_s3(s3, bucket.strip(), key)
            logger.info("Downloaded %s (%dx%d)", key, image.width, image.height)

            # 2. Run SAM 3 inference
            detections = run_sam3_inference(
                processor=_SAM3_PROCESSOR,
                image=image,
                text_prompt=text_prompt.strip(),
                confidence_threshold=confidence_threshold,
            )
            logger.info("%s → %d detection(s)", image_name, len(detections))

            # 3. Build visualisations
            overlay = create_overlay_image(image, detections)
            dots    = create_dots_image(image, detections)

            # 4. Save all output files
            results_json = build_results_json(key, text_prompt.strip(), detections)
            paths = save_outputs(
                output_dir=output_dir,
                image_key=key,
                original=image,
                overlay=overlay,
                dots=dots,
                results=results_json,
                detections=detections,
                save_masks=True,
            )

            # 5. Accumulate gallery entries
            caption_base = f"{image_name} — {len(detections)} detection(s)"
            overlay_gallery.append((paths["overlay"], caption_base))
            dots_gallery.append((paths["dots"],    f"{image_name} — dots view"))

            # 6. Build Markdown summary row for this image
            if detections:
                rows = "\n".join(
                    f"| {d['label']} | {d['score']:.3f} | {d['center'][0]}, {d['center'][1]} |"
                    for d in detections
                )
                block = (
                    f"### {image_name}\n"
                    f"| Label | Score | Center (x, y) |\n"
                    f"|-------|-------|---------------|\n"
                    f"{rows}"
                )
            else:
                block = f"### {image_name}\n_No detections above threshold._"

            det_md_blocks.append(block)

        except Exception:
            err_summary = traceback.format_exc()
            logger.error("Error processing %s:\n%s", key, err_summary)
            det_md_blocks.append(f"### {image_name}\n**Error during processing.** Check server logs.")

        # Stream the updated galleries after every image
        yield (
            overlay_gallery,
            dots_gallery,
            "\n\n---\n\n".join(det_md_blocks),
            gr.update(visible=False),
            status,
        )

    # ── Package everything into a ZIP ────────────────────────────────────────
    zip_path = os.path.join(output_dir, "sam3_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(output_dir):
            for fname in files:
                if fname == "sam3_results.zip":
                    continue
                abs_path = os.path.join(root, fname)
                arcname  = os.path.relpath(abs_path, output_dir)
                zf.write(abs_path, arcname)

    final_status = f"Done! Processed **{total}** image(s)."
    yield (
        overlay_gallery,
        dots_gallery,
        "\n\n---\n\n".join(det_md_blocks),
        gr.update(value=zip_path, visible=True),
        final_status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    with gr.Blocks(
        title="SAM — S3 Image Segmentation",
    ) as demo:

        # ── Header ───────────────────────────────────────────────────────────
        gr.Markdown(
            "# SAM — S3 Image Segmentation\n"
            "Run text-prompted segmentation (**Grounding DINO + SAM**) on every image "
            "in an S3-compatible bucket."
        )

        # ── S3 Configuration ─────────────────────────────────────────────────
        with gr.Accordion("S3 Configuration", open=True):
            endpoint = gr.Textbox(
                label="Endpoint URL",
                placeholder="https://s3.amazonaws.com   (leave blank for standard AWS S3)",
                info="Custom endpoint for MinIO, Cloudflare R2, Backblaze B2, etc.",
            )
            with gr.Row():
                access_key = gr.Textbox(
                    label="Access Key ID",
                    type="password",
                    placeholder="AKIAIOSFODNN7EXAMPLE",
                )
                secret_key = gr.Textbox(
                    label="Secret Access Key",
                    type="password",
                    placeholder="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                )
            with gr.Row():
                bucket = gr.Textbox(
                    label="Bucket Name",
                    placeholder="my-images-bucket",
                )
                prefix = gr.Textbox(
                    label="Prefix / Folder",
                    placeholder="2024/field-photos/",
                    info="Leave blank to scan the entire bucket.",
                )

        # ── Prompt & settings ─────────────────────────────────────────────────
        with gr.Row():
            text_prompt = gr.Textbox(
                label="What are you looking for?",
                placeholder='"yellow school bus"  •  "person walking"  •  "red car"',
                lines=2,
                scale=3,
            )
            confidence = gr.Slider(
                label="Confidence Threshold",
                minimum=0.05,
                maximum=0.95,
                value=0.3,
                step=0.05,
                scale=1,
                info="Discard detections below this score.",
            )

        # ── Run button ────────────────────────────────────────────────────────
        run_btn = gr.Button("⚡ Process Images", variant="primary", size="lg")

        status_box = gr.Markdown(
            value="_Enter your S3 credentials and a prompt, then click **Process Images**._"
        )

        # ── Results ───────────────────────────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("Overlay Gallery"):
                gr.Markdown(
                    "_Each image shows the original with colored segmentation masks, "
                    "bounding boxes, and confidence scores._"
                )
                overlay_gallery = gr.Gallery(
                    label="Segmentation Overlays",
                    columns=3,
                    height=600,
                    object_fit="contain",
                    show_label=False,
                )

            with gr.Tab("Dots View"):
                gr.Markdown(
                    "_Clean spatial overview: one colored dot per detected object, "
                    "placed at its mask centroid._"
                )
                dots_gallery = gr.Gallery(
                    label="Dots Visualization",
                    columns=3,
                    height=600,
                    object_fit="contain",
                    show_label=False,
                )

            with gr.Tab("Detections Table"):
                detections_md = gr.Markdown(value="_No results yet._")

        # ── Download ──────────────────────────────────────────────────────────
        zip_download = gr.File(
            label="Download All Results (ZIP)",
            visible=False,
            interactive=False,
        )

        # ── Event wiring ──────────────────────────────────────────────────────
        run_btn.click(
            fn=process_images,
            inputs=[
                endpoint,
                access_key,
                secret_key,
                bucket,
                prefix,
                text_prompt,
                confidence,
            ],
            outputs=[
                overlay_gallery,
                dots_gallery,
                detections_md,
                zip_download,
                status_box,
            ],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load the model before the UI starts so the first request is fast.
    load_sam3_model()

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",  # bind to all interfaces — required for VAST.AI / Docker
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        theme=gr.themes.Soft(),
        share=False,
    )
