"""
app.py — SAM 3 Gradio Application
───────────────────────────────────
Runs Meta's SAM 3 text-prompted image segmentation on images stored in any
S3-compatible object store (AWS S3, MinIO, Cloudflare R2, Backblaze B2 …).

Usage:
    python app.py

The app will be available at http://0.0.0.0:7860 (or the machine's public IP
when deployed on VAST.AI or similar GPU cloud services).

Environment variables (optional overrides):
    SAM3_CHECKPOINT   Path to the SAM 3 .pt checkpoint file.
                      Default: checkpoints/sam3_l.pt
    SAM3_MODEL_CFG    Model configuration name passed to build_sam3_image_model.
                      Default: sam3_l
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


def load_sam3_model() -> None:
    """
    Import SAM 3, build the image model, and wrap it in a Sam3Processor.

    Call once at startup.  Sets the module-level _SAM3_PROCESSOR on success,
    or _MODEL_LOAD_ERROR on failure.
    """
    global _SAM3_PROCESSOR, _MODEL_LOAD_ERROR

    model_cfg = os.environ.get("SAM3_MODEL_CFG", "sam3_l")
    checkpoint = os.environ.get("SAM3_CHECKPOINT", "checkpoints/sam3_l.pt")

    logger.info("Loading SAM 3 model cfg=%s checkpoint=%s …", model_cfg, checkpoint)

    try:
        from sam3 import Sam3Processor, build_sam3_image_model  # type: ignore[import]
    except ImportError:
        _MODEL_LOAD_ERROR = (
            "SAM 3 is not installed.  Run:\n"
            "  pip install git+https://github.com/facebookresearch/sam3.git\n"
            "and download a checkpoint.  See README for details."
        )
        logger.error(_MODEL_LOAD_ERROR)
        return

    if not Path(checkpoint).is_file():
        _MODEL_LOAD_ERROR = (
            f"Checkpoint not found: {checkpoint}\n"
            "Set the SAM3_CHECKPOINT environment variable to the correct path.\n"
            "See README for download instructions."
        )
        logger.error(_MODEL_LOAD_ERROR)
        return

    try:
        model = build_sam3_image_model(model_cfg, checkpoint)
        model = model.to(DEVICE).eval()
        _SAM3_PROCESSOR = Sam3Processor(model)
        logger.info("SAM 3 model ready ✓")
    except Exception as exc:
        _MODEL_LOAD_ERROR = f"Failed to load SAM 3 model: {exc}"
        logger.exception(_MODEL_LOAD_ERROR)


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
        title="SAM 3 — S3 Image Segmentation",
    ) as demo:

        # ── Header ───────────────────────────────────────────────────────────
        gr.Markdown(
            "# SAM 3 — S3 Image Segmentation\n"
            "Run Meta's **SAM 3** text-prompted segmentation on every image "
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
