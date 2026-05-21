# SAM 3 — S3 Image Segmentation

Run Meta's **SAM 3** text-prompted instance segmentation on every image in
any S3-compatible bucket — directly from a clean Gradio web UI.

---

## Features

| Feature | Detail |
|---|---|
| **Text prompts** | Describe what to find: `"yellow school bus"`, `"person"`, `"red car"` |
| **S3-compatible** | AWS S3, MinIO, Cloudflare R2, Backblaze B2, any custom endpoint |
| **Batch processing** | Scans an entire bucket prefix and processes all images |
| **Live preview** | Results stream to the UI as each image finishes |
| **Rich outputs** | Overlay masks · dot map · bounding boxes · JSON · individual mask PNGs |
| **One-click download** | Everything packaged into a ZIP |

---

## Project Structure

```
sam3-gradio-app/
├── app.py            # Gradio application — UI and processing pipeline
├── utils.py          # Helpers: S3 I/O, SAM 3 inference, visualisation, output
├── requirements.txt  # Python dependencies (see note about SAM 3 install)
└── README.md
```

**Output per image** (written to a temp folder, downloadable as ZIP):

```
sam3_results/
└── <image_stem>/
    ├── original.jpg      # Original image
    ├── overlay.png       # Image + colored segmentation masks
    ├── dots.png          # Spatial overview — one colored dot per detection
    ├── results.json      # Prompt, detections, scores, bboxes, centers
    ├── mask_00.png       # Individual binary mask for detection 0
    └── mask_01.png       # …
```

---

## Requirements

- Python **3.10+**
- A CUDA-capable GPU is **strongly recommended** (CPU inference is very slow)
- A SAM 3 checkpoint file (see [Hugging Face access](#hugging-face-model-access) below)

---

## Hugging Face Model Access

SAM 3 is gated behind Meta's license on Hugging Face.

1. Create a [Hugging Face account](https://huggingface.co/join) if you don't have one.
2. Go to **<https://huggingface.co/facebook/sam3-large>** and click **"Agree and access"**.
3. Generate an access token at <https://huggingface.co/settings/tokens> (read permission is enough).
4. Download the checkpoint:

```bash
pip install huggingface_hub
huggingface-cli login           # paste your token when prompted
mkdir -p checkpoints
huggingface-cli download facebook/sam3-large --local-dir checkpoints/
```

Set the `SAM3_CHECKPOINT` environment variable if you save it elsewhere:

```bash
export SAM3_CHECKPOINT=/path/to/your/sam3_l.pt
```

---

## Local Installation & Running

```bash
# 1. Clone / download the project
git clone https://github.com/YOUR_USERNAME/sam3-gradio-app.git
cd sam3-gradio-app

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install PyTorch (pick the right CUDA build for your GPU driver)
#    See https://pytorch.org/get-started/locally/ for the correct command.
#    Example for CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Install SAM 3 from source
pip install git+https://github.com/facebookresearch/sam3.git

# 6. Download the checkpoint (see above)
mkdir -p checkpoints
huggingface-cli download facebook/sam3-large --local-dir checkpoints/

# 7. Launch the app
python app.py
```

Open **<http://localhost:7860>** in your browser.

---

## Running on VAST.AI

VAST.AI is a GPU marketplace that makes it easy to rent cloud GPUs by the hour.

### Step 1 — Create an Instance

| Setting | Recommended value |
|---|---|
| **Template** | PyTorch (CUDA 12.x) |
| **Disk** | ≥ 30 GB |
| **GPU** | RTX 3090 / A100 / H100 (≥ 16 GB VRAM) |
| **Open ports** | **7860** (TCP) |

> In the instance configuration page, add port 7860 under **"Extra ports"** so
> it is accessible from the internet.

### Step 2 — On-Start Script

Paste the following into the **"On-Start Script"** field when renting the
instance (or run it manually in the instance terminal):

```bash
#!/bin/bash
set -euo pipefail

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -qq && apt-get install -y -qq git

# ── App setup ────────────────────────────────────────────────────────────────
cd /workspace
git clone https://github.com/YOUR_USERNAME/sam3-gradio-app.git
cd sam3-gradio-app

# ── Python dependencies ───────────────────────────────────────────────────────
# PyTorch is usually pre-installed in the VAST.AI PyTorch template.
# If not, uncomment the next line and adjust the CUDA version:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam3.git

# ── SAM 3 checkpoint ─────────────────────────────────────────────────────────
export HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"   # replace with your token
mkdir -p checkpoints
huggingface-cli download facebook/sam3-large --local-dir checkpoints/

# ── Launch (detached so the script exits cleanly) ────────────────────────────
nohup python app.py > /workspace/app.log 2>&1 &
echo "App starting — tail /workspace/app.log for progress"
```

### Step 3 — Access the App

Once the instance is running and the start script has finished, open:

```
http://<INSTANCE_PUBLIC_IP>:7860
```

The public IP is shown on the VAST.AI dashboard under your running instance.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SAM3_CHECKPOINT` | `checkpoints/sam3_l.pt` | Path to the SAM 3 `.pt` checkpoint |
| `SAM3_MODEL_CFG` | `sam3_l` | Model config name passed to `build_sam3_image_model` |

---

## Usage

1. Expand **S3 Configuration** and fill in:
   - **Endpoint URL** — leave blank for AWS S3, or enter your MinIO / R2 endpoint
   - **Access Key ID** and **Secret Access Key**
   - **Bucket Name** and optional **Prefix** (virtual folder)
2. Type a **text prompt** describing what to find (e.g. `"yellow school bus"`).
3. Adjust **Confidence Threshold** if needed (default `0.3`).
4. Click **⚡ Process Images**.
5. Watch results stream in:
   - **Overlay Gallery** — segmentation masks drawn over the original images
   - **Dots View** — spatial dot map of all detections
   - **Detections Table** — label, score, and center coordinates per image
6. Click **Download All Results (ZIP)** when processing is complete.

---

## Troubleshooting

**`ImportError: No module named 'sam3'`**
→ Install SAM 3: `pip install git+https://github.com/facebookresearch/sam3.git`

**`FileNotFoundError: checkpoints/sam3_l.pt`**
→ Download the checkpoint (see [Hugging Face access](#hugging-face-model-access))
and point to it with `export SAM3_CHECKPOINT=/path/to/checkpoint.pt`.

**`403 Forbidden` / `InvalidAccessKeyId`**
→ Double-check your S3 credentials and make sure the IAM policy includes
`s3:ListBucket` and `s3:GetObject` on the target bucket.

**Out of GPU memory**
→ Use a smaller model (`export SAM3_MODEL_CFG=sam3_b`) or reduce image
resolution before uploading to S3.

**No images found**
→ Check the prefix — it must match the folder path exactly.  Trailing `/` is
added automatically by the app, but the bucket must contain objects under that
prefix.

---

## License

This project is released under the MIT License.
SAM 3 itself is subject to Meta's [license terms](https://huggingface.co/facebook/sam3-large).
