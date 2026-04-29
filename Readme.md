# StyleForge — Neural Style Transfer

> Transform any photograph into a painting using AdaIN-based neural style transfer.  
> Built with PyTorch and served via a clean Flask web interface.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training](#training)
- [Resume Training](#resume-training)
- [Running the Web App](#running-the-web-app)
- [Tech Stack](#tech-stack)
- [Results](#results)
- [FAQ](#faq)

---

## Overview

StyleForge implements **Adaptive Instance Normalisation (AdaIN)** style transfer — a real-time approach that blends the content of one image with the artistic style of another.

Unlike slow optimization-based methods (Gatys et al.), AdaIN runs in a single forward pass through the network, making it fast enough to serve through a web interface.

The model is **trained from scratch** on large-scale content and style datasets.

---

## How It Works

```
Content Image ──┐
                ├──► VGG Encoder ──► AdaIN ──► Decoder ──► Stylised Output
Style Image ────┘
```

1. Both images are passed through a shared **VGG encoder** to extract deep features.
2. **AdaIN** aligns the mean and variance of the content features to match the style features.
3. A lightweight **decoder** reconstructs the final stylised image from the normalised features.
4. An **alpha** parameter (0.0 → 1.0) controls how strongly the style is applied.

---

## Project Structure

```
Neural_Style_Transfer_Project/
│
├── app.py                        # Flask web application
├── train.py                      # Training script
├── vgg_normalised.pth            # Pre-trained VGG encoder weights
│
├── utils/
│   ├── models.py                 # VGGEncoder and Decoder architecture
│   └── utils.py                  # AdaIN, calc_mean_std helpers
│
├── weights/
│   └── decoder_final.pth         # Trained decoder weights (after training)
│
├── experiments/
│   └── exp2/
│       ├── checkpoint.pth        # Latest checkpoint (for resuming)
│       └── decoder_final.pth     # Best model saved at end of training
│
├── examples/                     # Sample content/style/output images
│
├── templates/
│   └── index.html                # Jinja2 HTML template
│
└── static/
    ├── uploads/                  # Auto-created — stores user uploads
    ├── css/
    │   └── style.css
    └── js/
        └── main.js
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Mearnab01/Neural_Style_Transfer_Project.git
cd Neural_Style_Transfer_Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables (optional but recommended)

```bash
export SECRET_KEY="your-secret-key"
export DECODER_PATH="weights/decoder_final.pth"
export ENCODER_PATH="weights/vgg_normalised.pth"
```

> If not set, the app falls back to safe defaults automatically.

---

## Training

Training is designed to run on **Google Colab** with datasets stored in Google Drive.

### Step 1 — Mount Drive and navigate to project

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/drive/MyDrive/ai_nst/Neural_Style_Transfer_Project
```

### Step 2 — Pull latest changes (if already cloned)

```bash
!git pull origin main
```

### Step 3 — Run training

```bash
!python train.py \
  --batch_size 4 \
  --epochs 10 \
  --experiment exp2 \
  --content_dir /content/drive/MyDrive/ai_nst/big_datasets/content_dataset \
  --style_dir   /content/drive/MyDrive/ai_nst/big_datasets/style_dataset
```

| Argument        | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| `--batch_size`  | Images per training step (4 recommended for Colab)                     |
| `--epochs`      | Number of full passes through the dataset                              |
| `--experiment`  | Name of the experiment — checkpoints saved under `experiments/<name>/` |
| `--content_dir` | Path to content image dataset                                          |
| `--style_dir`   | Path to style image dataset                                            |

---

## Resume Training

If your Colab session disconnects or you want to continue from a checkpoint:

```bash
!python train.py \
  --batch_size 4 \
  --epochs 10 \
  --experiment exp2 \
  --content_dir /content/drive/MyDrive/ai_nst/big_datasets/content_dataset \
  --style_dir   /content/drive/MyDrive/ai_nst/big_datasets/style_dataset \
  --resume \
  --decoder_path /content/drive/MyDrive/ai_nst/Neural_Style_Transfer_Project/experiments/exp2/checkpoint.pth
```

> Checkpoints are saved to Google Drive, so they survive session disconnects.

### Keep Colab alive during long training runs

Run this in a separate cell **before** starting training:

```javascript
%%javascript
function ClickConnect(){
  console.log("Keeping session alive...");
  document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```

This clicks the Connect button every 60 seconds to prevent Colab from timing out.

---

## Running the Web App

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

Upload a content image and a style image, adjust the **Style Strength** slider, and click **Transfer Style**.

### Health check

```
GET /health
→ { "status": "ok", "device": "cuda" }
```

---

## Tech Stack

| Layer            | Technology                                           |
| ---------------- | ---------------------------------------------------- |
| Model            | PyTorch — VGG encoder + custom AdaIN decoder         |
| Training         | Google Colab (T4 GPU) + Google Drive storage         |
| Web backend      | Python 3, Flask, Flask-WTF                           |
| Web frontend     | HTML5, CSS3, Vanilla JS, Bootstrap 5, Font Awesome 6 |
| Image processing | Pillow, torchvision                                  |

---

## Results

| Content        | Style                 | Output                   |
| -------------- | --------------------- | ------------------------ |
| Portrait photo | Pencil sketch         | Sketch-stylised portrait |
| Portrait photo | Picasso — Seated Nude | Cubist-stylised portrait |

Example images are available in the `examples/` directory.

---

## FAQ

**Is this a pretrained model?**  
No. The decoder is trained from scratch. Only the VGG encoder uses pretrained ImageNet weights as a frozen feature extractor.

**Which styles work best?**  
Paintings with strong, distinct textures work best — impressionist, watercolour, cubist, ink sketch. Photorealistic style images produce subtle results.

**What does the alpha slider do?**  
Alpha controls the blend between content and style features. `1.0` = full style applied. `0.0` = content features unchanged. Values around `0.7–0.9` often give the best balance.

**What is the tech stack?**  
Python, PyTorch, Flask. The model uses VGG-based feature extraction with AdaIN normalisation.

**Why Google Drive for training?**  
Colab's local `/content/` storage is wiped on every disconnect. Storing datasets and checkpoints in Google Drive (`/content/drive/MyDrive/...`) ensures nothing is lost between sessions.

---

## Author

**Arnab Nath**  
[GitHub](https://github.com/Mearnab01)

---

_Reference: Huang & Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization", ICCV 2017._
