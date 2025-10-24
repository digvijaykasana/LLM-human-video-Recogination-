# Project Overview

## Human Activity Recognition with CLIP (Streamlit + PyTorch)

A complete, project that **classifies human actions in short videos** using a fine‑tuned **CLIP ViT‑B/32** image encoder and a lightweight classification layer. this project is when you upload a video to the model in streamlit web UI it predict whats the person is doing in the video is it sitting, standing, reading or walkind etc.
It includes:
- a **training pipeline** to fine‑tune on your custom dataset, and
- a **Streamlit app** to upload a video and get an instant action prediction.

---

## What this project does

- **Takes a video** (MP4/AVI), **extracts representative frames** with OpenCV.
- **Encodes frames** using the CLIP image encoder (`openai/clip-vit-base-patch32`).
- **Aggregates frame embeddings** to form a video‑level representation.
- **Classifies** the resulting representation into one of the configured actions.
- Provides a **web UI (Streamlit)** for easy demo and recruiter review.

---

## Repo structure (actual)

```
codes and output/
├─ code current/
│  ├─ app.py                # Streamlit inference app (current version)
│  ├─ training.py           # End‑to‑end training/validation/testing pipeline
│  ├─ logs.txt              # Training/eval logs (accuracy/loss etc.)
│  └─ .ipynb_checkpoints/   # Auto‑generated checkpoints
├─ code pre/
│  ├─ app.py                # Earlier app version (same idea, older paths)
│  └─ training.py           # Earlier training script
```

> You’ll run either `code current/app.py` (for the demo) or `code current/training.py` (to fine‑tune on your dataset).

---

## Environment

- **Python 3.10+**
- Install dependencies:
```bash
pip install -r requirements.txt
```

> Key libs: `torch`, `transformers`, `opencv-python`, `Pillow`, `numpy`, `streamlit`, `matplotlib`, `seaborn`, `scikit-learn`.

---

## Dataset layout

Training expects a **folder per class** with **videos inside**:

```
HumanActivityRecognition-VideoDataset/
├─ clapping/
│  ├─ vid_001.mp4
│  └─ ...
├─ meeting_and_splitting/
│  ├─ ms_001.mp4
│  └─ ...
├─ sitting/
├─ walking_while_reading_a_book/
└─ walking_while_using_phone/
```

Name the dataset folder exactly as passed in `training.py` (default: `HumanActivityRecognition-VideoDataset`).

---

## How the training pipeline works (`code current/training.py`)

1. **VideoDataset**
   - Walks the dataset directory (class‑subfolders).
   - **Extracts frames** with `cv2.VideoCapture` (evenly spaced sampling per clip).
   - Uses **`CLIPProcessor`** to preprocess each frame to the ViT‑B/32 input size.
   - Stacks frames into a tensor shaped `[T, C, H, W]` and then batches to `[B, T, C, H, W]`.

2. **Embedding & aggregation**
   - Passes frames through **`CLIPModel.get_image_features`** to get `[B, T, D]` embeddings.
   - **Temporal aggregation** (mean‑pool) reduces to `[B, D]` per video.

3. **Classification head**
   - A **linear classifier** maps `[B, D] → [B, num_classes]`.
   - **Loss**: `CrossEntropyLoss`.
   - **Optimizer**: `Adam(lr=1e-4)`.
   - **Metrics**: per‑epoch train/val loss and **validation accuracy**.

4. **Data split & loaders**
   - Uses a stratified **train/val/test** split.
   - `DataLoader` with batching and shuffling for training.

5. **Validation & early model saving**
   - Tracks the **best validation loss** and saves weights to `fine_tuned_clip_best.pth`.
   - Final model snapshot is also saved to `fine_tuned_clip.pth`.

6. **Evaluation**
   - Runs on the **test set** and prints **Test Accuracy**.
   - Generates a **confusion matrix** and **training curves** (loss/accuracy) with Matplotlib/Seaborn.
   - Writes progress to `logs.txt` (e.g., `Epoch x/y`, `Training loss`, `Test Accuracy` lines).

### Train it

```bash
cd "codes and output/code current"
python training.py
```
- Make sure your dataset is at `../HumanActivityRecognition-VideoDataset` or update the path in `training.py`.
- At the end, you’ll have `fine_tuned_clip.pth` and `fine_tuned_clip_best.pth` next to the script, plus plots and logs.

---

## How inference works (`code current/app.py`)

1. **Model bootstrap**
   - Loads `openai/clip-vit-base-patch32`.
   - Loads the **fine‑tuned checkpoint** (`fine_tuned_clip.pth`) onto **CUDA if available** else CPU.
   - Sets the model to `eval()`.

2. **Video input**
   - Streamlit file uploader accepts a **.mp4** (or other supported) video.
   - Video is saved temporarily and previewed with `st.video(...)`.

3. **Frame extraction + preprocessing**
   - Extracts frames via OpenCV.
   - Uses `CLIPProcessor` to convert frames to tensors.

4. **Embedding + aggregation**
   - Gets CLIP image features for each frame, **mean‑pools** across time.

5. **Classification**
   - Applies the fine‑tuned classifier to produce **class logits**.
   - Picks the argmax and **returns the predicted action**.

6. **UX**
   - A single **Predict** button runs the pipeline and shows:
     - The **top prediction** and (optionally) a small probability/score.
     - The uploaded video for visual confirmation.

### Run the app

```bash
cd "codes and output/code current"
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## Reproducing results

- Ensure your dataset is in the described structure.
- Run `training.py` to produce model weights.
- Keep the **checkpoint filenames** as used in `app.py` (`fine_tuned_clip.pth`).

If you change class names or counts, update them in `training.py` and (if listed) in the app’s label mappings.

---

## Files, clearly

- **`code current/training.py`** — dataset loader, CLIP feature extraction, temporal pooling, linear head, training loop, validation, test evaluation, plots, and checkpoints.
- **`code current/app.py`** — Streamlit UI, video upload/preview, preprocessing, model load, forward pass, and final prediction.
- **`code current/logs.txt`** — real training/evaluation logs from previous runs.
- **`code pre/*`** — an older iteration of the same idea; kept for reference.

---

## Why this approach

- **CLIP** provides strong image representations out of the box.
- **Temporal mean‑pooling** is simple and reliable for short actions.
- **Fine‑tuning a linear head** on top is fast, data‑efficient, and easy to iterate.
- Streamlit makes the demo effortless for **recruiters** and **stakeholders**.

---

## Commands recap

```bash
# 1) Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Train
cd "codes and output/code current"
python training.py

# 3) Demo
streamlit run app.py
```

---

## Notes for reviewers (what to look for)

- Code clarity: clean separation between **data ingestion**, **feature extraction**, **training loop**, and **inference**.
- Metrics: reported **test accuracy** and plots for loss/accuracy.
- Reproducibility: a single entry script for training and a **one‑click** demo for inference.

---
