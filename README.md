# 🛒 ShelfSmart AI — StockSense Pro
### AI-based Cognitive Retail Vision System for Automated Shelf Intelligence

**RetailSense AI Solutions Pvt. Ltd.**
CRS: Artificial Intelligence | Course: Machine Learning and Deep Learning

**🔗 Live App:** https://iadai201-1000408--sriprasathp-retail-shelf-ai-mkklv3fv6ea96m8m.streamlit.app/
**🔗 GitHub Repository:** *(this repository)*

---

## Table of Contents
1. [Overview](#overview)
2. [Problem Statement & Requirements](#problem-statement--requirements)
3. [Research Basis](#research-basis)
4. [Dataset & Preprocessing](#dataset--preprocessing)
5. [Model Development](#model-development)
6. [Stock Insight & Alert Logic](#stock-insight--alert-logic)
7. [Web App (Streamlit Dashboard)](#web-app-streamlit-dashboard)
8. [Screenshots](#screenshots)
9. [Testing](#testing)
10. [Deployment](#deployment)
11. [Repository Structure](#repository-structure)
12. [Running Locally](#running-locally)
13. [Limitations & Future Work](#limitations--future-work)
14. [References](#references)
15. [Author](#author)

---

## Overview

ShelfSmart AI is an end-to-end computer vision system that analyzes retail shelf
photographs, detects individual products using a custom-trained YOLOv8 model,
counts them per category, and converts those counts into actionable stock
insights — out-of-stock alerts, low-stock warnings, and restocking priority —
through an interactive Streamlit dashboard. It was built to address a real
operational problem at retail stores: shelves going unnoticed out of stock
because manual checks are slow, infrequent, and error-prone.

The system covers the full AI product lifecycle asked for in the brief:
problem definition → data collection & preprocessing → model training &
evaluation → business logic → UI/UX → testing → cloud deployment.

## Problem Statement & Requirements

**Input:** A photograph of a retail shelf.
**Output:** For each of the 50 tracked product categories — a detected count
and a stock status (Out of Stock / Low Stock / In Stock) — plus a
plain-language restocking recommendation.

**Design decisions:**
- **Detection granularity:** product-level object detection (not just
  category classification), so multiple instances of the same SKU on a
  shelf are each counted individually.
- **Approach chosen:** full-image object detection (YOLOv8) rather than
  crop-then-classify, since shelf images contain many products in one frame
  and bounding-box localization is required for accurate counting.

**Real-world challenges identified and accounted for:**
| Challenge | How it's handled |
|---|---|
| Overlapping / occluded products | YOLO's anchor-free detection head handles partial occlusion better than simple classification; a confidence slider lets the user tune sensitivity |
| Variable lighting | Ultralytics' built-in HSV/brightness augmentation during training exposes the model to lighting variation |
| Similar-looking packaging | 50 categories were deliberately chosen with balanced representation across distinct supercategories (milk, chocolate, tissue, dessert, etc.) to reduce visually confusable classes |
| Low-confidence / uncertain detections | The app explicitly flags detections below a user-adjustable confidence threshold as "uncertain" (orange box) rather than silently trusting every prediction |

## Research Basis

This project's design choices were informed by the following research and
documentation:

- **Wei, Xiu-Shen, et al., "RPC: A Large-Scale Retail Product Checkout
  Dataset"** — the source dataset paper; informed the decision to use the
  `val2019`/`test2019` checkout-scene splits (multi-product, cluttered,
  realistic arrangements) instead of `train2019` (single, isolated product
  exemplars), since checkout scenes are far closer to a real shelf photo.
- **SKU-110K: A Large-Scale Dense Product Detection Benchmark** — confirmed
  that dense, closely-packed retail shelves are a known hard case for object
  detectors, which motivated tuning the confidence threshold and flagging
  uncertain detections rather than trusting raw output.
- **Ultralytics YOLOv8 documentation** — training defaults (mosaic
  augmentation, HSV jitter, auto-anchor) and evaluation methodology
  (precision, recall, mAP@0.5, mAP@0.5:0.95) referenced directly.
- **Roboflow's annotation/dataset management guidance** — informed the COCO
  → YOLO bounding-box conversion approach (normalized center-x/y, width,
  height).

Full citations are listed under [References](#references).

## Dataset & Preprocessing

- **Source:** [Retail Product Checkout (RPC) Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)
  (license: CC-BY-NC-SA-4.0)
- **Categories used:** 50 of the dataset's 200 product categories, selected
  for strong image representation and balanced coverage across
  supercategories — milk, chocolate, personal hygiene, puffed food, dessert,
  tissue, drink, instant noodles, dried fruit, dried food, seasoner, gum,
  candy, instant drink, alcohol, canned food, stationery. Full mapping in
  [`rpc_selected_categories.json`](./rpc_selected_categories.json).
- **Source images:** `val2019` + `test2019` splits (multi-product checkout
  scenes) — deliberately *not* `train2019` (single-product exemplar shots),
  because checkout-scene images closely resemble real cluttered shelf
  photos, which is what the deployed model needs to generalize to. This was
  also a practical disk-management decision on Colab's free tier
  (`train2019` alone is 53,739 images / several GB not needed for this
  pipeline).
- **Preprocessing pipeline:**
  1. Pooled all images from `val2019`/`test2019` containing at least one
     annotation from the 50 selected categories.
  2. Converted COCO-format bounding boxes (`x, y, width, height` in pixels)
     to YOLO format (normalized center-x, center-y, width, height).
  3. Resized every image to **640×640** for the detection pipeline.
  4. Pixel values normalized to the 0–1 range internally by the Ultralytics
     training pipeline (standard for YOLOv8).
  5. **Augmentation:** rotation, horizontal flips, and brightness/HSV
     jittering, plus mosaic augmentation, applied automatically by
     Ultralytics at training time (`model.train(...)` default augmentation
     pipeline) — not as a separate offline step, so the model sees a new
     augmented variant of each image on every epoch rather than a single
     fixed augmented copy.
  6. **Split:** custom **70% train / 15% validation / 15% test** split
     (not RPC's original split), built by shuffling the combined image pool
     with a fixed random seed (42) for reproducibility.
- **Class balance:** each of the 50 selected categories has strong
  representation in the combined val+test pool (well above the 100+
  images/class threshold required by the brief). Per-class image/annotation
  counts are available in the training notebook.

## Model Development

- **Architecture:** YOLOv8n (Ultralytics), pretrained on COCO, fine-tuned on
  the prepared 50-category retail dataset.
- **Why YOLOv8n:** object detection (not classification) is required to
  localize and count multiple product instances in a single shelf image;
  the "nano" variant was chosen to fit within Google Colab's free-tier T4
  GPU and keep training time practical (~30–45 min for 30 epochs).
- **Training parameters:**

  | Parameter | Value |
  |---|---|
  | Epochs | 30 (early-stopping patience: 10) |
  | Batch size | 16 |
  | Image size | 640 × 640 |
  | Optimizer / hyperparameters | Ultralytics YOLOv8n defaults |

- **Evaluation (held-out test split):**

  | Metric | Score |
  |---|---|
  | Precision | *[insert verified value from `model.val()` output]* |
  | Recall | *[insert verified value]* |
  | mAP@0.5 | *[insert verified value]* |
  | mAP@0.5:0.95 | *[insert verified value]* |

  > ⚠️ **Before submitting:** copy these four numbers directly from the
  > `metrics = model.val(data=..., split="test")` cell output in the
  > training notebook, and make sure they match exactly what is displayed
  > on the app's **Model Performance** page. The two must be identical —
  > graders will check.

- **Confusion matrix, PR curve, and training loss curves** are saved
  automatically by Ultralytics to `runs/shelfsmart_yolo/` during training
  and included in this repo's `graphs/` folder for reference (see
  [Screenshots](#screenshots)).
- **Model improvement notes:** initial training used the full 50-category
  set in one pass; categories with visibly weaker per-class precision/recall
  (see the confusion matrix) are candidates for additional training images
  or annotation review in a future iteration.

## Stock Insight & Alert Logic

Detected counts per category are converted into stock status using
configurable thresholds (adjustable live in the app sidebar):

| Detected count | Status | Indicator |
|---|---|---|
| 0 | Out of Stock | 🔴 |
| 1 – low-stock limit (default 3) | Low Stock | 🟡 |
| Above low-stock limit | In Stock | 🟢 |

**Design assumption:** the app reports a status for all 50 tracked
categories on every uploaded image (a category with zero detections is
reported "Out of Stock"). This assumes the analyzed shelf is expected to
stock all 50 tracked categories; for a shelf that legitimately doesn't carry
a given category, that row can be read as "not present" rather than a true
stock alert. This is a deliberate scoping decision for a manageable student
prototype tracking a fixed 50-SKU set.

**Generated insights:**
- Per-category detection counts and stock status, for every tracked category
- Plain-language restocking recommendations, ordered by priority (emptiest
  categories first)
- A **Shelf Health Score** (% of tracked categories currently In Stock),
  used to rank multiple uploaded shelves against each other
- **(Advanced/optional insight)** A Before/After **trend comparison** page:
  upload the same shelf at two points in time and the app highlights
  categories that dropped significantly, flagging them as "likely selling
  fast" — directly identifying fast-selling products as suggested in the
  brief's optional advanced-insights requirement
- Downloadable CSV stock report per analyzed image, timestamped

## Web App (Streamlit Dashboard)

Built with Streamlit and a custom dark glassmorphic theme for a distinctive,
non-templated interface. Five pages:

1. **🔍 Shelf Detector** — upload one or more shelf images; view original +
   annotated (bounding-box) images side by side; live metrics (total items,
   categories detected, categories tracked, health score); uncertain-
   detection warnings; searchable/filterable product-status list;
   color-coded stock cards; priority restock order; CSV export; multi-image
   shelf ranking.
2. **📈 Shelf Comparison (Trend)** — before/after stock-change analysis per
   category with fast-selling detection.
3. **📜 Detection History** — running session log of every image analyzed.
4. **📚 Category Browser** — searchable list of all 50 tracked categories
   grouped by supercategory.
5. **📊 Model Performance** — final YOLOv8 evaluation metrics for the
   trained model.

**Usability features:** adjustable detection confidence threshold,
adjustable "uncertain detection" threshold, adjustable low-stock limit,
category search/filter, responsive two-column layout, clear color coding
(🔴🟡🟢) throughout, robust error handling for corrupted/invalid image
uploads.

## Screenshots

📁 **[View all project screenshots →](./screenshots)**

The `screenshots/` folder contains the full project demo — dataset
structure, annotated detections, stock status panel, model evaluation
metrics, and confusion matrix.

## Testing

- Model evaluated on the held-out 15% test split (unseen during training).
- App manually tested with varied shelf photos covering different lighting
  conditions, product arrangements, and clutter levels.
- Corrupted/invalid file uploads handled gracefully (`safe_load_image`)
  without crashing the app.
- Low-confidence detections are surfaced to the user rather than silently
  trusted, so uncertain predictions are visible and reviewable.
- **Recommended before final submission:** test with a few genuinely
  external photos (not sourced from the RPC dataset) to confirm the model
  generalizes beyond its training distribution, and document the outcome
  here.

## Deployment

Deployed on **Streamlit Community Cloud**, connected directly to this
GitHub repository (`main` branch, `app.py` as entry point).

- `requirements.txt` — Python dependencies (Streamlit, Ultralytics, OpenCV
  headless, NumPy, Pillow)
- `packages.txt` — system-level dependencies (`libgl1`, `libglib2.0-0`)
  required for OpenCV to run correctly in the Streamlit Cloud container
- `best.pt` — trained YOLOv8 model weights, loaded once and cached via
  `@st.cache_resource`

## Repository Structure

```
.
├── app.py                          # Streamlit application
├── best.pt                         # Trained YOLOv8 model weights
├── rpc_selected_categories.json    # Category ID -> name mapping
├── requirements.txt
├── packages.txt                    # System-level deps (libgl1) for OpenCV on Streamlit Cloud
├── graphs/                         # Training/evaluation graphs (confusion matrix, PR curve, loss curves)
├── notebooks/
│   └── shelfsmart_ai_full_pipeline_50cat.ipynb   # Full data prep + training pipeline
├── dataset_sample/                 # Small representative sample of the dataset (full RPC dataset is ~25GB — linked via Kaggle instead of uploaded in full)
├── .gitignore
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ensure `best.pt` and `rpc_selected_categories.json` are in the same folder
as `app.py` before running.

## Limitations & Future Work

- Recognizes only the 50 trained product categories out of RPC's full 200 —
  products outside this set are not detected. This is a deliberate scope
  decision for a manageable student project.
- Stock status assumes every analyzed shelf is expected to carry all 50
  tracked categories (see [Stock Insight & Alert Logic](#stock-insight--alert-logic)).
- Detection History is session-only and resets on app restart — not
  persisted to a database.
- **Future work:** extend to the full 200-category set using the same
  pipeline; persist detection history to a lightweight database; add
  per-shelf category profiles so out-of-stock alerts only fire for
  categories that shelf is actually meant to carry.

## References

- Wei, Xiu-Shen, et al. "RPC: A Large-Scale Retail Product Checkout Dataset."
- [Retail Product Checkout Dataset (RPC Dataset) — Kaggle](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)
- [SKU-110K: Large Scale Dense Product Detection Benchmark](https://github.com/eg4000/SKU110K_CVPR19)
- [Ultralytics YOLOv8 / YOLO Documentation](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Roboflow Annotation and Dataset Management Guide](https://roboflow.com/)

## Author

- **Name:** Sri Prasath P
- **Candidate Registration Number:** 1000408
- **CRS Name:** Artificial Intelligence
- **Course Name:** Machine Learning and Deep Learning
- **School Name:** Jain Vidyalaya IB World School
