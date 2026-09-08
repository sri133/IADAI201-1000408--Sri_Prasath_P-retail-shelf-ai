# ShelfSmart AI — StockSense Pro
### AI-based Cognitive Retail Vision System for Automated Shelf Intelligence

## Overview
This project analyzes retail shelf images to detect products, count them per
category, and generate stock-status insights (Out of Stock / Low Stock / In
Stock) with restocking recommendations — built for RetailSense AI Solutions
Pvt. Ltd.'s ShelfSmart AI initiative.

**Live app:** https://iadai201-1000408--sriprasathp-retail-shelf-ai-mkklv3fv6ea96m8m.streamlit.app/

## Problem Statement
Manual shelf monitoring in retail stores is slow and error-prone, leading to
undetected out-of-stock situations that hurt sales and customer satisfaction.
This system automates that process using computer vision: a store manager
uploads a shelf photo and instantly sees which products are present, how many
of each, and what needs restocking — without manually counting anything.

## Dataset
- **Source:** [Retail Product Checkout (RPC) Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)
- **License:** CC-BY-NC-SA-4.0
- **Categories used:** 50 of the dataset's 200 categories, selected for
  strong representation and coverage across supercategories (milk,
  chocolate, personal hygiene, puffed food, dessert, tissue, drink, instant
  noodles, dried fruit, dried food, seasoner, gum, candy, instant drink,
  alcohol, canned food, stationery). Full category list: see
  `rpc_selected_categories.json` in this repo.
- **Source images:** `val2019` + `test2019` splits of RPC (multi-product
  checkout scenes), not `train2019` (single-product exemplar photos), since
  the checkout scenes closely resemble real cluttered shelf images.
- **Preprocessing:**
  - Images resized to 640x640
  - Annotations converted from COCO bounding-box format
    (`x, y, width, height` in pixels) to YOLO format (normalized center x/y,
    width, height)
  - Augmentation handled by Ultralytics' built-in training-time pipeline
    (rotation, flips, brightness/HSV jitter, mosaic)
- **Split:** custom 70% train / 15% validation / 15% test split (not RPC's
  original split), built from the combined pool of images containing the
  50 selected categories.

## Model
- **Architecture:** YOLOv8n (Ultralytics), pretrained on COCO, fine-tuned on
  the prepared dataset.
- **Training parameters:**
  - Epochs: 30 (early stopping patience: 10)
  - Batch size: 16
  - Image size: 640x640
  - Optimizer / hyperparameters: Ultralytics defaults for YOLOv8n

## Results (test set)

| Metric | Score |
|---|---|
| Precision | *(fill in from your 50-category evaluation output)* |
| Recall | *(fill in)* |
| mAP@0.5 | *(fill in)* |
| mAP@0.5:0.95 | *(fill in)* |

Per-category metrics and training curves (loss, confusion matrix,
precision-recall curve) are available in the `graphs/` folder and displayed
in the app's **Model Performance & EDA** page.

## Stock Status Logic
| Detected count | Status |
|---|---|
| 0 | Out of Stock |
| 1–3 | Low Stock |
| 4+ | In Stock |

The app flags Out of Stock and Low Stock categories with restock
recommendations, and highlights well-stocked categories, for every one of
the 50 tracked categories per uploaded image.

## App Features
- Upload a shelf image (JPG/PNG)
- View the original and annotated (bounding-box) images side by side
- Per-category detection counts and stock status for all 50 tracked
  categories
- Plain-language restocking recommendations
- Adjustable detection confidence threshold
- Dedicated Model Performance & EDA page showing training graphs and metrics

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Ensure `best.pt` (trained model weights) and `rpc_selected_categories.json`
are in the same folder as `app.py`.

## Repository Structure
```
.
├── app.py                          # Streamlit application
├── best.pt                         # Trained YOLOv8 model weights
├── rpc_selected_categories.json    # Category ID -> name mapping
├── requirements.txt
├── packages.txt                    # System-level deps (libgl1) for OpenCV on Streamlit Cloud
├── graphs/                         # Training/evaluation graphs (confusion matrix, PR curve, etc.)
├── notebooks/
│   └── shelfsmart_training_pipeline.ipynb   # Full data prep + training pipeline
├── dataset_sample/                 # Small sample of the dataset (not full 25GB)
└── README.md
```

## Deployment
Deployed on Streamlit Community Cloud, connected directly to this GitHub
repository (`main` branch, `app.py` as the entry point). System-level
dependencies (`libgl1`, `libglib2.0-0`) are installed via `packages.txt` to
support OpenCV on the deployment image.

## Testing
The model was tested on unseen shelf images (held-out test split) covering
varied lighting and product arrangements. Detection was also manually
verified against sample checkout-scene images to confirm bounding boxes and
category labels aligned correctly with the products present.

## Limitations
This prototype recognizes only the 50 trained product categories out of RPC's
full 200 — products outside this set are not detected. This is a deliberate
scope decision for a manageable student project; the same pipeline
(category selection → annotation conversion → training) can be repeated with
additional categories to extend coverage.

## References
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Retail Product Checkout Dataset (RPC Dataset)](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)
- [Streamlit Documentation](https://docs.streamlit.io/)
- Wei, Xiu-Shen, et al. "RPC: A large-scale retail product checkout dataset."

## Screenshots
_Add screenshots of the app demo here before submission — upload screen,
a detection result with bounding boxes, and the stock-status/recommendations
panel._

## Author
- **Name:** Sri Prasath P
- **Candidate Registration Number:** 1000408
- **Course:** Artificial Intelligence — Machine Learning and Deep Learning
- **School:**Jain Vidyalaya IB World School
