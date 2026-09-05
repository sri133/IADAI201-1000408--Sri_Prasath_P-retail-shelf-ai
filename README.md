# ShelfSmart AI — StockSense Pro
### AI-based Cognitive Retail Vision System for Automated Shelf Intelligence

## Overview
This project analyzes retail shelf images to detect products, count them per
category, and generate stock-status insights (Out of Stock / Low Stock / In
Stock) with restocking recommendations — built for RetailSense AI Solutions
Pvt. Ltd.'s ShelfSmart AI initiative.

**Live app:** _add your Streamlit Cloud link here_

## Problem Statement
Manual shelf monitoring in retail stores is slow and error-prone, leading to
undetected out-of-stock situations that hurt sales and customer satisfaction.
This system automates that process using computer vision.

## Dataset
- **Source:** [Retail Product Checkout (RPC) Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)
- **Categories used:** 8 of the dataset's 200 categories, selected for strong
  representation and visual distinctiveness across supercategories:
  - 97_milk
  - 129_chocolate
  - 173_personal_hygiene
  - 8_puffed_food
  - 67_dessert
  - 181_tissue
  - 82_drink
  - 48_instant_noodles
- **Source images:** `val2019` + `test2019` splits of RPC (multi-product
  checkout scenes), not `train2019` (single-product exemplar photos), since
  the checkout scenes closely resemble real cluttered shelf images.
- **Preprocessing:** images resized to 640x640, annotations converted from
  COCO bounding-box format to YOLO format (normalized center x/y, width,
  height).
- **Split:** custom 70% train / 15% validation / 15% test split (not RPC's
  original split), 8,031 images total.

## Model
- **Architecture:** YOLOv8n (Ultralytics), pretrained on COCO, fine-tuned on
  the prepared dataset.
- **Training parameters:** 30 epochs, batch size 16, image size 640x640,
  early stopping patience 10.

## Results (test set)

| Metric | Score |
|---|---|
| Precision | 99.7% |
| Recall | 99.7% |
| mAP@0.5 | 99.4% |
| mAP@0.5:0.95 | 87.5% |

Per-category performance was consistently strong (mAP@0.5 between 0.991 and
0.995 across all 8 categories), indicating no single category dominates or
underperforms.

## Stock Status Logic
| Detected count | Status |
|---|---|
| 0 | Out of Stock |
| 1–3 | Low Stock |
| 4+ | In Stock |

The app flags Out of Stock and Low Stock categories with restock
recommendations, and highlights well-stocked categories.

## App Features
- Upload a shelf image (JPG/PNG)
- View the original and annotated (bounding-box) images side by side
- See per-category detection counts and stock status
- Get restocking recommendations
- Adjustable detection confidence threshold

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
├── notebooks/
│   └── shelfsmart_training_pipeline.ipynb   # Full data prep + training pipeline
├── dataset_sample/                 # Small sample of the dataset (not full 25GB)
└── README.md
```

## References
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Retail Product Checkout Dataset (RPC Dataset)](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)
- [Streamlit Documentation](https://docs.streamlit.io/)
- Wei, Xiu-Shen, et al. "RPC: A large-scale retail product checkout dataset." (SKU-110K / RPC benchmark reference)

## Screenshots
_Add screenshots of the app demo here before submission._

## Author
- **Name:**
- **Candidate Registration Number:**
- **Course:** Artificial Intelligence — Machine Learning and Deep Learning
- **School:**
