"""
StockSense Pro / ShelfSmart AI
AI-based Cognitive Retail Vision System for Automated Shelf Intelligence
"""

import json
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ShelfSmart AI - Retail Shelf Monitoring",
    page_icon="🛒",
    layout="wide",
)

MODEL_PATH = "best.pt"
CATEGORIES_JSON = "rpc_selected_categories.json"
GRAPHS_DIR = "graphs"

OUT_OF_STOCK_MAX = 0
LOW_STOCK_MAX = 3

STATUS_COLORS = {
    "Out of Stock": "#e74c3c",
    "Low Stock": "#f1c40f",
    "In Stock": "#2ecc71",
}


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


@st.cache_data
def load_category_names(json_path: str):
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return {}


def get_stock_status(count: int) -> str:
    if count <= OUT_OF_STOCK_MAX:
        return "Out of Stock"
    elif count <= LOW_STOCK_MAX:
        return "Low Stock"
    else:
        return "In Stock"


def draw_detections(image: np.ndarray, boxes, class_names):
    annotated = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)

        color = (46, 204, 113)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("🛒 ShelfSmart AI")
st.sidebar.markdown(
    "**RetailSense AI Solutions Pvt. Ltd.**\n\n"
    "AI-based Cognitive Retail Vision System for Automated Shelf Intelligence."
)
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", ["🔍 Shelf Detector", "📊 Model Performance & EDA"])

st.sidebar.markdown("---")

if page == "🔍 Shelf Detector":
    confidence_threshold = st.sidebar.slider(
        "Detection confidence threshold", min_value=0.1, max_value=0.9, value=0.4, step=0.05
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Stock status rules**\n"
        "- 🔴 Out of Stock: 0 items\n"
        "- 🟡 Low Stock: 1-3 items\n"
        "- 🟢 In Stock: 4+ items"
    )

# ------------------------------------------------------------------
# PAGE 1 — Shelf Detector
# ------------------------------------------------------------------
if page == "🔍 Shelf Detector":
    st.title("ShelfSmart AI — Automated Shelf Intelligence")
    st.write(
        "Upload a shelf image to detect products, count them by category, "
        "and get instant restocking insights."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found. Place your trained `best.pt` in the repo root.")
        st.stop()

    model = load_model(MODEL_PATH)
    class_names = model.names if isinstance(model.names, list) else [model.names[i] for i in range(len(model.names))]

    uploaded_file = st.file_uploader("Upload a shelf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Running detection..."):
            results = model.predict(image_bgr, conf=confidence_threshold, verbose=False)
            result = results[0]

        annotated_bgr = draw_detections(image_bgr, result.boxes, class_names)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Detected Products")
            st.image(annotated_rgb, use_container_width=True)

        counts = {}
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
            counts[label] = counts.get(label, 0) + 1

        st.markdown("---")
        st.subheader("📊 Stock Insights")

        if not counts:
            st.warning("No products detected. Try lowering the confidence threshold in the sidebar.")
        else:
            total_items = sum(counts.values())
            num_categories_detected = len(counts)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Items Detected", total_items)
            m2.metric("Categories Detected", num_categories_detected)
            m3.metric("Categories Tracked", len(class_names))

            st.markdown("### Product-wise Stock Status")
            for label in class_names:
                count = counts.get(label, 0)
                status = get_stock_status(count)
                color = STATUS_COLORS[status]
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                padding:10px 14px; margin-bottom:6px; border-radius:8px;
                                background-color:#1e1e1e; border-left:6px solid {color};">
                        <span style="font-weight:600;">{label}</span>
                        <span>{count} detected</span>
                        <span style="color:{color}; font-weight:700;">{status}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("### 🔔 Recommendations")
            out_of_stock = [c for c in class_names if get_stock_status(counts.get(c, 0)) == "Out of Stock"]
            low_stock = [c for c in class_names if get_stock_status(counts.get(c, 0)) == "Low Stock"]
            in_stock = [c for c in class_names if get_stock_status(counts.get(c, 0)) == "In Stock"]

            if out_of_stock:
                st.error(f"🔴 Restock needed immediately: {', '.join(out_of_stock)}")
            if low_stock:
                st.warning(f"🟡 Running low, plan restock soon: {', '.join(low_stock)}")
            if in_stock:
                st.success(f"🟢 Well stocked: {', '.join(in_stock)}")
    else:
        st.info("👆 Upload a shelf image to get started.")

# ------------------------------------------------------------------
# PAGE 2 — Model Performance & EDA
# ------------------------------------------------------------------
else:
    st.title("📊 Model Performance & Dataset EDA")
    st.write(
        "Training results, evaluation metrics, and dataset exploration graphs "
        "generated during model development (Ultralytics YOLOv8)."
    )

    graph_files = {
        "results.png": "Training curves — loss, precision, recall, mAP over epochs",
        "confusion_matrix.png": "Confusion matrix (raw counts)",
        "confusion_matrix_normalized.png": "Confusion matrix (normalized)",
        "PR_curve.png": "Precision-Recall curve",
        "F1_curve.png": "F1-Confidence curve",
        "P_curve.png": "Precision-Confidence curve",
        "R_curve.png": "Recall-Confidence curve",
        "category_distribution.png": "Dataset category distribution (EDA)",
        "val_batch0_pred.jpg": "Sample validation predictions",
    }

    if not os.path.exists(GRAPHS_DIR):
        st.warning(
            f"No `{GRAPHS_DIR}/` folder found in the repo. "
            f"Add your exported training graphs there (see README) to display them here."
        )
    else:
        found_any = False
        for filename, caption in graph_files.items():
            filepath = os.path.join(GRAPHS_DIR, filename)
            if os.path.exists(filepath):
                found_any = True
                st.subheader(caption)
                st.image(filepath, use_container_width=True)
                st.markdown("---")
        if not found_any:
            st.warning(f"No matching graph files found inside `{GRAPHS_DIR}/`.")

    st.subheader("📈 Final Evaluation Metrics")
    st.markdown(
        """
        | Metric | Score |
        |---|---|
        | Precision | *(fill in from your Colab evaluation output)* |
        | Recall | *(fill in)* |
        | mAP@0.5 | *(fill in)* |
        | mAP@0.5:0.95 | *(fill in)* |
        """
    )
