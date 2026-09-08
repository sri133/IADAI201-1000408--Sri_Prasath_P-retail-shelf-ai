"""
StockSense Pro / ShelfSmart AI
AI-based Cognitive Retail Vision System for Automated Shelf Intelligence
"""

import csv
import io
import json
import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
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

STATUS_COLORS = {
    "Out of Stock": "#e74c3c",
    "Low Stock": "#f1c40f",
    "In Stock": "#2ecc71",
}
STATUS_ICON = {"Out of Stock": "🔴", "Low Stock": "🟡", "In Stock": "🟢"}


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


@st.cache_data
def load_category_names(json_path: str):
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return {}


# ------------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------------
def get_stock_status(count: int, low_stock_max: int) -> str:
    if count == 0:
        return "Out of Stock"
    elif count <= low_stock_max:
        return "Low Stock"
    else:
        return "In Stock"


def draw_detections(image: np.ndarray, boxes, class_names, low_conf_flag: float):
    """Draw boxes; low-confidence detections drawn in orange as 'uncertain'."""
    annotated = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)

        uncertain = conf < low_conf_flag
        color = (0, 165, 255) if uncertain else (46, 204, 113)  # BGR: orange vs green
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        tag = "?" if uncertain else ""
        text = f"{label} {conf:.2f}{tag}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


def run_detection(image_bgr, model, class_names, conf_threshold, low_conf_flag):
    results = model.predict(image_bgr, conf=conf_threshold, verbose=False)
    result = results[0]

    counts = {}
    uncertain_detections = []
    for box in result.boxes:
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        counts[label] = counts.get(label, 0) + 1
        if conf < low_conf_flag:
            uncertain_detections.append((label, conf))

    annotated_bgr = draw_detections(image_bgr, result.boxes, class_names, low_conf_flag)
    return annotated_bgr, counts, uncertain_detections


def build_stock_table(counts, class_names, low_stock_max):
    rows = []
    for label in class_names:
        count = counts.get(label, 0)
        status = get_stock_status(count, low_stock_max)
        rows.append({"category": label, "count": count, "status": status})
    return rows


def stock_health_score(stock_table):
    """% of tracked categories that are In Stock — used for shelf ranking."""
    if not stock_table:
        return 0.0
    in_stock = sum(1 for r in stock_table if r["status"] == "In Stock")
    return round(100 * in_stock / len(stock_table), 1)


def generate_csv_report(stock_table, image_name):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Image", "Category", "Count", "Status", "Generated At"])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in stock_table:
        writer.writerow([image_name, row["category"], row["count"], row["status"], ts])
    return buf.getvalue()


def priority_restock_list(stock_table):
    """Out-of-stock and low-stock items, worst (emptiest) first."""
    needs_restock = [r for r in stock_table if r["status"] in ("Out of Stock", "Low Stock")]
    return sorted(needs_restock, key=lambda r: r["count"])


def safe_load_image(uploaded_file):
    """Returns a PIL image or None if the file isn't a valid/openable image."""
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image.load()
        return image
    except (UnidentifiedImageError, OSError):
        return None


# ------------------------------------------------------------------
# Session state init (for detection history)
# ------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {name, timestamp, total_items, categories_detected, health_score}


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("🛒 ShelfSmart AI")
st.sidebar.markdown(
    "**RetailSense AI Solutions Pvt. Ltd.**\n\n"
    "AI-based Cognitive Retail Vision System for Automated Shelf Intelligence."
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🔍 Shelf Detector",
        "📈 Shelf Comparison (Trend)",
        "📜 Detection History",
        "📚 Category Browser",
        "📊 Model Performance",
    ],
)

st.sidebar.markdown("---")

if page in ("🔍 Shelf Detector", "📈 Shelf Comparison (Trend)"):
    st.sidebar.subheader("Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Detection confidence threshold", min_value=0.1, max_value=0.9, value=0.4, step=0.05
    )
    low_conf_flag = st.sidebar.slider(
        "Flag detections below this confidence as 'uncertain'",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05
    )
    low_stock_max = st.sidebar.slider(
        "Low Stock upper limit (items)", min_value=1, max_value=10, value=3, step=1
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Stock status rules**\n"
        f"- 🔴 Out of Stock: 0 items\n"
        f"- 🟡 Low Stock: 1-{low_stock_max} items\n"
        f"- 🟢 In Stock: {low_stock_max + 1}+ items"
    )

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file `{MODEL_PATH}` not found. Place your trained `best.pt` in the repo root.")
    st.stop()

model = load_model(MODEL_PATH)
class_names = model.names if isinstance(model.names, list) else [model.names[i] for i in range(len(model.names))]


# ------------------------------------------------------------------
# PAGE 1 — Shelf Detector (single + batch)
# ------------------------------------------------------------------
if page == "🔍 Shelf Detector":
    st.title("ShelfSmart AI — Automated Shelf Intelligence")
    st.write(
        "Upload one or more shelf images to detect products, count them by category, "
        "and get instant restocking insights."
    )

    uploaded_files = st.file_uploader(
        "Upload shelf image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        batch_results = []  # for ranking across multiple images

        for uploaded_file in uploaded_files:
            image = safe_load_image(uploaded_file)
            if image is None:
                st.error(f"⚠️ Could not read `{uploaded_file.name}` — it may not be a valid image file. Skipped.")
                continue

            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            with st.spinner(f"Running detection on {uploaded_file.name}..."):
                try:
                    annotated_bgr, counts, uncertain = run_detection(
                        image_bgr, model, class_names, confidence_threshold, low_conf_flag
                    )
                except Exception as e:
                    st.error(f"⚠️ Detection failed on `{uploaded_file.name}`: {e}")
                    continue

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            st.markdown(f"## 🖼️ {uploaded_file.name}")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("Detected Products")
                st.image(annotated_rgb, use_container_width=True)
                st.caption("🟢 confident detection   🟠 low-confidence / uncertain")

            stock_table = build_stock_table(counts, class_names, low_stock_max)
            total_items = sum(counts.values())
            categories_detected = len(counts)
            health = stock_health_score(stock_table)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Items Detected", total_items)
            m2.metric("Categories Detected", categories_detected)
            m3.metric("Categories Tracked", len(class_names))
            m4.metric("Shelf Health Score", f"{health}%")

            if uncertain:
                st.warning(
                    f"⚠️ {len(uncertain)} detection(s) had low confidence and may need a manual check: "
                    + ", ".join(f"{lbl} ({conf:.2f})" for lbl, conf in uncertain[:10])
                    + (" ..." if len(uncertain) > 10 else "")
                )

            # Category search/filter
            search_term = st.text_input(
                f"🔎 Filter categories (image: {uploaded_file.name})", key=f"filter_{uploaded_file.name}"
            )
            filtered_table = [
                r for r in stock_table if search_term.lower() in r["category"].lower()
            ] if search_term else stock_table

            st.markdown("### Product-wise Stock Status")
            for row in filtered_table:
                color = STATUS_COLORS[row["status"]]
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                padding:10px 14px; margin-bottom:6px; border-radius:8px;
                                background-color:#1e1e1e; border-left:6px solid {color};">
                        <span style="font-weight:600;">{row['category']}</span>
                        <span>{row['count']} detected</span>
                        <span style="color:{color}; font-weight:700;">{STATUS_ICON[row['status']]} {row['status']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Priority restock ordering
            restock_priority = priority_restock_list(stock_table)
            if restock_priority:
                st.markdown("### 🔔 Priority Restock Order (emptiest first)")
                for i, row in enumerate(restock_priority, 1):
                    st.markdown(f"{i}. **{row['category']}** — {row['count']} detected ({row['status']})")
            else:
                st.success("🟢 All tracked categories are adequately stocked.")

            # Downloadable report
            csv_data = generate_csv_report(stock_table, uploaded_file.name)
            st.download_button(
                label="⬇️ Download stock report (CSV)",
                data=csv_data,
                file_name=f"stock_report_{uploaded_file.name}.csv",
                mime="text/csv",
                key=f"download_{uploaded_file.name}",
            )

            # Log to session history
            st.session_state.history.append({
                "name": uploaded_file.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_items": total_items,
                "categories_detected": categories_detected,
                "health_score": health,
            })

            batch_results.append({"name": uploaded_file.name, "health_score": health, "total_items": total_items})
            st.markdown("---")

        # Shelf ranking, only meaningful with 2+ images
        if len(batch_results) > 1:
            st.markdown("## 🏆 Shelf Ranking")
            ranked = sorted(batch_results, key=lambda r: r["health_score"], reverse=True)
            for i, r in enumerate(ranked, 1):
                st.markdown(f"{i}. **{r['name']}** — Health Score: {r['health_score']}% ({r['total_items']} items detected)")
            st.caption("Health Score = % of tracked categories currently In Stock on that shelf.")
    else:
        st.info("👆 Upload one or more shelf images to get started.")


# ------------------------------------------------------------------
# PAGE 2 — Shelf Comparison / Trend
# ------------------------------------------------------------------
elif page == "📈 Shelf Comparison (Trend)":
    st.title("📈 Shelf Comparison — Stock Trend")
    st.write(
        "Upload the same shelf at two different times to see how stock levels "
        "have changed — useful for spotting fast-selling products."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Before")
        before_file = st.file_uploader("Upload earlier shelf image", type=["jpg", "jpeg", "png"], key="before")
    with col_b:
        st.subheader("After")
        after_file = st.file_uploader("Upload later shelf image", type=["jpg", "jpeg", "png"], key="after")

    if before_file and after_file:
        before_img = safe_load_image(before_file)
        after_img = safe_load_image(after_file)

        if before_img is None or after_img is None:
            st.error("⚠️ One or both files could not be read as valid images. Please re-upload.")
        else:
            before_bgr = cv2.cvtColor(np.array(before_img), cv2.COLOR_RGB2BGR)
            after_bgr = cv2.cvtColor(np.array(after_img), cv2.COLOR_RGB2BGR)

            with st.spinner("Analyzing both images..."):
                _, before_counts, _ = run_detection(before_bgr, model, class_names, confidence_threshold, low_conf_flag)
                _, after_counts, _ = run_detection(after_bgr, model, class_names, confidence_threshold, low_conf_flag)

            col1, col2 = st.columns(2)
            with col1:
                st.image(before_img, caption="Before", use_container_width=True)
            with col2:
                st.image(after_img, caption="After", use_container_width=True)

            st.markdown("### 📊 Stock Change by Category")
            for label in class_names:
                before_c = before_counts.get(label, 0)
                after_c = after_counts.get(label, 0)
                delta = after_c - before_c
                if delta == 0 and before_c == 0:
                    continue  # skip categories with no activity either time
                arrow = "⬆️" if delta > 0 else ("⬇️" if delta < 0 else "➡️")
                note = ""
                if delta < 0:
                    note = " — likely selling fast" if abs(delta) >= 2 else ""
                st.markdown(f"**{label}**: {before_c} → {after_c}  {arrow} ({delta:+d}){note}")
    else:
        st.info("👆 Upload both a 'before' and 'after' image of the same shelf to compare.")


# ------------------------------------------------------------------
# PAGE 3 — Detection History
# ------------------------------------------------------------------
elif page == "📜 Detection History":
    st.title("📜 Detection History (this session)")
    st.write("A running log of every shelf image analyzed in this session.")

    if not st.session_state.history:
        st.info("No detections yet. Run the Shelf Detector to build a history.")
    else:
        st.markdown(
            "| # | Image | Time | Total Items | Categories Detected | Health Score |\n"
            "|---|---|---|---|---|---|"
        )
        for i, h in enumerate(reversed(st.session_state.history), 1):
            st.markdown(
                f"| {i} | {h['name']} | {h['timestamp']} | {h['total_items']} | "
                f"{h['categories_detected']} | {h['health_score']}% |"
            )
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()


# ------------------------------------------------------------------
# PAGE 4 — Category Browser
# ------------------------------------------------------------------
elif page == "📚 Category Browser":
    st.title("📚 Tracked Product Categories")
    st.write(
        f"This system recognizes **{len(class_names)} product categories**. "
        "Products outside this list will not be detected — this is a scoped "
        "prototype, not full-catalog coverage."
    )

    search = st.text_input("🔎 Search categories")

    supercats = {}
    for name in class_names:
        parts = name.split("_", 1)
        supercat = parts[1] if len(parts) > 1 else "other"
        supercats.setdefault(supercat, []).append(name)

    for supercat, items in sorted(supercats.items()):
        filtered = [i for i in items if search.lower() in i.lower()] if search else items
        if not filtered:
            continue
        st.subheader(supercat.replace("_", " ").title())
        st.write(", ".join(filtered))


# ------------------------------------------------------------------
# PAGE 5 — Model Performance & EDA
# ------------------------------------------------------------------
else:
    st.title("📊 Model Performance")
    st.write(
        "Final evaluation metrics for the trained YOLOv8 model, computed once "
        "on the held-out test set during training. These numbers are fixed — "
        "they describe the model itself, not whatever image you upload on the "
        "Shelf Detector page."
    )

    st.subheader("📈 Final Evaluation Metrics")
    st.markdown(
        """
        | Metric | Score |
        |---|---|
        | Precision | 99.6% |
        | Recall | 99.6% |
        | mAP@0.5 | 99.5% |
        | mAP@0.5:0.95 | 87.0% |
        """
    )
    st.caption("Evaluated on a held-out test set of 3,796 images across 50 product categories.")
