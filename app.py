"""
StockSense Pro / ShelfSmart AI
AI-based Cognitive Retail Vision System for Automated Shelf Intelligence

UI theme: "Aurora" — a soft-futuristic pastel glassmorphism design
(mint / sky / lavender / rose aurora glows on deep indigo, frosted-glass
cards, gradient typography, slow relaxing motion).
All detection & analytics logic is unchanged from the original build.
"""

import csv
import io
import json
import os
from datetime import datetime
from html import escape as esc

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


# ------------------------------------------------------------------
# Aurora theme — soft futuristic pastel glassmorphism.
# Pure CSS, no external JS, renders reliably on Streamlit Cloud.
# ------------------------------------------------------------------
AURORA_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --mint:  #7ef9d0;
    --sky:   #7ec8ff;
    --lav:   #b388ff;
    --rose:  #ff9ad5;
    --peach: #ffd39c;
    --red:   #ff8fab;
    --amber: #ffd166;
    --green: #5ef2c0;
    --ink:      #eaf2ff;
    --ink-soft: #c6d1ec;
    --ink-mute: #93a0c4;
    --glass: rgba(255, 255, 255, 0.045);
    --glass-border: rgba(255, 255, 255, 0.12);
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
}

p, li { color: var(--ink-soft); line-height: 1.6; }
.stApp a { color: var(--sky); text-decoration: none; }
.stApp a:hover { color: var(--mint); }

::selection { background: rgba(126, 249, 208, 0.30); }

/* Scrollbar — soft gradient */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, rgba(126, 249, 208, 0.35), rgba(179, 136, 255, 0.35));
    border-radius: 999px;
}

/* ---------------------------------------------------------------
   Background — deep indigo sky, faint stars, and a slowly drifting
   aurora layer that lives BEHIND all content (never tints photos).
   --------------------------------------------------------------- */
.stApp {
    background:
        radial-gradient(1.6px 1.6px at 12% 18%, rgba(255,255,255,.30), transparent 100%),
        radial-gradient(1.2px 1.2px at 34% 64%, rgba(255,255,255,.22), transparent 100%),
        radial-gradient(1.8px 1.8px at 58% 10%, rgba(255,255,255,.25), transparent 100%),
        radial-gradient(1.2px 1.2px at 72% 46%, rgba(255,255,255,.20), transparent 100%),
        radial-gradient(1.6px 1.6px at 90% 76%, rgba(255,255,255,.28), transparent 100%),
        radial-gradient(1.2px 1.2px at 44% 90%, rgba(255,255,255,.18), transparent 100%),
        linear-gradient(180deg, #0c1128 0%, #0a0e22 55%, #0c1128 100%);
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: fixed;
    inset: -22%;
    z-index: -1;
    pointer-events: none;
    background:
        radial-gradient(44% 34% at 24% 30%, rgba(126, 249, 208, 0.12), transparent 70%),
        radial-gradient(50% 40% at 76% 18%, rgba(126, 200, 255, 0.14), transparent 70%),
        radial-gradient(48% 42% at 82% 78%, rgba(179, 136, 255, 0.15), transparent 70%),
        radial-gradient(46% 38% at 18% 82%, rgba(255, 154, 213, 0.12), transparent 70%),
        radial-gradient(36% 28% at 50% 56%, rgba(255, 211, 156, 0.06), transparent 70%);
    animation: auroraDrift 26s ease-in-out infinite alternate;
}
@keyframes auroraDrift {
    0%   { transform: translate(0, 0) scale(1) rotate(0deg); }
    50%  { transform: translate(2.4%, -1.8%) scale(1.06) rotate(1.6deg); }
    100% { transform: translate(-1.8%, 2.4%) scale(1.03) rotate(-1.6deg); }
}

/* ---------------------------------------------------------------
   Typography
   --------------------------------------------------------------- */
h1 {
    font-weight: 800 !important;
    letter-spacing: .3px;
    background: linear-gradient(92deg, #7ef9d0 0%, #7ec8ff 32%, #b388ff 64%, #ff9ad5 100%);
    background-size: 220% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: titleFlow 9s ease-in-out infinite;
    padding-top: .2rem;
}
@keyframes titleFlow {
    0%, 100% { background-position: 0% center; }
    50%      { background-position: 100% center; }
}

h2 {
    color: #f0f4ff !important;
    font-weight: 700 !important;
    letter-spacing: .2px;
}

h3 {
    position: relative;
    color: #eef3ff !important;
    font-weight: 700 !important;
    padding-left: 15px !important;
}
h3::before {
    content: "";
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 5px;
    height: 17px;
    border-radius: 3px;
    background: linear-gradient(180deg, var(--mint), var(--lav));
    box-shadow: 0 0 10px rgba(126, 249, 208, 0.55);
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(126, 249, 208, 0.5), rgba(179, 136, 255, 0.5), transparent);
    margin: 1.3rem 0;
}

/* Top header bar — soft fade instead of hard edge */
div[data-testid="stHeader"] {
    background: linear-gradient(180deg, rgba(12, 17, 40, 0.85), transparent);
    backdrop-filter: blur(8px);
}

/* ---------------------------------------------------------------
   Sidebar — deep glass panel with a mint hairline
   --------------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(17, 22, 48, 0.96), rgba(11, 15, 34, 0.98));
    border-right: 1px solid rgba(126, 200, 255, 0.20);
    box-shadow: 8px 0 34px rgba(4, 7, 18, 0.55);
}
section[data-testid="stSidebar"] hr {
    background: linear-gradient(90deg, rgba(126, 200, 255, 0.35), rgba(179, 136, 255, 0.3), transparent);
}

.brand-block {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 4px 2px 4px;
    margin-bottom: 12px;
}
.brand-logo {
    width: 46px;
    height: 46px;
    flex: 0 0 46px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 23px;
    background: linear-gradient(135deg, rgba(126, 249, 208, 0.22), rgba(126, 200, 255, 0.22) 50%, rgba(179, 136, 255, 0.28));
    border: 1px solid rgba(255, 255, 255, 0.20);
    box-shadow: 0 0 22px rgba(126, 200, 255, 0.35), inset 0 0 12px rgba(255, 255, 255, 0.08);
}
.brand-name {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 1.16rem;
    color: #ffffff;
    letter-spacing: .2px;
    line-height: 1.15;
}
.brand-name span {
    background: linear-gradient(90deg, var(--mint), var(--sky), var(--lav));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.brand-sub {
    font-size: .66rem;
    color: var(--ink-mute);
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-top: 3px;
}
.brand-card {
    margin: 2px 0 4px;
    padding: 10px 12px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.10);
    font-size: .76rem;
    color: var(--ink-mute);
    line-height: 1.55;
}
.brand-card b { color: var(--ink-soft); }

.side-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Sora', sans-serif;
    font-size: .70rem;
    font-weight: 700;
    letter-spacing: .20em;
    text-transform: uppercase;
    color: var(--ink-mute);
    margin: 2px 0 10px;
}
.side-label::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 3px;
    background: linear-gradient(135deg, var(--mint), var(--lav));
    box-shadow: 0 0 9px rgba(126, 249, 208, 0.7);
}

/* Sidebar navigation — soft pill buttons */
div[role="radiogroup"] { display: flex; flex-direction: column; gap: 3px; }
div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 8px 12px;
    margin: 0;
    transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
    cursor: pointer;
}
div[role="radiogroup"] label p { color: var(--ink-soft); font-weight: 600; }
div[role="radiogroup"] label:hover {
    background: rgba(126, 200, 255, 0.08);
    transform: translateX(3px);
}
div[role="radiogroup"] label:hover p { color: #eaf2ff; }
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(126, 249, 208, 0.15), rgba(126, 200, 255, 0.13) 60%, rgba(179, 136, 255, 0.14));
    border-color: rgba(126, 249, 208, 0.45);
    box-shadow: 0 0 18px rgba(126, 249, 208, 0.14);
}
div[role="radiogroup"] label:has(input:checked) p { color: #eafff9; font-weight: 700; }

/* Widget labels */
[data-testid="stWidgetLabel"] p {
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    color: #a9b6d8;
    font-size: .87rem;
}
section[data-testid="stSidebar"] p { color: var(--ink-mute); }

/* Sliders — aurora track */
.stSlider [data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, var(--mint), var(--sky), var(--lav)) !important;
    border-radius: 999px;
}

/* Text inputs — glass fields */
.stTextInput input, div[data-baseweb="input"] input {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    border-radius: 12px !important;
    color: var(--ink) !important;
    font-family: 'Manrope', sans-serif;
}
.stTextInput input:focus, div[data-baseweb="input"] input:focus {
    border-color: var(--sky) !important;
    box-shadow: 0 0 0 3px rgba(126, 200, 255, 0.18) !important;
}
.stTextInput input::placeholder, div[data-baseweb="input"] input::placeholder {
    color: #7c88a8 !important;
}

/* Legend card (stock rules) */
.legend-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 12px 14px;
    margin-top: 10px;
}
.legend-card .lc-title {
    font-family: 'Sora', sans-serif;
    font-size: .78rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 8px;
}
.lc-row {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: .84rem;
    color: var(--ink-soft);
    padding: 3px 0;
}
.lc-row i {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--sc);
    box-shadow: 0 0 9px var(--sc);
    flex: 0 0 9px;
}

/* ---------------------------------------------------------------
   Buttons — glass pills with mint→lavender glow
   --------------------------------------------------------------- */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, rgba(126, 249, 208, 0.12), rgba(126, 200, 255, 0.10) 50%, rgba(179, 136, 255, 0.14)) !important;
    color: #eafff8 !important;
    border: 1px solid rgba(126, 249, 208, 0.45) !important;
    border-radius: 999px !important;
    padding: 0.5rem 1.35rem !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: .02em;
    box-shadow: 0 4px 18px rgba(4, 7, 18, 0.35), 0 0 16px rgba(126, 249, 208, 0.12) !important;
    transition: all 0.22s ease !important;
    backdrop-filter: blur(8px);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    color: #ffffff !important;
    border-color: rgba(179, 136, 255, 0.65) !important;
    box-shadow: 0 8px 26px rgba(4, 7, 18, 0.45), 0 0 26px rgba(179, 136, 255, 0.28) !important;
    transform: translateY(-2px);
}
.stButton > button:active, .stDownloadButton > button:active {
    transform: translateY(0);
}

/* ---------------------------------------------------------------
   File uploader — soft lavender dropzone
   --------------------------------------------------------------- */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(179, 136, 255, 0.40);
    border-radius: 16px;
    padding: 0.55rem;
    background: rgba(179, 136, 255, 0.05);
    transition: all 0.25s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(126, 249, 208, 0.60);
    background: rgba(126, 249, 208, 0.05);
    box-shadow: 0 0 24px rgba(126, 249, 208, 0.12);
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: var(--ink-mute) !important;
}

/* ---------------------------------------------------------------
   Metric cards — frosted glass with gradient hairline
   --------------------------------------------------------------- */
[data-testid="stMetric"] {
    position: relative;
    overflow: hidden;
    background: linear-gradient(160deg, rgba(126, 200, 255, 0.08), rgba(255, 255, 255, 0.02) 60%, rgba(179, 136, 255, 0.06));
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 1.05rem 1.15rem;
    box-shadow: 0 10px 30px rgba(5, 8, 20, 0.40);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: linear-gradient(90deg, var(--mint), var(--sky), var(--lav));
    opacity: 0.85;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(126, 249, 208, 0.5);
    box-shadow: 0 14px 40px rgba(126, 200, 255, 0.18);
}
[data-testid="stMetricLabel"] {
    color: #9fb0d8 !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700;
    font-size: 1.55rem !important;
    text-shadow: 0 0 18px rgba(126, 200, 255, 0.45);
}

/* ---------------------------------------------------------------
   Alerts — rounded glass
   --------------------------------------------------------------- */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 10px 28px rgba(5, 8, 20, 0.40);
    backdrop-filter: blur(10px);
}

/* ---------------------------------------------------------------
   Images — framed with a soft glow
   --------------------------------------------------------------- */
[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid rgba(126, 200, 255, 0.28);
    box-shadow: 0 12px 34px rgba(5, 8, 20, 0.50), 0 0 20px rgba(126, 200, 255, 0.12);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.012);
    border-color: rgba(126, 249, 208, 0.55);
    box-shadow: 0 16px 44px rgba(5, 8, 20, 0.55), 0 0 30px rgba(126, 249, 208, 0.20);
}

/* Generic markdown tables (safety net) */
table { border-collapse: separate; border-spacing: 0 8px; width: 100%; }
th {
    font-family: 'Manrope', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-size: 0.72rem;
    color: var(--ink-mute);
    text-align: left;
}
td {
    background: rgba(255, 255, 255, 0.03);
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding: 10px 14px;
    color: var(--ink-soft);
}
td:first-child {
    border-left: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px 0 0 12px;
}
td:last-child {
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0 12px 12px 0;
}

/* ---------------------------------------------------------------
   HERO — glass card with a slowly orbiting aurora border
   --------------------------------------------------------------- */
.hero {
    position: relative;
    border-radius: 22px;
    padding: 20px 22px;
    overflow: hidden;
    margin: 4px 0 18px;
}
.hero::before {
    content: "";
    position: absolute;
    left: 50%;
    top: 50%;
    width: 220%;
    aspect-ratio: 1 / 1;
    background: conic-gradient(
        from 0deg,
        transparent 0 58%,
        rgba(126, 249, 208, 0.55) 70%,
        rgba(126, 200, 255, 0.65) 79%,
        rgba(179, 136, 255, 0.65) 87%,
        rgba(255, 154, 213, 0.50) 94%,
        transparent 100%
    );
    transform: translate(-50%, -50%) rotate(0turn);
    animation: heroOrbit 11s linear infinite;
}
.hero::after {
    content: "";
    position: absolute;
    inset: 1.5px;
    border-radius: 20.5px;
    background: linear-gradient(155deg, rgba(22, 28, 56, 0.94), rgba(15, 19, 42, 0.95) 45%, rgba(26, 20, 52, 0.94));
}
.hero > * { position: relative; z-index: 1; }
@keyframes heroOrbit {
    from { transform: translate(-50%, -50%) rotate(0turn); }
    to   { transform: translate(-50%, -50%) rotate(1turn); }
}
.hero-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    color: var(--ink-soft);
    background: rgba(255, 255, 255, 0.05);
}
.hero-chip i {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--c);
    box-shadow: 0 0 8px var(--c);
}
.hero p {
    color: var(--ink-soft);
    margin: 0;
    font-size: 1.02rem;
    line-height: 1.65;
}

/* Soft glass note card used for page intros */
.note-card {
    border: 1px solid rgba(255, 255, 255, 0.10);
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 16px;
    box-shadow: 0 8px 24px rgba(5, 8, 20, 0.30);
}
.note-card p { margin: 0; color: var(--ink-soft); }

/* ---------------------------------------------------------------
   STOCK CARDS — tinted glass rails with meter bars
   --------------------------------------------------------------- */
.stock-card {
    --sc: var(--green);
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 6px 18px;
    align-items: center;
    padding: 13px 16px 13px 18px;
    margin-bottom: 9px;
    border-radius: 16px;
    background: linear-gradient(90deg, color-mix(in srgb, var(--sc) 8%, transparent), rgba(255, 255, 255, 0.03) 42%);
    border: 1px solid color-mix(in srgb, var(--sc) 24%, transparent);
    border-left: 4px solid var(--sc);
    box-shadow: 0 6px 20px rgba(5, 8, 20, 0.35), 0 0 16px color-mix(in srgb, var(--sc) 12%, transparent);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.stock-card:hover {
    transform: translateX(5px);
    box-shadow: 0 10px 26px rgba(5, 8, 20, 0.45), 0 0 26px color-mix(in srgb, var(--sc) 28%, transparent);
}
.sc-name { font-weight: 700; color: var(--ink); font-size: 0.97rem; }
.sc-meter {
    height: 6px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
    margin-top: 7px;
}
.sc-meter span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, color-mix(in srgb, var(--sc) 55%, var(--sky)), var(--sc));
    box-shadow: 0 0 10px color-mix(in srgb, var(--sc) 55%, transparent);
}
.sc-side { display: flex; align-items: center; gap: 12px; }
.sc-count { color: var(--ink-mute); font-size: 0.82rem; white-space: nowrap; }

/* Shared status pill */
.pill {
    --sc: var(--green);
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-weight: 800;
    font-size: 0.79rem;
    padding: 5px 12px;
    border-radius: 999px;
    color: var(--sc);
    background: color-mix(in srgb, var(--sc) 13%, transparent);
    border: 1px solid color-mix(in srgb, var(--sc) 38%, transparent);
    white-space: nowrap;
    text-shadow: 0 0 12px color-mix(in srgb, var(--sc) 55%, transparent);
}
.pill i {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--sc);
    box-shadow: 0 0 9px var(--sc);
}

/* ---------------------------------------------------------------
   RESTOCK rows — gradient rank badges
   --------------------------------------------------------------- */
.restock-row {
    --sc: var(--red);
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.09);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.restock-row:hover {
    transform: translateX(4px);
    border-color: color-mix(in srgb, var(--sc) 45%, transparent);
    box-shadow: 0 0 18px color-mix(in srgb, var(--sc) 18%, transparent);
}
.rs-rank {
    width: 30px;
    height: 30px;
    flex: 0 0 30px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    color: #0b1024;
    background: linear-gradient(135deg, var(--sc), color-mix(in srgb, var(--sc) 55%, #ffffff));
    box-shadow: 0 0 14px color-mix(in srgb, var(--sc) 45%, transparent);
}
.rs-name { flex: 1; font-weight: 700; color: var(--ink); }
.rs-meta { color: var(--ink-mute); font-size: 0.84rem; white-space: nowrap; }

/* ---------------------------------------------------------------
   RANKING podium
   --------------------------------------------------------------- */
.rank-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.11);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.rank-card:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(5, 8, 20, 0.50); }
.rank-card.r1 {
    background: linear-gradient(90deg, rgba(126, 249, 208, 0.10), rgba(255, 255, 255, 0.03) 45%);
    border-color: rgba(126, 249, 208, 0.40);
    box-shadow: 0 0 24px rgba(126, 249, 208, 0.12);
}
.rank-card.r2 {
    background: linear-gradient(90deg, rgba(126, 200, 255, 0.10), rgba(255, 255, 255, 0.03) 45%);
    border-color: rgba(126, 200, 255, 0.40);
}
.rank-card.r3 {
    background: linear-gradient(90deg, rgba(179, 136, 255, 0.10), rgba(255, 255, 255, 0.03) 45%);
    border-color: rgba(179, 136, 255, 0.40);
}
.rk-badge { font-size: 1.45rem; filter: drop-shadow(0 0 10px rgba(255, 211, 156, 0.45)); flex: 0 0 auto; }
.rk-info { flex: 1; min-width: 140px; }
.rk-name { font-weight: 800; color: var(--ink); font-family: 'Sora', sans-serif; font-size: 0.95rem; }
.rk-bar {
    height: 7px;
    width: 100%;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
    margin-top: 8px;
}
.rk-bar span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--mint), var(--sky));
    box-shadow: 0 0 12px rgba(126, 200, 255, 0.50);
}
.rk-score {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 1.28rem;
    color: var(--ink);
    white-space: nowrap;
    text-align: right;
}
.rk-score small {
    display: block;
    font-size: 0.60rem;
    color: var(--ink-mute);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 700;
}

/* ---------------------------------------------------------------
   TREND rows — before/after mini bars + delta chips
   --------------------------------------------------------------- */
.trend-row {
    display: grid;
    grid-template-columns: minmax(150px, 1.15fr) 2fr auto auto;
    gap: 16px;
    align-items: center;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.09);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.trend-row:hover {
    transform: translateX(4px);
    border-color: rgba(126, 200, 255, 0.35);
    box-shadow: 0 0 18px rgba(126, 200, 255, 0.10);
}
.tr-name { font-weight: 700; color: var(--ink); }
.tr-note {
    display: inline-block;
    margin-top: 5px;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--peach);
    background: rgba(255, 211, 156, 0.09);
    border: 1px solid rgba(255, 211, 156, 0.30);
    border-radius: 999px;
    padding: 2px 9px;
}
.tr-bars { display: flex; flex-direction: column; gap: 5px; }
.tr-bar { height: 7px; border-radius: 999px; background: rgba(255, 255, 255, 0.07); overflow: hidden; }
.tr-bar span { display: block; height: 100%; border-radius: 999px; }
.tr-bar.before span { background: linear-gradient(90deg, var(--lav), var(--rose)); }
.tr-bar.after span  { background: linear-gradient(90deg, var(--mint), var(--sky)); }
.tr-counts { font-family: 'Sora', sans-serif; font-weight: 700; color: var(--ink-mute); white-space: nowrap; font-size: 0.92rem; }
.tr-counts b { color: var(--ink); }
.tr-delta {
    font-weight: 800;
    font-size: 0.83rem;
    padding: 4px 11px;
    border-radius: 999px;
    white-space: nowrap;
}
.tr-delta.up   { color: var(--green); background: rgba(94, 242, 192, 0.10); border: 1px solid rgba(94, 242, 192, 0.35); }
.tr-delta.down { color: var(--red);   background: rgba(255, 143, 171, 0.10); border: 1px solid rgba(255, 143, 171, 0.35); }
.tr-delta.flat { color: var(--ink-mute); background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.15); }

/* ---------------------------------------------------------------
   HISTORY table — glass rows
   --------------------------------------------------------------- */
.hist-table {
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.03);
    margin-bottom: 14px;
}
.ht-row {
    display: grid;
    grid-template-columns: 52px 1.7fr 1.1fr 1fr 1fr 0.9fr;
    padding: 11px 16px;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.ht-row:last-child { border-bottom: none; }
.ht-head {
    background: rgba(255, 255, 255, 0.05);
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-mute);
    font-weight: 800;
}
.ht-row:not(.ht-head):hover { background: rgba(126, 200, 255, 0.05); }
.ht-num { font-family: 'Sora', sans-serif; color: var(--ink-mute); font-weight: 700; }
.ht-name { font-weight: 700; color: var(--ink); overflow-wrap: anywhere; }
.ht-plain { color: var(--ink-soft); }

/* ---------------------------------------------------------------
   CATEGORY chips — colourful cloud
   --------------------------------------------------------------- */
.supercat { display: flex; align-items: baseline; gap: 10px; margin: 10px 0 10px; }
.supercat span { font-family: 'Sora', sans-serif; font-weight: 700; color: var(--ink); font-size: 1.0rem; }
.supercat em { font-style: normal; font-size: 0.73rem; color: var(--ink-mute); }
.supercat::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(126, 249, 208, 0.4), rgba(179, 136, 255, 0.3), transparent);
    margin-left: 6px;
}
.chip-cloud { display: flex; flex-wrap: wrap; gap: 9px; margin: 2px 0 18px; }
.chip {
    --c: var(--sky);
    font-size: 0.86rem;
    font-weight: 600;
    color: #d5e2ff;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.11);
    cursor: default;
    transition: all 0.2s ease;
}
.chip:hover {
    color: #ffffff;
    border-color: color-mix(in srgb, var(--c) 60%, transparent);
    background: color-mix(in srgb, var(--c) 12%, transparent);
    box-shadow: 0 4px 18px color-mix(in srgb, var(--c) 25%, transparent);
    transform: translateY(-2px);
}
.chip:nth-child(5n+1) { --c: var(--mint); }
.chip:nth-child(5n+2) { --c: var(--sky); }
.chip:nth-child(5n+3) { --c: var(--lav); }
.chip:nth-child(5n+4) { --c: var(--rose); }
.chip:nth-child(5n+5) { --c: var(--peach); }

/* ---------------------------------------------------------------
   PERFORMANCE tiles — big gradient numbers
   --------------------------------------------------------------- */
.perf-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px;
    margin: 14px 0 8px;
}
.perf-tile {
    position: relative;
    overflow: hidden;
    padding: 20px 20px 18px;
    border-radius: 18px;
    background: linear-gradient(160deg, rgba(126, 249, 208, 0.07), rgba(126, 200, 255, 0.05) 50%, rgba(179, 136, 255, 0.08));
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 10px 30px rgba(5, 8, 20, 0.40);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.perf-tile::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: linear-gradient(90deg, var(--mint), var(--sky), var(--lav));
    opacity: 0.85;
}
.perf-tile:hover {
    transform: translateY(-4px);
    border-color: rgba(126, 249, 208, 0.45);
    box-shadow: 0 16px 40px rgba(126, 200, 255, 0.18);
}
.pt-val {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 1.95rem;
    background: linear-gradient(90deg, var(--mint), var(--sky) 55%, var(--lav));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    filter: drop-shadow(0 0 12px rgba(126, 200, 255, 0.35));
}
.pt-name {
    margin-top: 4px;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ink-mute);
    font-weight: 800;
}
.pt-bar {
    margin-top: 12px;
    height: 7px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
}
.pt-bar span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--mint), var(--sky), var(--lav));
    box-shadow: 0 0 12px rgba(126, 249, 208, 0.40);
}

/* Preview-build badge (used only in the demo preview build) */
.preview-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--peach);
    border: 1px dashed rgba(255, 211, 156, 0.5);
    border-radius: 999px;
    padding: 4px 10px;
    margin: 0 0 8px;
    background: rgba(255, 211, 156, 0.06);
}

/* Respect users who prefer calm */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}
"""


def inject_custom_css():
    st.markdown(f"<style>{AURORA_CSS}</style>", unsafe_allow_html=True)


MODEL_PATH = "best.pt"
CATEGORIES_JSON = "rpc_selected_categories.json"

STATUS_COLORS = {
    "Out of Stock": "#ff8fab",  # soft coral
    "Low Stock": "#ffd166",     # warm amber
    "In Stock": "#5ef2c0",      # mint green
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
# Core helpers (unchanged from original build)
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
# UI helpers — presentation only (no analytics logic here)
# ------------------------------------------------------------------
def status_pill(status: str) -> str:
    color = STATUS_COLORS[status]
    return f'<span class="pill" style="--sc:{color};"><i></i>{esc(status)}</span>'


def stock_card_html(row, meter_scale: int = 8) -> str:
    color = STATUS_COLORS[row["status"]]
    meter = min(row["count"] / meter_scale, 1.0) * 100
    return f'''
<div class="stock-card" style="--sc:{color};">
    <div>
        <div class="sc-name">{esc(row["category"])}</div>
        <div class="sc-meter"><span style="width:{meter:.0f}%;"></span></div>
    </div>
    <div class="sc-side">
        <span class="sc-count">{row["count"]} detected</span>
        {status_pill(row["status"])}
    </div>
</div>'''


def restock_row_html(rank: int, row) -> str:
    color = STATUS_COLORS[row["status"]]
    return f'''
<div class="restock-row" style="--sc:{color};">
    <span class="rs-rank">{rank}</span>
    <span class="rs-name">{esc(row["category"])}</span>
    <span class="rs-meta">{row["count"]} detected</span>
    {status_pill(row["status"])}
</div>'''


def rank_card_html(rank: int, item) -> str:
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}")
    medal_html = f'<div class="rk-badge">{medal}</div>'
    return f'''
<div class="rank-card r{min(rank, 3)}">
    {medal_html}
    <div class="rk-info">
        <div class="rk-name">{esc(item["name"])}</div>
        <div class="rk-bar"><span style="width:{item["health_score"]:.0f}%;"></span></div>
    </div>
    <div class="rk-score">{item["health_score"]}%<small>{item["total_items"]} items · Health</small></div>
</div>'''


def trend_row_html(label, before_c, after_c, delta, note) -> str:
    scale = 10.0
    dir_class = "up" if delta > 0 else ("down" if delta < 0 else "flat")
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "■")
    note_html = f'<div class="tr-note">🔥 {esc(note)}</div>' if note else ""
    return f'''
<div class="trend-row">
    <div class="tr-name">{esc(label)}{note_html}</div>
    <div class="tr-bars">
        <div class="tr-bar before"><span style="width:{min(before_c / scale, 1.0) * 100:.0f}%;"></span></div>
        <div class="tr-bar after"><span style="width:{min(after_c / scale, 1.0) * 100:.0f}%;"></span></div>
    </div>
    <div class="tr-counts">{before_c} → <b>{after_c}</b></div>
    <div class="tr-delta {dir_class}">{arrow} {delta:+d}</div>
</div>'''


def history_table_html(history) -> str:
    rows = ['''
<div class="hist-table">
    <div class="ht-row ht-head">
        <span>#</span><span>Image</span><span>Time</span>
        <span>Total Items</span><span>Categories</span><span>Health</span>
    </div>''']
    for i, h in enumerate(reversed(history), 1):
        rows.append(f'''
    <div class="ht-row">
        <span class="ht-num">{i}</span>
        <span class="ht-name">{esc(h["name"])}</span>
        <span class="ht-plain">{esc(h["timestamp"])}</span>
        <span class="ht-plain">{h["total_items"]}</span>
        <span class="ht-plain">{h["categories_detected"]}</span>
        <span>{status_pill("In Stock" if h["health_score"] >= 50 else ("Low Stock" if h["health_score"] > 0 else "Out of Stock"))}</span>
    </div>''')
    rows.append("</div>")
    return "".join(rows)


def perf_tile_html(value: str, name: str, pct: float) -> str:
    return f'''
<div class="perf-tile">
    <div class="pt-val">{value}%</div>
    <div class="pt-name">{esc(name)}</div>
    <div class="pt-bar"><span style="width:{pct:.1f}%;"></span></div>
</div>'''


BRAND_HTML = '''
<div class="brand-block">
    <div class="brand-logo">🛒</div>
    <div>
        <div class="brand-name">ShelfSmart<span>AI</span></div>
        <div class="brand-sub">Cognitive Retail Vision</div>
    </div>
</div>
<div class="brand-card">
    <b>RetailSense AI Solutions Pvt. Ltd.</b><br>
    AI-based Cognitive Retail Vision System for Automated Shelf Intelligence.
</div>
'''


def hero_html() -> str:
    return '''
<div class="hero">
    <div class="hero-chips">
        <span class="hero-chip"><i style="--c:#7ef9d0;"></i>YOLOv8 vision engine</span>
        <span class="hero-chip"><i style="--c:#7ec8ff;"></i>50 product categories</span>
        <span class="hero-chip"><i style="--c:#b388ff;"></i>Live stock intelligence</span>
        <span class="hero-chip"><i style="--c:#ff9ad5;"></i>Restock priorities</span>
    </div>
    <p>Upload one or more shelf images to detect products, count them by category,
    and get instant restocking insights — powered by a YOLOv8 model trained on
    50 retail product categories.</p>
</div>'''


# ------------------------------------------------------------------
# Session state init (for detection history)
# ------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {name, timestamp, total_items, categories_detected, health_score}


inject_custom_css()

# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.markdown(BRAND_HTML, unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown('<div class="side-label">Menu</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigate",
    [
        "🔍 Shelf Detector",
        "📈 Shelf Comparison (Trend)",
        "📜 Detection History",
        "📚 Category Browser",
        "📊 Model Performance",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

if page in ("🔍 Shelf Detector", "📈 Shelf Comparison (Trend)"):
    st.sidebar.markdown('<div class="side-label">⚙️ Detection Settings</div>', unsafe_allow_html=True)
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
        f'''
<div class="legend-card">
    <div class="lc-title">Stock status rules</div>
    <div class="lc-row" style="--sc:{STATUS_COLORS['Out of Stock']};"><i></i> Out of Stock · 0 items</div>
    <div class="lc-row" style="--sc:{STATUS_COLORS['Low Stock']};"><i></i> Low Stock · 1–{low_stock_max} items</div>
    <div class="lc-row" style="--sc:{STATUS_COLORS['In Stock']};"><i></i> In Stock · {low_stock_max + 1}+ items</div>
</div>''',
        unsafe_allow_html=True,
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
    st.markdown(hero_html(), unsafe_allow_html=True)

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
                st.markdown(stock_card_html(row, meter_scale=max(low_stock_max + 4, 8)),
                            unsafe_allow_html=True)

            # Priority restock ordering
            restock_priority = priority_restock_list(stock_table)
            if restock_priority:
                st.markdown("### 🔔 Priority Restock Order (emptiest first)")
                for i, row in enumerate(restock_priority, 1):
                    st.markdown(restock_row_html(i, row), unsafe_allow_html=True)
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
                st.markdown(rank_card_html(i, r), unsafe_allow_html=True)
            st.caption("Health Score = % of tracked categories currently In Stock on that shelf.")
    else:
        st.info("👆 Upload one or more shelf images to get started.")


# ------------------------------------------------------------------
# PAGE 2 — Shelf Comparison / Trend
# ------------------------------------------------------------------
elif page == "📈 Shelf Comparison (Trend)":
    st.title("📈 Shelf Comparison — Stock Trend")
    st.markdown(
        '''
<div class="note-card">
    <p>Upload the same shelf at two different times to see how stock levels
    have changed — useful for spotting fast-selling products.</p>
</div>''',
        unsafe_allow_html=True,
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
            st.markdown(
                '<div class="tr-bars" style="max-width:420px; margin-bottom:14px;">'
                '<div class="tr-bar before"><span style="width:100%;"></span></div>'
                '<div class="tr-bar after"><span style="width:100%;"></span></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.caption("Top bar = before · Bottom bar = after")
            for label in class_names:
                before_c = before_counts.get(label, 0)
                after_c = after_counts.get(label, 0)
                delta = after_c - before_c
                if delta == 0 and before_c == 0:
                    continue  # skip categories with no activity either time
                note = ""
                if delta < 0:
                    note = "likely selling fast" if abs(delta) >= 2 else ""
                st.markdown(trend_row_html(label, before_c, after_c, delta, note), unsafe_allow_html=True)
    else:
        st.info("👆 Upload both a 'before' and 'after' image of the same shelf to compare.")


# ------------------------------------------------------------------
# PAGE 3 — Detection History
# ------------------------------------------------------------------
elif page == "📜 Detection History":
    st.title("📜 Detection History (this session)")
    st.markdown(
        '''
<div class="note-card">
    <p>A running log of every shelf image analyzed in this session.</p>
</div>''',
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.info("No detections yet. Run the Shelf Detector to build a history.")
    else:
        st.markdown(history_table_html(st.session_state.history), unsafe_allow_html=True)
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()


# ------------------------------------------------------------------
# PAGE 4 — Category Browser
# ------------------------------------------------------------------
elif page == "📚 Category Browser":
    st.title("📚 Tracked Product Categories")
    st.markdown(
        f'''
<div class="note-card">
    <p>This system recognizes <b>{len(class_names)} product categories</b>.
    Products outside this list will not be detected — this is a scoped
    prototype, not full-catalog coverage.</p>
</div>''',
        unsafe_allow_html=True,
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
        st.markdown(
            f'<div class="supercat"><span>{esc(supercat.replace("_", " ").title())}</span>'
            f'<em>{len(filtered)} categories</em></div>',
            unsafe_allow_html=True,
        )
        chips = "".join(f'<span class="chip">{esc(i)}</span>' for i in filtered)
        st.markdown(f'<div class="chip-cloud">{chips}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# PAGE 5 — Model Performance & EDA
# ------------------------------------------------------------------
else:
    st.title("📊 Model Performance")
    st.markdown(
        '''
<div class="note-card">
    <p>Final evaluation metrics for the trained YOLOv8 model, computed once
    on the held-out test set during training. These numbers are fixed —
    they describe the model itself, not whatever image you upload on the
    Shelf Detector page.</p>
</div>''',
        unsafe_allow_html=True,
    )

    st.subheader("📈 Final Evaluation Metrics")
    st.markdown(
        '<div class="perf-grid">'
        + perf_tile_html("99.6", "Precision", 99.6)
        + perf_tile_html("99.6", "Recall", 99.6)
        + perf_tile_html("99.5", "mAP@0.5", 99.5)
        + perf_tile_html("87.0", "mAP@0.5:0.95", 87.0)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Evaluated on a held-out test set of 3,796 images across 50 product categories.")
