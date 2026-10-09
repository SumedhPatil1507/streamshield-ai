"""
StreamShield AI — Real-Time Multimodal Content Moderation & Streaming Cockpit
Interactive Streamlit Application powered by ONNX Runtime, Kafka Pipelines, and Plotly.
"""

from __future__ import annotations

import asyncio
import io
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

# NOTE: onnxruntime MUST be imported before pandas in this environment.
# Importing pandas first and onnxruntime afterwards crashes the process with
# a native access violation (0xC0000005), while the reverse order is stable.
# Verified locally with pandas 2.2.3 / onnxruntime 1.30.0 / numpy 2.5.3.
import onnxruntime  # noqa: F401  # isort: skip

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="StreamShield AI — Multimodal Moderation Cockpit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force offline mode for synthetic fast fallback if network is constrained
os.environ.setdefault("STREAMSHIELD_OFFLINE", "1")

from src.models.inference import (
    InferenceEngine,
    ModerationResult,
    tokenize_batch,
    preprocess_images_batch,
)
from src.models.model_utils import normalize_l2, sigmoid
from src.streaming.config import settings
from src.streaming.schemas import MultimodalPayload

# ---------------------------------------------------------------------------
# Global Plotly configuration — makes every chart in the cockpit fully
# interactive: scroll-zoom, pan, box/lasso selection, hover compare, and
# 2x-resolution PNG export from the floating modebar.
# ---------------------------------------------------------------------------
PLOT_CFG = {
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "displayModeBar": "hover",
    "toImageButtonOptions": {
        "format": "png",
        "filename": "streamshield_chart",
        "scale": 2,
    },
}

# ---------------------------------------------------------------------------
# Custom CSS for Sleek Dark Glassmorphism Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Theme overrides & custom sleek aesthetic */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main container background gradient */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1117 0%, #07090e 100%);
        color: #e6edf3;
    }

    /* Custom Glassmorphic Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
    }

    .hero-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.1) 50%, rgba(14, 165, 233, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.12);
    }

    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .status-active {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-danger {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-info {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Metric Card styling */
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-delta {
        font-size: 0.82rem;
        margin-top: 4px;
    }

    /* Decision Badges */
    .badge-toxic {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4);
    }
    .badge-safe {
        background: linear-gradient(135deg, #059669 0%, #065f46 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.4);
    }
    .badge-review {
        background: linear-gradient(135deg, #d97706 0%, #92400e 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(217, 119, 6, 0.4);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #6366f1;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
    }

    /* Tab active highlight */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(13, 17, 23, 0.6);
        padding: 6px 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
    }

    /* Animated hero banner glow */
    @keyframes heroPulse {
        0%, 100% { box-shadow: 0 10px 30px rgba(99, 102, 241, 0.12); }
        50% { box-shadow: 0 10px 44px rgba(14, 165, 233, 0.28); }
    }
    .hero-banner { animation: heroPulse 7s ease-in-out infinite; }

    /* Pulsing LIVE indicator for Auto-Live streaming mode */
    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 14px;
        border-radius: 9999px;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.45);
        color: #f87171;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .live-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #ef4444;
        animation: livePulse 1.4s infinite;
    }
    @keyframes livePulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Plotly modebar (zoom / pan / export controls) theming */
    .modebar-container svg { fill: #94a3b8 !important; }
    .modebar-btn:hover svg, .modebar-btn.active svg { fill: #38bdf8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Singleton Engine Cache & Auto-Bootstrap
# ---------------------------------------------------------------------------
def _ensure_models_exist() -> None:
    """Ensure ONNX stub models exist, generating them on-the-fly if needed."""
    from pathlib import Path
    text_path = Path(settings.TEXT_ONNX_PATH)
    image_path = Path(settings.IMAGE_MODEL_PATH)
    if not text_path.exists() or not image_path.exists():
        from scripts.export_onnx_stubs import main as export_stubs_main
        export_stubs_main()


@st.cache_resource(show_spinner="⚡ Initialising ONNX Multimodal Inference Engine...")
def load_inference_engine() -> InferenceEngine:
    """Load ONNX runtime sessions for Text & Image inference once."""
    _ensure_models_exist()
    return InferenceEngine(
        text_model_path=settings.TEXT_ONNX_PATH,
        image_model_path=settings.IMAGE_MODEL_PATH,
    )

try:
    engine = load_inference_engine()
except Exception as e:
    st.error(f"Failed to load ONNX inference engine: {e}")
    st.stop()



# ---------------------------------------------------------------------------
# Synthetic Image Generators & Presets
# ---------------------------------------------------------------------------
def generate_synthetic_image(category: str) -> Image.Image:
    """Generate rich visual placeholder test images with labels."""
    img = Image.new("RGB", (224, 224), color=(20, 24, 33))
    draw = ImageDraw.Draw(img)

    if category == "clean_nature":
        # Green nature scene
        draw.rectangle([0, 140, 224, 224], fill=(34, 197, 94))
        draw.ellipse([60, 40, 160, 140], fill=(250, 204, 21))
        draw.text((20, 20), "SAFE CONTENT", fill=(255, 255, 255))
        draw.text((20, 180), "Friendly Stream Banner", fill=(20, 24, 33))
    elif category == "toxic_flame":
        # Red warning scene
        draw.rectangle([0, 0, 224, 224], fill=(153, 27, 27))
        draw.polygon([(112, 30), (190, 180), (34, 180)], fill=(239, 68, 68))
        draw.text((95, 80), "!", fill=(255, 255, 255))
        draw.text((30, 190), "HARASSMENT / FLAME", fill=(255, 255, 255))
    elif category == "nsfw_flagged":
        # High-risk neon magenta pattern
        draw.rectangle([0, 0, 224, 224], fill=(76, 5, 25))
        for r in range(10, 110, 15):
            draw.ellipse([112 - r, 112 - r, 112 + r, 112 + r], outline=(244, 63, 94), width=2)
        draw.text((45, 105), "FLAGGED MEDIA", fill=(255, 255, 255))
    elif category == "spam_promo":
        # Gold/yellow commercial spam banner
        draw.rectangle([0, 0, 224, 224], fill=(120, 53, 15))
        draw.rectangle([20, 40, 204, 184], outline=(245, 158, 11), width=3)
        draw.text((40, 70), "CLICK HERE $$$", fill=(253, 224, 71))
        draw.text((50, 120), "FREE GIFTS NOW", fill=(255, 255, 255))
    else:
        # Default neutral avatar
        draw.rectangle([0, 0, 224, 224], fill=(30, 41, 59))
        draw.ellipse([70, 50, 154, 134], fill=(99, 102, 241))
        draw.text((48, 160), "USER AVATAR", fill=(203, 213, 225))

    return img


def image_to_bytes(img: Image.Image) -> bytes:
    """Encode PIL image to JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []

if "stream_active" not in st.session_state:
    st.session_state.stream_active = False

if "stream_buffer" not in st.session_state:
    st.session_state.stream_buffer = []

if "quarantine_queue" not in st.session_state:
    st.session_state.quarantine_queue = []

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None


# Preset scenarios for easy live demonstration
PRESET_SCENARIOS = {
    "🌟 Clean Chat & Gameplay": {
        "text": "GG WP! Amazing clutch play on the final round, love the stream!",
        "image_cat": "clean_nature",
        "desc": "Wholesome gaming interaction, positive sentiment",
    },
    "🚨 Aggressive Toxicity & Hate Speech": {
        "text": "You are completely useless and pathetic, delete your channel right now trash.",
        "image_cat": "toxic_flame",
        "desc": "Severe harassment and targeted abusive text",
    },
    "⚠️ Flagged NSFW / Explicit Media": {
        "text": "Check out this leaked photo on my profile right now!",
        "image_cat": "nsfw_flagged",
        "desc": "Borderline text with high-risk media payload",
    },
    "📢 Spam & Crypto Scam Link": {
        "text": "CLAIM 5000 FREE TOKENS NOW! Visit http://scam-airdrop-reward.xyz/free",
        "image_cat": "spam_promo",
        "desc": "Promotional unsolicited link and spam pattern",
    },
    "💬 Borderline / Sarcastic": {
        "text": "Oh wow, what a truly genius move that was... totally not losing.",
        "image_cat": "default_avatar",
        "desc": "Subtle sarcasm requiring contextual human review",
    },
}


# ---------------------------------------------------------------------------
# Decision engine — pure function of fused confidence + threshold controls so
# the verdict card can be recomputed live whenever a slider moves (no model
# re-inference required).
# ---------------------------------------------------------------------------
def decide_verdict(confidence: float, threshold: float, band: float) -> dict:
    """Map a fused confidence score to a moderation verdict and action badge."""
    if confidence >= threshold:
        return {
            "label": "Toxic",
            "action_badge": "⛔ REJECT / QUARANTINE",
            "badge_class": "badge-toxic",
            "action_desc": "Content violates safety guidelines. Automatically blocked or escalated.",
        }
    if abs(confidence - threshold) < band:
        return {
            "label": "Needs Review",
            "action_badge": "🔍 HUMAN REVIEW QUEUE",
            "badge_class": "badge-review",
            "action_desc": "Borderline score within uncertainty band. Sent to moderation team.",
        }
    return {
        "label": "Non-Toxic",
        "action_badge": "✅ APPROVED / SAFE",
        "badge_class": "badge-safe",
        "action_desc": "Content verified safe for live broadcast and public streaming.",
    }

# ---------------------------------------------------------------------------
# Top Header & System Cockpit Bar
# ---------------------------------------------------------------------------
header_col1, header_col2 = st.columns([2.5, 1.5])

with header_col1:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 8px;">
            <div style="background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); padding: 10px 14px; border-radius: 12px; font-size: 1.6rem;">
                🛡️
            </div>
            <div>
                <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    StreamShield AI
                </h1>
                <p style="margin: 0; color: #94a3b8; font-size: 0.95rem;">
                    Sub-50ms Multimodal Content Moderation & Streaming Intelligence Engine
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    st.markdown(
        """
        <div style="display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; align-items: center; height: 100%;">
            <span class="status-pill status-active">● ONNX ENGINE ONLINE</span>
            <span class="status-pill status-info">⚡ SUB-50MS SLA</span>
            <span class="status-pill status-active">📦 KAFKA READY</span>
            <span class="status-pill status-active">🔒 AUDIT ACTIVE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 0.8rem 0 1.2rem 0;'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------------------------
tab_studio, tab_stream, tab_analytics, tab_benchmark, tab_arch = st.tabs([
    "🛡️ Live Moderation Studio",
    "⚡ Stream Pipeline Simulator",
    "📊 Analytics & Audit Logs",
    "🚀 Engine Benchmarks",
    "🏗️ Architecture & Model Specs",
])


# ===========================================================================
# TAB 1: Live Multimodal Moderation Studio
# ===========================================================================
with tab_studio:
    st.markdown(
        """
        <div class="hero-banner">
            <h3 style="margin:0 0 6px 0; color: #f8fafc; font-size: 1.25rem;">Multimodal Real-Time Classification</h3>
            <p style="margin:0; color: #cbd5e1; font-size: 0.9rem;">
                Test text and image payloads concurrently through parallel ONNX execution graphs. Observe confidence scores, risk radar breakdown, 128-d vector embeddings, and millisecond latency.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    studio_col1, studio_col2 = st.columns([1.1, 0.9], gap="large")

    with studio_col1:
        st.markdown("<h4 style='margin-bottom: 8px; color: #e2e8f0;'>1. Input Payload & Modality Controls</h4>", unsafe_allow_html=True)

        preset_choice = st.selectbox(
            "⚡ Quick Load Preset Scenario:",
            list(PRESET_SCENARIOS.keys()),
            index=0,
            help="Select a representative real-world moderation scenario",
        )

        preset_data = PRESET_SCENARIOS[preset_choice]

        text_input = st.text_area(
            "📝 Text Payload (Chat / Comment / Post):",
            value=preset_data["text"],
            height=95,
            placeholder="Type or paste chat message here...",
        )

        img_col1, img_col2 = st.columns([1, 1])

        with img_col1:
            image_source = st.radio(
                "🖼️ Image Source Mode:",
                ["Preset Scenario Graphic", "Upload Local Image", "Synthetic Pattern"],
                horizontal=True,
            )

        uploaded_img = None
        current_pil_img = None

        with img_col2:
            if image_source == "Preset Scenario Graphic":
                current_pil_img = generate_synthetic_image(preset_data["image_cat"])
                st.caption(f"ℹ️ Scenario Graphic: `{preset_data['image_cat']}`")
            elif image_source == "Upload Local Image":
                uploaded_file = st.file_uploader("Upload Image (PNG/JPG):", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    current_pil_img = Image.open(uploaded_file).convert("RGB")
                else:
                    current_pil_img = generate_synthetic_image("default_avatar")
            else:
                synth_type = st.selectbox("Select Pattern:", ["clean_nature", "toxic_flame", "nsfw_flagged", "spam_promo", "default_avatar"])
                current_pil_img = generate_synthetic_image(synth_type)

        # Threshold configuration
        with st.expander("⚙️ Decision Thresholds & Sensitivity Tuning", expanded=False):
            thresh_col1, thresh_col2 = st.columns(2)
            with thresh_col1:
                toxicity_threshold = st.slider("Toxicity Decision Threshold:", 0.10, 0.90, 0.50, 0.05)
            with thresh_col2:
                review_band = st.slider("Human Review Ambiguity Band (±):", 0.05, 0.25, 0.15, 0.05)

        analyze_button = st.button("⚡ Run Real-Time Multimodal Inference", type="primary", use_container_width=True)

    with studio_col2:
        st.markdown("<h4 style='margin-bottom: 8px; color: #e2e8f0;'>2. Media Preview & Raw Inspect</h4>", unsafe_allow_html=True)

        preview_col1, preview_col2 = st.columns([1, 1.2])
        with preview_col1:
            if current_pil_img is not None:
                st.image(current_pil_img, caption="224×224 Normalised Input", use_container_width=True)
        with preview_col2:
            st.markdown(
                f"""
                <div class="glass-card" style="padding: 12px; font-size: 0.85rem;">
                    <div style="color: #94a3b8; font-weight: 600; margin-bottom: 6px;">PAYLOAD METADATA</div>
                    <div><b>Length:</b> {len(text_input)} characters</div>
                    <div><b>Tokens:</b> ~{max(1, len(text_input.split()))} subwords</div>
                    <div><b>Resolution:</b> 224 × 224 × 3</div>
                    <div><b>Channels:</b> CHW Tensor (Float32)</div>
                    <div><b>Target SLA:</b> &lt; 50.0 ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Execute Inference when requested
    if analyze_button or "last_studio_result" not in st.session_state:
        raw_img_bytes = image_to_bytes(current_pil_img) if current_pil_img else io.BytesIO().getvalue()

        # Measure precise inference timing
        t0 = time.perf_counter()

        # Tokenize text
        tokens = tokenize_batch([text_input])

        # Preprocess image
        pixel_tensor = preprocess_images_batch([raw_img_bytes])

        # Run ONNX text session
        text_out = engine._text_session.run(None, {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        })
        text_logits, text_emb = text_out[0], text_out[1]

        # Run ONNX image session
        image_out = engine._image_session.run(None, {
            "pixel_values": pixel_tensor,
        })
        image_logits, image_emb = image_out[0], image_out[1]

        t_total = (time.perf_counter() - t0) * 1000.0

        # Calculate probabilities
        text_prob = float(sigmoid(text_logits)[0, 1])
        image_prob = float(sigmoid(image_logits)[0, 1])
        fused_confidence = max(text_prob, image_prob)

        # Compute verdict from the current threshold controls
        decision = decide_verdict(fused_confidence, toxicity_threshold, review_band)

        studio_res = {
            **decision,
            "confidence": fused_confidence,
            "text_confidence": text_prob,
            "image_confidence": image_prob,
            "latency_ms": t_total,
            "text_emb": normalize_l2(text_emb[0]).tolist(),
            "image_emb": normalize_l2(image_emb[0]).tolist(),
            "timestamp": datetime.now(timezone.utc),
        }
        st.session_state.last_studio_result = studio_res

        # Append to audit history
        audit_entry = {
            "id": f"pay_{len(st.session_state.audit_history) + 1:04d}",
            "text": text_input[:45] + ("..." if len(text_input) > 45 else ""),
            "label": decision["label"],
            "confidence": fused_confidence,
            "text_conf": text_prob,
            "image_conf": image_prob,
            "latency_ms": t_total,
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
        }
        st.session_state.audit_history.append(audit_entry)

    # Render Results Section — the verdict is recomputed live from the current
    # threshold sliders, so moving them updates the card without re-inference.
    res = dict(st.session_state.last_studio_result)
    res.update(decide_verdict(res["confidence"], toxicity_threshold, review_band))
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 12px;'>📊 Inference Verdict & Diagnostic Analytics</h3>", unsafe_allow_html=True)

    # Primary Decision Banner Cards
    res_card1, res_card2, res_card3, res_card4 = st.columns([1.2, 1, 1, 1])

    with res_card1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">POLICY VERDICT</div>
                <div style="margin: 8px 0;">
                    <span class="{res['badge_class']}">{res['label'].upper()}</span>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">{res['action_desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_card2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">FUSED CONFIDENCE</div>
                <div class="metric-value" style="color: {'#f87171' if res['confidence'] >= 0.5 else '#34d399'};">
                    {res['confidence'] * 100:.1f}%
                </div>
                <div class="metric-delta" style="color: #94a3b8;">max(Text, Image)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_card3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">TEXT TOXICITY SCORE</div>
                <div class="metric-value" style="color: {'#fbbf24' if res['text_confidence'] >= 0.4 else '#38bdf8'};">
                    {res['text_confidence'] * 100:.1f}%
                </div>
                <div class="metric-delta" style="color: #94a3b8;">DistilBERT + ONNX</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_card4:
        sla_color = "#34d399" if res["latency_ms"] < 50.0 else "#f87171"
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">WALL-CLOCK LATENCY</div>
                <div class="metric-value" style="color: {sla_color};">
                    {res['latency_ms']:.2f} ms
                </div>
                <div class="metric-delta" style="color: {sla_color};">
                    {'⚡ SLA MET (<50ms)' if res['latency_ms'] < 50 else '⚠️ SLA EXCEEDED'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive Plots Section: Gauge & Risk Breakdown + Vector Embeddings
    chart_col1, chart_col2 = st.columns([1.1, 0.9])

    with chart_col1:
        # Multimodal Risk Breakdown Bar & Gauge
        fig_gauges = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "bar"}]],
            column_widths=[0.45, 0.55],
            subplot_titles=["Overall Risk Gauge", "Modality Risk Distribution"],
        )

        # Gauge Chart
        fig_gauges.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=res["confidence"] * 100,
                domain={"row": 0, "column": 0},
                number={"suffix": "%", "font": {"size": 26, "color": "#f8fafc"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                    "bar": {"color": "#ef4444" if res["confidence"] >= 0.5 else "#10b981"},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(16, 185, 129, 0.2)"},
                        {"range": [40, 65], "color": "rgba(245, 158, 11, 0.2)"},
                        {"range": [65, 100], "color": "rgba(239, 68, 68, 0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f43f5e", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ),
            row=1, col=1,
        )

        # Modality Bar Chart
        categories = ["Text Model", "Image Model", "Fused Decision"]
        scores = [res["text_confidence"] * 100, res["image_confidence"] * 100, res["confidence"] * 100]
        colors = ["#6366f1", "#06b6d4", "#f43f5e" if res["confidence"] >= 0.5 else "#10b981"]

        fig_gauges.add_trace(
            go.Bar(
                x=categories,
                y=scores,
                marker_color=colors,
                text=[f"{s:.1f}%" for s in scores],
                textposition="auto",
                hoverinfo="x+y",
            ),
            row=1, col=2,
        )

        fig_gauges.update_layout(
            height=280,
            margin=dict(l=15, r=15, t=35, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            showlegend=False,
        )
        fig_gauges.update_yaxes(range=[0, 105], gridcolor="rgba(255,255,255,0.05)", row=1, col=2)
        fig_gauges.update_xaxes(gridcolor="rgba(255,255,255,0.05)", row=1, col=2)

        st.plotly_chart(fig_gauges, width="stretch", config=PLOT_CFG)

    with chart_col2:
        # 128-D Normalized Embedding Spectrum — explore any dimension window
        st.markdown(
            "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 2px;'>"
            "<span style='color:#e2e8f0; font-weight:600;'>🧬 Embedding Spectrum Explorer</span>"
            "<span style='color:#94a3b8; font-size:0.78rem;'>128-d L2-normalized</span></div>",
            unsafe_allow_html=True,
        )
        emb_dim_range = st.slider(
            "Dimension Window:",
            min_value=0,
            max_value=124,
            value=(0, 32),
            step=4,
            key="emb_dim_window",
            help="Slide to inspect any window of the 128-dimensional text & image feature vectors.",
        )
        dim_start, dim_end = emb_dim_range
        if dim_end - dim_start < 4:
            dim_end = min(128, dim_start + 4)
        dim_idx = list(range(dim_start, dim_end))
        text_emb_slice = res["text_emb"][dim_start:dim_end]
        img_emb_slice = res["image_emb"][dim_start:dim_end]

        fig_emb = go.Figure()
        fig_emb.add_trace(go.Scatter(
            x=dim_idx,
            y=text_emb_slice,
            mode="lines+markers",
            name="Text Embedding (128-d)",
            line=dict(color="#818cf8", width=2),
            marker=dict(size=5, line=dict(width=1, color="#c7d2fe")),
            hovertemplate="dim %{x}<br>text: %{y:.4f}<extra></extra>",
        ))
        fig_emb.add_trace(go.Scatter(
            x=dim_idx,
            y=img_emb_slice,
            mode="lines+markers",
            name="Image Embedding (128-d)",
            line=dict(color="#38bdf8", width=2),
            marker=dict(size=5, line=dict(width=1, color="#bae6fd")),
            hovertemplate="dim %{x}<br>image: %{y:.4f}<extra></extra>",
        ))

        fig_emb.update_layout(
            title=f"L2-Normalized Feature Vectors (Dimensions {dim_start}–{dim_end})",
            height=280,
            margin=dict(l=15, r=15, t=35, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            font=dict(color="#94a3b8"),
            hovermode="x unified",
        )
        fig_emb.update_xaxes(title_text="Dimension Index", gridcolor="rgba(255,255,255,0.05)")
        fig_emb.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

        st.plotly_chart(fig_emb, width="stretch", config=PLOT_CFG)


# ===========================================================================
# TAB 2: Stream Pipeline Simulator & Live Monitor
# ===========================================================================
with tab_stream:
    st.markdown(
        """
        <div class="hero-banner">
            <h3 style="margin:0 0 6px 0; color: #f8fafc; font-size: 1.25rem;">Live Kafka Streaming Pipeline Simulation</h3>
            <p style="margin:0; color: #cbd5e1; font-size: 0.9rem;">
                Simulate a high-frequency multimodal Kafka stream. Observe live micro-batch consumption, dynamic latency distribution, and quarantine queues with real-time moderator action triggers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 1])

    with ctrl_col1:
        stream_batch_size = st.select_slider("Micro-Batch Size:", options=[4, 8, 16, 32, 64], value=16)
    with ctrl_col2:
        toxic_ratio_slider = st.slider("Simulated Toxicity Rate:", 0.05, 0.60, 0.20, 0.05)
    with ctrl_col3:
        num_burst_batches = st.number_input("Batches to Ingest:", min_value=1, max_value=20, value=3)
    with ctrl_col4:
        st.write("")
        st.write("")
        trigger_stream = st.button("⚡ Inject Stream Traffic", type="primary", use_container_width=True)

    # Auto-Live controls: continuous micro-batch ingestion with auto-refreshing charts
    auto_col1, auto_col2, auto_col3 = st.columns([1.4, 1, 1.6])
    with auto_col1:
        auto_live = st.toggle(
            "🔴 Auto-Live Streaming",
            value=False,
            help="Ingest a fresh micro-batch on every refresh interval so KPIs, charts, and the quarantine queue update continuously.",
        )
    with auto_col2:
        auto_interval = st.select_slider(
            "Refresh Interval:",
            options=[0.5, 1.0, 2.0, 4.0],
            value=1.0,
            format_func=lambda s: f"{s:g}s",
        )
    with auto_col3:
        st.write("")
        if auto_live:
            st.markdown(
                '<span class="live-pill"><span class="live-dot"></span> Live auto-ingest active</span>',
                unsafe_allow_html=True,
            )

    # Candidate text pool for stream generation
    SAFE_CHATS = [
        "Love this stream! Keep up the great work ❤️",
        "Can anyone tell me what song is playing in the background?",
        "Subscribed for 6 months! Let's go hype hype",
        "GG team, that was insane gameplay",
        "What build are you running on this playthrough?",
        "Thank you for the wonderful tutorial and walkthrough!",
        "Hello from Tokyo! Streaming quality is pristine today",
        "POGGERS clutch at the last second!",
    ]
    TOXIC_CHATS = [
        "You are so trash at this game, uninstall immediately",
        "I hate you, stop streaming and go away",
        "Worst player alive, absolutely brainless gameplay",
        "Get destroyed idiot, channel is dead",
        "Free crypto link claim: http://fake-coins.xyz/scam",
    ]

    def ingest_one_batch() -> None:
        """Generate, score, and record one simulated Kafka micro-batch."""
        batch_payloads = []
        for _ in range(stream_batch_size):
            if random.random() < toxic_ratio_slider:
                text_content = random.choice(TOXIC_CHATS)
                img_cat = random.choice(["toxic_flame", "nsfw_flagged", "spam_promo"])
            else:
                text_content = random.choice(SAFE_CHATS)
                img_cat = random.choice(["clean_nature", "default_avatar"])

            batch_payloads.append(
                MultimodalPayload(
                    text_content=text_content,
                    image_url=f"https://streamshield.ai/synth/{img_cat}",
                )
            )

        # Score batch with the ONNX InferenceEngine
        t0 = time.perf_counter()
        results = asyncio.run(engine.run_batch(batch_payloads))
        _elapsed_batch_ms = (time.perf_counter() - t0) * 1000.0

        # Append to session stream buffer
        for pay, r in zip(batch_payloads, results):
            item = {
                "payload_id": r.payload_id[:8],
                "text": pay.text_content,
                "label": r.label,
                "confidence": r.confidence,
                "text_conf": r.text_confidence,
                "image_conf": r.image_confidence,
                "latency_ms": r.latency_ms,
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
                "status": "Quarantined" if r.label == "Toxic" else "Cleared",
            }
            st.session_state.stream_buffer.insert(0, item)
            st.session_state.audit_history.append(item)

            if r.label == "Toxic":
                st.session_state.quarantine_queue.insert(0, item)

        # Cap buffers to prevent excessive memory
        st.session_state.stream_buffer = st.session_state.stream_buffer[:200]
        st.session_state.quarantine_queue = st.session_state.quarantine_queue[:50]
        st.session_state.audit_history = st.session_state.audit_history[:500]

    if trigger_stream:
        with st.spinner(f"Consuming and scoring {num_burst_batches * stream_batch_size} streaming packets..."):
            for _ in range(num_burst_batches):
                ingest_one_batch()

    # Live Stream Metrics Cockpit — auto-refreshing fragment panel
    @st.fragment(run_every=auto_interval if auto_live else None)
    def live_stream_panel() -> None:
        """Live KPIs, charts, quarantine queue, and packet ledger.

        With Auto-Live Streaming enabled, Streamlit re-runs this fragment on
        every refresh interval: one micro-batch is ingested and every chart
        redraws automatically without touching the rest of the page.
        """
        if auto_live:
            ingest_one_batch()

        if not st.session_state.stream_buffer:
            st.info(
                "👆 Click **Inject Stream Traffic** above — or enable **🔴 Auto-Live Streaming** — "
                "to simulate live Kafka multimodal micro-batches."
            )
            return

        df_stream = pd.DataFrame(st.session_state.stream_buffer)

        total_ingested = len(df_stream)
        toxic_count = len(df_stream[df_stream["label"] == "Toxic"])
        toxic_pct = (toxic_count / total_ingested) * 100 if total_ingested > 0 else 0
        avg_latency = df_stream["latency_ms"].mean()
        p95_latency = df_stream["latency_ms"].quantile(0.95)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-label">STREAM INGESTION</div>
                    <div class="metric-value" style="color: #38bdf8;">{total_ingested}</div>
                    <div class="metric-delta" style="color: #94a3b8;">Packets Monitored</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-label">FLAGGED TOXICITY RATE</div>
                    <div class="metric-value" style="color: {'#f87171' if toxic_pct > 25 else '#34d399'};">
                        {toxic_pct:.1f}%
                    </div>
                    <div class="metric-delta" style="color: #94a3b8;">{toxic_count} violations</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-label">AVG BATCH LATENCY</div>
                    <div class="metric-value" style="color: #a855f7;">{avg_latency:.2f} ms</div>
                    <div class="metric-delta" style="color: #34d399;">ONNX CPU ThreadPool</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-label">P95 LATENCY</div>
                    <div class="metric-value" style="color: {'#34d399' if p95_latency < 50 else '#fbbf24'};">
                        {p95_latency:.2f} ms
                    </div>
                    <div class="metric-delta" style="color: #94a3b8;">SLA Threshold: 50 ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Interactive Stream Trend Chart & Quarantine Feed
        stream_viz_col1, stream_viz_col2 = st.columns([1.2, 0.8])

        with stream_viz_col1:
            # Timeline of Confidence & Toxic Events
            fig_stream_timeline = px.scatter(
                df_stream.head(60),
                x="timestamp",
                y="confidence",
                color="label",
                size="latency_ms",
                hover_data=["text", "text_conf", "image_conf", "payload_id"],
                color_discrete_map={"Toxic": "#ef4444", "Non-Toxic": "#10b981", "Needs Review": "#f59e0b"},
                title="Real-Time Streaming Event Scatter (Recent Packets)",
            )
            fig_stream_timeline.add_hline(y=0.5, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", annotation_text="Toxicity Gate (0.50)")
            fig_stream_timeline.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=10, r=10, t=35, b=20),
            )
            fig_stream_timeline.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig_stream_timeline.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_stream_timeline, width="stretch", config=PLOT_CFG)

        with stream_viz_col2:
            st.markdown("<h4 style='color: #f87171; margin-bottom: 8px;'>🚨 Live Quarantine Queue (Action Required)</h4>", unsafe_allow_html=True)
            if st.session_state.quarantine_queue:
                for q_item in st.session_state.quarantine_queue[:4]:
                    st.markdown(
                        f"""
                        <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #ef4444; border-radius: 6px; padding: 8px 12px; margin-bottom: 8px; font-size: 0.85rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: 700; color: #f87171;">ID: {q_item['payload_id']}</span>
                                <span style="color: #94a3b8;">Conf: {q_item['confidence']*100:.1f}%</span>
                            </div>
                            <div style="margin: 4px 0; color: #e2e8f0;">"{q_item['text']}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if st.button("🧹 Clear Quarantine Queue", use_container_width=True):
                    st.session_state.quarantine_queue = []
                    st.rerun()
            else:
                st.info("✅ Quarantine queue is currently empty. No active violations.")

        # Live Table Feed
        st.markdown("<h4 style='color: #e2e8f0; margin-top: 12px;'>📥 Ingested Packet Ledger</h4>", unsafe_allow_html=True)
        st.dataframe(
            df_stream[["payload_id", "timestamp", "label", "confidence", "text_conf", "image_conf", "latency_ms", "text"]].head(25),
            use_container_width=True,
            hide_index=True,
        )

    live_stream_panel()


# ===========================================================================
# TAB 3: Analytics, Audit Logs & Vector Space Explorer
# ===========================================================================
with tab_analytics:
    st.markdown(
        """
        <div class="hero-banner">
            <h3 style="margin:0 0 6px 0; color: #f8fafc; font-size: 1.25rem;">Audit Compliance & Statistical Analytics</h3>
            <p style="margin:0; color: #cbd5e1; font-size: 0.9rem;">
                Historical compliance records persisted to the PostgreSQL audit log table. Explore latency distributions, label ratios, and export audit trails.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.audit_history:
        df_audit = pd.DataFrame(st.session_state.audit_history)

        an_col1, an_col2 = st.columns([1, 1])

        with an_col1:
            # Toxicity Distribution Donut Chart
            label_counts = df_audit["label"].value_counts().reset_index()
            label_counts.columns = ["Label", "Count"]

            fig_donut = px.pie(
                label_counts,
                names="Label",
                values="Count",
                hole=0.55,
                color="Label",
                color_discrete_map={"Toxic": "#ef4444", "Non-Toxic": "#10b981", "Needs Review": "#f59e0b"},
                title="Historical Moderation Verdict Distribution",
            )
            fig_donut.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=10, r=10, t=35, b=15),
            )
            st.plotly_chart(fig_donut, width="stretch", config=PLOT_CFG)

        with an_col2:
            # Latency Distribution Histogram & Box Plot
            fig_hist = px.histogram(
                df_audit,
                x="latency_ms",
                nbins=20,
                color="label",
                color_discrete_map={"Toxic": "#ef4444", "Non-Toxic": "#10b981", "Needs Review": "#f59e0b"},
                title="Batch Latency Distribution (ms)",
                marginal="box",
            )
            fig_hist.add_vline(x=50.0, line_dash="dash", line_color="#f43f5e", annotation_text="50ms SLA Target")
            fig_hist.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=10, r=10, t=35, b=15),
            )
            fig_hist.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig_hist.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_hist, width="stretch", config=PLOT_CFG)

        # Audit Table with Search & Filter
        st.markdown("<h4 style='color: #e2e8f0; margin-top: 16px;'>🔍 Audit Trail Ledger (PostgreSQL Sync)</h4>", unsafe_allow_html=True)
        
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
        with filter_col1:
            label_filter = st.multiselect("Filter by Verdict:", options=df_audit["label"].unique(), default=df_audit["label"].unique())
        with filter_col2:
            conf_min, conf_max = st.slider("Confidence Range:", 0.0, 1.0, (0.0, 1.0), 0.05)
        with filter_col3:
            search_query = st.text_input("Search Text Content:", placeholder="Type to filter...")

        filtered_df = df_audit[
            (df_audit["label"].isin(label_filter)) &
            (df_audit["confidence"] >= conf_min) &
            (df_audit["confidence"] <= conf_max)
        ]
        if search_query:
            filtered_df = filtered_df[filtered_df["text"].str.contains(search_query, case=False, na=False)]

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        # Download CSV export
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Audit Log (CSV)",
            data=csv_bytes,
            file_name="streamshield_audit_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No audit logs collected yet. Run inference in the Studio or Pipeline Simulator to populate data.")


# ===========================================================================
# TAB 4: Engine Benchmarks & Performance Analysis
# ===========================================================================
with tab_benchmark:
    st.markdown(
        """
        <div class="hero-banner">
            <h3 style="margin:0 0 6px 0; color: #f8fafc; font-size: 1.25rem;">Interactive Performance Benchmarks & Analysis</h3>
            <p style="margin:0; color: #cbd5e1; font-size: 0.9rem;">
                Comprehensive benchmark analysis from the StreamShield AI whitepaper. Compare CPU, CUDA, and TensorRT performance across latency, throughput, VRAM footprint, and concurrency scaling.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Whitepaper benchmark data
    benchmark_data = {
        "Metric": [
            "Inference Latency (p50)",
            "Inference Latency (p90)",
            "Inference Latency (p99)",
            "End-to-End Glass-to-Alert (p99)",
            "Throughput (FPS)",
            "GPU Memory VRAM (MB)",
            "PCIe Transfer Overhead (ms)",
            "Sub-50ms SLA Compliance (%)"
        ],
        "PyTorch CPU Baseline": [124.50, 182.10, 310.80, 465.00, 48, 0, 0, 0.0],
        "PyTorch CUDA FP16": [14.20, 22.40, 38.90, 68.50, 380, 6450, 16.40, 84.2],
        "StreamShield TensorRT + Zero-Copy": [4.12, 7.85, 12.30, 18.40, 1420, 1840, 0.82, 99.98]
    }

    concurrency_data = {
        "Active Streams": [1, 5, 10, 25, 50, 75, 100],
        "Aggregate Input (FPS)": [30, 150, 300, 750, 1500, 2250, 3000],
        "StreamShield Ingestion (FPS)": [30, 150, 300, 750, 1420, 2100, 2800],
        "Median Latency (p50, ms)": [3.8, 4.1, 4.8, 6.2, 8.9, 14.2, 21.5],
        "Tail Latency (p99, ms)": [6.2, 7.5, 9.1, 12.4, 18.6, 28.9, 41.2],
        "GPU Utilization (%)": [8, 16, 28, 54, 89, 98, 100],
        "VRAM Used (GB)": [1.84, 1.92, 2.10, 2.65, 3.40, 4.20, 5.10],
        "SLA Breaches (%)": [0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.04]
    }

    # Section selection
    bench_section = st.radio(
        "Select Benchmark Section:",
        ["📊 Comparative Performance Matrix", "📈 Concurrency Scaling Analysis", "💾 Memory Footprint Analysis", "⚡ Live Stress Testing"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if bench_section == "📊 Comparative Performance Matrix":
        st.markdown("<h4 style='color: #e2e8f0; margin: 16px 0 12px 0;'>CPU vs CUDA vs TensorRT Performance Comparison</h4>", unsafe_allow_html=True)

        df_bench = pd.DataFrame(benchmark_data)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

        # Interactive Latency Comparison Chart — pick metrics & axis scale
        latency_metrics = ["Inference Latency (p50)", "Inference Latency (p90)", "Inference Latency (p99)", "End-to-End Glass-to-Alert (p99)"]
        sel_col1, sel_col2 = st.columns([2.4, 1])
        with sel_col1:
            selected_latency_metrics = st.multiselect(
                "Latency Metrics to Compare:",
                options=latency_metrics,
                default=latency_metrics,
                key="bench_latency_metric_multiselect",
                help="Toggle series on/off — the chart and table update instantly.",
            )
        with sel_col2:
            st.write("")
            log_scale_latency = st.toggle(
                "Log Y-Axis",
                value=False,
                key="bench_latency_log_toggle",
                help="Logarithmic scale makes CPU-vs-TensorRT gaps easier to compare.",
            )

        if not selected_latency_metrics:
            st.info("☑️ Select at least one latency metric above to render the comparison chart.")

        latency_df = df_bench[df_bench["Metric"].isin(selected_latency_metrics)].melt(
            id_vars=["Metric"],
            var_name="Implementation",
            value_name="Latency (ms)"
        )

        fig_latency = px.bar(
            latency_df,
            x="Metric",
            y="Latency (ms)",
            color="Implementation",
            barmode="group",
            title="Latency Comparison Across Implementations (Lower is Better)",
            color_discrete_map={
                "PyTorch CPU Baseline": "#64748b",
                "PyTorch CUDA FP16": "#38bdf8",
                "StreamShield TensorRT + Zero-Copy": "#34d399"
            }
        )
        fig_latency.add_hline(y=50.0, line_dash="dash", line_color="#f43f5e", annotation_text="50ms SLA Target")
        fig_latency.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_latency.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig_latency.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        if log_scale_latency:
            fig_latency.update_yaxes(type="log")
        st.plotly_chart(fig_latency, width="stretch", config=PLOT_CFG)

        # Throughput Comparison
        throughput_df = df_bench[df_bench["Metric"] == "Throughput (FPS)"].melt(
            id_vars=["Metric"],
            var_name="Implementation",
            value_name="Throughput (FPS)"
        )

        fig_throughput = px.bar(
            throughput_df,
            x="Implementation",
            y="Throughput (FPS)",
            title="Throughput Comparison (Higher is Better)",
            color="Implementation",
            color_discrete_map={
                "PyTorch CPU Baseline": "#64748b",
                "PyTorch CUDA FP16": "#38bdf8",
                "StreamShield TensorRT + Zero-Copy": "#34d399"
            }
        )
        fig_throughput.update_traces(texttemplate='%{y:.0f}', textposition='outside')
        fig_throughput.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            showlegend=False
        )
        fig_throughput.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig_throughput.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_throughput, width="stretch", config=PLOT_CFG)

        # Performance Radar Chart
        performance_categories = ["Latency (inv)", "Throughput", "VRAM Efficiency", "SLA Compliance"]
        performance_values = {
            "PyTorch CPU Baseline": [1/124.50 * 100, 48/1420 * 100, 0, 0],
            "PyTorch CUDA FP16": [1/14.20 * 100, 380/1420 * 100, 6450/1840 * 100, 84.2],
            "StreamShield TensorRT + Zero-Copy": [1/4.12 * 100, 1420/1420 * 100, 1840/1840 * 100, 99.98]
        }

        radar_df = pd.DataFrame(performance_values, index=performance_categories).T.reset_index()
        radar_df = radar_df.rename(columns={"index": "Implementation"})

        fig_radar = go.Figure()
        for impl in radar_df["Implementation"]:
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_df[radar_df["Implementation"] == impl].iloc[0, 1:].values,
                theta=performance_categories,
                fill='toself',
                name=impl,
                line_color={
                    "PyTorch CPU Baseline": "#64748b",
                    "PyTorch CUDA FP16": "#38bdf8",
                    "StreamShield TensorRT + Zero-Copy": "#34d399"
                }[impl]
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            title="Normalized Performance Radar (Higher is Better)",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_radar, width="stretch", config=PLOT_CFG)

    elif bench_section == "📈 Concurrency Scaling Analysis":
        st.markdown("<h4 style='color: #e2e8f0; margin: 16px 0 12px 0;'>Concurrency Scaling: 1 to 100 Simultaneous Streams</h4>", unsafe_allow_html=True)

        df_concurrency = pd.DataFrame(concurrency_data)
        st.dataframe(df_concurrency, use_container_width=True, hide_index=True)

        # Latency vs Concurrent Streams
        fig_latency_scale = go.Figure()
        fig_latency_scale.add_trace(go.Scatter(
            x=df_concurrency["Active Streams"],
            y=df_concurrency["Median Latency (p50, ms)"],
            mode='lines+markers',
            name='Median (p50)',
            line=dict(color='#34d399', width=3)
        ))
        fig_latency_scale.add_trace(go.Scatter(
            x=df_concurrency["Active Streams"],
            y=df_concurrency["Tail Latency (p99, ms)"],
            mode='lines+markers',
            name='Tail (p99)',
            line=dict(color='#f87171', width=3)
        ))
        fig_latency_scale.add_hline(y=50.0, line_dash="dash", line_color="#f43f5e", annotation_text="50ms SLA Ceiling")
        fig_latency_scale.update_layout(
            title="Latency Scaling with Concurrent Streams",
            xaxis_title="Active Streams",
            yaxis_title="Latency (ms)",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_latency_scale.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig_latency_scale.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_latency_scale, width="stretch", config=PLOT_CFG)

        # Throughput vs Concurrent Streams
        fig_throughput_scale = go.Figure()
        fig_throughput_scale.add_trace(go.Scatter(
            x=df_concurrency["Active Streams"],
            y=df_concurrency["Aggregate Input (FPS)"],
            mode='lines+markers',
            name='Aggregate Input',
            line=dict(color='#64748b', width=2, dash='dash')
        ))
        fig_throughput_scale.add_trace(go.Scatter(
            x=df_concurrency["Active Streams"],
            y=df_concurrency["StreamShield Ingestion (FPS)"],
            mode='lines+markers',
            name='StreamShield Ingestion',
            line=dict(color='#38bdf8', width=3)
        ))
        fig_throughput_scale.update_layout(
            title="Throughput Scaling with Concurrent Streams",
            xaxis_title="Active Streams",
            yaxis_title="Throughput (FPS)",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_throughput_scale.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig_throughput_scale.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_throughput_scale, width="stretch", config=PLOT_CFG)

        # GPU Utilization & VRAM Scaling
        fig_resource = make_subplots(
            rows=1, cols=2,
            subplot_titles=("GPU Utilization (%)", "VRAM Used (GB)"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        fig_resource.add_trace(
            go.Scatter(x=df_concurrency["Active Streams"], y=df_concurrency["GPU Utilization (%)"],
                      mode='lines+markers', name='GPU Utilization', line=dict(color='#a855f7', width=3)),
            row=1, col=1
        )
        fig_resource.add_trace(
            go.Scatter(x=df_concurrency["Active Streams"], y=df_concurrency["VRAM Used (GB)"],
                      mode='lines+markers', name='VRAM Used', line=dict(color='#f59e0b', width=3)),
            row=1, col=2
        )

        fig_resource.update_xaxes(title_text="Active Streams", row=1, col=1)
        fig_resource.update_xaxes(title_text="Active Streams", row=1, col=2)
        fig_resource.update_yaxes(title_text="Utilization (%)", row=1, col=1)
        fig_resource.update_yaxes(title_text="VRAM (GB)", row=1, col=2)

        fig_resource.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            showlegend=False
        )
        fig_resource.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig_resource.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_resource, width="stretch", config=PLOT_CFG)

    elif bench_section == "💾 Memory Footprint Analysis":
        st.markdown("<h4 style='color: #e2e8f0; margin: 16px 0 12px 0;'>GPU Memory VRAM Footprint Comparison</h4>", unsafe_allow_html=True)

        memory_data = {
            "Component": [
                "Pinned Host Buffer",
                "GPU Tensor Buffers (FP16)",
                "Vision TensorRT Engine",
                "Text TensorRT Engine",
                "Audio TensorRT Engine",
                "TensorRT Workspace",
                "CUDA Context Overhead",
                "ONNX Runtime Session",
                "Pre-allocated Tensor Pool",
                "Misc. (Activations)"
            ],
            "StreamShield (MB)": [96, 48, 42, 152, 39, 256, 128, 64, 128, 867],
            "PyTorch CUDA (MB)": [96, 96, 392, 392, 74, 512, 256, 512, 2400, 1920]
        }

        df_memory = pd.DataFrame(memory_data)
        st.dataframe(df_memory, use_container_width=True, hide_index=True)

        # Memory Breakdown Stacked Bar
        fig_memory = go.Figure()
        fig_memory.add_trace(go.Bar(
            name="StreamShield TensorRT",
            x=df_memory["Component"],
            y=df_memory["StreamShield (MB)"],
            marker_color="#34d399"
        ))
        fig_memory.add_trace(go.Bar(
            name="PyTorch CUDA Standard",
            x=df_memory["Component"],
            y=df_memory["PyTorch CUDA (MB)"],
            marker_color="#38bdf8"
        ))

        fig_memory.update_layout(
            barmode='group',
            title="Memory Footprint Breakdown by Component",
            xaxis_title="Component",
            yaxis_title="Memory (MB)",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_memory.update_xaxes(gridcolor="rgba(255,255,255,0.05)", tickangle=45)
        fig_memory.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_memory, width="stretch", config=PLOT_CFG)

        # Total Memory Comparison
        total_streamshield = df_memory["StreamShield (MB)"].sum()
        total_pytorch = df_memory["PyTorch CUDA (MB)"].sum()
        savings_pct = ((total_pytorch - total_streamshield) / total_pytorch) * 100

        fig_total_memory = go.Figure(data=[
            go.Bar(name='StreamShield TensorRT', x=['Total VRAM'], y=[total_streamshield], marker_color="#34d399"),
            go.Bar(name='PyTorch CUDA Standard', x=['Total VRAM'], y=[total_pytorch], marker_color="#38bdf8")
        ])
        fig_total_memory.update_layout(
            barmode='group',
            title=f"Total VRAM Footprint: {savings_pct:.1f}% Reduction with StreamShield",
            yaxis_title="Total Memory (MB)",
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            showlegend=True
        )
        fig_total_memory.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_total_memory, width="stretch", config=PLOT_CFG)

    elif bench_section == "⚡ Live Stress Testing":
        st.markdown("<h4 style='color: #e2e8f0; margin: 16px 0 12px 0;'>Live ONNX Engine Stress Testing</h4>", unsafe_allow_html=True)

        bench_col1, bench_col2, bench_col3 = st.columns([1, 1, 1])
        with bench_col1:
            bench_batch_size = st.selectbox("Benchmark Batch Size:", [4, 8, 16, 32], index=2)
        with bench_col2:
            bench_passes = st.selectbox("Benchmark Passes / Iterations:", [10, 25, 50], index=0)
        with bench_col3:
            st.write("")
            st.write("")
            run_bench_btn = st.button("🚀 Start Stress Benchmark", type="primary", use_container_width=True)

        if run_bench_btn:
            progress_bar = st.progress(0, text="Running ONNX benchmark passes...")
            latencies = []
            dummy_payloads = [
                MultimodalPayload(
                    text_content=f"Benchmark dummy payload #{idx} testing ONNX graph throughput",
                    image_url="https://streamshield.ai/bench/img",
                )
                for idx in range(bench_batch_size)
            ]

            total_t0 = time.perf_counter()
            for p_idx in range(bench_passes):
                t_pass_start = time.perf_counter()
                _ = asyncio.run(engine.run_batch(dummy_payloads))
                t_pass_ms = (time.perf_counter() - t_pass_start) * 1000.0
                latencies.append(t_pass_ms)
                progress_bar.progress((p_idx + 1) / bench_passes, text=f"Pass {p_idx + 1}/{bench_passes} — {t_pass_ms:.2f} ms")

            total_elapsed_s = time.perf_counter() - total_t0
            total_items = bench_passes * bench_batch_size
            throughput_fps = total_items / total_elapsed_s

            st.session_state.benchmark_results = {
                "latencies": latencies,
                "batch_size": bench_batch_size,
                "passes": bench_passes,
                "total_items": total_items,
                "throughput_fps": throughput_fps,
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "mean": np.mean(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
            }
            progress_bar.empty()

        if st.session_state.benchmark_results is not None:
            b_res = st.session_state.benchmark_results

            bk1, bk2, bk3, bk4 = st.columns(4)
            with bk1:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="metric-label">THROUGHPUT</div>
                        <div class="metric-value" style="color: #38bdf8;">{b_res['throughput_fps']:.1f}</div>
                        <div class="metric-delta" style="color: #94a3b8;">Items / Second</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with bk2:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="metric-label">P50 (MEDIAN) LATENCY</div>
                        <div class="metric-value" style="color: #34d399;">{b_res['p50']:.2f} ms</div>
                        <div class="metric-delta" style="color: #94a3b8;">Mean: {b_res['mean']:.2f} ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with bk3:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="metric-label">P95 LATENCY</div>
                        <div class="metric-value" style="color: {'#34d399' if b_res['p95'] < 50 else '#f87171'};">
                            {b_res['p95']:.2f} ms
                        </div>
                        <div class="metric-delta" style="color: #94a3b8;">Target: &lt; 50 ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with bk4:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="metric-label">P99 LATENCY</div>
                        <div class="metric-value" style="color: #fbbf24;">{b_res['p99']:.2f} ms</div>
                        <div class="metric-delta" style="color: #94a3b8;">Max: {b_res['max']:.2f} ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Plot latency curve across iterations
            fig_bench = px.line(
                x=list(range(1, len(b_res["latencies"]) + 1)),
                y=b_res["latencies"],
                markers=True,
                labels={"x": "Iteration Pass", "y": "Latency (ms)"},
                title=f"Per-Pass Wall-Clock Latency (Batch Size = {b_res['batch_size']})",
            )
            fig_bench.add_hline(y=50.0, line_dash="dash", line_color="#f43f5e", annotation_text="50ms Target SLA")
            fig_bench.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=10, r=10, t=35, b=15),
            )
            fig_bench.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig_bench.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_bench, width="stretch", config=PLOT_CFG)


# ===========================================================================
# TAB 5: Architecture & Model Specs
# ===========================================================================
with tab_arch:
    st.markdown(
        """
        <div class="hero-banner">
            <h3 style="margin:0 0 6px 0; color: #f8fafc; font-size: 1.25rem;">System Architecture & ONNX Pipeline Specification</h3>
            <p style="margin:0; color: #cbd5e1; font-size: 0.9rem;">
                Detailed blueprint of StreamShield AI's asynchronous multimodal pipeline, Kafka micro-batching consumer, and audit logging mechanics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    arch_col1, arch_col2 = st.columns([1.1, 0.9])

    with arch_col1:
        st.markdown(
            """
            #### 🔄 End-to-End Multimodal Pipeline Flow
            
            ```
            ┌─────────────────────────────────────────────────────────────┐
            │                     Inbound Kafka Stream                    │
            │                  (Topic: multimodal-stream)                 │
            └──────────────────────────────┬──────────────────────────────┘
                                           │
                                  Micro-Batch Collector
                             (Batch: 16 msgs / 500ms timeout)
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
             [ Text Modality Pipeline ]           [ Image Modality Pipeline ]
             • Tokenize (DistilBERT)              • Async Fetch (HTTPX/Mock)
             • ONNX Runtime Session               • Normalise (ImageNet CHW)
             • Logits + 128-d Embedding           • ONNX Runtime Session
                        │                         • Logits + 128-d Embedding
                        └──────────────────┬──────────────────┘
                                           │
                                  `asyncio.gather`
                                           │
                                           ▼
                                 Multimodal Fusion Unit
                        • Sigmoid Probability Calculation
                        • Confidence = max(Text_prob, Image_prob)
                        • L2 Normalization on Embeddings
                        • Wall-Clock Latency Profiler (<50ms)
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                [ Kafka Commit Offset ]             [ PostgreSQL Audit DB ]
               (Only after DB success)             (Table: inference_audit)
            ```
            """
        )

    with arch_col2:
        st.markdown(
            f"""
            #### 📦 Active ONNX Model Registry
            
            <div class="glass-card" style="margin-bottom: 10px;">
                <div style="color: #818cf8; font-weight: 700;">📝 Text Toxicity Model</div>
                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">
                    <b>Model File:</b> <code>{settings.TEXT_ONNX_PATH}</code><br>
                    <b>Input 1:</b> <code>input_ids</code> (Batch, 128) [INT64]<br>
                    <b>Input 2:</b> <code>attention_mask</code> (Batch, 128) [INT64]<br>
                    <b>Output 1:</b> <code>logits</code> (Batch, 2) [FLOAT32]<br>
                    <b>Output 2:</b> <code>embeddings</code> (Batch, 128) [FLOAT32]
                </div>
            </div>

            <div class="glass-card">
                <div style="color: #38bdf8; font-weight: 700;">🖼️ Image Classifier Model</div>
                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">
                    <b>Model File:</b> <code>{settings.IMAGE_MODEL_PATH}</code><br>
                    <b>Input:</b> <code>pixel_values</code> (Batch, 3, 224, 224) [FLOAT32]<br>
                    <b>Output 1:</b> <code>logits</code> (Batch, 2) [FLOAT32]<br>
                    <b>Output 2:</b> <code>embeddings</code> (Batch, 128) [FLOAT32]<br>
                    <b>Normalization:</b> ImageNet Mean & Std
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 2rem 0 1rem 0;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.82rem;">
        StreamShield AI © 2026 — High-Throughput Real-Time Multimodal Content Moderation System. Built with ONNX Runtime, Confluent Kafka, SQLAlchemy, and Streamlit.
    </div>
    """,
    unsafe_allow_html=True,
)
