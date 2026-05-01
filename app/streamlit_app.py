from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtsrb_robustness.corruptions import DEMO_CORRUPTIONS
from gtsrb_robustness.inference import gradcam_overlay, load_model_for_inference, predict_topk


st.set_page_config(
    page_title="GTSRB Reliability Workbench",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f7f9; color: #14171f; }
      [data-testid="stSidebar"] { background: #111827; color: white; }
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] p { color: white; }
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] textarea,
      [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #111827 !important;
      }
      .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
      .metric-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0 1.25rem;
      }
      .metric-cell {
        border-left: 3px solid #2f6fed;
        padding: 0.55rem 0.8rem;
        background: #ffffff;
      }
      .metric-cell span { display:block; color:#667085; font-size:0.78rem; }
      .metric-cell strong { font-size:1.2rem; }
      .safety-note {
        border-left: 4px solid #d92d20;
        padding: 0.8rem 1rem;
        background: #fff6f5;
        color: #7a271a;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_model(checkpoint: str, model_name: str):
    return load_model_for_inference(Path(checkpoint), model_name)


def resolve_checkpoint() -> str | None:
    candidates = [
        os.getenv("GTSRB_CHECKPOINT"),
        str(ROOT / "runs" / "resnet18_robust" / "best.pt"),
        str(ROOT / "checkpoints" / "best.pt"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


with st.sidebar:
    st.title("GTSRB Workbench")
    model_name = st.selectbox("Model", ["resnet18", "efficientnet_b0", "baseline_cnn"], index=0)
    checkpoint = st.text_input("Checkpoint path", value=resolve_checkpoint() or "")
    temperature = st.number_input(
        "Calibration temperature",
        min_value=0.05,
        max_value=10.0,
        value=float(os.getenv("GTSRB_TEMPERATURE", "1.0")),
        step=0.05,
    )
    corruption_name = st.selectbox("Apply corruption", list(DEMO_CORRUPTIONS.keys()), index=0)
    st.caption("Use a trained checkpoint for live predictions. Without one, the app still previews the interaction shell.")

st.title("Traffic Sign Reliability Workbench")
st.write("Upload a cropped traffic sign image, inspect top predictions, and compare the model's attention under realistic corruptions.")

uploaded = st.file_uploader("Upload traffic sign image", type=["png", "jpg", "jpeg", "webp"])

if not checkpoint or not Path(checkpoint).exists():
    st.warning("No checkpoint found. Train a model first or set `GTSRB_CHECKPOINT` to a valid `.pt` file.")

if uploaded is None:
    st.info("Upload an image to run inference.")
else:
    original = Image.open(uploaded).convert("RGB")
    shown = DEMO_CORRUPTIONS[corruption_name](original)
    left, middle, right = st.columns([1.05, 1.05, 1.0], gap="large")

    with left:
        st.subheader("Input")
        st.image(shown, use_container_width=True)
        st.caption(f"Corruption: {corruption_name}")

    if checkpoint and Path(checkpoint).exists():
        with st.spinner("Running model..."):
            model, device = cached_model(checkpoint, model_name)
            predictions = predict_topk(model, device, shown, k=5, temperature=temperature)
            try:
                cam_image = gradcam_overlay(model, device, shown)
            except Exception as exc:
                cam_image = None
                st.warning(f"Grad-CAM unavailable for this model: {exc}")

        top = predictions[0]
        st.markdown(
            f"""
            <div class="metric-strip">
              <div class="metric-cell"><span>Top class</span><strong>{top["class_id"]}</strong></div>
              <div class="metric-cell"><span>Confidence</span><strong>{top["probability"]:.1%}</strong></div>
              <div class="metric-cell"><span>Device</span><strong>{device.type.upper()}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with middle:
            st.subheader("Grad-CAM")
            if cam_image is not None:
                st.image(cam_image, use_container_width=True)
            else:
                st.info("Grad-CAM could not be rendered for this input.")

        with right:
            st.subheader("Top predictions")
            frame = pd.DataFrame(predictions)
            st.dataframe(
                frame.assign(probability=frame["probability"].map(lambda value: f"{value:.2%}")),
                hide_index=True,
                use_container_width=True,
            )
            st.bar_chart(pd.DataFrame({"probability": [p["probability"] for p in predictions]}, index=[p["label"] for p in predictions]))

st.markdown(
    """
    <div class="safety-note">
      <strong>Educational use only.</strong> This project is a controlled course prototype. It is not validated for autonomous driving, traffic enforcement, or safety-critical deployment.
    </div>
    """,
    unsafe_allow_html=True,
)
