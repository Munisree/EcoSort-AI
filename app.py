"""
app.py — EcoSort AI  |  Phase 2: Waste Classification + RAG Disposal Guidance

Streamlit application entry point.

Run with:
    streamlit run app.py

Phase 2 scope:
    Image upload -> preprocessing -> EfficientNet-B0 inference
    -> waste category + confidence score display
    -> knowledge-base disposal guidance via rag.retriever (RAG, no LLM).

The application will refuse to display predictions if the trained model
checkpoint is not present. Users are directed to run training first.

LLM assistant and advanced features are not part of Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from classifier.model import load_trained_model, predict
from classifier.preprocess import preprocess_image
from rag.retriever import get_guidance_by_category

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH        = Path("models/ecosort_efficientnet_b0.pth")
CONFIDENCE_WARN   = 0.50   # below this → yellow low-confidence warning
CONFIDENCE_REFUSE = 0.20   # below this → show a more prominent caution

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load model — cached so it is not reloaded on every interaction.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading waste-classification model …")
def get_model():
    """
    Load the trained EfficientNet-B0 checkpoint once and cache it.

    Returns (model, class_names) or raises FileNotFoundError.
    """
    return load_trained_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("ℹ️ About EcoSort AI")
    st.markdown(
        """
        **EcoSort AI** is an AI-powered waste-segregation assistant.

        It uses a fine-tuned **EfficientNet-B0** image classifier to identify
        waste items from photographs and suggest responsible disposal methods.

        **Current phase:** Phase 2 — Classification + Disposal Guidance
        **Coming soon:** Conversational AI assistant (Phase 3)

        ---
        **UN SDG alignment**  
        🎯 SDG 12 — Responsible Consumption & Production  
        🏙️ SDG 11 — Sustainable Cities & Communities

        ---
        **Developed with IBM Bob**  
        *1M1B AI for Sustainability Virtual Internship*  
        *IBM SkillsBuild × AICTE*

        ---
        **Privacy notice**  
        Uploaded images are processed in memory only and are never stored or
        transmitted to any external server.
        """
    )
    st.markdown("---")
    st.caption("Phase 2 of 3 — classifier + RAG disposal guidance")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("♻️ EcoSort AI")
st.subheader("Intelligent Waste Segregation Assistant")
st.markdown(
    "Upload a photo of a waste item and EcoSort AI will classify it and "
    "suggest how to dispose of it responsibly."
)
st.markdown("---")

# ---------------------------------------------------------------- Model check
model_available = MODEL_PATH.exists()

if not model_available:
    st.error(
        "**Trained model not found.**\n\n"
        f"Expected: `{MODEL_PATH}`\n\n"
        "Please train the classifier first by running:\n\n"
        "```bash\n"
        "python -m classifier.train\n"
        "```\n\n"
        "Make sure the Garbage Classification dataset is placed at "
        "`data/garbage_classification/` before training."
    )
    st.stop()   # Do not render anything below — prevents random predictions.

# Load model (cached after first call).
try:
    model, class_names = get_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

# ------------------------------------------------------------- File uploader
uploaded_file = st.file_uploader(
    "Choose an image …",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WebP",
)

if uploaded_file is not None:
    # ------------------------------------------------- Load and display image
    pil_image = Image.open(uploaded_file)

    # Correct EXIF orientation (e.g. rotated phone photos) before processing.
    pil_image = ImageOps.exif_transpose(pil_image)

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.image(pil_image, caption="Uploaded image", use_container_width=True)

    # --------------------------------------------------- Run classifier
    with st.spinner("Classifying …"):
        tensor           = preprocess_image(pil_image)
        label, confidence, all_scores = predict(model, tensor, class_names)

    # --------------------------------------------------- Display results
    with col_results:
        st.markdown("### Classification Result")

        # Predicted category badge
        st.markdown(
            f"""
            <div style="
                background:#e8f5e9; border-left:5px solid #2e7d32;
                padding:12px 16px; border-radius:4px; margin-bottom:12px;
            ">
                <span style="font-size:0.85rem; color:#555;">Predicted category</span><br>
                <span style="font-size:1.6rem; font-weight:700; color:#1b5e20;">
                    {label.title()}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence bar
        conf_pct = confidence * 100
        bar_color = "#2e7d32" if confidence >= CONFIDENCE_WARN else "#f9a825"

        st.markdown(
            f"""
            <div style="margin-bottom:4px;">
                <span style="font-size:0.85rem; color:#555;">Model confidence</span>
                <span style="font-size:0.85rem; font-weight:600; float:right;">
                    {conf_pct:.1f}%
                </span>
            </div>
            <div style="background:#e0e0e0; border-radius:6px; height:14px; width:100%;">
                <div style="
                    background:{bar_color}; width:{conf_pct:.1f}%;
                    height:14px; border-radius:6px;">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Confidence reflects the model's internal probability score, "
            "not a guarantee of correctness. Always verify before disposal."
        )

        # Low-confidence warnings
        if confidence < CONFIDENCE_REFUSE:
            st.error(
                "⚠️ **Very low confidence.** The model is highly uncertain "
                "about this image. Please verify the waste type manually "
                "before disposal."
            )
        elif confidence < CONFIDENCE_WARN:
            st.warning(
                "⚠️ **Low confidence.** The model is uncertain about this "
                "prediction. Please verify the waste category before disposal."
            )

    # ------------------------------------------ Full score breakdown (expander)
    with st.expander("Show full score breakdown"):
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for cls_name, score in sorted_scores:
            bar_w = int(score * 100)
            is_top = cls_name == label
            weight = "bold" if is_top else "normal"
            st.markdown(
                f"""
                <div style="margin-bottom:6px;">
                    <span style="font-size:0.82rem; font-weight:{weight};
                                 width:120px; display:inline-block;">
                        {cls_name.title()}
                    </span>
                    <span style="font-size:0.82rem; color:#888;">
                        {score*100:.1f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ------------------------------------------------- Disposal Guidance (RAG)
    guidance = get_guidance_by_category(label)

    st.markdown("### Disposal Guidance")

    if guidance is None:
        st.warning(
            "No specific guidance found for this category in the knowledge base. "
            "Please follow your local municipal solid waste disposal instructions "
            "and consult your city's waste-management authority."
        )
    else:
        # Short description
        st.markdown(
            f"""
            <div style="
                background:#f7f8fa; border-left:4px solid #3b82d4;
                padding:10px 14px; border-radius:4px; margin-bottom:14px;
            ">
                <span style="font-size:0.82rem; color:#57606a; text-transform:uppercase;
                             letter-spacing:0.04em;">About this item</span><br>
                <span style="font-size:0.93rem; color:#1f2328;">{guidance['short_description']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Disposal guidance
        st.markdown(
            f"""
            <div style="
                background:#e8f5e9; border-left:4px solid #2e7d32;
                padding:10px 14px; border-radius:4px; margin-bottom:10px;
            ">
                <span style="font-size:0.82rem; color:#57606a; text-transform:uppercase;
                             letter-spacing:0.04em;">How to dispose</span><br>
                <span style="font-size:0.93rem; color:#1b5e20;">{guidance['disposal_guidance']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Safety note
        st.markdown(
            f"""
            <div style="
                background:#fff8e1; border-left:4px solid #f9a825;
                padding:10px 14px; border-radius:4px; margin-bottom:10px;
            ">
                <span style="font-size:0.82rem; color:#57606a; text-transform:uppercase;
                             letter-spacing:0.04em;">Safety note</span><br>
                <span style="font-size:0.93rem; color:#4a3800;">{guidance['safety_note']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Recycling / reuse tip
        st.markdown(
            f"""
            <div style="
                background:#e8eaf6; border-left:4px solid #7c5cd8;
                padding:10px 14px; border-radius:4px; margin-bottom:10px;
            ">
                <span style="font-size:0.82rem; color:#57606a; text-transform:uppercase;
                             letter-spacing:0.04em;">Recycling / reuse tip</span><br>
                <span style="font-size:0.93rem; color:#1f1640;">{guidance['recycling_or_reuse_tip']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Guidance is sourced from the EcoSort AI knowledge base. "
            "Always follow your local municipal waste-disposal instructions, "
            "which may differ from the general guidance shown here."
        )

else:
    # No image uploaded yet — show instructions.
    st.markdown(
        """
        ### How to use EcoSort AI

        1. Click **Browse files** above and select a photo of your waste item.
        2. EcoSort AI will classify the waste into one of 12 categories.
        3. You will see the predicted category and the model's confidence score.
        4. Disposal guidance, safety notes, and recycling tips will appear
           automatically based on the identified category.

        **Supported waste categories (12 classes):**
        Battery · Biological · Brown glass · Cardboard · Clothes
        Green glass · Metal · Paper · Plastic · Shoes · Trash · White glass

        > *Guidance is sourced from the EcoSort AI knowledge base and is intended
        > as general information. Always follow your local municipal waste-disposal
        > instructions.*
        """
    )
