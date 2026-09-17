import json
import os
from typing import Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/cropvision_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMAGE_SIZE = (224, 224)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="CropVision",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       PAGE
    -------------------------------------------------------- */

    .stApp {
        background: #0b0f0d;
        color: #f5f7f5;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }


    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .eyebrow {
        font-size: 0.75rem;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        color: #8fb99a;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .hero {
        font-size: clamp(4rem, 10vw, 8.5rem);
        line-height: 0.82;
        letter-spacing: -0.075em;
        font-weight: 850;
        margin: 0;
    }

    .subtitle {
        max-width: 650px;
        margin-top: 1.4rem;
        font-size: 1.1rem;
        line-height: 1.7;
        color: #aab4ad;
    }

    .line {
        width: 100%;
        height: 1px;
        background: rgba(255,255,255,0.10);
        margin: 2.8rem 0;
    }


    /* --------------------------------------------------------
       SECTIONS
    -------------------------------------------------------- */

    .section-number {
        color: #7fa98a;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .section-heading {
        font-size: 2rem;
        font-weight: 750;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }

    .section-text {
        color: #89928c;
        margin-bottom: 1.2rem;
    }

    .muted {
        color: #858f89;
    }


    /* --------------------------------------------------------
       UPLOAD
    -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(143,185,154,0.35);
        border-radius: 18px;
        padding: 1rem;
        background: rgba(255,255,255,0.025);
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(143,185,154,0.65);
    }


    /* --------------------------------------------------------
       IMAGES
    -------------------------------------------------------- */

    [data-testid="stImage"] img {
        border-radius: 16px;
    }

    .visual-label {
        color: #7fa98a;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }


    /* --------------------------------------------------------
       RESULT
    -------------------------------------------------------- */

    .result-name {
        font-size: clamp(2rem, 4vw, 3.4rem);
        font-weight: 800;
        letter-spacing: -0.05em;
        line-height: 1;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .confidence-number {
        font-size: 4.5rem;
        line-height: 1;
        font-weight: 850;
        letter-spacing: -0.07em;
        color: #91cc9e;
    }

    .confidence-status {
        margin-top: 0.45rem;
        margin-bottom: 0.9rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #91cc9e;
    }

    .confidence-status.low {
        color: #d6ae71;
    }

    .confidence-status.moderate {
        color: #b8c987;
    }


    /* --------------------------------------------------------
       INFO
    -------------------------------------------------------- */

    .info-title {
        font-size: 1.3rem;
        font-weight: 750;
        margin-bottom: 0.7rem;
    }

    .info-copy {
        color: #a3ada6;
        line-height: 1.7;
    }

    .attention-note {
        color: #7f8983;
        font-size: 0.83rem;
        line-height: 1.6;
        margin-top: 0.7rem;
    }


    /* --------------------------------------------------------
       BUTTON
    -------------------------------------------------------- */

    .stButton > button {
        width: 100%;
        min-height: 3.1rem;
        border-radius: 999px;
        border: 1px solid rgba(143,185,154,0.30);
        background: #d9eadc;
        color: #101511;
        font-weight: 800;
        letter-spacing: 0.03em;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        background: #edf7ef;
        border-color: rgba(143,185,154,0.7);
    }


    /* --------------------------------------------------------
       PROGRESS
    -------------------------------------------------------- */

    .stProgress > div > div > div > div {
        background-color: #91cc9e;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        margin-top: 4rem;
        text-align: center;
        color: #626b65;
        font-size: 0.75rem;
        letter-spacing: 0.10em;
        text-transform: uppercase;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model not found. Check the models folder."
        )
        st.stop()

    return keras.models.load_model(MODEL_PATH)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(
            "class_names.json not found."
        )
        st.stop()

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


model = load_model()
class_names = load_class_names()


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Pepper,_bell___Bacterial_spot": {
        "crop": "Bell Pepper",
        "status": "Possible bacterial spot",
        "description":
            "The model detected visual patterns associated "
            "with bacterial spot in bell pepper leaves.",
        "tips": [
            "Inspect nearby leaves for similar symptoms.",
            "Avoid unnecessary overhead watering.",
            "Remove severely affected material where appropriate.",
            "Use local agricultural guidance for treatment."
        ]
    },

    "Pepper,_bell___healthy": {
        "crop": "Bell Pepper",
        "status": "Healthy",
        "description":
            "The model detected visual patterns associated "
            "with a healthy bell pepper leaf.",
        "tips": [
            "Continue regular plant monitoring.",
            "Maintain healthy growing conditions.",
            "Watch for changes in leaf color and texture."
        ]
    },

    "Potato___Early_blight": {
        "crop": "Potato",
        "status": "Possible early blight",
        "description":
            "The model detected visual patterns associated "
            "with early blight in potato leaves.",
        "tips": [
            "Inspect nearby leaves for similar symptoms.",
            "Avoid prolonged leaf wetness.",
            "Remove severely affected material where appropriate.",
            "Consult local agricultural guidance."
        ]
    },

    "Potato___Late_blight": {
        "crop": "Potato",
        "status": "Possible late blight",
        "description":
            "The model detected visual patterns associated "
            "with late blight in potato leaves.",
        "tips": [
            "Inspect the plant carefully.",
            "Check surrounding plants for similar symptoms.",
            "Reduce prolonged moisture on foliage.",
            "Seek local agricultural guidance promptly."
        ]
    },

    "Potato___healthy": {
        "crop": "Potato",
        "status": "Healthy",
        "description":
            "The model detected visual patterns associated "
            "with a healthy potato leaf.",
        "tips": [
            "Continue regular monitoring.",
            "Maintain healthy growing conditions.",
            "Check leaves periodically for changes."
        ]
    }
}


# ============================================================
# GRAD-CAM SETUP
# ============================================================

@st.cache_resource
def create_gradcam_model():

    base_model = None
    augmentation_layer = None
    pooling_layer = None
    dropout_layer = None
    classifier_layer = None

    for layer in model.layers:

        layer_name = layer.name.lower()

        if isinstance(layer, keras.Model):

            if "mobilenet" in layer_name:
                base_model = layer

        if layer_name == "data_augmentation":
            augmentation_layer = layer

        if isinstance(
            layer,
            keras.layers.GlobalAveragePooling2D
        ):
            pooling_layer = layer

        if isinstance(
            layer,
            keras.layers.Dropout
        ):
            dropout_layer = layer

        if isinstance(
            layer,
            keras.layers.Dense
        ):
            classifier_layer = layer

    if base_model is None:
        raise RuntimeError(
            "MobileNetV2 base model was not found."
        )

    if pooling_layer is None:
        raise RuntimeError(
            "GlobalAveragePooling2D layer was not found."
        )

    if classifier_layer is None:
        raise RuntimeError(
            "Classifier layer was not found."
        )

    grad_input = keras.Input(
        shape=(224, 224, 3)
    )

    x = grad_input

    if augmentation_layer is not None:

        x = augmentation_layer(
            x,
            training=False
        )

    x = preprocess_input(x)

    conv_outputs = base_model(
        x,
        training=False
    )

    x = pooling_layer(
        conv_outputs
    )

    if dropout_layer is not None:

        x = dropout_layer(
            x,
            training=False
        )

    predictions = classifier_layer(
        x
    )

    grad_model = keras.Model(
        inputs=grad_input,
        outputs=[
            conv_outputs,
            predictions
        ]
    )

    return grad_model


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    image: Image.Image
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:

    grad_model = create_gradcam_model()

    resized = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        resized,
        dtype=np.float32
    )

    input_image = np.expand_dims(
        image_array,
        axis=0
    )

    input_image = preprocess_input(
        input_image
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_image,
            training=False
        )

        predicted_index = tf.argmax(
            predictions[0]
        )

        predicted_score = predictions[
            0,
            predicted_index
        ]

    grads = tape.gradient(
        predicted_score,
        conv_outputs
    )

    if grads is None:

        raise RuntimeError(
            "Gradients could not be calculated."
        )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_heatmap = tf.reduce_max(
        heatmap
    )

    if float(max_heatmap.numpy()) > 0:

        heatmap = (
            heatmap / max_heatmap
        )

    heatmap = heatmap.numpy()

    # Resize Grad-CAM to original image size
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        image.size
    )

    heatmap = (
        np.array(
            heatmap_image,
            dtype=np.float32
        ) / 255.0
    )

    return (
        heatmap_image,
        heatmap,
        predictions[0].numpy()
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="eyebrow">'
    'AI · COMPUTER VISION · AGRICULTURE'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<h1 class="hero">CROP VISION 🌱</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI-powered plant disease detection system that '
    'analyzes leaf images using transfer learning.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="line"></div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-number">01 / Upload</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-heading">Analyze a leaf</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-text">'
    'Upload a clear image of a plant leaf.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE + ANALYSIS
# ============================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown(
        '<div style="height:1.2rem;"></div>',
        unsafe_allow_html=True
    )

    image_col, control_col = st.columns(
        [1.25, 0.75],
        gap="large"
    )

    with image_col:

        st.markdown(
            '<div class="visual-label">INPUT IMAGE</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            width="stretch"
        )

    with control_col:

        st.markdown(
            '<div class="section-number">AI ANALYSIS</div>',
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            "### Ready to analyze"
        )

        st.markdown(
            '<p class="muted">'
            f'CropVision currently recognizes '
            f'<b>{len(class_names)}</b> plant health classes.'
            '</p>',
            unsafe_allow_html=True
        )

        st.write("")

        analyze = st.button(
            "ANALYZE LEAF →"
        )

        st.write("")

        st.caption(
            "For demonstration and research purposes."
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if analyze:

        with st.spinner(
            "Analyzing leaf..."
        ):

            try:

                # --------------------------------------------
                # Prediction + Grad-CAM
                # --------------------------------------------

                (
                    heatmap_image,
                    heatmap,
                    predictions
                ) = generate_gradcam(image)

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

                st.stop()


        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index]
        ) * 100

        top_indices = np.argsort(
            predictions
        )[::-1][:3]

        info = DISEASE_INFO.get(
            predicted_class,
            {}
        )


        # ====================================================
        # CONFIDENCE INTERPRETATION
        # ====================================================

        if confidence >= 80:

            confidence_status = "HIGH CONFIDENCE"
            confidence_class = ""

        elif confidence >= 55:

            confidence_status = "MODERATE CONFIDENCE"
            confidence_class = "moderate"

        else:

            confidence_status = "LOW CONFIDENCE"
            confidence_class = "low"


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="line"></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-number">02 / Result</div>',
            unsafe_allow_html=True
        )

        result_col, confidence_col = st.columns(
            [1.5, 1],
            gap="large"
        )


        with result_col:

            st.markdown(
                '<div class="muted">MODEL PREDICTION</div>',
                unsafe_allow_html=True
            )

            clean_name = (
                predicted_class
                .replace("___", " · ")
                .replace("_", " ")
                .replace(",", "")
            )

            st.markdown(
                f'<div class="result-name">'
                f'{clean_name}'
                f'</div>',
                unsafe_allow_html=True
            )

            if info:

                st.markdown(
                    f'**Crop:** {info["crop"]}'
                )

                st.write(
                    info["description"]
                )


        with confidence_col:

            st.markdown(
                '<div class="muted">MODEL CONFIDENCE</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="confidence-number">'
                f'{confidence:.1f}%'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="confidence-status '
                f'{confidence_class}">'
                f'{confidence_status}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(
                min(confidence / 100, 1.0)
            )

            if confidence < 55:

                st.caption(
                    "The model is uncertain between multiple "
                    "classes. Consider uploading a clearer "
                    "leaf image."
                )

            elif confidence < 80:

                st.caption(
                    "The prediction has moderate confidence. "
                    "Image quality and visual similarity may "
                    "affect the result."
                )

            else:

                st.caption(
                    "The model shows strong confidence for "
                    "this image."
                )


        # ====================================================
        # MODEL EXPLANATION
        # ====================================================

        st.write("")

        st.markdown(
            '<div class="section-number">'
            'MODEL EXPLANATION'
            '</div>',
            unsafe_allow_html=True
        )

        attention_col, note_col = st.columns(
            [1.25, 0.75],
            gap="large"
        )

        with attention_col:

            # Convert grayscale heatmap into RGB color map
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap("jet")

            colored_heatmap = cmap(
                heatmap
            )[..., :3]

            colored_heatmap = (
                colored_heatmap * 255
            ).astype(
                np.uint8
            )

            # Blend heatmap with original image
            original_array = np.array(
                image.resize(
                    image.size
                ),
                dtype=np.float32
            )

            overlay = (
                original_array * 0.55
                +
                colored_heatmap * 0.45
            )

            overlay = np.clip(
                overlay,
                0,
                255
            ).astype(
                np.uint8
            )

            overlay_image = Image.fromarray(
                overlay
            )

            st.markdown(
                '<div class="visual-label">'
                'GRAD-CAM ATTENTION'
                '</div>',
                unsafe_allow_html=True
            )

            st.image(
                overlay_image,
                width="stretch"
            )

        with note_col:

            st.markdown(
                "### Where the model looked"
            )

            st.markdown(
                '<div class="info-copy">'
                'Grad-CAM highlights image regions that '
                'contributed to the selected prediction. '
                'Warmer areas indicate stronger relative '
                'activation for the predicted class.'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="attention-note">'
                'This visualization explains model attention; '
                'it does not independently verify a disease '
                'diagnosis.'
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # TOP PREDICTIONS
        # ====================================================

        st.write("")

        st.markdown(
            '<div class="section-number">'
            'MODEL DISTRIBUTION'
            '</div>',
            unsafe_allow_html=True
        )

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            probability = (
                float(
                    predictions[index]
                ) * 100
            )

            name = (
                class_names[index]
                .replace("___", " · ")
                .replace("_", " ")
                .replace(",", "")
            )

            left, right = st.columns(
                [4, 1]
            )

            with left:

                st.write(
                    f"**{rank}  {name}**"
                )

            with right:

                st.write(
                    f"{probability:.2f}%"
                )

            st.progress(
                min(
                    probability / 100,
                    1.0
                )
            )


        # ====================================================
        # INSIGHTS
        # ====================================================

        if info:

            st.markdown(
                '<div class="line"></div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-number">'
                '03 / Insights'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="info-title">'
                f'{info["status"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="info-copy">'
                f'{info["description"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                "**General considerations**"
            )

            for tip in info["tips"]:

                st.write(
                    f"• {tip}"
                )

            st.caption(
                "CropVision is an image-based research/demo "
                "classifier and should not replace professional "
                "agricultural diagnosis."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'CROPVISION · TRANSFER LEARNING · MOBILENETV2 · 2026'
    '</div>',
    unsafe_allow_html=True
)