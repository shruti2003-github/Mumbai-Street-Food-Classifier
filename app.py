import streamlit as st
import numpy as np
from PIL import Image
import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import tensorflow as tf
import io
import base64

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mumbai Street Food Classifier",
    page_icon="🍽️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

.main {
    background: #fdf6ec;
}

.stApp {
    background: linear-gradient(135deg, #fdf6ec 0%, #fff8f0 100%);
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    color: #1a1a1a;
    text-align: center;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}

.hero-subtitle {
    text-align: center;
    color: #7a5c3b;
    font-size: 1rem;
    margin-bottom: 2rem;
    letter-spacing: 0.04em;
}

.badge {
    display: inline-block;
    background: #f4a435;
    color: white;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.result-card {
    background: white;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    border-left: 5px solid #f4a435;
    margin-top: 1.5rem;
}

.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #1a1a1a;
    margin: 0;
}

.result-confidence {
    font-size: 1rem;
    color: #7a5c3b;
    margin-top: 0.2rem;
}

.food-info {
    background: #fffbf5;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #4a3520;
    border: 1px solid #f0e0c8;
}

.stFileUploader > div {
    border: 2px dashed #f4a435 !important;
    border-radius: 14px !important;
    background: #fffbf5 !important;
}

.model-info {
    font-size: 0.78rem;
    color: #aaa;
    text-align: center;
    margin-top: 2rem;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Grilled Sandwich", "Idli", "Masala Dosa", "Paani Puri", "Samosa", "Vada Pav"]
IMG_SIZE    = (224, 224)
MODEL_PATH  = "model_resaved.h5"

FOOD_INFO = {
    "Vada Pav":  "🥔 Mumbai's iconic street burger — spiced potato fritter in a bun.",
    "Grilled Sandwich":  "🥪 Mumbai-style toasted sandwich, often layered with chutney & veggies.",
    "Samosa":    "🔺 Crispy pastry stuffed with spiced potatoes and peas.",
    "Paani Puri": "💧 Hollow crisp shells filled with tangy tamarind water.",
    "Masala Dosa":      "🫓 Thin crispy South Indian crepe made from fermented rice batter.",
    "Idli":      "⚪ Soft steamed rice cakes, a South Indian breakfast staple.",
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🍽️ Mumbai Street Food</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="color:#f4a435;">Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Deep Learning · MobileNetV2 · 91.33% Accuracy</div>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a food image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True, caption="Uploaded image")

    if model is None:
        st.error("⚠️ `model.h5` not found. Place your trained model file in the same directory as `app.py`.")
    else:
        with st.spinner("Analysing..."):
            img_array = np.array(image.resize(IMG_SIZE)) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            preds     = model.predict(img_array)[0]

        top_idx    = int(np.argmax(preds))
        top_label  = CLASS_NAMES[top_idx]
        confidence = float(preds[top_idx]) * 100

        st.markdown(f"""
        <div class="result-card">
            <p class="result-label">{top_label}</p>
            <p class="result-confidence">Confidence: <strong>{confidence:.1f}%</strong></p>
            <div class="food-info">{FOOD_INFO[top_label]}</div>
        </div>
        """, unsafe_allow_html=True)

        # ✅ NEW ADDITION (Top 3 predictions with %)
        st.markdown("#### 🔍 Top Predictions")
        top3_idx = preds.argsort()[-3:][::-1]
        for i in top3_idx:
            st.write(f"{CLASS_NAMES[i]} — {preds[i]*100:.2f}%")

        # Probability bar chart
        st.markdown("#### All class probabilities")
        prob_dict = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(prob_dict)

else:
    st.info("👆 Upload an image of Mumbai street food to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="model-info">
    Model: MobileNetV2 (Transfer Learning) &nbsp;|&nbsp;
    Dataset: 2,400 images, 6 classes &nbsp;|&nbsp;
    Research by Shruti Kesharwani
</div>
""", unsafe_allow_html=True)
