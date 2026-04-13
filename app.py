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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mumbai Street Food Classifier",
    page_icon="🍽️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: linear-gradient(145deg, #1a0a00 0%, #2d1200 50%, #1a0800 100%); }

.hero-wrap {
    text-align: center;
    padding: 2rem 0 1.2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.6rem, 5vw, 2.8rem);
    color: #fff;
    line-height: 1.15;
    margin: 0;
}
.hero-accent { color: #f4a435; }
.hero-sub {
    color: #c8956a;
    font-size: clamp(0.7rem, 2vw, 0.9rem);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.stFileUploader > div {
    border: 2px dashed #f4a435 !important;
    border-radius: 16px !important;
    background: rgba(244,164,53,0.06) !important;
}

/* Result card */
.result-outer {
    background: linear-gradient(135deg, #f4a435 0%, #ff6b00 100%);
    border-radius: 20px;
    padding: 2px;
    margin-top: 2rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}
.result-inner {
    background: #1e0d00;
    border-radius: 18px;
    overflow: hidden;
}
.result-image-wrap {
    width: 100%;
    background: #1e0d00;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
}
.result-image-wrap img {
    max-width: 100%;
    max-height: 260px;
    width: auto;
    height: auto;
    object-fit: contain;
    border-radius: 10px;
    display: block;
}
.result-info {
    padding: 1.2rem 1.4rem 1.6rem;
}
.result-tag {
    display: inline-block;
    background: #f4a435;
    color: #1a0800;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-food-name {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.4rem, 4vw, 2.2rem);
    color: #fff;
    margin: 0 0 0.3rem;
    line-height: 1.1;
}
.result-desc {
    color: #c8956a;
    font-size: clamp(0.8rem, 2vw, 0.9rem);
    margin: 0 0 1.2rem;
    line-height: 1.5;
}

/* Confidence block */
.conf-block {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: rgba(244,164,53,0.1);
    border: 1px solid rgba(244,164,53,0.25);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 1.2rem;
}
.conf-num {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.8rem, 5vw, 2.6rem);
    color: #f4a435;
    line-height: 1;
    font-weight: 900;
    flex-shrink: 0;
}
.conf-right { flex: 1; min-width: 0; }
.conf-lbl {
    color: #e0b080;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 5px;
}
.conf-track {
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    height: 9px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #f4a435, #ff6b00);
}

/* Prob bars */
.probs-lbl {
    color: #c8956a;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0 0 0.7rem;
    font-weight: 600;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.prob-name {
    color: #e0b080;
    font-size: clamp(0.72rem, 2vw, 0.82rem);
    width: 110px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prob-track {
    flex: 1;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    height: 7px;
    overflow: hidden;
    min-width: 0;
}
.prob-fill { height: 100%; border-radius: 99px; background: rgba(244,164,53,0.35); }
.prob-fill.top { background: linear-gradient(90deg,#f4a435,#ff6b00); }
.prob-pct {
    color: #c8956a;
    font-size: 0.76rem;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

.footer {
    text-align: center;
    color: #6b3d1a;
    font-size: 0.72rem;
    margin-top: 2rem;
    padding-bottom: 1.5rem;
}
footer { visibility: hidden; }

/* Mobile tweaks */
@media (max-width: 480px) {
    .result-info { padding: 1rem 1rem 1.2rem; }
    .conf-block { padding: 0.7rem 0.8rem; }
    .prob-name { width: 90px; }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Grilled Sandwich", "Idli", "Masala Dosa", "Paani Puri", "Samosa", "Vada Pav"]
IMG_SIZE    = (224, 224)
MODEL_PATH  = "model_resaved.h5"

FOOD_DESC = {
    "Vada Pav":        "Mumbai's iconic street burger — spiced potato fritter in a soft bun with chutney.",
    "Grilled Sandwich":"Mumbai-style toasted sandwich, layered with green chutney, veggies & cheese.",
    "Samosa":          "Crispy golden pastry stuffed with spiced potatoes and peas.",
    "Paani Puri":      "Hollow crisp shells filled with tangy tamarind water and chickpeas.",
    "Masala Dosa":     "Thin crispy South Indian crepe made from fermented rice & lentil batter.",
    "Idli":            "Soft steamed rice cakes, a South Indian breakfast staple served with sambar.",
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <p class="hero-sub">Deep Learning · MobileNetV2 · 91.33% Accuracy</p>
    <h1 class="hero-title">Mumbai <span class="hero-accent">Street Food</span><br>Classifier</h1>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ── Predict & Display ─────────────────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    if model is None:
        st.error("⚠️ Model file not found. Make sure `model_resaved.h5` is in your repo.")
    else:
        with st.spinner("Classifying..."):
            img_array = np.array(image.resize(IMG_SIZE)) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            preds     = model.predict(img_array)[0]

        top_idx    = int(np.argmax(preds))
        top_label  = CLASS_NAMES[top_idx]
        confidence = float(preds[top_idx]) * 100

        # Encode image to base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Probability rows sorted high to low
        prob_rows_html = ""
        for i in np.argsort(preds)[::-1]:
            pct    = float(preds[i]) * 100
            is_top = "top" if i == top_idx else ""
            prob_rows_html += f"""
            <div class="prob-row">
                <span class="prob-name">{CLASS_NAMES[i]}</span>
                <div class="prob-track">
                    <div class="prob-fill {is_top}" style="width:{pct:.1f}%"></div>
                </div>
                <span class="prob-pct">{pct:.1f}%</span>
            </div>"""

        desc = FOOD_DESC.get(top_label, "A delicious Mumbai street food.")

        st.markdown(f"""
        <div class="result-outer">
          <div class="result-inner">

            <div class="result-image-wrap">
              <img src="data:image/jpeg;base64,{b64}" alt="{top_label}" />
            </div>

            <div class="result-info">
              <span class="result-tag">Prediction</span>
              <p class="result-food-name">{top_label}</p>
              <p class="result-desc">{desc}</p>

              <div class="conf-block">
                <div class="conf-num">{confidence:.0f}%</div>
                <div class="conf-right">
                  <div class="conf-lbl">Confidence Score</div>
                  <div class="conf-track">
                    <div class="conf-fill" style="width:{confidence:.1f}%"></div>
                  </div>
                </div>
              </div>

              <p class="probs-lbl">All Classes</p>
              {prob_rows_html}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown(
        "<p style='text-align:center;color:#6b3d1a;margin-top:1.5rem;font-size:0.95rem;'>👆 Upload a street food image to classify it.</p>",
        unsafe_allow_html=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MobileNetV2 · 2,400 images · 6 classes · 91.33% accuracy<br>
    Research by Shruti Kesharwani · B.K. Birla College, Kalyan
</div>
""", unsafe_allow_html=True)
