"""
Plant Disease Detection System
================================
- TensorFlow/Keras model for disease prediction
- AUTO LEAF CROP (rembg + OpenCV)
- OpenWeatherMap weather integration
- Groq LLaMA AI recommendations (FREE!)
- LangChain + Groq CropBot chatbot
- Frontend matching the reference Plant AI design
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from streamlit_js_eval import get_geolocation
import cv2
import io

# ── Groq imports ──
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── rembg import (safe) ──
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Plant AI — Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS  ← sirf GROQ_API_KEY daalo neeche
# ─────────────────────────────────────────────

MODEL_PATH       = "plant_disease_model_rahul.keras"
WEATHER_API_KEY  = st.secrets["WEATHER_API_KEY"]
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
GROQ_API_KEY     = st.secrets["GROQ_API_KEY"]

CLASS_NAMES = [
    "Apple - Apple Scab",          "Apple - Black Rot",
    "Apple - Cedar Apple Rust",    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry - Powdery Mildew",     "Cherry - Healthy",
    "Corn - Cercospora / Gray Leaf Spot", "Corn - Common Rust",
    "Corn - Northern Leaf Blight", "Corn - Healthy",
    "Grape - Black Rot",           "Grape - Esca (Black Measles)",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)", "Grape - Healthy",
    "Orange - Huanglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",      "Peach - Healthy",
    "Pepper (Bell) - Bacterial Spot", "Pepper (Bell) - Healthy",
    "Potato - Early Blight",       "Potato - Late Blight",  "Potato - Healthy",
    "Raspberry - Healthy",         "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",     "Tomato - Early Blight",
    "Tomato - Late Blight",        "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites / Two-Spotted Spider Mite",
    "Tomato - Target Spot",        "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",       "Tomato - Healthy",
]

# ─────────────────────────────────────────────
# GROQ CLIENT + LANGCHAIN LLM
# ─────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def get_langchain_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY,
        temperature=0.7,
    )

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None

# ─────────────────────────────────────────────
# AUTO LEAF CROP
# ─────────────────────────────────────────────
def remove_background_rembg(pil_image):
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    result_bytes = rembg_remove(img_bytes.read())
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")

def paste_on_white(rgba_image):
    white_bg = Image.new("RGB", rgba_image.size, (255, 255, 255))
    white_bg.paste(rgba_image, mask=rgba_image.split()[3])
    return white_bg

def crop_leaf_opencv(pil_image):
    img_rgb = np.array(pil_image.convert("RGB"))
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(img_hsv, np.array([15, 20, 20]), np.array([100, 255, 255]))
    mask_brown = cv2.inRange(img_hsv, np.array([5,  20, 20]), np.array([30,  255, 200]))
    mask_grey  = cv2.inRange(img_hsv, np.array([0,  0,  40]), np.array([180, 40,  200]))
    mask = cv2.bitwise_or(cv2.bitwise_or(mask_green, mask_brown), mask_grey)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask   = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_image
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < img_rgb.shape[0] * img_rgb.shape[1] * 0.02:
        return pil_image
    x, y, w, h = cv2.boundingRect(largest)
    pad = 25
    return pil_image.crop((max(0, x-pad), max(0, y-pad),
                           min(img_rgb.shape[1], x+w+pad), min(img_rgb.shape[0], y+h+pad)))

def auto_crop_leaf(pil_image):
    try:
        if REMBG_AVAILABLE:
            cropped = crop_leaf_opencv(paste_on_white(remove_background_rembg(pil_image)))
            method  = "AI Background Remove + Leaf Crop"
        else:
            cropped = crop_leaf_opencv(pil_image)
            method  = "OpenCV Leaf Crop"
    except Exception as e:
        cropped = pil_image
        method  = f"Original (crop failed: {e})"
    return cropped, method

# ─────────────────────────────────────────────
# IMAGE PREDICTION
# ─────────────────────────────────────────────
def predict_disease(image_file, model):
    pil_image = Image.open(image_file).convert("RGB")
    cropped_image, crop_method = auto_crop_leaf(pil_image)
    arr   = np.expand_dims(tf.keras.preprocessing.image.img_to_array(
                cropped_image.resize((224, 224))), axis=0)
    preds = model.predict(arr, verbose=0)
    idx   = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(np.max(preds)) * 100, cropped_image, crop_method

# ─────────────────────────────────────────────
# WEATHER
# ─────────────────────────────────────────────
def get_weather_by_coords(lat, lon):
    try:
        resp = requests.get(WEATHER_BASE_URL,
                            params={"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"},
                            timeout=10)
        d = resp.json()
        if resp.status_code == 200:
            return {"city": d["name"], "country": d["sys"]["country"],
                    "temp": d["main"]["temp"], "feels_like": d["main"]["feels_like"],
                    "humidity": d["main"]["humidity"], "condition": d["weather"][0]["main"],
                    "description": d["weather"][0]["description"].capitalize(),
                    "wind_speed": d["wind"]["speed"], "lat": lat, "lon": lon}
        st.warning(f"⚠️ {d.get('message','Unknown error')}")
    except Exception as e:
        st.warning(f"⚠️ {e}")
    return None

def get_weather(city):
    try:
        resp = requests.get(WEATHER_BASE_URL,
                            params={"q": city, "appid": WEATHER_API_KEY, "units": "metric"},
                            timeout=10)
        d = resp.json()
        if resp.status_code == 200:
            return {"city": d["name"], "country": d["sys"]["country"],
                    "temp": d["main"]["temp"], "feels_like": d["main"]["feels_like"],
                    "humidity": d["main"]["humidity"], "condition": d["weather"][0]["main"],
                    "description": d["weather"][0]["description"].capitalize(),
                    "wind_speed": d["wind"]["speed"]}
        st.warning(f"⚠️ {d.get('message','Unknown error')}")
    except Exception as e:
        st.warning(f"⚠️ {e}")
    return None

# ─────────────────────────────────────────────
# AI RECOMMENDATION — Groq LLaMA
# ─────────────────────────────────────────────
def generate_recommendation(disease, weather):
    client = get_groq_client()
    weather_context = (
        f"Temperature {weather['temp']}°C (feels like {weather['feels_like']}°C), "
        f"Humidity {weather['humidity']}%, Condition: {weather['condition']} "
        f"({weather['description']}), Wind {weather['wind_speed']} m/s."
        if weather else "No weather data available."
    )
    prompt = f"""You are an expert agricultural advisor AI helping farmers identify and treat plant diseases.

Detected Disease / Condition: {disease}
{weather_context}

Provide a structured recommendation with these EXACT four sections.
IMPORTANT: Write complete detailed content under each section. Do NOT use numbered lists starting with just "1." - write full sentences and paragraphs.

**SUMMARY:**
One or two sentence overview of the situation and urgency.

**TREATMENT:**
Write 3-4 specific actionable treatment steps in full sentences. Include fungicide names, application methods, and timing.

**FERTILIZER:**
Write 2-3 sentences of fertilizer advice tailored to this disease and current weather.

**WEATHER ALERTS:**
Write 2-3 specific weather-based risks and precautions based on the given temperature, humidity and wind data.

Keep advice practical, concise, and farmer-friendly."""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000, temperature=0.7,
        )
        raw = response.choices[0].message.content

        def extract(text, heading):
            import re
    
            # Method 1: **HEADING:** format
            m = re.search(
                    rf"\*\*{heading}:\*\*\s*(.*?)(?=\*\*[A-Z][A-Z ]+:\*\*|$)",
                    text, re.DOTALL | re.IGNORECASE
            )   
            if m:
                content = m.group(1).strip()
                if content and len(content) > 10:
                    return content
    
            # Method 2: HEADING: format (bina asterisk)
            m2 = re.search(
                rf"(?:^|\n){heading}:\s*(.*?)(?=\n[A-Z][A-Z ]+:|$)",
                text, re.DOTALL | re.IGNORECASE
            )
            if m2:
                content = m2.group(1).strip()
                if content and len(content) > 10:
                    return content
    
            return None

        alerts_raw = extract(raw, "WEATHER ALERTS") or ""
        return {
            "summary":    extract(raw, "SUMMARY")    or raw[:300],
            "treatment":  extract(raw, "TREATMENT")  or "Please consult a local agronomist.",
            "fertilizer": extract(raw, "FERTILIZER") or "Use a balanced NPK fertiliser.",
            "alerts":     [l.strip().lstrip("•-* ").strip() for l in alerts_raw.split("\n")
                           if l.strip() and len(l.strip()) > 5],
        }
    except Exception as e:
        return {
            "treatment":  f"⚠️ AI unavailable: {e}. Consult a local agronomist.",
            "fertilizer": "Use a balanced NPK fertiliser (e.g. 14-14-14), soil pH 6.0–7.0.",
            "alerts":     ["Could not fetch weather-based alerts."],
            "summary":    f"Disease detected: **{disease}**. AI recommendations temporarily unavailable.",
        }

# ─────────────────────────────────────────────
# CROPBOT — LangChain + Groq
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are CropBot, a friendly and knowledgeable agricultural AI assistant.
You help farmers with:
- Plant disease identification and treatment
- Crop care and best practices
- Fertilizer and pesticide advice
- Irrigation and soil health
- Weather impacts on crops
- Pest management
- General plant health questions

Keep answers practical, concise, and farmer-friendly. Use simple language.
When recommending chemicals, always mention safety precautions.
If asked about something unrelated to agriculture, politely redirect the conversation."""

def chatbot_response(user_message, lc_history):
    llm = get_langchain_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + lc_history + [HumanMessage(content=user_message)]
    try:
        return llm.invoke(messages).content
    except Exception as e:
        return f"⚠️ CropBot temporarily unavailable: {e}"

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] { font-family: 'Inter', sans-serif !important; }
    [data-testid="stAppViewContainer"] > .main > div:first-child { padding-top: 0 !important; }
    .main .block-container { padding-top: 0 !important; padding-left: 0 !important; padding-right: 0 !important; max-width: 100% !important; }
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%) !important; border-right: none; }
    [data-testid="stSidebar"] * { color: #d8f3dc !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 15px !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }
    .hero-section { background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)), url('https://images.unsplash.com/photo-1558618666-fcd25c85cd64?auto=format&fit=crop&w=1600&q=80') center/cover no-repeat; min-height: 420px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 70px 40px 80px; }
    .hero-brand { font-size: 14px; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; color: #b7e4c7; margin-bottom: 20px; }
    .hero-title { font-size: clamp(32px, 5vw, 62px); font-weight: 800; color: #ffffff; line-height: 1.15; margin-bottom: 36px; max-width: 680px; }
    .hero-btn { display: inline-flex; align-items: center; gap: 10px; padding: 14px 36px; border: 2px solid rgba(255,255,255,0.85); border-radius: 40px; background: rgba(255,255,255,0.1); backdrop-filter: blur(8px); color: #ffffff !important; font-size: 16px; font-weight: 600; cursor: pointer; }
    .how-section { background: #2d6a4f; padding: 60px 40px 70px; text-align: center; }
    .how-title { font-size: 34px; font-weight: 700; color: #ffffff; margin-bottom: 50px; }
    .how-grid { display: flex; justify-content: center; gap: 60px; flex-wrap: wrap; }
    .how-step { max-width: 200px; display: flex; flex-direction: column; align-items: center; }
    .how-icon { font-size: 52px; margin-bottom: 16px; line-height: 1; }
    .how-num { width: 34px; height: 34px; border-radius: 50%; background: #ffffff; color: #1b4332; font-size: 15px; font-weight: 700; display: flex; align-items: center; justify-content: center; margin-bottom: 16px; }
    .how-step-title { font-size: 18px; font-weight: 700; color: #ffffff; margin-bottom: 10px; }
    .how-step-desc { font-size: 14px; color: rgba(255,255,255,0.8); line-height: 1.6; }
    .plants-section { background: #ffffff; padding: 60px 80px; display: flex; align-items: center; gap: 60px; }
    .plants-text-col { flex: 1; }
    .plants-title { font-size: 30px; font-weight: 700; color: #1b4332; margin-bottom: 16px; }
    .plants-desc { font-size: 15px; color: #4a5568; line-height: 1.8; }
    .crops-section { background: #f0fdf4; padding: 50px 80px; text-align: center; }
    .crops-title { font-size: 28px; font-weight: 700; color: #1b4332; margin-bottom: 6px; }
    .crops-sub { font-size: 14px; color: #6b7280; margin-bottom: 28px; }
    .crops-grid { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
    .crop-chip { padding: 8px 20px; border-radius: 24px; background: #ffffff; border: 1.5px solid #86efac; font-size: 14px; color: #166534; font-weight: 500; }
    .page-header { background: linear-gradient(135deg, #1b4332, #2d6a4f); padding: 32px 40px; color: white; }
    .page-header h1 { font-size: 30px; font-weight: 700; color: white !important; margin: 0; }
    .page-header p { color: rgba(255,255,255,0.75); margin: 6px 0 0; font-size: 14px; }
    .metric-card { background: linear-gradient(135deg, #f0fff4, #dcfce7); border-left: 4px solid #22c55e; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
    .metric-card h4 { color: #166534; margin: 0 0 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card p { color: #14532d; margin: 0; font-size: 1.4rem; font-weight: 700; }
    .disease-badge { display: inline-block; background: #dc2626; color: white !important; padding: 7px 20px; border-radius: 24px; font-weight: 700; font-size: 15px; margin: 10px 0; }
    .healthy-badge { background: #16a34a !important; }
    .alert-box { background: #fff5f5; border-left: 4px solid #f87171; border-radius: 6px; padding: 10px 14px; margin: 6px 0; color: #7f1d1d; font-size: 14px; line-height: 1.55; }
    .crop-info-box { background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 6px; padding: 8px 14px; margin: 8px 0; font-size: 13px; color: #166534; }
    .chat-container { max-height: 440px; overflow-y: auto; padding: 8px; }
    .chat-user { background: linear-gradient(135deg, #166534, #22c55e); color: white; padding: 11px 18px; border-radius: 18px 18px 4px 18px; margin: 6px 0 6px 80px; font-size: 14px; line-height: 1.55; }
    .chat-bot { background: linear-gradient(135deg, #f0fdf4, #dcfce7); color: #1a202c; border: 1px solid #bbf7d0; padding: 11px 18px; border-radius: 18px 18px 18px 4px; margin: 6px 80px 6px 0; font-size: 14px; line-height: 1.55; }
    .chat-label-user { text-align: right; font-size: 11px; color: #9ca3af; margin: 2px 0; }
    .chat-label-bot  { text-align: left;  font-size: 11px; color: #9ca3af; margin: 2px 0; }
    .location-badge { background: #eff6ff; border: 1px solid #93c5fd; border-radius: 8px; padding: 8px 14px; margin-bottom: 10px; font-size: 13px; color: #1d4ed8; }
    .stButton > button { background: linear-gradient(135deg, #22c55e, #16a34a) !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 10px 28px !important; font-weight: 600 !important; font-size: 14px !important; box-shadow: 0 4px 14px rgba(34,197,94,0.3) !important; }
    .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
    [data-testid="stFileUploader"] { border: 2px dashed #86efac !important; border-radius: 12px !important; background: #f0fdf4 !important; }
    h1, h2, h3 { color: #1b4332 !important; }
    .streamlit-expanderHeader { background: #f0fdf4 !important; border-radius: 8px !important; color: #166534 !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WEATHER CARDS HELPER
# ─────────────────────────────────────────────
def display_weather_cards(w):
    if not w: return
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><h4>🌡 Temperature</h4><p>{w["temp"]} °C</p></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h4>💧 Humidity</h4><p>{w["humidity"]} %</p></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><h4>☁️ Condition</h4><p>{w["condition"]}</p></div>', unsafe_allow_html=True)
    st.caption(f"Feels like {w['feels_like']} °C · {w['description']} · Wind {w['wind_speed']} m/s")

# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-brand">🌿 Plant AI</div>
        <div class="hero-title">Try our AI Powered<br>Disease Detection</div>
        <div class="hero-btn">☁️&nbsp; Try Now</div>
    </div>
    <div class="how-section">
        <div class="how-title">How it works?</div>
        <div class="how-grid">
            <div class="how-step"><div class="how-icon">📱</div><div class="how-num">1</div><div class="how-step-title">Click a Pic</div><div class="how-step-desc">Take a picture of your plant leaf</div></div>
            <div class="how-step"><div class="how-icon">☁️</div><div class="how-num">2</div><div class="how-step-title">Upload on Plant AI</div><div class="how-step-desc">Visit Plant AI and click Try Now to upload your picture</div></div>
            <div class="how-step"><div class="how-icon">📋</div><div class="how-num">3</div><div class="how-step-title">Get final Report</div><div class="how-step-desc">Plant AI will analyze your plant and display a detailed report</div></div>
        </div>
    </div>
    <div class="plants-section">
        <div class="plants-text-col">
            <div class="plants-title">🌱 Your plants need you!</div>
            <div class="plants-desc">Human society needs to increase food production by an estimated 70% by 2050. Currently, infectious diseases reduce the potential yield by an average of 40%.<br><br>Our AI-powered system helps farmers <strong>detect diseases early</strong>, get <strong>weather-aware treatment advice</strong> from Groq LLaMA AI, and consult <strong>CropBot</strong> — an always-on agricultural assistant — all in under 3 seconds.</div>
        </div>
    </div>
    <div class="crops-section">
        <div class="crops-title">Supported Crops</div>
        <div class="crops-sub">14 crops · 38 disease classes · PlantVillage dataset</div>
        <div class="crops-grid">
            <div class="crop-chip">🍎 Apple</div><div class="crop-chip">🍇 Grape</div><div class="crop-chip">🌽 Corn</div>
            <div class="crop-chip">🥔 Potato</div><div class="crop-chip">🍅 Tomato</div><div class="crop-chip">🍑 Peach</div>
            <div class="crop-chip">🫑 Bell Pepper</div><div class="crop-chip">🍓 Strawberry</div><div class="crop-chip">🫐 Blueberry</div>
            <div class="crop-chip">🍊 Orange</div><div class="crop-chip">🌱 Soybean</div><div class="crop-chip">🥦 Squash</div>
            <div class="crop-chip">🍒 Cherry</div><div class="crop-chip">🫒 Raspberry</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
def page_about():
    st.markdown("""<div class="page-header"><h1>📖 About This Project</h1><p>PlantVillage dataset · EfficientNetB3 · Groq LLaMA · LangChain · OpenWeatherMap</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🗃️ Dataset")
        st.markdown("""| Split | Images |\n|-------|--------|\n| Training | 70,295 |\n| Validation | 17,572 |\n\n**87,000+ RGB images** across **38 classes**.""")
    with c2:
        st.markdown("### 🤖 Model")
        st.markdown("- **Architecture:** EfficientNetB3 (Transfer Learning)\n- **Input:** 224×224px\n- **Auto Crop:** rembg + OpenCV\n\n### 🌦️ Weather\n- OpenWeatherMap · Auto GPS · Manual city")
    st.markdown("### 💡 AI Stack")
    st.markdown("""| Component | Technology |\n|-----------|------------|\n| Disease model | TensorFlow/Keras (EfficientNetB3) |\n| Leaf Auto-Crop | rembg + OpenCV |\n| Recommendations | **Groq LLaMA 3.1 8B** |\n| CropBot | **LangChain + ChatGroq** |\n| Weather | OpenWeatherMap |\n| Framework | Streamlit |""")

# ─────────────────────────────────────────────
# PAGE: DISEASE RECOGNITION
# ─────────────────────────────────────────────
def page_recognition():
    st.markdown("""<div class="page-header"><h1>🔬 Disease Recognition</h1><p>Upload a leaf image — AI will auto-crop and detect the disease instantly</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please check MODEL_PATH.")
        return

    if REMBG_AVAILABLE:
        st.markdown('<div class="crop-info-box">✅ <b>Auto Leaf Crop:</b> AI background remove (rembg) + OpenCV — active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="crop-info-box">⚠️ <b>Auto Leaf Crop:</b> OpenCV only — run <code>pip install rembg onnxruntime</code> for better results</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("📷 Upload Plant Image")
        uploaded = st.file_uploader("Choose a leaf image (JPG / PNG / JPEG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="📤 Uploaded Image", use_container_width=True)

        st.subheader("🌍 Weather Data")
        weather_data = None

        st.markdown("**Option 1 — Use My Current Location**")
        if st.button("📍 Detect My Location & Fetch Weather"):
            with st.spinner("Requesting location..."):
                location = get_geolocation()
            if location and "coords" in location:
                lat, lon = location["coords"]["latitude"], location["coords"]["longitude"]
                st.markdown(f'<div class="location-badge">📌 Detected: Lat <b>{lat:.4f}</b>, Lon <b>{lon:.4f}</b></div>', unsafe_allow_html=True)
                with st.spinner("Fetching weather..."):
                    weather_data = get_weather_by_coords(lat, lon)
                if weather_data:
                    st.success(f"✅ Weather for **{weather_data['city']}, {weather_data['country']}**")
                    display_weather_cards(weather_data)
                else:
                    st.error("❌ Could not fetch weather.")
            else:
                st.warning("⚠️ Location denied. Use manual option below.")

        st.markdown("---")
        st.markdown("**Option 2 — Enter City Manually**")
        city = st.text_input("City name", placeholder="e.g. Mumbai, Delhi, Kolkata")
        if st.button("🌤️ Fetch Weather by City") and city.strip():
            with st.spinner("Fetching weather..."):
                weather_data = get_weather(city.strip())
            if weather_data:
                st.success(f"Weather for **{weather_data['city']}, {weather_data['country']}**")
                display_weather_cards(weather_data)

        if weather_data:
            st.session_state["weather_data"] = weather_data

    with right_col:
        st.subheader("🧠 AI Analysis")
        if st.button("🚀 Predict Disease", use_container_width=True):
            if uploaded is None:
                st.warning("⚠️ Please upload an image first.")
            else:
                with st.spinner("✂️ Auto-cropping leaf & predicting..."):
                    disease_name, confidence, cropped_img, crop_method = predict_disease(uploaded, model)

                st.markdown("**🌿 Auto-Cropped Leaf:**")
                st.image(cropped_img, caption=f"✂️ {crop_method}", use_container_width=True)
                st.markdown("---")

                is_healthy  = "healthy" in disease_name.lower()
                badge_class = "healthy-badge disease-badge" if is_healthy else "disease-badge"
                st.markdown("#### 📋 Prediction Result")
                st.markdown(f'<span class="{badge_class}">{disease_name}</span>', unsafe_allow_html=True)
                st.progress(int(confidence))
                st.caption(f"Confidence: **{confidence:.1f}%**")
                st.markdown("---")

                stored_weather = st.session_state.get("weather_data", None)
                st.markdown("#### 💡 Groq AI Recommendations")
                with st.spinner("🤖 Generating recommendations..."):
                    rec = generate_recommendation(disease_name, stored_weather)

                st.info(rec["summary"])
                with st.expander("🩺 Treatment Plan", expanded=True):
                    st.markdown(rec["treatment"])
                with st.expander("🌱 Fertiliser Advice", expanded=True):
                    st.markdown(rec["fertilizer"])
                if rec["alerts"]:
                    with st.expander("🚨 Weather-Based Alerts", expanded=True):
                        for alert in rec["alerts"]:
                            if alert.strip():
                                st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)
                elif stored_weather:
                    st.success("✅ Weather conditions look favourable for your crop.")
                else:
                    st.caption("💡 Add your city above to get weather-aware alerts.")

# ─────────────────────────────────────────────
# PAGE: CROPBOT
# ─────────────────────────────────────────────
def page_chatbot():
    st.markdown("""<div class="page-header"><h1>🤖 CropBot — AI Farming Assistant</h1><p>Powered by LangChain + Groq LLaMA 3.1</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("Ask me **anything** about plant diseases, crop care, fertilizers, pests, irrigation, and more! 🌾")

    if "chat_history_display" not in st.session_state:
        st.session_state["chat_history_display"] = []
    if "lc_history" not in st.session_state:
        st.session_state["lc_history"] = []

    if st.button("🗑️ Clear Chat"):
        st.session_state["chat_history_display"] = []
        st.session_state["lc_history"] = []
        st.rerun()

    if st.session_state["chat_history_display"]:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for role, text in st.session_state["chat_history_display"]:
            if role == "user":
                st.markdown(f'<p class="chat-label-user">You</p><div class="chat-user">{text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="chat-label-bot">🌿 CropBot</p><div class="chat-bot">{text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""<div style="text-align:center;padding:40px 20px;color:#6b7280;border:2px dashed #86efac;border-radius:14px;margin:20px 0;background:#f0fdf4;">
            <h3 style="color:#166534;font-size:22px;">👋 Hello, Farmer!</h3>
            <p>Start by typing a question below or pick a quick example.</p>
            <p style="font-size:13px;font-style:italic;color:#9ca3af;">"My tomato leaves are turning yellow" · "Best fertilizer for potatoes?" · "How to prevent late blight?"</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💬 Quick Questions:**")
    qs_col1, qs_col2, qs_col3 = st.columns(3)
    quick_questions = ["How do I treat tomato late blight?", "Best fertilizer for corn in humid weather?", "How to prevent powdery mildew on grapes?"]
    for i, (col, q) in enumerate(zip([qs_col1, qs_col2, qs_col3], quick_questions)):
        with col:
            if st.button(q, key=f"quick_{i}"):
                with st.spinner("🌿 CropBot is thinking..."):
                    bot_reply = chatbot_response(q, st.session_state["lc_history"])
                st.session_state["chat_history_display"].extend([("user", q), ("bot", bot_reply)])
                st.session_state["lc_history"].extend([HumanMessage(content=q), AIMessage(content=bot_reply)])
                st.rerun()

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your question:", placeholder="e.g. My potato leaves have dark spots. What disease could it be?", height=80, label_visibility="collapsed")
        send_btn = st.form_submit_button("📤 Send", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("🌿 CropBot is thinking..."):
            bot_reply = chatbot_response(user_input.strip(), st.session_state["lc_history"])
        st.session_state["chat_history_display"].extend([("user", user_input.strip()), ("bot", bot_reply)])
        st.session_state["lc_history"].extend([HumanMessage(content=user_input.strip()), AIMessage(content=bot_reply)])
        st.rerun()
    elif send_btn:
        st.warning("⚠️ Please type a question before sending.")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    inject_css()
    st.sidebar.title("🌿 Plant AI")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", ["🏠 Home", "📖 About", "🔬 Disease Recognition", "🤖 CropBot Chat"])
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Model:** TensorFlow / Keras  \n"
        "**Auto Crop:** rembg + OpenCV  \n"
        "**Weather:** OpenWeatherMap  \n"
        "**Recommendations:** Groq LLaMA 3.1  \n"
        "**Chatbot:** LangChain + Groq  \n"
        "**Framework:** Streamlit"
    )
    if page == "🏠 Home":                  page_home()
    elif page == "📖 About":               page_about()
    elif page == "🔬 Disease Recognition": page_recognition()
    elif page == "🤖 CropBot Chat":        page_chatbot()

if __name__ == "__main__":
    main()