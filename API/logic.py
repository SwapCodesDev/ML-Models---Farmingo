import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
from functools import lru_cache

import tensorflow as tf


# ============================================================
# Average Daily Rainfall Version (mm/day)
# ============================================================

CROP_REQUIREMENTS = {
    # Cereals
    "rice": {"temperature": (20, 35), "humidity": (70, 90), "rainfall": (1.25, 2.5), "ph": (5.5, 7.0)},
    "maize": {"temperature": (18, 30), "humidity": (50, 80), "rainfall": (0.67, 1.25), "ph": (5.8, 7.0)},
    "wheat": {"temperature": (10, 25), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},
    "barley": {"temperature": (12, 25), "humidity": (40, 60), "rainfall": (0.33, 0.83), "ph": (6.0, 7.5)},
    "millet": {"temperature": (25, 35), "humidity": (40, 60), "rainfall": (0.25, 0.83), "ph": (5.5, 7.0)},
    "sorghum": {"temperature": (25, 35), "humidity": (40, 60), "rainfall": (0.33, 1.0), "ph": (6.0, 7.5)},

    # Pulses
    "chickpea": {"temperature": (10, 30), "humidity": (40, 60), "rainfall": (0.42, 0.83), "ph": (6.0, 8.0)},
    "kidneybeans": {"temperature": (15, 30), "humidity": (50, 70), "rainfall": (0.5, 1.0), "ph": (6.0, 7.5)},
    "blackgram": {"temperature": (20, 35), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},
    "lentil": {"temperature": (10, 30), "humidity": (40, 60), "rainfall": (0.33, 0.67), "ph": (6.0, 7.5)},
    "mungbean": {"temperature": (25, 35), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.2, 7.2)},
    "mothbeans": {"temperature": (25, 40), "humidity": (20, 50), "rainfall": (0.17, 0.5), "ph": (6.0, 7.0)},
    "pigeonpeas": {"temperature": (20, 35), "humidity": (50, 70), "rainfall": (0.42, 1.0), "ph": (6.0, 7.5)},

    # Commercial
    "cotton": {"temperature": (25, 35), "humidity": (50, 80), "rainfall": (0.42, 1.25), "ph": (6.0, 8.0)},
    "jute": {"temperature": (20, 35), "humidity": (70, 90), "rainfall": (1.25, 2.08), "ph": (6.4, 7.2)},
    "sugarcane": {"temperature": (20, 35), "humidity": (70, 85), "rainfall": (0.83, 2.08), "ph": (6.0, 7.5)},
    "coffee": {"temperature": (20, 30), "humidity": (60, 90), "rainfall": (1.25, 2.08), "ph": (6.0, 6.8)},
    "tea": {"temperature": (18, 30), "humidity": (70, 90), "rainfall": (1.25, 2.5), "ph": (4.5, 6.0)},
    "rubber": {"temperature": (25, 35), "humidity": (70, 90), "rainfall": (1.25, 2.5), "ph": (4.5, 6.5)},
    "tobacco": {"temperature": (20, 30), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (5.5, 6.5)},
    "groundnut": {"temperature": (25, 35), "humidity": (50, 70), "rainfall": (0.42, 1.0), "ph": (6.0, 7.0)},
    "sunflower": {"temperature": (20, 30), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},
    "soybean": {"temperature": (20, 30), "humidity": (60, 80), "rainfall": (0.5, 1.0), "ph": (6.0, 7.5)},
    "mustard": {"temperature": (10, 25), "humidity": (40, 60), "rainfall": (0.25, 0.83), "ph": (6.0, 7.5)},

    # Fruits
    "banana": {"temperature": (25, 30), "humidity": (70, 90), "rainfall": (0.83, 1.67), "ph": (6.0, 7.5)},
    "mango": {"temperature": (24, 35), "humidity": (50, 70), "rainfall": (0.42, 1.25), "ph": (5.5, 7.5)},
    "orange": {"temperature": (20, 30), "humidity": (50, 70), "rainfall": (0.5, 1.0), "ph": (5.5, 7.0)},
    "grapes": {"temperature": (20, 30), "humidity": (50, 70), "rainfall": (0.33, 0.83), "ph": (6.0, 7.5)},
    "papaya": {"temperature": (25, 35), "humidity": (60, 80), "rainfall": (0.67, 1.25), "ph": (6.0, 6.5)},
    "pomegranate": {"temperature": (25, 35), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},
    "guava": {"temperature": (23, 30), "humidity": (50, 70), "rainfall": (0.5, 0.83), "ph": (6.0, 7.5)},
    "apple": {"temperature": (10, 25), "humidity": (50, 70), "rainfall": (0.42, 1.25), "ph": (6.0, 7.5)},
    "pineapple": {"temperature": (22, 32), "humidity": (70, 90), "rainfall": (1.25, 2.08), "ph": (4.5, 6.5)},
    "watermelon": {"temperature": (25, 35), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},
    "muskmelon": {"temperature": (24, 32), "humidity": (50, 70), "rainfall": (0.42, 0.83), "ph": (6.0, 7.5)},

    # Plantation
    "coconut": {"temperature": (25, 35), "humidity": (70, 90), "rainfall": (1.25, 2.08), "ph": (5.5, 7.0)},
    "cashew": {"temperature": (24, 35), "humidity": (50, 70), "rainfall": (0.42, 1.67), "ph": (5.0, 7.0)},
}

# ============================================================
# State Average Environmental Conditions (Daily)
# ============================================================

STATE_CONDITIONS = {
    "Andhra Pradesh": {"temperature": 28.5, "humidity": 75, "ph": 6.8, "rainfall": 5.2},
    "Arunachal Pradesh": {"temperature": 22.0, "humidity": 85, "ph": 6.2, "rainfall": 6.5},
    "Assam": {"temperature": 26.5, "humidity": 88, "ph": 6.0, "rainfall": 7.0},
    "Bihar": {"temperature": 27.0, "humidity": 70, "ph": 6.5, "rainfall": 4.5},
    "Chhattisgarh": {"temperature": 27.5, "humidity": 75, "ph": 6.6, "rainfall": 5.0},
    "Goa": {"temperature": 29.0, "humidity": 85, "ph": 6.5, "rainfall": 6.2},
    "Gujarat": {"temperature": 30.0, "humidity": 60, "ph": 7.0, "rainfall": 3.5},
    "Haryana": {"temperature": 26.0, "humidity": 55, "ph": 7.2, "rainfall": 2.8},
    "Himachal Pradesh": {"temperature": 18.0, "humidity": 65, "ph": 6.8, "rainfall": 4.0},
    "Jharkhand": {"temperature": 26.5, "humidity": 70, "ph": 6.4, "rainfall": 4.6},
    "Karnataka": {"temperature": 27.5, "humidity": 80, "ph": 6.4, "rainfall": 4.0},
    "Kerala": {"temperature": 28.0, "humidity": 88, "ph": 6.2, "rainfall": 7.5},
    "Madhya Pradesh": {"temperature": 27.0, "humidity": 65, "ph": 6.7, "rainfall": 3.8},
    "Maharashtra": {"temperature": 28.0, "humidity": 70, "ph": 6.6, "rainfall": 4.1},
    "Manipur": {"temperature": 23.0, "humidity": 80, "ph": 6.1, "rainfall": 6.2},
    "Meghalaya": {"temperature": 22.0, "humidity": 90, "ph": 5.8, "rainfall": 8.0},
    "Mizoram": {"temperature": 23.5, "humidity": 85, "ph": 6.0, "rainfall": 6.8},
    "Nagaland": {"temperature": 24.0, "humidity": 80, "ph": 6.1, "rainfall": 6.0},
    "Odisha": {"temperature": 28.0, "humidity": 80, "ph": 6.5, "rainfall": 5.5},
    "Punjab": {"temperature": 26.5, "humidity": 60, "ph": 7.3, "rainfall": 3.0},
    "Rajasthan": {"temperature": 31.0, "humidity": 45, "ph": 7.5, "rainfall": 2.0},
    "Sikkim": {"temperature": 20.0, "humidity": 85, "ph": 6.0, "rainfall": 6.5},
    "Tamil Nadu": {"temperature": 29.0, "humidity": 75, "ph": 6.7, "rainfall": 4.0},
    "Telangana": {"temperature": 28.0, "humidity": 70, "ph": 6.5, "rainfall": 4.2},
    "Tripura": {"temperature": 25.5, "humidity": 85, "ph": 6.3, "rainfall": 6.0},
    "Uttar Pradesh": {"temperature": 27.0, "humidity": 65, "ph": 7.0, "rainfall": 3.5},
    "Uttarakhand": {"temperature": 21.0, "humidity": 70, "ph": 6.8, "rainfall": 4.2},
    "West Bengal": {"temperature": 27.5, "humidity": 80, "ph": 6.3, "rainfall": 5.0},
    # Union Territories
    "Andaman and Nicobar Islands": {"temperature": 27.0, "humidity": 85, "ph": 6.5, "rainfall": 7.2},
    "Chandigarh": {"temperature": 26.0, "humidity": 60, "ph": 7.2, "rainfall": 3.0},
    "Dadra and Nagar Haveli and Daman and Diu": {"temperature": 28.0, "humidity": 75, "ph": 6.8, "rainfall": 5.0},
    "Delhi": {"temperature": 27.0, "humidity": 55, "ph": 7.3, "rainfall": 2.5},
    "Jammu and Kashmir": {"temperature": 16.0, "humidity": 65, "ph": 6.8, "rainfall": 3.8},
    "Ladakh": {"temperature": 10.0, "humidity": 40, "ph": 7.0, "rainfall": 1.5},
    "Lakshadweep": {"temperature": 28.0, "humidity": 85, "ph": 6.5, "rainfall": 7.0},
    "Puducherry": {"temperature": 29.0, "humidity": 80, "ph": 6.8, "rainfall": 4.8}
}


# ============================================================
# 1. REVERSE GEOCODING (CACHED)
# ============================================================

@lru_cache(maxsize=256)
def reverse_geocode_state(lat: float, lon: float) -> str:
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": lat,
        "lon": lon,
        "zoom": 10,
        "addressdetails": 1,
    }

    try:
        r = requests.get(url, params=params, timeout=10, headers={"User-Agent": "Farmingo/1.0"})
        r.raise_for_status()
        address = r.json().get("address", {})
    except Exception:
        return "Unknown"

    for k in ("state", "region", "state_district", "province", "county"):
        if k in address:
            return address[k].title()

    return address.get("country", "Unknown").title()


# ============================================================
# 2. WEATHER API
# ============================================================

@lru_cache(maxsize=256)
def fetch_open_meteo(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "hourly": "relativehumidity_2m,soil_temperature_0cm,soil_moisture_0_to_1cm,temperature_2m",
        "forecast_days": 7,
        "timezone": "auto",
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def compute_features(df: pd.DataFrame) -> dict:
    cols = [
        "temp_max", "temp_min", "precipitation",
        "rh_mean", "soil_temp_mean", "soil_moist_mean", "temp_mean"
    ]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    temp_avg = df["temp_mean"].mean() if df["temp_mean"].notna().any() \
               else ((df["temp_max"] + df["temp_min"]) / 2).mean()

    humidity_avg = df["rh_mean"].mean()
    rainfall_avg = df["precipitation"].sum() / 7
    soil_moist = df["soil_moist_mean"].mean() or 0.15
    soil_temp = df["soil_temp_mean"].mean()

    N = 200 * soil_moist * (1 - abs(soil_moist - 0.25) * 2)
    P = 40 * np.exp(-((temp_avg - 30) ** 2) / 100)
    K = 250 * soil_moist * (1 - abs(soil_moist - 0.25) * 2)

    ph = 6.8 - 0.05 * (rainfall_avg / 5) - 0.02 * (humidity_avg / 100)
    ph = float(np.clip(ph, 5.0, 8.0))

    return {
        "N": round(N, 2),
        "P": round(P, 2),
        "K": round(K, 2),
        "temperature": round(temp_avg, 2),
        "humidity": round(humidity_avg, 2),
        "ph": ph,
        "rainfall": round(rainfall_avg, 2),
    }


# ============================================================
# 4. SEASON DETECTION
# ============================================================

def get_season() -> int:
    month = datetime.now().strftime("%b")
    if month in ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]:
        return 1
    if month in ["Jun", "Jul", "Aug", "Sep"]:
        return 2
    return 3


# ============================================================
# 5. MODEL LOADING (CACHED)
# ============================================================

MODEL_PATH = r"D:\Projects\ML Models - Farmingo\Weather Prediction\backup\xgboost_model.pkl"
STATE_ENCODER_PATH = r"D:\Projects\ML Models - Farmingo\Weather Prediction\backup\state_encoder.pkl"
CROP_ENCODER_PATH = r"D:\Projects\ML Models - Farmingo\Weather Prediction\backup\crop_encoder.pkl"

@lru_cache(maxsize=1)
def load_main_model():
    return (
        pickle.load(open(MODEL_PATH, "rb")),
        pickle.load(open(STATE_ENCODER_PATH, "rb")),
        pickle.load(open(CROP_ENCODER_PATH, "rb")),
    )


def predict_crop(features: dict, state: str) -> str:
    model, state_le, crop_le = load_main_model()

    clean_state = state.replace(" State", "").replace(" District", "").strip()
    enc_state = state_le.transform([clean_state])[0] if clean_state in state_le.classes_ else 0

    X = np.array([[
        features["N"], features["P"], features["K"],
        features["temperature"], features["humidity"],
        features["ph"], features["rainfall"],
        enc_state, features["season_code"]
    ]])

    pred = model.predict(X)
    return crop_le.inverse_transform(pred)[0].capitalize()


# ============================================================
# 6. ALTERNATIVE CROPS (merged)
# ============================================================

def recommend_alternatives(predicted: str, state: str):
    state_data = next((s for s in STATE_CONDITIONS if state.lower() in s.lower()), None)
    if not state_data:
        return []

    env = STATE_CONDITIONS[state_data]
    ranked = []

    for crop, req in CROP_REQUIREMENTS.items():
        if crop == predicted.lower():
            continue

        score = (
            abs(env["temperature"] - np.mean(req["temperature"])) +
            abs(env["humidity"] - np.mean(req["humidity"])) / 2 +
            abs(env["ph"] - np.mean(req["ph"])) * 5 +
            abs(env["rainfall"] - np.mean(req["rainfall"]) * 7) / 2
        )
        ranked.append((crop, score))

    ranked.sort(key=lambda x: x[1])
    return [c.capitalize() for c, _ in ranked[:5]]


# ============================================================
# 7. DISEASE PREDICTION
# ============================================================

INFO_JSON_FOLDER = r"D:\Projects\ML Models - Farmingo\Crop Disease Prediction\backup\info_json"
MODEL_FOLDER = r"D:\Projects\ML Models - Farmingo\Crop Disease Prediction\backup\trained_models"

@lru_cache(maxsize=128)
def load_disease_info(crop):
    path = os.path.join(INFO_JSON_FOLDER, f"{crop}_disease_info.json")
    with open(path, "r") as f:
        return json.load(f)

@lru_cache(maxsize=64)
def load_crop_model(crop):
    path = os.path.join(MODEL_FOLDER, f"{crop}_leaf_disease_classifier.h5")
    return tf.keras.models.load_model(path)

def predict_disease(crop: str, image_path: str) -> dict:
    info = load_disease_info(crop)
    model = load_crop_model(crop)

    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)[None]
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    class_names = list(info.keys())

    predicted = class_names[idx]
    details = info[predicted]

    return {
        "crop": crop,
        "predicted_disease": predicted,
        "confidence": round(float(preds[idx] * 100), 2),
        "cause": details.get("Cause"),
        "symptoms": details.get("Symptoms"),
        "precautions": details.get("Precautions", []),
        "cure": {
            "chemical": details.get("Cure", {}).get("Chemical", []),
            "organic": details.get("Cure", {}).get("Organic", []),
        },
    }


# ============================================================
# 8. PRICE PREDICTION
# ============================================================

PRICE_MODEL_PATH = r"D:\Projects\ML Models - Farmingo\Crop Price Prediction\backup\crop_price_model_01.pkl"

@lru_cache(maxsize=1)
def load_price_model():
    return pickle.load(open(PRICE_MODEL_PATH, "rb"))

def predict_crop_price(crop, region, date=None):
    model = load_price_model()
    date_obj = pd.to_datetime(date) if date else datetime.today()

    df = pd.DataFrame([{
        "District": region,
        "Market": "null",
        "Commodity": crop,
        "Year": date_obj.year,
        "Month": date_obj.month,
        "Day": date_obj.day,
        "DayOfWeek": date_obj.dayofweek,
        "WeekOfYear": date_obj.isocalendar()[1],
    }])

    return round(float(model.predict(df)[0]), 2)
