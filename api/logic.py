import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # MUST be before importing TF

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import pickle
import json
import re
import time
from functools import lru_cache
from pathlib import Path


# Path setup

# Base: project root
BASE_DIR = Path(__file__).resolve().parent.parent

PRICE_DATA_PATH = BASE_DIR / "crop-price-prediction" / "datasets" / "wholesale_commodity_prices.xlsx"
df_prices = pd.read_excel(PRICE_DATA_PATH)
df_prices['State'] = df_prices['State'].astype(str).str.strip().str.lower()
df_prices['Commodity'] = df_prices['Commodity'].astype(str).str.strip().str.lower()

# Model directories

CROP_DISEASE_PREDICTION_DIR = BASE_DIR / "crop-disease-prediction" / "backup"
WEATHER_PREDICTION_DIR = BASE_DIR / "weather-prediction" / "backup"

# --- Weather Prediction ---
MODEL_PATH = WEATHER_PREDICTION_DIR / "xgboost_model.pkl"
STATE_ENCODER_PATH = WEATHER_PREDICTION_DIR / "state_encoder.pkl"
CROP_ENCODER_PATH = WEATHER_PREDICTION_DIR / "crop_encoder.pkl"

# --- Crop Disease Prediction ---
INFO_JSON_FOLDER = CROP_DISEASE_PREDICTION_DIR / "info_json"
MODEL_FOLDER = CROP_DISEASE_PREDICTION_DIR / "trained_models"




# Average Daily Rainfall Version (mm/day)
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

# State Average Environmental Conditions (Daily)
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


# 1. REVERSE GEOCODING (CACHED)
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


# 2. WEATHER API
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


# 3. FEATURE ENGINEERING
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


# 4. SEASON DETECTION
def get_season() -> int:
    month = datetime.now().strftime("%b")
    if month in ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]:
        return 1
    if month in ["Jun", "Jul", "Aug", "Sep"]:
        return 2
    return 3


# 5. MODEL LOADING (CACHED)
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


# 6. ALTERNATIVE CROPS (merged)
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


# 7. DISEASE PREDICTION
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


# ==============================
# 🌶️ Commodity Map
# ==============================
COMMODITY_MAP = {
    "onion": ["onion", "onion dry", "onion green"],
    "tomato": ["tomato", "tomato hybrid"],
    "potato": ["potato"],
    "cabbage": ["cabbage"],
    "carrot": ["carrot"],
    "chilli": ["chilli", "green chilli", "red chilli"],
    "brinjal": ["brinjal"],
    "cucumber": ["cucumber"],
    "cauliflower": ["cauliflower"],
    "beetroot": ["beetroot", "beet"],
    "bhindi": ["bhindi", "bhendi", "ladies finger"],
    "garlic": ["garlic"],
    "ginger": ["ginger"],
    "sweet potato": ["sweet potato"],
    "spring onion": ["spring onion"],
    "spinach": ["spinach"],
    "methi": ["methi", "fenugreek"],
    "coriander leaves": ["coriander leaves", "dhaniya"],
    "bottle gourd": ["bottle gourd", "lauki"],
    "ridge gourd": ["ridge gourd", "turai"],
    "bitter gourd": ["bitter gourd", "karela"],
    "snake gourd": ["snake gourd"],
    "drumstick": ["drumstick"],
    "pumpkin": ["pumpkin"],
    "capsicum": ["capsicum", "bell pepper"],
}


# ==============================
# 📦 Fetch Data
# ==============================
API_KEY = "579b464db66ec23bdd0000019cbc42efd27b401673aa06ae28eb5b4d"

_API_CACHE = {}
CACHE_TTL = 3600  # 1 hour

def fetch_data(state):
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    current_time = time.time()

    # Check Cache
    if state in _API_CACHE:
        cached_time, cached_data = _API_CACHE[state]
        if current_time - cached_time < CACHE_TTL:
            print(f"API Cache hit for state: {state}")
            return cached_data

    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 1000,
        "filters[state]": state
    }

    retries = 3
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            records = data.get("records", [])

            # Save to cache
            _API_CACHE[state] = (current_time, records)
            return records

        except Exception as e:
            if attempt == retries - 1:
                print(f"API Error after {retries} attempts for state {state}:", e)
                return []
            
            # Exponential Backoff: 2s, 4s
            sleep_time = 2 ** (attempt + 1)
            print(f"API attempt {attempt+1} failed ({e}). Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    return []


# ==============================
# 🔍 Filter Commodity
# ==============================
def filter_data(data, commodity):
    aliases = COMMODITY_MAP.get(commodity.lower(), [commodity.lower()])
    result = []

    for row in data:
        item = row.get("commodity", "").lower()

        if any(alias in item for alias in aliases):
            result.append({
                "date": row.get("arrival_date"),
                "district": row.get("district"),
                "market": row.get("market"),
                "commodity": row.get("commodity"),
                "price": row.get("modal_price")
            })

    return result


# ==============================
# 💰 NOTEBOOK PRICE LOGIC (UNCHANGED)
# ==============================
def compute_prices(filtered):
    all_prices = []

    for rec in filtered:
        try:
            price = float(rec.get("price", 0))

            if 200 <= price <= 6000:
                all_prices.append(price)

        except:
            continue

    if len(all_prices) == 0:
        return 0, 0

    prices = np.array(all_prices)

    if len(prices) > 10:
        low = np.percentile(prices, 5)
        high = np.percentile(prices, 95)
        prices = prices[(prices >= low) & (prices <= high)]

    base_price = np.percentile(prices, 10)
    max_price = np.percentile(prices, 90)

    return round(base_price, 2), round(max_price, 2)


# ==============================
# 📊 EXCEL FUNCTION (NEW ONLY)
# ==============================
def get_excel_prices(state, commodity):
    try:
        state_name = state.strip().lower()
        commodity_name = re.sub(r'[^a-zA-Z ]', '', commodity).strip().lower()
        current_month = datetime.now().month

        filtered = df_prices[
            (df_prices['State'] == state_name) &
            (df_prices['Commodity'] == commodity_name) &
            (df_prices['Month'] == current_month)
        ]

        if not filtered.empty:
            return float(filtered['MinPrice'].min()), float(filtered['MaxPrice'].max())
        else:
            return None, None

    except Exception as e:
        print("Excel Error:", e)
        return None, None


# ==============================
# 🏠 PREDICT LOGIC
# ==============================
def predict_price_logic(lat: float, lon: float, commodity: str = "chilli"):
    state = reverse_geocode_state(lat, lon)

    if state == "Unknown":
        raise Exception("State detection failed")

    all_data = fetch_data(state)
    filtered = filter_data(all_data, commodity)

    base_price, max_price = compute_prices(filtered)

    excel_min, excel_max = get_excel_prices(state, commodity)

    output = {
        "status": "success",
        "state": state,
        "total_records": len(all_data),
        "filtered_count": len(filtered),
        "data": filtered[:10],

        "base_price": base_price,
        "max_price": max_price,
        "base_price_kg": round(base_price / 100, 2),
        "max_price_kg": round(max_price / 100, 2),

        "excel_min": excel_min,
        "excel_max": excel_max
    }
    
    return output

# ==========================================
# 📈 DEMAND & SUPPLY: AGMARKNET API WRAPPER
# ==========================================

states = {
  20: {
      "name": "Maharashtra",
      "districts": {
            338: "Ahmednagar",
            339: "Akola",
            340: "Amarawati",
            342: "Beed",
            343: "Bhandara",
            344: "Buldhana",
            345: "Chandrapur",
            346: "Chattrapati Sambhajinagar",
            347: "Dharashiv(Usmanabad)",
            348: "Dhule",
            349: "Gadchiroli",
            350: "Gondiya",
            351: "Hingoli",
            352: "Jalana",
            353: "Jalgaon",
            354: "Kolhapur",
            355: "Latur",
            356: "Mumbai",
            358: "Nagpur",
            359: "Nanded",
            360: "Nandurbar",
            361: "Nashik",
            363: "Parbhani",
            364: "Pune",
            365: "Raigad",
            366: "Ratnagiri",
            367: "Sangli",
            368: "Satara",
            369: "Sholapur",
            370: "Sindhudurg",
            371: "Thane",
            372: "Vashim",
            373: "Wardha",
            374: "Yavatmal"
      }
  }
}

commodities = {
    1: {
        "name": "Cereals",
        "commodities": {
            28: "Bajra(Pearl Millet/Cumbu)",
            4: "Maize",
            3: "Rice",
            1: "Wheat"
        }
    },
    2: {
        "name": "Pulses",
        "commodities": {
            214: "Arhar Dal(Tur Dal)",
            217: "Bengal Gram Dal(Chana Dal)",
            219: "Green Gram Dal(Moong Dal)",
            213: "Masur Dal",
            79: "Mataki",
            491: "Kidney Beans(Rajma)"
        }
    },
    5: {
        "name": "Fruits",
        "commodities": {
            18: "Orange",
            60: "Water Melon",
            17: "Apple",
            19: "Banana",
            20: "Mango",
            22: "Grapes"
        }
    },
    6: {
        "name": "Vegetables",
        "commodities": {
            24: "Potato",
            23: "Onion",
            65: "Tomato",
            70: "Pumpkin",
            73: "Green Chilli",
            133: "Raddish"
        }
    },
    7: {
        "name": "Spices",
        "commodities": {
            34: "Black pepper",
            26: "Chili Red",
            267: "Cinamon(Dalchini)",
            27: "Ginger(Dry)",
            35: "Turmeric"
        }
    }
}


def lookup(data="default"):
    reverse_dict = {}
    if data == "states":
        for state_id, state_info in states.items():
            reverse_dict[state_info["name"].lower()] = state_id
            for district_id, district_name in state_info["districts"].items():
                reverse_dict[district_name.lower()] = district_id
    elif data == "commodities":
        for category_id, category_info in commodities.items():
            reverse_dict[category_info["name"].lower()] = category_id
            for item_id, item_name in category_info["commodities"].items():
                reverse_dict[item_name.lower()] = item_id
    return reverse_dict


def get_id(name, data, lookup_type="default"):
    if lookup_type == "default":
        lookup_dict = lookup(data)
    elif lookup_type == "lookup":
        lookup_dict = data
    else:
        return None
    if lookup_dict:
        return lookup_dict.get(name.lower())
    return None


def get_response(
        req_type=3, msp=0, period="date", page=1, options=2, 
        limit=10, 
        state=[], district=[], market=[], 
        group=[], commodity=[], 
        from_date=date.today(), to_date=date.today()
):
    url = "https://api.agmarknet.gov.in/v1/all-type-report/all-type-report-agm"
    params = {
        "type": req_type,
        "state": state,
        "district": district if district else ["99999"],
        "market": market,
        "group": group,
        "commodity": commodity if commodity else ["99999"],
        "from_date": str(from_date),
        "to_date": str(to_date),
        "msp": msp,
        "period": period,
        "page": page,
        "options": options,
        "itemsPerPage": limit
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://agmarknet.gov.in",
        "Referer": "https://agmarknet.gov.in/"
    }
    return requests.post(url, json=params, headers=headers, timeout=30)


# ==========================================
# 📈 DEMAND & SUPPLY: ANALYTICS ENGINE
# ==========================================

class MarketAnalytics:
    def __init__(self, master_data_path):
        self.df = pd.read_csv(master_data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])

    def get_precision_baseline(self, district, commodity, target_date):
        day_of_year = target_date.timetuple().tm_yday
        subset = self.df[
            (self.df['District'].str.lower() == district.lower()) & 
            (self.df['Commodity'].str.lower() == commodity.lower())
        ].copy()
        subset['day_of_year'] = subset['Date'].dt.dayofyear
        window_mask = (subset['day_of_year'] >= day_of_year - 7) & (subset['day_of_year'] <= day_of_year + 7)
        precision_data = subset[window_mask]

        if precision_data.empty:
            return {"status": "Off-Season"}

        mean_qty = precision_data['Arrival_Quantity'].mean()
        std_qty = precision_data['Arrival_Quantity'].std()
        mean_price = precision_data['Modal_Price'].mean()

        return {
            "baseline_qty": round(mean_qty, 2),
            "std_qty": std_qty if std_qty > 0 else 1,
            "baseline_price": round(mean_price, 2),
            "status": "Active"
        }

    def calculate_precision_gap(self, live_qty, live_price, baseline):
        if baseline['status'] == "Off-Season":
            return {"condition": "Off-Season", "confidence": "N/A"}

        z_score = (live_qty - baseline['baseline_qty']) / baseline['std_qty']
        price_shift = ((live_price - baseline['baseline_price']) / baseline['baseline_price']) * 100

        Z_THRESH = 1.0
        BUFFER = 0.15
        PRICE_SENSITIVITY = 5.0
        SHOCK_THRESHOLD = 15.0

        res = ""
        conf = "High"

        if z_score < -(Z_THRESH + BUFFER):
            if price_shift > PRICE_SENSITIVITY:
                res = "Confirmed Shortage (High Demand Pull)"
                conf = "High"
            else:
                res = "Supply Dip (Weak Market Interest)"
                conf = "Moderate"
        elif -(Z_THRESH + BUFFER) <= z_score <= -(Z_THRESH - BUFFER):
            res = "Trending Towards Shortage"
            conf = "Moderate"
        elif z_score > (Z_THRESH + BUFFER):
            if price_shift < -PRICE_SENSITIVITY:
                res = "Confirmed Surplus (Market Glut)"
                conf = "High"
            else:
                res = "Supply Surge (Stable Prices)"
                conf = "Moderate"
        elif (Z_THRESH - BUFFER) <= z_score <= (Z_THRESH + BUFFER):
            res = "Trending Towards Surplus"
            conf = "Moderate"
        else:
            if price_shift > SHOCK_THRESHOLD:
                res = "Demand Shock (Supply Normal, Price High)"
                conf = "Moderate"
            elif price_shift < -SHOCK_THRESHOLD:
                res = "Demand Slump (Supply Normal, Price Low)"
                conf = "Moderate"
            else:
                res = "Market Equilibrium (Normal Variance)"
                conf = "High"

        if "Shortage" in res and price_shift < 0:
            res += " - Warning: Price Not Rising"
            conf = "Low"
        elif "Surplus" in res and price_shift > 0:
            res += " - Warning: Price Resilient"
            conf = "Low"

        return {
            "z_score": round(z_score, 2),
            "supply_gap_pct": round(((live_qty - baseline['baseline_qty'])/baseline['baseline_qty'])*100, 2),
            "price_shift_pct": round(price_shift, 2),
            "condition": res,
            "confidence": conf
        }

# Initialize the "Market Memory"
MARKET_DATA_PATH = BASE_DIR / "demand-and-supply" / "datasets" / "unified_market_data.csv"
market_engine = MarketAnalytics(MARKET_DATA_PATH)
states_lookup = lookup(data="states")
comm_lookup = lookup(data="commodities")

def run_market_report(district_name, commodity_name, category_name, input_date):
    dist_id = get_id(district_name, states_lookup, lookup_type="lookup")
    comm_id = get_id(commodity_name, comm_lookup, lookup_type="lookup")
    group_id = get_id(category_name, comm_lookup, lookup_type="lookup")
    
    if dist_id is None or comm_id is None or group_id is None:
        return {"error": f"Invalid Selection: Check if '{district_name}', '{commodity_name}' or '{category_name}' is correct."}
    
    maharashtra_id = 20
    live_data = None
    found_date = None
    
    for i in range(0, 6):
        search_date = input_date - timedelta(days=i)
        try:
            response = get_response(
                state=[maharashtra_id], 
                district=[dist_id], 
                group=[group_id], 
                commodity=[comm_id], 
                from_date=search_date, 
                to_date=search_date
            )
            
            if response.status_code == 200:
                res_json = response.json()
                if res_json.get("success") and res_json.get("rows"):
                    live_data = res_json.get("rows")[0]
                    found_date = search_date
                    break
        except Exception as e:
            continue
            
    if not live_data:
        return {"error": "No recent data found in Agmarknet for this selection."}

    current_qty = float(live_data.get("cumm_arr", 0))
    current_price_tonne = float(live_data.get("model_price_wt", 0)) * 10
    baseline = market_engine.get_precision_baseline(district_name, commodity_name, found_date)
    analysis = market_engine.calculate_precision_gap(current_qty, current_price_tonne, baseline)
    
    return {
        "date_found": str(found_date),
        "live_supply": current_qty,
        "live_price": current_price_tonne,
        "baseline_qty": baseline.get('baseline_qty', 0),
        "baseline_price": baseline.get('baseline_price', 0),
        "analysis": analysis
    }


# ==========================================
# 📈 DEMAND & SUPPLY: RECOMMENDATION ENGINE
# ==========================================

class FarmerAdvisor:
    def __init__(self):
        self.api_key = "gsk_dqHS3Af187AIaKP0hqilWGdyb3FYasxRpxJ8aLMOjlk5jLsyxpY"
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_advice(self, crop, district, analysis_results):
        condition = analysis_results['condition']
        confidence = analysis_results['confidence']
        gap = analysis_results['supply_gap_pct']
        price_shift = analysis_results['price_shift_pct']

        system_msg = (
            "You are a precise agricultural decision-support tool. "
            "Provide exactly 3 direct, actionable sentences. "
            "Each sentence MUST be on a new line. "
            "Do not use bolding (**), italics, or introductory phrases like 'Based on...' or 'I recommend...'. "
            "Do not refer to yourself as an AI or an economist. Speak directly to the farmer."
        )

        user_msg = f"""
        Data for {crop} in {district}:
        - Market Status: {condition}
        - Supply Gap: {gap}% (Negative means less supply than usual and Positive means more supply than usual)
        - Price Shift: {price_shift}% (Negative means price is falling and Positive means price is rising)
        - Data Confidence: {confidence}

        Instructions:
        1. Evaluate if they should sell, hold, or wait.
        2. Identify the risk/opportunity clearly.
        3. Suggest one practical next step.
        Keep it professional, helpful and precise.
        """

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.5
        }
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=10)
            data = response.json()
            if "choices" in data:
                return data['choices'][0]['message']['content'].strip()
            return f"Strategic Status: {condition}. Confidence: {confidence}."
        except Exception:
            return "Market analysis suggests cautious trading. Monitor local price trends daily."


# ==========================================
# 📈 DEMAND & SUPPLY: MAIN LOGIC WRAPPER
# ==========================================

def execute_demand_supply(district: str, commodity: str, category: str, target_date_str: str = None) -> dict:
    if target_date_str:
        try:
            query_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        except ValueError:
            query_date = date.today()
    else:
        query_date = date.today()
        
    report = run_market_report(district, commodity, category, query_date)
    
    if "error" in report:
        return {"status": "error", "error": report["error"]}
        
    if report["analysis"]["condition"] != "Off-Season":
        advisor = FarmerAdvisor()
        advice = advisor.generate_advice(commodity, district, report["analysis"])
    else:
        advice = "The crop is currently out of season; no actionable insights."

    return {
        "status": "success",
        "date_found": report["date_found"],
        "live_supply": report["live_supply"],
        "live_price": report["live_price"],
        "baseline_qty": report["baseline_qty"],
        "baseline_price": report["baseline_price"],
        "analysis": report["analysis"],
        "recommendation": advice
    }



