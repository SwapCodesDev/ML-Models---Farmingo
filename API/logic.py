import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logs

import warnings
warnings.filterwarnings("ignore")  # Disable all warnings
# OR selective:
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import requests
import pandas as pd
import numpy as np
from datetime import datetime
from utility import load_pickle_model
import pickle
import os
from requests.exceptions import RequestException

import tensorflow as tf

import json

# ------------ 1. Reverse Geocoding ------------
def reverse_geocode_state(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": lat,
        "lon": lon,
        "zoom": 10,
        "addressdetails": 1
    }
    headers = {"User-Agent": "CropRecommender/1.0 (contact@farmingo.ai)"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("⚠️ Reverse geocoding failed:", e)
        return "Unknown"

    address = data.get("address", {})

    # ✅ Try multiple fields for flexibility
    for key in ("state", "region", "state_district", "province", "county"):
        if key in address:
            return address[key].title()

    # ✅ Handle code-style responses like "IN-MH"
    for k, v in address.items():
        if isinstance(v, str) and v.startswith("IN-"):
            code = v.split("-")[-1]
            state_map = {
                "MH": "Maharashtra", "KA": "Karnataka", "TN": "Tamil Nadu",
                "UP": "Uttar Pradesh", "MP": "Madhya Pradesh", "GJ": "Gujarat",
                "RJ": "Rajasthan", "BR": "Bihar", "WB": "West Bengal", "KL": "Kerala",
                "TG": "Telangana", "AP": "Andhra Pradesh", "OR": "Odisha",
                "PB": "Punjab", "HR": "Haryana", "AS": "Assam", "CT": "Chhattisgarh"
            }
            return state_map.get(code, "Unknown")

    return address.get("country", "Unknown").title()



# ------------ 2. Weather API ------------
def fetch_open_meteo(lat, lon):
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "hourly": "relativehumidity_2m,soil_temperature_0cm,soil_moisture_0_to_1cm,temperature_2m",
        "forecast_days": 7,
        "timezone": "auto"
    }

    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# ------------ 3. Feature Computation ------------
def compute_features(df):
    for c in ["temp_max", "temp_min", "precipitation", "rh_mean", "soil_temp_mean", "soil_moist_mean", "temp_mean"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # temp
    if df["temp_mean"].notnull().any():
        temp_avg = df["temp_mean"].mean()
    else:
        temp_avg = ((df["temp_max"] + df["temp_min"]) / 2).mean()

    humidity_avg = df["rh_mean"].mean()
    rainfall_avg = df["precipitation"].sum() / 7
    soil_moist = df["soil_moist_mean"].mean()
    soil_temp = df["soil_temp_mean"].mean()

    if soil_moist is None or np.isnan(soil_moist) or soil_moist == 0:
        soil_moist = 0.15

    # NPK + pH formulas remain unchanged from your Flask code
    N = round(200 * soil_moist * (1 - abs(soil_moist - 0.25) * 2), 2)
    P = round(40 * np.exp(-((temp_avg - 30)**2) / 100), 2)
    K = round(250 * soil_moist * (1 - abs(soil_moist - 0.25) * 2), 2)
    ph = round(6.8 - 0.05 * (rainfall_avg / 5.0) - 0.02 * (humidity_avg / 100.0), 2)
    ph = np.clip(ph, 5.0, 8.0)

    return {
        "N": N,
        "P": P,
        "K": K,
        "temperature": float(round(temp_avg, 2)),
        "humidity": float(round(humidity_avg, 2)),
        "ph": float(ph),
        "rainfall": float(round(rainfall_avg, 2))
    }


# ------------ 4. Season Code ------------
def get_season():
    month = datetime.now().strftime("%b")  # <-- returns Jan, Feb, Mar...
    if month in ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]:
        return 1
    if month in ["Jun", "Jul", "Aug", "Sep"]:
        return 2
    return 3



# ------------ 5. Load Model & Predict ------------
# BASE = os.path.dirname(os.path.abspath(__file__))
model_path = r"D:\Projects\ML Models - Farmingo\Weather Prediction\backup\xgboost_model.pkl"
state_encoder_path = r"D:\Projects\ML Models - Farmingo\Weather Prediction\backup\state_encoder.pkl"
crop_encoder_path = r"D:\Projects\ML Models - Farmingo\Weather Prediction\backup\crop_encoder.pkl"

model = pickle.load(open(model_path, "rb"))
state_le = pickle.load(open(state_encoder_path, "rb"))
crop_le = pickle.load(open(crop_encoder_path, "rb"))


def predict_crop(features, state):
    state_name = state.replace(" State", "").replace(" District", "").strip()

    if state_name in state_le.classes_:
        enc_state = state_le.transform([state_name])[0]
    else:
        enc_state = 0

    X = np.array([[
        features["N"], features["P"], features["K"],
        features["temperature"], features["humidity"],
        features["ph"], features["rainfall"],
        enc_state, features["season_code"]
    ]])

    pred = model.predict(X)
    return crop_le.inverse_transform(pred)[0].capitalize()

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

# --- State Average Environmental Conditions (Daily) ---
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


# ------------ 6. Alternative Crops ------------
def recommend_alternatives(predicted, state):
    # Normalize state names
    matched_state = None
    for known_state in STATE_CONDITIONS:
        if state.lower() in known_state.lower() or known_state.lower() in state.lower():
            matched_state = known_state
            break

    if not matched_state:
        return []

    env = STATE_CONDITIONS[matched_state]
    ranked = []

    for crop, req in CROP_REQUIREMENTS.items():
        if crop == predicted.lower():
            continue

        # Calculate "distance" between state climate and crop requirement
        score = (
            abs(env["temperature"] - np.mean(req["temperature"])) +
            abs(env["humidity"] - np.mean(req["humidity"])) / 2 +
            abs(env["ph"] - np.mean(req["ph"])) * 5 +
            abs(env["rainfall"] - np.mean(req["rainfall"]) * 7) / 2
        )
        ranked.append((crop, score))

    ranked.sort(key=lambda x: x[1])
    return [c.capitalize() for c, _ in ranked[:5]]



# ------------ 7. Disease Prediction ------------



# 📁 Folder Paths (adjust these as per your setup)
INFO_JSON_FOLDER = r"D:\Projects\ML Models - Farmingo\Crop Disease Prediction\backup\info_json"
MODEL_FOLDER = r"D:\Projects\ML Models - Farmingo\Crop Disease Prediction\backup\trained_models"


# ---------------------------
# 🔹 Load JSON Info
# ---------------------------
def load_disease_info(crop_name: str):
    """
    Load disease info JSON for the given crop.
    """
    json_path = os.path.join(INFO_JSON_FOLDER, f"{crop_name}_disease_info.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Disease info not found for crop: {crop_name}")

    with open(json_path, "r") as f:
        return json.load(f)


# ---------------------------
# 🔹 Load Trained Model
# ---------------------------
def load_crop_model(crop_name: str):
    """
    Load trained disease classifier model for the given crop.
    """
    model_path = os.path.join(MODEL_FOLDER, f"{crop_name}_leaf_disease_classifier.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for crop: {crop_name}")

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model for {crop_name}: {str(e)}")


# ---------------------------
# 🔹 Predict Disease
# ---------------------------
def predict_disease(crop_name: str, img_path: str):
    """
    Predict crop disease and return structured JSON result.
    """

    # Load required assets
    disease_info = load_disease_info(crop_name)
    model = load_crop_model(crop_name)

    # Preprocess image
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(preds[0])
    confidence = float(np.max(preds[0]) * 100)

    # Get class names
    class_names = list(disease_info.keys())
    if predicted_index >= len(class_names):
        raise ValueError("Model output size doesn't match JSON class list.")

    predicted_class = class_names[predicted_index]
    info = disease_info[predicted_class]

    # Format structured API response
    result = {
        "crop": crop_name,
        "predicted_disease": predicted_class,
        "confidence": round(confidence, 2),
        "cause": info.get("Cause", "N/A"),
        "symptoms": info.get("Symptoms", "N/A"),
        "precautions": info.get("Precautions", []),
        "cure": {
            "chemical": info.get("Cure", {}).get("Chemical", []),
            "organic": info.get("Cure", {}).get("Organic", []),
        },
    }

    return result
# ------------ 8. Crop Price Prediction ------------

MODEL_PATH = r"D:\Projects\ML Models - Farmingo\Crop Price Prediction\backup\crop_price_model_01.pkl"

def predict_crop_price(crop: str, region: str, date: str = None):
    """Predict crop price for given crop, region, and date."""
    model = load_pickle_model(MODEL_PATH)
    if model is None:
        return -1.0  # Error indicator

    if date is None:
        date_obj = datetime.today()
    else:
        date_obj = pd.to_datetime(date, dayfirst=True)

    # Extract date features
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    day_of_week = date_obj.dayofweek
    week_of_year = date_obj.isocalendar()[1]

    # Fallback market: use first available for region
    market = "null"

    # Prepare input for model
    input_df = pd.DataFrame([{
        "District": region,
        "Market": market,
        "Commodity": crop,
        "Year": year,
        "Month": month,
        "Day": day,
        "DayOfWeek": day_of_week,
        "WeekOfYear": week_of_year
    }])

    pred_price = model.predict(input_df)[0]
    return round(pred_price, 2)