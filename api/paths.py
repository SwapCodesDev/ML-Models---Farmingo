from pathlib import Path

# API directory path
API_DIR = Path(__file__).resolve().parent

# Base / Project Root directory path
BASE_DIR = API_DIR.parent

# --- Static/Frontend Paths ---
INDEX_HTML_PATH = API_DIR / "index.html"

# --- Crop Price Prediction ---
PRICE_DATA_PATH = BASE_DIR / "crop-price-prediction" / "datasets" / "wholesale_commodity_prices.xlsx"

# --- Crop Disease Prediction ---
CROP_DISEASE_PREDICTION_DIR = BASE_DIR / "crop-disease-prediction" / "backup"
INFO_JSON_FOLDER = CROP_DISEASE_PREDICTION_DIR / "info_json"
MODEL_FOLDER = CROP_DISEASE_PREDICTION_DIR / "trained_models"

# --- Weather Prediction ---
WEATHER_PREDICTION_DIR = BASE_DIR / "weather-prediction" / "backup"
MODEL_PATH = WEATHER_PREDICTION_DIR / "xgboost_model.pkl"
STATE_ENCODER_PATH = WEATHER_PREDICTION_DIR / "state_encoder.pkl"
CROP_ENCODER_PATH = WEATHER_PREDICTION_DIR / "crop_encoder.pkl"

# --- Demand and Supply ---
MARKET_DATA_PATH = BASE_DIR / "demand-and-supply" / "datasets" / "unified_market_data.csv"
