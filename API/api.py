from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime
import pandas as pd
import numpy as np
import requests
import shutil
import uuid
import os

from utility import load_pickle_model

from schema import (
    CropPriceInput, 
    CropPriceOutput, 
    WeatherSoilData,
    CropRequest, 
    CropResponse, 
    DiseaseRequest, 
    DiseaseResponse
)

from logic import (
    reverse_geocode_state,
    fetch_open_meteo,
    compute_features,
    get_season,
    predict_crop,
    recommend_alternatives,
    predict_disease
)

# Path to your trained model
MODEL_PATH = r"D:\Projects\ML Models - Farmingo\Crop Price Prediction\backup\crop_price_model_01.pkl"

app = FastAPI(title="Farmingo")

# ✅ Allow your frontend to call FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8000", "http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the latest received coordinates
user_location = {"latitude": None, "longitude": None}

# Define base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Serve index.html at the root URL
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/location")
async def receive_location(request: Request):
    data = await request.json()
    lat = data.get("latitude")
    lon = data.get("longitude")

    # Store globally
    user_location["latitude"] = lat
    user_location["longitude"] = lon

    print(f"✅ User location updated: {lat}, {lon}")
    return {"latitude": lat, "longitude": lon}



@app.post("/crop_disease_prediction", response_model=DiseaseResponse)
async def predict_crop_disease(crop_name: str, file: UploadFile = File(...)):
    # Validate crop string
    crop_name = crop_name.lower()

    # Save temp image
    temp_name = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_disease(crop_name, temp_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Delete temp image
        if file:
            try:
                os.remove(temp_name)
            except:
                pass

    return DiseaseResponse(**result)


@app.post("/recommend", response_model=CropResponse)
def recommend(data: CropRequest):

    lat = user_location["latitude"]
    lon = user_location["longitude"]

    # weather API
    weather_json = fetch_open_meteo(lat, lon)

    # Convert to DataFrame exactly like your Flask version
    daily = weather_json["daily"]
    hourly = weather_json["hourly"]

    df_daily = pd.DataFrame({
        "date": daily["time"],
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "precipitation": daily["precipitation_sum"]
    })

    df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.date

    df = df_daily.copy()
    df["rh_mean"] = np.mean(hourly["relativehumidity_2m"])
    df["soil_temp_mean"] = np.mean(hourly["soil_temperature_0cm"])
    df["soil_moist_mean"] = np.mean(hourly["soil_moisture_0_to_1cm"])
    df["temp_mean"] = np.mean(hourly["temperature_2m"])

    features = compute_features(df)
    season_code = get_season()
    state = reverse_geocode_state(lat, lon)

    features["season_code"] = season_code

    predicted_crop = predict_crop(features, state)
    alternatives = recommend_alternatives(predicted_crop, state)

    features_obj = WeatherSoilData(**features)

    return CropResponse(
        status="success",
        coords={"latitude": lat, "longitude": lon},
        weather=features_obj,
        predicted_crop=predicted_crop,
        predicted_score=1.0,
        fully_suitable=[],
        partially_suitable=alternatives,
        state=state,
        season_code=season_code
    )


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


@app.post("/crop_price", response_model=CropPriceOutput)
def predict(data: CropPriceInput):
    price = predict_crop_price(data.crop, data.region, data.date)
    return CropPriceOutput(
        crop=data.crop,
        region=data.region,
        date=data.date,
        price=float(price)
    )



# To run the app, use the command: uvicorn api:app --reload
# cd to the directory containing api.py before running the command.
# Example = uvicorn api:app --reload --port 8000 
# File name: api.py
# http://127.0.0.1:8000/predict
# http://127.0.0.1:8000/docs # A testing playground
# http://127.0.0.1:8000/redoc # Documentation page

