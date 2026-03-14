from fastapi import (
    FastAPI, 
    Request, 
    UploadFile, 
    HTTPException,
    Depends,
    File
)

from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import pandas as pd
import numpy as np
import shutil
import uuid
import tempfile

# Internal imports

from schema import (
    CropPriceInput, 
    CropPriceOutput, 
    WeatherSoilData,
    CropRequest, 
    CropResponse, 
    DiseaseResponse
)

from authorization import validate_api_key

from logic import (
    reverse_geocode_state,
    fetch_open_meteo,
    compute_features,
    get_season,
    predict_crop,
    recommend_alternatives,
    predict_disease,
    predict_crop_price
)

app = FastAPI(
    title="Farmingo API", 
    version="1.0", 
    description="API endpoints for Farmingo App",
    # docs_url=None,
    # redoc_url=None,
    # openapi_url=None,
    # dependencies=[Depends(validate_api_key)]
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:9002",
        "http://127.0.0.1:8000",
        "https://farmingo-coral.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Use app.state (thread-safe)
app.state.user_location = {"latitude": None, "longitude": None}

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# Endpoint: Health Check
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Farmingo API is running"}

# Endpoint: API Version
@app.get("/version")
def get_version():
    return {"version": app.version}

# Endpoint: List Routes
@app.get("/routes")
def list_routes():
    return [{"path": route.path, "name": route.name} for route in app.router.routes]

# Endpoint: API Info
@app.get("/info")
def api_info():
    return {
        "app": "Farmingo API",
        "version": app.version,
        "description": "Backend API for Farmingo App",
        "docs": "/docs"
    }


# Endpoint: Receive and Store Location
@app.post("/location")
async def receive_location(request: Request):
    data = await request.json()
    lat = data.get("latitude")
    lon = data.get("longitude")

    if lat is None or lon is None:
        raise HTTPException(400, "Latitude & Longitude required")

    app.state.user_location = {"latitude": float(lat), "longitude": float(lon)}

    print(f"Location saved: {lat}, {lon}")
    return {"latitude": lat, "longitude": lon}


# Endpoint: Crop Recommendation
@app.post("/recommend", response_model=CropResponse)
def recommend(data: CropRequest):

    if data.auto_location:
        lat = app.state.user_location["latitude"]
        lon = app.state.user_location["longitude"]
    else:
        lat = data.latitude
        lon = data.longitude

    if lat is None or lon is None:
        raise HTTPException(400, "Location not received yet. Call /location first.")

    # Fetch weather
    weather = fetch_open_meteo(lat, lon)
    daily, hourly = weather["daily"], weather["hourly"]

    # Convert to DataFrame
    df_daily = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).date,
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "precipitation": daily["precipitation_sum"]
    })

    df = df_daily.copy()
    df["rh_mean"] = np.mean(hourly["relativehumidity_2m"])
    df["soil_temp_mean"] = np.mean(hourly["soil_temperature_0cm"])
    df["soil_moist_mean"] = np.mean(hourly["soil_moisture_0_to_1cm"])
    df["temp_mean"] = np.mean(hourly["temperature_2m"])

    # Compute processed features
    features = compute_features(df)
    features["season_code"] = get_season()

    # Reverse geocoding
    state = reverse_geocode_state(lat, lon)

    # Predict crop
    predicted_crop = predict_crop(features, state)
    alternatives = recommend_alternatives(predicted_crop, state)

    return CropResponse(
        status="success",
        coords={"latitude": lat, "longitude": lon},
        weather=WeatherSoilData(**features),
        predicted_crop=predicted_crop,
        predicted_score=1.0,
        fully_suitable=[],
        partially_suitable=alternatives,
        state=state,
        season_code=features["season_code"]
    )


# Endpoint: Crop Disease Prediction
@app.post("/crop_disease_prediction", response_model=DiseaseResponse)
async def predict_crop_disease(crop_name: str, file: UploadFile = File(...)):
    crop_name = crop_name.lower().strip()

    # Use temp directory for safe cleanup
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = os.path.join(tmp, f"crop_{uuid.uuid4()}.jpg")

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            result = predict_disease(crop_name, temp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return DiseaseResponse(**result)


# Endpoint: Crop Price Prediction
@app.post("/crop_price", response_model=CropPriceOutput)
def predict(data: CropPriceInput):
    price = predict_crop_price(data.crop, data.region, data.date)
    return CropPriceOutput(
        crop=data.crop,
        region=data.region,
        date=data.date,
        price=float(price)
    )
