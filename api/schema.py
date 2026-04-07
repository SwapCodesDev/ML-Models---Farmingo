from pydantic import BaseModel
from typing import List, Dict, Optional

# Price Prediction Models

class PricePredictionRequest(BaseModel):
    lat: float
    lon: float
    commodity: str = "chilli"


class PricePredictionResponse(BaseModel):
    status: str
    state: Optional[str] = None
    total_records: Optional[int] = None
    filtered_count: Optional[int] = None
    data: Optional[list] = None
    base_price: Optional[float] = None
    max_price: Optional[float] = None
    base_price_kg: Optional[float] = None
    max_price_kg: Optional[float] = None
    excel_min: Optional[float] = None
    excel_max: Optional[float] = None
    message: Optional[str] = None

# Weather Data Models

class CropRequest(BaseModel):
    auto_location: bool
    latitude: float
    longitude: float


class WeatherSoilData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class CropResponse(BaseModel):
    status: str
    coords: dict
    weather: WeatherSoilData
    predicted_crop: str
    predicted_score: float
    fully_suitable: list
    partially_suitable: list
    state: str
    season_code: int

# Crop disease Models


class DiseaseRequest(BaseModel):
    crop_name: str


class DiseaseResponse(BaseModel):
    predicted_disease: str
    confidence: float
    cause: str
    symptoms: str
    precautions: List[str]
    cure: Dict[str, List[str]]  # { "Chemical": [], "Organic": [] }