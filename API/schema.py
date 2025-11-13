from pydantic import BaseModel
from typing import List, Dict

# Crop Price Prediction Models

class CropPriceInput(BaseModel):
    crop: str
    region: str
    date: str  # ISO format date string

class CropPriceOutput(BaseModel):
    crop: str
    region: str
    date: str   # ISO format date string
    price: float

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