from pydantic import BaseModel

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

class WeatherDataInput(BaseModel):
    location: str
    date: str  # ISO format date string

class WeatherDataOutput(BaseModel):
    temperature: float
    humidity: float
    precipitation: float

# Crop Disease Detection Models

class CropDiseaseInput(BaseModel):
    crop: str
    symptoms: str  # Description of symptoms observed

class CropDiseaseOutput(BaseModel):
    disease: str
    confidence: float