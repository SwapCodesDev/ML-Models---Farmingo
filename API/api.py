from fastapi import FastAPI
from datetime import datetime
import pandas as pd

from utility import load_pickle_model
from schema import CropPriceInput, CropPriceOutput

# Path to your trained model
MODEL_PATH = r"D:\Projects\ML Models - Farmingo\Crop Price Prediction\backup\crop_price_model_01.pkl"

app = FastAPI(title="Farmingo")


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
