
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="House Price Prediction API")

# Load the trained model
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "house_price_model.pkl"
)

with open(MODEL_PATH, "rb") as file:
    model = joblib.load(MODEL_PATH)


class HouseFeatures(BaseModel):
    Area: float = Field(gt=0)
    Bedrooms: int = Field(ge=0)
    Bathrooms: int = Field(ge=0)
    Parking: int = Field(ge=0)
    Age: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}


@app.post("/predict")
def predict_price(house: HouseFeatures):
    features = np.array([[
        house.Area,
        house.Bedrooms,
        house.Bathrooms,
        house.Parking,
        house.Age
    ]])

    prediction = model.predict(features)[0]

    return {
        "predicted_price_lakhs": round(float(prediction), 2)
    }
