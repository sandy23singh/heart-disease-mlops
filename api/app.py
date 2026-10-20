from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.predict import HeartDiseasePredictor


app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment - Heart Disease Risk Prediction",
    version="1.0.0"
)

try:
    predictor = HeartDiseasePredictor()
except Exception as e:
    predictor = None
    print(f"Model loading error: {e}")


class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


@app.get("/")
def home():
    return {
        "status": "running",
        "service": "Heart Disease Prediction API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: HeartData):

    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )

    result = predictor.predict(
        data.model_dump()
    )

    risk = (
        "Heart Disease Risk"
        if result["prediction"] == 1
        else "No Heart Disease Risk"
    )

    return {
        "prediction": result["prediction"],
        "risk": risk,
        "confidence": round(
            result["confidence"],
            4
        )
    }