from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# --------------------------------------------------
# App initialization
# --------------------------------------------------
app = FastAPI(title="ML Prediction API", version="1.0.0")

# --------------------------------------------------
# CORS (safe for frontend + backend communication)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict this
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load model & threshold safely
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    threshold = joblib.load(os.path.join(BASE_DIR, "threshold.pkl"))
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {e}")

# --------------------------------------------------
# Request schema (VERY IMPORTANT)
# --------------------------------------------------
class PredictionInput(BaseModel):
    features: list[float]

# --------------------------------------------------
# Health check (DO NOT SKIP)
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "ML API running"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]

        return {
            "prediction": float(prediction),
            "result": "High Risk" if prediction >= threshold else "Low Risk"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
