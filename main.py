from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

# -------------------------
# LOAD MODEL SAFELY
# -------------------------
model = None
model_path = os.path.join(os.path.dirname(__file__), "models", "fraud_model.pkl")

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
    else:
        print("❌ Model file not found at:", model_path)
except Exception as e:
    print("❌ Error loading model:", str(e))


# -------------------------
# INPUT SCHEMA
# -------------------------
class InputData(BaseModel):
    features: list


# -------------------------
# ROUTES
# -------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.get("/test")
def test():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        x = np.array(data.features).reshape(1, -1)
        prob = model.predict_proba(x)[0][1]

        return {
            "fraud": int(prob > 0.5),
            "probability": float(prob)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))