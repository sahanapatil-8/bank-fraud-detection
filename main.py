from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

# 👇 FORCE enable docs
app = FastAPI(
    title="Fraud Detection API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# -------------------------
# SAFE MODEL LOAD
# -------------------------
model = None
model_path = os.path.join(os.path.dirname(__file__), "models", "fraud_model.pkl")

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded")
    except Exception as e:
        print("❌ Model load error:", e)
else:
    print("❌ Model file not found")

# -------------------------
# INPUT
# -------------------------
class InputData(BaseModel):
    features: list

# -------------------------
# ROUTES
# -------------------------
@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/test")
def test():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    x = np.array(data.features).reshape(1, -1)
    prob = model.predict_proba(x)[0][1]

    return {
        "fraud": int(prob > 0.5),
        "probability": float(prob)
    }