from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.pkl")
model = joblib.load(model_path)

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.get("/test")
def test():
    return {"status": "working"}

@app.post("/predict")
def predict(data: InputData):
    input_data = np.array(data.features).reshape(1, -1)

    prob = model.predict_proba(input_data)[0][1]
    prediction = 1 if prob > 0.5 else 0

    return {
        "fraud": int(prediction),
        "probability": float(prob)
    }