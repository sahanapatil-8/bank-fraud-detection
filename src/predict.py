from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import numpy as np
import joblib
import os

app = FastAPI()

@app.get("/")
def root():
    return {"status": "predict.py is loaded"}

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.pkl")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at: {model_path}")

EXPECTED_FEATURES = 30  # time(1) + V1-V28(28) + amount(1)


class InputData(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v):
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_FEATURES} features, got {len(v)}"
            )
        return v


@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array(data.features).reshape(1, -1)
        prob = model.predict_proba(input_array)[0][1]
        prediction = int(prob > 0.5)
        return {"fraud": prediction, "probability": float(prob)}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")