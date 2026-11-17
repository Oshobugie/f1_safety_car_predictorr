from contextlib import asynccontextmanager
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier

# In-memory cache for loaded ML artifacts to avoid disk I/O on every request
artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the fitted preprocessor and trained model once at startup and cleans up on shutdown."""
    artifacts["preprocessor"] = joblib.load("artifacts/data_transformation/preprocessor.joblib")

    # Use XGBoost's native loader since the artifact was exported via booster.save_model()
    model = XGBClassifier()
    model.load_model("artifacts/model_trainer/model.joblib")
    artifacts["model"] = model

    yield
    artifacts.clear()


app = FastAPI(title="F1 Safety Car Predictor API", lifespan=lifespan)


class LapTelemetry(BaseModel):
    """Schema validating incoming single-driver lap telemetry payloads."""
    position: int
    lap_time_sec: float


@app.get("/")
def health_check():
    """Lightweight liveness probe confirming the API is up and model artifacts are in memory."""
    return {"status": "healthy", "model_loaded": "model" in artifacts}


@app.post("/predict")
def predict_safety_car(telemetry: LapTelemetry):
    """Scores live lap telemetry and returns both binary classification and Safety Car probability."""
    try:
        # Preserve the exact feature column order used during training: [position, lap_time_sec]
        raw_features = np.array([[telemetry.position, telemetry.lap_time_sec]])

        # Apply the same median imputation and StandardScaler parameters fitted in Stage 2
        scaled_features = artifacts["preprocessor"].transform(raw_features)

        # Extract binary flag (0 or 1) and positive-class (Safety Car) probability
        prediction = int(artifacts["model"].predict(scaled_features)[0])
        probability = float(artifacts["model"].predict_proba(scaled_features)[0][1])

        return {
            "is_safety_car": prediction,
            "safety_car_probability": round(probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))