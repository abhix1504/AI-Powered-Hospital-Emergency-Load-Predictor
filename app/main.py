"""
main.py
-------
FastAPI application for the Mediload Emergency Arrivals predictor.

Flow:
    1. GET  "/"        -> serves index.html (the frontend)
    2. POST "/predict" -> receives raw JSON, validates it with Pydantic,
                           builds a single-row pandas DataFrame with the
                           EXACT column names/order the pipeline was trained
                           on, and calls pipeline.predict(df).

The sklearn Pipeline (ColumnTransformer -> OneHotEncoder/StandardScaler ->
model) is the ONLY place preprocessing happens. This file never encodes or
scales anything by hand.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import PredictionInput, PredictionOutput

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Mediload Emergency Arrivals Predictor",
    description="Predicts next-hour emergency department arrivals from a trained scikit-learn Pipeline.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# --------------------------------------------------------------------------
# Load the trained pipeline ONCE at startup (not on every request)
# --------------------------------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "mediload_best_pipeline.pkl"
pipeline = None
model_load_error = None

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    # We don't crash the whole app on import -- we surface a clear error
    # on the /predict endpoint instead, which is much easier to debug.
    model_load_error = str(exc)

# Column order MUST match exactly what the pipeline's ColumnTransformer
# was fit with. This list is the single source of truth for turning the
# validated Pydantic object into a DataFrame row.
FEATURE_COLUMNS = [
    "hour",
    "month",
    "is_weekend",
    "is_holiday",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_24",
    "day_name",
]


def classify_load_level(predicted_arrivals: float) -> str:
    """
    Convenience label for the UI only -- NOT used by the model.
    Thresholds below were reverse-engineered from the training data's
    'load_level' column (LOW 0-8, MODERATE 9-18, HIGH 19-30, CRITICAL 31+).
    Adjust these to match your hospital's real triage definitions.
    """
    if predicted_arrivals <= 8:
        return "LOW"
    if predicted_arrivals <= 18:
        return "MODERATE"
    if predicted_arrivals <= 30:
        return "HIGH"
    return "CRITICAL"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the landing page + prediction form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=PredictionOutput)
async def predict(payload: PredictionInput):
    """
    Receives raw feature values, validated by Pydantic (PredictionInput).
    Builds a DataFrame with the same column names the pipeline expects,
    then lets the pipeline handle ALL preprocessing (scaling + encoding).
    """
    if pipeline is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model not loaded",
                "detail": model_load_error or "Unknown error loading model file.",
            },
        )

    try:
        # 1) Pydantic model -> plain dict -> single-row DataFrame.
        #    payload.day_name is a DayName enum; .value gives the plain string
        #    ("Friday") the OneHotEncoder was originally fit on.
        row = {
            "hour": payload.hour,
            "month": payload.month,
            "is_weekend": payload.is_weekend,
            "is_holiday": payload.is_holiday,
            "lag_1": payload.lag_1,
            "lag_2": payload.lag_2,
            "lag_3": payload.lag_3,
            "lag_24": payload.lag_24,
            "rolling_mean_3": payload.rolling_mean_3,
            "rolling_mean_6": payload.rolling_mean_6,
            "rolling_mean_24": payload.rolling_mean_24,
            "day_name": payload.day_name.value,
        }
        input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        # 2) Pipeline handles OneHotEncoder + StandardScaler + model internally.
        raw_prediction = pipeline.predict(input_df)[0]
        prediction = round(float(raw_prediction), 2)

        return PredictionOutput(
            prediction=prediction,
            load_level=classify_load_level(prediction),
            message="Prediction generated successfully",
        )

    except Exception as exc:  # noqa: BLE001
        # Never leak raw Python tracebacks to the client.
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction failed",
                "detail": "The model could not process the provided input. Please check your values and try again.",
            },
        )


@app.get("/health")
async def health_check():
    """Simple readiness probe -- also useful for confirming the model loaded."""
    return {
        "status": "ok" if pipeline is not None else "degraded",
        "model_loaded": pipeline is not None,
    }
