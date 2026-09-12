"""
schemas.py
----------
Pydantic models that validate the RAW input coming from the browser.

These fields map 1:1 to the columns your sklearn Pipeline was trained on:

    numerical_features = [
        "hour", "month", "is_weekend", "is_holiday",
        "lag_1", "lag_2", "lag_3", "lag_24",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_24",
    ]
    categorical_features = ["day_name"]

IMPORTANT: We do NOT one-hot encode or scale anything here. We only validate
types/ranges. The ColumnTransformer inside the saved pipeline does the
OneHotEncoder + StandardScaler work when we call pipeline.predict(df).
"""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class DayName(str, Enum):
    """Restricts day_name to exactly the categories the OneHotEncoder saw at
    training time. handle_unknown='ignore' in the encoder means an unseen
    category wouldn't crash the pipeline, but it also wouldn't mean anything
    useful to the model -- so we validate it strictly here instead."""
    monday = "Monday"
    tuesday = "Tuesday"
    wednesday = "Wednesday"
    thursday = "Thursday"
    friday = "Friday"
    saturday = "Saturday"
    sunday = "Sunday"


class PredictionInput(BaseModel):
    """
    Raw feature values exactly as a human/browser would enter them.
    No encoding, no scaling -- that is the pipeline's job.
    """

    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    month: int = Field(..., ge=1, le=12, description="Month number (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if Saturday/Sunday else 0")
    is_holiday: int = Field(..., ge=0, le=1, description="1 if a public holiday else 0")

    lag_1: float = Field(..., ge=0, description="Arrivals 1 hour ago")
    lag_2: float = Field(..., ge=0, description="Arrivals 2 hours ago")
    lag_3: float = Field(..., ge=0, description="Arrivals 3 hours ago")
    lag_24: float = Field(..., ge=0, description="Arrivals 24 hours ago (same hour, previous day)")

    rolling_mean_3: float = Field(..., ge=0, description="Average arrivals over the last 3 hours")
    rolling_mean_6: float = Field(..., ge=0, description="Average arrivals over the last 6 hours")
    rolling_mean_24: float = Field(..., ge=0, description="Average arrivals over the last 24 hours")

    day_name: DayName = Field(..., description="Day of the week")

    @field_validator("is_weekend", "is_holiday")
    @classmethod
    def must_be_binary_flag(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("must be 0 or 1")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "hour": 18,
                "month": 7,
                "is_weekend": 0,
                "is_holiday": 0,
                "lag_1": 15,
                "lag_2": 13,
                "lag_3": 12,
                "lag_24": 14,
                "rolling_mean_3": 13.3,
                "rolling_mean_6": 12.8,
                "rolling_mean_24": 12.1,
                "day_name": "Friday",
            }
        }


class PredictionOutput(BaseModel):
    """Clean, predictable response shape returned to the frontend."""

    prediction: float
    load_level: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
