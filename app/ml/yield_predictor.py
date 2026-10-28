
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "crop_yield_model.joblib"
)

_model = None


def get_yield_model():
    """Load the trained model once and reuse it."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained yield model not found: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def predict_crop_yield(
    crop: str,
    crop_year: int,
    season: str,
    state: str,
    area: float,
    annual_rainfall: float,
    fertilizer: float,
    pesticide: float,
) -> float:
    """Predict crop yield using the trained regression pipeline."""
    model = get_yield_model()

    input_data = pd.DataFrame(
        [
            {
                "Crop": crop,
                "Crop_Year": crop_year,
                "Season": season,
                "State": state,
                "Area": area,
                "Annual_Rainfall": annual_rainfall,
                "Fertilizer": fertilizer,
                "Pesticide": pesticide,
            }
        ]
    )

    prediction = model.predict(input_data)[0]
    return max(0.0, float(prediction))
