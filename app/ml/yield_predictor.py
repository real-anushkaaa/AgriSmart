
import logging
import math
from pathlib import Path

import joblib
import pandas as pd

from app.ml.errors import ModelUnavailableError


logger = logging.getLogger(__name__)

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
            raise ModelUnavailableError("Trained yield model is missing.")
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as exc:
            logger.exception("Could not load crop yield model")
            raise ModelUnavailableError("Trained yield model is invalid.") from exc

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
                "Crop": crop.strip(),
                "Crop_Year": crop_year,
                "Season": season.strip(),
                "State": state.strip(),
                "Area": area,
                "Annual_Rainfall": annual_rainfall,
                "Fertilizer": fertilizer,
                "Pesticide": pesticide,
            }
        ]
    )

    try:
        prediction = float(model.predict(input_data)[0])
        if not math.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction.")
    except Exception as exc:
        logger.exception("Crop yield inference failed")
        raise ModelUnavailableError("Crop yield inference failed.") from exc
    return max(0.0, prediction)
