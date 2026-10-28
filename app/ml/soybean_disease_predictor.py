from pathlib import Path

from app.ml.disease_predictor import predict_disease

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "soybean_disease_model.keras"
CLASSES_PATH = ARTIFACTS_DIR / "soybean_disease_classes.json"


def predict_soybean_disease(image_bytes: bytes) -> dict[str, str | float]:
    return predict_disease(image_bytes, MODEL_PATH, CLASSES_PATH)
