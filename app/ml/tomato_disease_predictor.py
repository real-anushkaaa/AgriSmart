
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "tomato_disease_model.keras"
CLASSES_PATH = ARTIFACTS_DIR / "tomato_disease_classes.json"

IMAGE_SIZE = (224, 224)

_model = None
_class_names = None


def load_tomato_model():
    global _model, _class_names

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Tomato disease model not found: {MODEL_PATH}"
            )

        if not CLASSES_PATH.exists():
            raise FileNotFoundError(
                f"Tomato class names not found: {CLASSES_PATH}"
            )

        _model = tf.keras.models.load_model(MODEL_PATH)

        with open(CLASSES_PATH, encoding="utf-8") as file:
            _class_names = json.load(file)

    return _model, _class_names


def predict_tomato_disease(image_bytes: bytes) -> dict:
    model, class_names = load_tomato_model()

    try:
        with Image.open(__import__("io").BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)
            image_array = np.asarray(image, dtype=np.float32)
    except Exception as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    probabilities = model.predict(
        np.expand_dims(image_array, axis=0),
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))

    return {
        "disease": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
    }
