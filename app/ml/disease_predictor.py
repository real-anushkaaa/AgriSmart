import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.ml.errors import InvalidImageError, ModelUnavailableError


logger = logging.getLogger(__name__)
IMAGE_SIZE = (224, 224)


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ModelUnavailableError("TensorFlow is not installed.") from exc
    return tf


@lru_cache(maxsize=4)
def load_disease_model(model_path: str, classes_path: str):
    model_file = Path(model_path)
    classes_file = Path(classes_path)
    if not model_file.is_file() or not classes_file.is_file():
        raise ModelUnavailableError("Disease model artifacts are missing.")

    try:
        tf = _tensorflow()
        class_names = json.loads(classes_file.read_text(encoding="utf-8"))
        if not isinstance(class_names, list) or not class_names or not all(
            isinstance(name, str) and name for name in class_names
        ):
            raise ValueError("Class metadata must be a non-empty list of names.")

        model = tf.keras.models.load_model(model_file)
        if model.output_shape[-1] != len(class_names):
            raise ValueError("Model output count does not match class metadata.")
        return model, tuple(class_names)
    except ModelUnavailableError:
        raise
    except Exception as exc:
        logger.exception("Could not load disease model artifact %s", model_file.name)
        raise ModelUnavailableError("Disease model artifacts are invalid.") from exc


def predict_disease(
    image_bytes: bytes,
    model_path: Path,
    classes_path: Path,
) -> dict[str, str | float]:
    if not image_bytes:
        raise InvalidImageError("Uploaded image is empty.")

    tf = _tensorflow()
    try:
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            image,
            IMAGE_SIZE,
            method="bilinear",
            antialias=False,
        )
        image = tf.cast(image, tf.float32)
    except Exception as exc:
        raise InvalidImageError("Uploaded file is not a valid image.") from exc

    model, class_names = load_disease_model(str(model_path), str(classes_path))
    try:
        probabilities = np.asarray(
            model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
        )
        if (
            probabilities.ndim != 1
            or len(probabilities) != len(class_names)
            or not np.all(np.isfinite(probabilities))
        ):
            raise ValueError("Model prediction does not match class metadata.")
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
    except Exception as exc:
        logger.exception("Disease model inference failed for %s", model_path.name)
        raise ModelUnavailableError("Disease model inference failed.") from exc

    return {
        "disease": class_names[predicted_index],
        "confidence": confidence,
    }
