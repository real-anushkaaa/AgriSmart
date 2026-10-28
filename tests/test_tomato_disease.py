
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def test_tomato_disease_prediction_success():
    with patch(
        "app.api.endpoints.predictions.predict_tomato_disease",
        return_value={
            "disease": "Tomato___Bacterial_spot",
            "confidence": 0.9643,
        },
    ):
        response = client.post(
            "/api/v1/diseases/tomato/predict",
            files={
                "file": (
                    "tomato.jpg",
                    b"fake-image-data",
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200
    assert response.json()["disease"] == "Tomato___Bacterial_spot"
    assert response.json()["confidence"] == 0.9643


def test_tomato_disease_rejects_non_image():
    response = client.post(
        "/api/v1/diseases/tomato/predict",
        files={
            "file": (
                "notes.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_tomato_disease_maps_invalid_image_to_400():
    from app.ml.errors import InvalidImageError

    with patch(
        "app.api.endpoints.predictions.predict_tomato_disease",
        side_effect=InvalidImageError("Uploaded file is not a valid image."),
    ):
        response = client.post(
            "/api/v1/diseases/tomato/predict",
            files={"file": ("tomato.jpg", b"not an image", "image/jpeg")},
        )

    assert response.status_code == 400


def test_tomato_disease_enforces_upload_limit(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_image_upload_bytes", 4)
    response = client.post(
        "/api/v1/diseases/tomato/predict",
        files={"file": ("tomato.jpg", b"12345", "image/jpeg")},
    )
    assert response.status_code == 413


def test_tomato_disease_rejects_oversized_request_before_parsing(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_image_upload_bytes", 4)
    response = client.post(
        "/api/v1/diseases/tomato/predict",
        files={"file": ("tomato.jpg", b"x" * 70_000, "image/jpeg")},
    )
    assert response.status_code == 413
