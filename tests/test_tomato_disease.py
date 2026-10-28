
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def test_tomato_disease_prediction_success():
    with patch(
        "app.api.routes.predict_tomato_disease",
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
