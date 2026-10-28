from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_soybean_disease_prediction_success():
    with patch(
        "app.api.routes.predict_soybean_disease",
        return_value={
            "disease": "Rust",
            "confidence": 0.7682,
        },
    ):
        response = client.post(
            "/api/v1/diseases/soybean/predict",
            files={
                "file": (
                    "soybean.jpg",
                    b"fake-image-data",
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200
    assert response.json()["disease"] == "Rust"
    assert response.json()["confidence"] == 0.7682


def test_soybean_disease_rejects_non_image():
    response = client.post(
        "/api/v1/diseases/soybean/predict",
        files={
            "file": (
                "notes.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
