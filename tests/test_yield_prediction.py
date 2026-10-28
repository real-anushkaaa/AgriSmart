
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_yield_prediction_success():
    with patch(
        "app.api.endpoints.predictions.predict_crop_yield",
        return_value=2.6655,
    ):
        response = client.post(
            "/api/v1/yield/predict",
            json={
                "crop": "Rice",
                "crop_year": 2019,
                "season": "Kharif",
                "state": "Assam",
                "area": 5000,
                "annual_rainfall": 2000,
                "fertilizer": 500000,
                "pesticide": 1500,
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["crop"] == "Rice"
    assert data["state"] == "Assam"
    assert data["crop_year"] == 2019
    assert data["predicted_yield"] == 2.6655
    assert data["model_type"] == "RandomForestRegressor"


def test_yield_prediction_rejects_invalid_area():
    response = client.post(
        "/api/v1/yield/predict",
        json={
            "crop": "Rice",
            "crop_year": 2019,
            "season": "Kharif",
            "state": "Assam",
            "area": 0,
            "annual_rainfall": 2000,
            "fertilizer": 500000,
            "pesticide": 1500,
        },
    )

    assert response.status_code == 422


def test_yield_prediction_maps_unavailable_model_to_503():
    from app.ml.errors import ModelUnavailableError

    with patch(
        "app.api.endpoints.predictions.predict_crop_yield",
        side_effect=ModelUnavailableError("corrupt model"),
    ):
        response = client.post(
            "/api/v1/yield/predict",
            json={
                "crop": "Rice",
                "crop_year": 2019,
                "season": "Kharif",
                "state": "Assam",
                "area": 5000,
                "annual_rainfall": 2000,
                "fertilizer": 500000,
                "pesticide": 1500,
            },
        )

    assert response.status_code == 503
