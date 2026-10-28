
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.errors import GeminiServiceError


client = TestClient(app)


def test_price_prediction_success():
    with patch(
        "app.api.endpoints.gemini.predict_crop_price",
        return_value=(
            "Price information\n"
            "AI-generated guidance only; live price not verified."
        ),
    ):
        response = client.post(
            "/api/v1/prices/predict",
            json={
                "crop": "Wheat",
                "state": "Madhya Pradesh",
                "market": "Indore",
                "date": "2026-10-28",
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["crop"] == "Wheat"
    assert data["state"] == "Madhya Pradesh"
    assert data["market"] == "Indore"
    assert data["date"] == "2026-10-28"
    assert "price_guidance" in data
    assert "AI-generated guidance" in data["price_guidance"]


def test_price_prediction_rejects_invalid_date_format():
    response = client.post(
        "/api/v1/prices/predict",
        json={
            "crop": "Wheat",
            "state": "Madhya Pradesh",
            "market": "Indore",
            "date": "28-10-2026",
        },
    )

    assert response.status_code == 422


def test_price_prediction_rejects_impossible_calendar_date():
    response = client.post(
        "/api/v1/prices/predict",
        json={
            "crop": "Wheat",
            "state": "Madhya Pradesh",
            "market": "Indore",
            "date": "2026-13-45",
        },
    )

    assert response.status_code == 422


def test_price_prediction_handles_gemini_failure():
    with patch(
        "app.api.endpoints.gemini.predict_crop_price",
        side_effect=GeminiServiceError("Simulated Gemini failure"),
    ):
        response = client.post(
            "/api/v1/prices/predict",
            json={
                "crop": "Wheat",
                "state": "Madhya Pradesh",
                "market": "Indore",
                "date": "2026-10-28",
            },
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Crop price guidance service is temporarily unavailable."
    )
