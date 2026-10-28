
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.errors import GeminiServiceError

client = TestClient(app)


def test_crop_calendar_generation_success():
    with patch(
        "app.api.endpoints.gemini.generate_crop_calendar",
        return_value="Land preparation: May. Sowing: June.",
    ):
        response = client.post(
            "/api/v1/crop-calendar/generate",
            json={
                "crop": "Soybean",
                "location": "Indore, Madhya Pradesh",
                "season": "Kharif",
                "start_month": 6,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Soybean"
    assert data["location"] == "Indore, Madhya Pradesh"
    assert data["season"] == "Kharif"
    assert "Sowing: June" in data["calendar"]


def test_crop_calendar_rejects_invalid_month():
    response = client.post(
        "/api/v1/crop-calendar/generate",
        json={
            "crop": "Soybean",
            "location": "Indore, Madhya Pradesh",
            "season": "Kharif",
            "start_month": 13,
        },
    )

    assert response.status_code == 422


def test_crop_calendar_handles_gemini_failure():
    with patch(
        "app.api.endpoints.gemini.generate_crop_calendar",
        side_effect=GeminiServiceError("Gemini unavailable"),
    ):
        response = client.post(
            "/api/v1/crop-calendar/generate",
            json={
                "crop": "Soybean",
                "location": "Indore, Madhya Pradesh",
                "season": "Kharif",
                "start_month": 6,
            },
        )

    assert response.status_code == 502
