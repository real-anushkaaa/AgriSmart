
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.errors import GeminiConfigurationError, GeminiServiceError


client = TestClient(app)


def test_chatbot_answer_success():
    with patch(
        "app.api.endpoints.gemini.get_agriculture_answer",
        return_value="Water tomato plants according to soil moisture and weather.",
    ):
        response = client.post(
            "/api/v1/chatbot/ask",
            json={"question": "How often should I water tomato plants?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "How often should I water tomato plants?"
    assert "soil moisture" in data["answer"]


def test_chatbot_rejects_short_question():
    response = client.post(
        "/api/v1/chatbot/ask",
        json={"question": "Hi"},
    )

    assert response.status_code == 422


def test_chatbot_handles_gemini_failure():
    with patch(
        "app.api.endpoints.gemini.get_agriculture_answer",
        side_effect=GeminiServiceError("Gemini unavailable"),
    ):
        response = client.post(
            "/api/v1/chatbot/ask",
            json={"question": "How can I protect my crops from pests?"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Agriculture chatbot service is temporarily unavailable."
    )


def test_chatbot_reports_missing_gemini_configuration_as_503():
    with patch(
        "app.api.endpoints.gemini.get_agriculture_answer",
        side_effect=GeminiConfigurationError("API key is missing"),
    ):
        response = client.post(
            "/api/v1/chatbot/ask",
            json={"question": "How can I protect my crops from pests?"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Gemini API is not configured."
