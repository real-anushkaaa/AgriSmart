from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.core.config import settings
from app.services import gemini_service
from app.services.errors import GeminiConfigurationError, GeminiServiceError


def test_generate_text_passes_model_and_system_instruction(monkeypatch):
    generate_content = Mock(return_value=SimpleNamespace(text="  Helpful answer.  "))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(gemini_service, "get_gemini_client", lambda: client)
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.8-flash")

    result = gemini_service.generate_text("user prompt", "system rules")

    assert result == "Helpful answer."
    kwargs = generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-3.8-flash"
    assert kwargs["contents"] == "user prompt"
    assert kwargs["config"].system_instruction == "system rules"


def test_generate_text_wraps_provider_errors(monkeypatch):
    generate_content = Mock(side_effect=RuntimeError("provider unavailable"))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(gemini_service, "get_gemini_client", lambda: client)

    with pytest.raises(GeminiServiceError, match="generation request failed"):
        gemini_service.generate_text("user prompt", "system rules")


def test_gemini_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    gemini_service.get_gemini_client.cache_clear()

    try:
        with pytest.raises(GeminiConfigurationError, match="API key"):
            gemini_service.get_gemini_client()
    finally:
        gemini_service.get_gemini_client.cache_clear()
