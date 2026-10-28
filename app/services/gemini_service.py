
import logging
from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import settings
from app.services.errors import GeminiConfigurationError, GeminiServiceError


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_gemini_client():
    """Create and reuse the Gemini API client."""
    if not settings.gemini_api_key:
        raise GeminiConfigurationError("Gemini API key is not configured.")

    try:
        return genai.Client(api_key=settings.gemini_api_key)
    except Exception as exc:
        logger.exception("Could not initialize Gemini client")
        raise GeminiConfigurationError("Gemini client initialization failed.") from exc


def generate_text(prompt: str, system_instruction: str) -> str:
    """Send a prompt to Gemini and return the generated text."""
    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )
    except Exception as exc:
        logger.exception("Gemini generation request failed")
        raise GeminiServiceError("Gemini generation request failed.") from exc

    if not response.text:
        raise GeminiServiceError("Gemini returned an empty response.")

    return response.text.strip()
