
from functools import lru_cache

from google import genai

from app.core.config import settings


@lru_cache(maxsize=1)
def get_gemini_client():
    """Create and reuse the Gemini API client."""
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini API key is not configured.")

    return genai.Client(api_key=settings.gemini_api_key)


def generate_text(prompt: str) -> str:
    """Send a prompt to Gemini and return the generated text."""
    client = get_gemini_client()

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()
