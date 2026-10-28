class GeminiConfigurationError(RuntimeError):
    """Gemini cannot be used because its configuration is incomplete."""


class GeminiServiceError(RuntimeError):
    """Gemini rejected or failed to complete a generation request."""
