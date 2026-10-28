class InvalidImageError(ValueError):
    """The uploaded bytes do not contain a supported image."""


class ModelUnavailableError(RuntimeError):
    """A model artifact could not be loaded or used for inference."""
