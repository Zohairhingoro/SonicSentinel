"""Custom exception classes for ML inference microservice."""

from typing import Optional


class AudioDecodeError(Exception):
    """Raised when audio payload cannot be decoded or parsed as valid audio."""

    def __init__(self, message: str = "Audio decoding failure", detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message


class ModelInferenceError(Exception):
    """Raised when model loading, execution, or prediction fails."""

    def __init__(self, message: str = "Model inference failure", detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message


class MalformedRequestError(Exception):
    """Raised when request body, headers, or payload structure is malformed."""

    def __init__(self, message: str = "Malformed request", detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message
