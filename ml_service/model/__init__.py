"""Model loading and inference module."""

from .loader import (
    FallbackModelStub,
    ModelArtifact,
    get_model,
    initialize_model,
    load_model_artifact,
)

__all__ = [
    "FallbackModelStub",
    "ModelArtifact",
    "get_model",
    "initialize_model",
    "load_model_artifact",
]
