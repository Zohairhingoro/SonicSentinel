"""Pytest fixtures and configuration for ml_service tests."""

import sys
from pathlib import Path
from typing import Any, Generator, List
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_SERVICE_DIR = REPO_ROOT / "ml_service"

if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import app
from model.loader import ModelArtifact

CATEGORIES: List[str] = [
    "Machinery Fault",
    "Glass Breaking",
    "Alarm or Siren",
    "Vehicle Horn",
    "Animal Sound",
    "Gunshot",
    "Panic Scream",
    "Aggression or Violent Conflict",
    "Person Asking for Help",
    "Background Noise",
]


class MockEstimator:
    """Mock ML estimator providing predict and predict_proba interfaces."""

    def __init__(self, classes: List[str]) -> None:
        self.classes_ = classes

    def predict(self, x: Any) -> List[str]:
        return [self.classes_[0]]

    def predict_proba(self, x: Any) -> List[List[float]]:
        n = len(self.classes_)
        return [[1.0 / n] * n]


@pytest.fixture
def sample_audio_dir() -> Path:
    """Return path to sample_audio directory."""
    return REPO_ROOT / "sample_audio"


@pytest.fixture
def mock_model_artifact(monkeypatch: pytest.MonkeyPatch) -> ModelArtifact:
    """Provide a deterministic in-memory mock ModelArtifact."""
    estimator = MockEstimator(classes=CATEGORIES)
    artifact = ModelArtifact(
        model=estimator,
        version="test-1.0.0",
        trained_on="2026-09-27T00:00:00Z",
        metrics={"accuracy": 0.95, "f1_score": 0.92},
        metadata={"classes": CATEGORIES},
    )

    import api.routes_predict
    import model.loader

    monkeypatch.setattr(model.loader, "get_model", lambda *args, **kwargs: artifact)
    monkeypatch.setattr(api.routes_predict, "get_model", lambda *args, **kwargs: artifact)
    return artifact


@pytest.fixture
def client(mock_model_artifact: ModelArtifact) -> Generator[TestClient, None, None]:
    """TestClient fixture with mocked model artifact."""
    with TestClient(app) as test_client:
        yield test_client
