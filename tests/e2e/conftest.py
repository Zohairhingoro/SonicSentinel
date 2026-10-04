"""Pytest configuration and fixtures for end-to-end integration tests."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import patch
import django
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = REPO_ROOT / "webapp"

if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings.dev")
django.setup()

from django.test import Client
from django.test.utils import setup_databases, teardown_databases

MOCK_ML_RESPONSE: Dict[str, Any] = {
    "predicted_class": "Machinery Fault",
    "confidence": {
        "Machinery Fault": 0.88,
        "Glass Breaking": 0.02,
        "Alarm or Siren": 0.02,
        "Vehicle Horn": 0.02,
        "Animal Sound": 0.01,
        "Gunshot": 0.01,
        "Panic Scream": 0.01,
        "Aggression or Violent Conflict": 0.01,
        "Person Asking for Help": 0.01,
        "Background Noise": 0.01,
    },
    "top3": [
        {"category": "Machinery Fault", "confidence": 0.88},
        {"category": "Glass Breaking", "confidence": 0.02},
        {"category": "Alarm or Siren", "confidence": 0.02},
    ],
    "quality": "Good",
    "waveform": [0.0, 0.1, 0.2],
    "spectrogram": [[0.1, 0.2], [0.3, 0.4]],
    "model_version": "test-1.0.0",
}


@pytest.fixture(scope="session", autouse=True)
def setup_django_test_database() -> Generator[None, None, None]:
    """Create and tear down an isolated Django test database for the test session."""
    old_config = setup_databases(verbosity=0, interactive=False)
    yield
    teardown_databases(old_config, verbosity=0)


@pytest.fixture
def client() -> Client:
    """Return configured Django test client."""
    return Client()


@pytest.fixture
def sample_audio_dir() -> Path:
    """Return path to sample_audio directory."""
    return REPO_ROOT / "sample_audio"


@pytest.fixture(autouse=True)
def mock_predict_audio_file() -> Generator[None, None, None]:
    """Mock ML microservice predict call across audio_events views."""
    with patch("audio_events.views.predict_audio_file", return_value=MOCK_ML_RESPONSE):
        yield
