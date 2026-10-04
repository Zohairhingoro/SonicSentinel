"""Pytest fixtures and configuration for ml_pipeline tests."""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_SERVICE_DIR = REPO_ROOT / "ml_service"
ML_PIPELINE_DIR = REPO_ROOT / "ml_pipeline"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))
if str(ML_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_DIR))


@pytest.fixture
def sample_audio_dir() -> Path:
    """Return path to sample_audio directory."""
    return REPO_ROOT / "sample_audio"
