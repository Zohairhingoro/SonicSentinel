"""Unit tests for POST /predict/live API endpoint."""

import base64
import time
from fastapi.testclient import TestClient
from main import app
from tests.test_predict import create_synthetic_wav

client = TestClient(app)


def test_predict_live_raw_bytes_success() -> None:
    """Test POST /predict/live with raw binary WAV audio bytes body returns 200."""
    wav_bytes = create_synthetic_wav(duration_seconds=1.0, frequency=440.0)

    start_time = time.time()
    response = client.post(
        "/predict/live",
        content=wav_bytes,
        headers={"Content-Type": "audio/wav"},
    )
    elapsed = time.time() - start_time

    assert response.status_code == 200
    assert elapsed < 3.0, f"Live prediction took {elapsed:.2f}s, exceeding 3.0s latency target."

    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "top3" in data
    assert "quality" in data
    assert "waveform" in data
    assert "spectrogram" in data
    assert "model_version" in data

    assert isinstance(data["confidence"], dict)
    assert len(data["confidence"]) == 10
    assert len(data["top3"]) == 3


def test_predict_live_base64_json_success() -> None:
    """Test POST /predict/live with JSON payload containing base64 audio string returns 200."""
    wav_bytes = create_synthetic_wav(duration_seconds=1.0, frequency=880.0)
    b64_str = base64.b64encode(wav_bytes).decode("utf-8")

    start_time = time.time()
    response = client.post(
        "/predict/live",
        json={"audio_data": b64_str, "filename": "live_mic_chunk.wav"},
    )
    elapsed = time.time() - start_time

    assert response.status_code == 200
    assert elapsed < 3.0

    data = response.json()
    assert "predicted_class" in data
    assert data["quality"] in ["Good", "Acceptable", "Poor", "Unusable"]


def test_predict_live_data_url_json_success() -> None:
    """Test POST /predict/live with data URL base64 string returns 200."""
    wav_bytes = create_synthetic_wav(duration_seconds=1.0, frequency=500.0)
    b64_str = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("utf-8")

    response = client.post(
        "/predict/live",
        json={"audio_data": b64_str},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data


def test_predict_live_empty_payload_returns_400() -> None:
    """Test POST /predict/live with empty request payload returns 400 Bad Request."""
    response = client.post("/predict/live", content=b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
