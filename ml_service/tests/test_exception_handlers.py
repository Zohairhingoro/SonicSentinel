"""Unit tests for centralized exception handlers in ml_service/main.py."""

from unittest.mock import patch
from fastapi.testclient import TestClient

from exceptions import AudioDecodeError, MalformedRequestError, ModelInferenceError
from main import app
from tests.test_predict import create_synthetic_wav

client = TestClient(app)


def test_audio_decode_failure_handler(caplog) -> None:
    """Test that AudioDecodeError is caught centrally and returns structured JSON error without leaking stack trace."""
    wav_bytes = create_synthetic_wav(duration_seconds=1.0)
    files = {"file": ("corrupt.wav", wav_bytes, "audio/wav")}

    with patch("api.routes_predict._decode_audio", side_effect=ValueError("Corrupted WAV header bytes")):
        with caplog.at_level("ERROR"):
            response = client.post("/predict", files=files)

    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "AUDIO_DECODE_FAILURE"
    assert data["message"] == "Audio decoding failure"
    assert "detail" in data
    assert "Corrupted WAV header bytes" in data["detail"]
    assert "Traceback" not in response.text
    assert "Audio decoding failure" in caplog.text


def test_model_inference_failure_handler(caplog) -> None:
    """Test that ModelInferenceError is caught centrally and returns 500 status code without leaking stack trace."""
    wav_bytes = create_synthetic_wav(duration_seconds=1.0)
    files = {"file": ("valid.wav", wav_bytes, "audio/wav")}

    with patch("api.routes_predict.get_model", side_effect=RuntimeError("Model weights corrupted")):
        with caplog.at_level("ERROR"):
            response = client.post("/predict", files=files)

    assert response.status_code == 500
    data = response.json()
    assert data["error_code"] == "MODEL_INFERENCE_FAILURE"
    assert data["message"] == "Model inference failure"
    assert "detail" in data
    assert "Model weights corrupted" in data["detail"]
    assert "Traceback" not in response.text
    assert "Model inference failure" in caplog.text


def test_malformed_request_handler(caplog) -> None:
    """Test that MalformedRequestError returns 400 with MALFORMED_REQUEST error code without leaking stack trace."""
    with caplog.at_level("WARNING"):
        response = client.post("/predict/live", content=b"")

    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "MALFORMED_REQUEST"
    assert data["message"] == "Malformed request"
    assert "detail" in data
    assert "Traceback" not in response.text
    assert "Malformed request" in caplog.text


def test_invalid_json_malformed_request(caplog) -> None:
    """Test that invalid JSON payload returns 400 with MALFORMED_REQUEST code."""
    with caplog.at_level("WARNING"):
        response = client.post(
            "/predict/live",
            content=b"{invalid_json_payload:",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "MALFORMED_REQUEST"
    assert "detail" in data
    assert "Traceback" not in response.text
