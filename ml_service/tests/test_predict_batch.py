"""Unit tests for POST /predict/batch API endpoint."""

from fastapi.testclient import TestClient
from main import app
from tests.test_predict import create_synthetic_wav

client = TestClient(app)


def test_predict_batch_success() -> None:
    """Test POST /predict/batch with multiple valid WAV clips returns 200 and list of per-file results."""
    wav1 = create_synthetic_wav(duration_seconds=1.0, frequency=440.0)
    wav2 = create_synthetic_wav(duration_seconds=1.5, frequency=880.0)

    files = [
        ("files", ("audio_1.wav", wav1, "audio/wav")),
        ("files", ("audio_2.wav", wav2, "audio/wav")),
    ]

    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200

    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 2

    for data in results:
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
        assert data["quality"] in ["Good", "Acceptable", "Poor", "Unusable"]
        assert len(data["waveform"]) > 0
        assert len(data["spectrogram"]) > 0


def test_predict_batch_invalid_file_returns_400() -> None:
    """Test POST /predict/batch containing an invalid file returns 400 Bad Request."""
    wav_bytes = create_synthetic_wav(duration_seconds=1.0, frequency=440.0)

    files = [
        ("files", ("valid.wav", wav_bytes, "audio/wav")),
        ("files", ("invalid.txt", b"not audio data", "text/plain")),
    ]

    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
