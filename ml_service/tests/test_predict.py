"""Unit tests for POST /predict API endpoint."""

import io
import math
import struct
import wave
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def create_synthetic_wav(
    duration_seconds: float = 1.0,
    sample_rate: int = 22050,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """Generate in-memory WAV audio file bytes for testing."""
    num_samples = int(duration_seconds * sample_rate)
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        samples = []
        for i in range(num_samples):
            t = float(i) / sample_rate
            value = int(amplitude * 32767.0 * math.sin(2.0 * math.pi * frequency * t))
            samples.append(struct.pack("<h", value))

        wav_file.writeframes(b"".join(samples))

    buffer.seek(0)
    return buffer.read()


def test_predict_success() -> None:
    """Test POST /predict with a valid WAV audio clip returns 200 and schema compliant JSON."""
    wav_bytes = create_synthetic_wav(duration_seconds=1.5, frequency=440.0)
    files = {"file": ("test_signal.wav", wav_bytes, "audio/wav")}

    response = client.post("/predict", files=files)
    assert response.status_code == 200

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
    assert data["quality"] in ["Good", "Acceptable", "Poor", "Unusable"]
    assert len(data["waveform"]) > 0
    assert len(data["spectrogram"]) > 0


def test_predict_unsupported_format() -> None:
    """Test POST /predict with unsupported file format returns 400 Bad Request."""
    files = {"file": ("document.txt", b"Hello world", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_predict_empty_file() -> None:
    """Test POST /predict with 0-byte empty file returns 400 Bad Request."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_silent_audio() -> None:
    """Test POST /predict with completely silent audio clip returns 400 Bad Request."""
    wav_bytes = create_synthetic_wav(duration_seconds=1.0, amplitude=0.0)
    files = {"file": ("silent.wav", wav_bytes, "audio/wav")}

    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "silence" in response.json()["detail"].lower() or "usable" in response.json()["detail"].lower()
