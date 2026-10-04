"""Unit tests for ml_service audio validation at FastAPI service level."""

from pathlib import Path
from typing import Any, Dict
import pytest
from fastapi.testclient import TestClient


def test_fixture_files_exist(sample_audio_dir: Path) -> None:
    assert (sample_audio_dir / "valid.wav").exists()
    assert (sample_audio_dir / "too_short.wav").exists()
    assert (sample_audio_dir / "silent.wav").exists()
    assert (sample_audio_dir / "unsupported.txt").exists()
    assert (sample_audio_dir / "corrupt.wav").exists()


def test_valid_file_acceptance(client: TestClient, sample_audio_dir: Path) -> None:
    audio_path = sample_audio_dir / "valid.wav"
    with open(audio_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("valid.wav", fh, "audio/wav")})

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert isinstance(data["confidence"], dict)
    assert len(data["confidence"]) == 10
    assert len(data["top3"]) == 3
    assert data["quality"] in ["Good", "Acceptable", "Poor", "Unusable"]
    assert len(data["waveform"]) > 0
    assert len(data["spectrogram"]) > 0
    assert data["model_version"] == "test-1.0.0"


def test_unsupported_format_txt_rejection(client: TestClient, sample_audio_dir: Path) -> None:
    txt_path = sample_audio_dir / "unsupported.txt"
    with open(txt_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("unsupported.txt", fh, "text/plain")})

    assert response.status_code == 400
    assert "Unsupported file format 'txt'" in response.json()["detail"]


@pytest.mark.parametrize("extension", ["xyz", "pdf", "exe", "doc"])
def test_unsupported_format_extensions_rejection(client: TestClient, extension: str) -> None:
    payload = b"Fake file content for testing extension rejection."
    filename = f"sample.{extension}"
    response = client.post("/predict", files={"file": (filename, payload, "application/octet-stream")})

    assert response.status_code == 400
    assert f"Unsupported file format '{extension}'" in response.json()["detail"]


def test_oversized_file_rejection_with_config_threshold(
    client: TestClient,
    sample_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import audio.validation

    original_load = audio.validation.load_audio_config

    def patched_load(path: Any = None) -> Dict[str, Any]:
        cfg = original_load(path).copy()
        cfg["max_file_size_mb"] = 0.01
        return cfg

    monkeypatch.setattr(audio.validation, "load_audio_config", patched_load)

    audio_path = sample_audio_dir / "valid.wav"
    with open(audio_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("valid.wav", fh, "audio/wav")})

    assert response.status_code == 400
    assert "exceeds limit of 0.01 MB" in response.json()["detail"]


def test_oversized_file_rejection_large_stream(client: TestClient) -> None:
    oversized_bytes = b"\x00" * (21 * 1024 * 1024)
    response = client.post("/predict", files={"file": ("large_audio.wav", oversized_bytes, "audio/wav")})

    assert response.status_code == 400
    assert "exceeds limit of 20 MB" in response.json()["detail"]


def test_too_short_duration_rejection(client: TestClient, sample_audio_dir: Path) -> None:
    audio_path = sample_audio_dir / "too_short.wav"
    with open(audio_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("too_short.wav", fh, "audio/wav")})

    assert response.status_code == 400
    assert "less than minimum required" in response.json()["detail"]


def test_too_long_duration_rejection(
    client: TestClient,
    sample_audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import audio.validation

    original_load = audio.validation.load_audio_config

    def patched_load(path: Any = None) -> Dict[str, Any]:
        cfg = original_load(path).copy()
        cfg["max_duration_seconds"] = 0.5
        return cfg

    monkeypatch.setattr(audio.validation, "load_audio_config", patched_load)

    audio_path = sample_audio_dir / "valid.wav"
    with open(audio_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("valid.wav", fh, "audio/wav")})

    assert response.status_code == 400
    assert "exceeds maximum allowed" in response.json()["detail"]


def test_silent_audio_rejection(client: TestClient, sample_audio_dir: Path) -> None:
    audio_path = sample_audio_dir / "silent.wav"
    with open(audio_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("silent.wav", fh, "audio/wav")})

    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "silence" in detail or "no usable acoustic signal" in detail


def test_empty_file_rejection(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("empty.wav", b"", "audio/wav")})

    assert response.status_code == 400
    body = response.json()
    assert body.get("error_code") == "MALFORMED_REQUEST"
    assert "empty" in body.get("detail", "").lower()


def test_corrupt_wav_file_rejection(client: TestClient, sample_audio_dir: Path) -> None:
    audio_path = sample_audio_dir / "corrupt.wav"
    with open(audio_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("corrupt.wav", fh, "audio/wav")})

    assert response.status_code == 400
    body = response.json()
    assert body.get("error_code") == "AUDIO_DECODE_FAILURE"


def test_corrupt_audio_arbitrary_bytes(client: TestClient) -> None:
    garbage_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00GARBAGE_PAYLOAD"
    response = client.post("/predict", files={"file": ("corrupt_signal.wav", garbage_bytes, "audio/wav")})

    assert response.status_code == 400
    body = response.json()
    assert body.get("error_code") == "AUDIO_DECODE_FAILURE"


def test_missing_file_field_rejection(client: TestClient) -> None:
    response = client.post("/predict", data={"other_param": "value"})

    assert response.status_code == 400
    body = response.json()
    assert body.get("error_code") == "MALFORMED_REQUEST"
