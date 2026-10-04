"""Unit tests for POST /predict API contract schema and 4xx error handling."""

import io
from pathlib import Path
from typing import Any, Dict, List
import pytest
from fastapi.testclient import TestClient

MANDATORY_CATEGORIES: List[str] = [
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

VALID_QUALITY_VALUES = {"Good", "Acceptable", "Poor", "Unusable"}


def test_predict_known_sample_clip_matches_agents_md_schema(
    client: TestClient,
    sample_audio_dir: Path,
) -> None:
    audio_path = sample_audio_dir / "valid.wav"
    assert audio_path.exists()

    with open(audio_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("valid.wav", fh, "audio/wav")})

    assert response.status_code == 200
    data: Dict[str, Any] = response.json()

    required_keys = {
        "predicted_class",
        "confidence",
        "top3",
        "quality",
        "waveform",
        "spectrogram",
        "model_version",
    }
    assert required_keys.issubset(data.keys())

    assert isinstance(data["predicted_class"], str)
    assert data["predicted_class"] in MANDATORY_CATEGORIES

    assert isinstance(data["confidence"], dict)
    assert len(data["confidence"]) == 10
    for category in MANDATORY_CATEGORIES:
        assert category in data["confidence"]
        score = data["confidence"][category]
        assert isinstance(score, (float, int))
        assert 0.0 <= score <= 1.0

    assert isinstance(data["top3"], list)
    assert len(data["top3"]) == 3
    for pred in data["top3"]:
        assert isinstance(pred, dict)
        assert "category" in pred
        assert "confidence" in pred
        assert pred["category"] in MANDATORY_CATEGORIES
        assert 0.0 <= pred["confidence"] <= 1.0

    assert data["top3"][0]["confidence"] >= data["top3"][1]["confidence"]
    assert data["top3"][1]["confidence"] >= data["top3"][2]["confidence"]
    assert data["predicted_class"] == data["top3"][0]["category"]

    assert isinstance(data["quality"], str)
    assert data["quality"] in VALID_QUALITY_VALUES

    assert isinstance(data["waveform"], list)
    assert len(data["waveform"]) > 0
    for val in data["waveform"][:50]:
        assert isinstance(val, (float, int))
        assert -1.0 <= val <= 1.0

    assert isinstance(data["spectrogram"], list)
    assert len(data["spectrogram"]) > 0
    assert isinstance(data["spectrogram"][0], list)
    assert len(data["spectrogram"][0]) > 0
    for val in data["spectrogram"][0][:50]:
        assert isinstance(val, (float, int))
        assert 0.0 <= val <= 1.0

    assert isinstance(data["model_version"], str)
    assert len(data["model_version"]) > 0


@pytest.mark.parametrize(
    ("filename", "payload", "content_type", "expected_code_or_substr"),
    [
        ("unsupported.txt", b"Plain text file content", "text/plain", "unsupported"),
        ("document.pdf", b"%PDF-1.4 binary data", "application/pdf", "unsupported"),
        ("empty.wav", b"", "audio/wav", "empty"),
        ("corrupt.wav", b"RIFF\x00\x00\x00\x00WAVEbad_bytes", "audio/wav", "AUDIO_DECODE_FAILURE"),
        ("not_audio.wav", b"ABCDEF1234567890INVALID", "audio/wav", "AUDIO_DECODE_FAILURE"),
    ],
)
def test_invalid_files_return_4xx_not_500(
    client: TestClient,
    filename: str,
    payload: bytes,
    content_type: str,
    expected_code_or_substr: str,
) -> None:
    response = client.post(
        "/predict",
        files={"file": (filename, io.BytesIO(payload), content_type)},
    )

    assert 400 <= response.status_code < 500
    assert response.status_code != 500

    resp_body = response.json()
    body_text = str(resp_body).lower()
    assert expected_code_or_substr.lower() in body_text


def test_invalid_silent_sample_audio_returns_4xx_not_500(
    client: TestClient,
    sample_audio_dir: Path,
) -> None:
    silent_path = sample_audio_dir / "silent.wav"
    with open(silent_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("silent.wav", fh, "audio/wav")})

    assert 400 <= response.status_code < 500
    assert response.status_code != 500
    assert "silence" in str(response.json()).lower() or "usable" in str(response.json()).lower()


def test_invalid_too_short_sample_audio_returns_4xx_not_500(
    client: TestClient,
    sample_audio_dir: Path,
) -> None:
    short_path = sample_audio_dir / "too_short.wav"
    with open(short_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("too_short.wav", fh, "audio/wav")})

    assert 400 <= response.status_code < 500
    assert response.status_code != 500
    assert "minimum" in str(response.json()).lower() or "too_short" in str(response.json()).lower()


def test_invalid_corrupt_sample_audio_returns_4xx_not_500(
    client: TestClient,
    sample_audio_dir: Path,
) -> None:
    corrupt_path = sample_audio_dir / "corrupt.wav"
    with open(corrupt_path, "rb") as fh:
        response = client.post("/predict", files={"file": ("corrupt.wav", fh, "audio/wav")})

    assert 400 <= response.status_code < 500
    assert response.status_code != 500
    assert response.json().get("error_code") == "AUDIO_DECODE_FAILURE"


def test_missing_payload_returns_4xx_not_500(client: TestClient) -> None:
    response = client.post("/predict", data={})

    assert 400 <= response.status_code < 500
    assert response.status_code != 500
    assert response.json().get("error_code") == "MALFORMED_REQUEST"
