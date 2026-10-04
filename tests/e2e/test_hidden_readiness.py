"""End-to-end test suite covering hidden-test-readiness scenarios from AGENTS.md."""

import io
import wave
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

from audio_events.models import AudioEvent
from audio_events.services.validation import validate_audio_file
from core.storage import are_fingerprints_near_duplicate, compute_audio_fingerprint


def _create_synthetic_wav(
    duration: float = 1.0,
    sample_rate: int = 22050,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """Generate in-memory WAV audio file bytes."""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    signal = (amplitude * np.sin(2.0 * np.pi * frequency * t) * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal.tobytes())

    return buffer.getvalue()


def test_silent_audio_rejection_and_quality(client: Client, sample_audio_dir: Path) -> None:
    silent_path = sample_audio_dir / "silent.wav"
    assert silent_path.exists()

    content = silent_path.read_bytes()
    uploaded = SimpleUploadedFile("silent.wav", content, content_type="audio/wav")

    val_res = validate_audio_file(uploaded)
    assert not val_res["accepted"]
    assert "silence" in val_res["reason"].lower() or "unusable" in val_res["reason"].lower()
    assert val_res["metadata"]["rms_energy"] < 1e-4

    uploaded.seek(0)
    response = client.post("/audio/upload/", {"audio_file": uploaded})

    assert response.status_code == 400
    data: Dict[str, Any] = response.json()
    assert data["accepted"] is False
    assert "silence" in data["reason"].lower() or "unusable" in data["reason"].lower()


def test_corrupted_file_rejection(client: Client, sample_audio_dir: Path) -> None:
    corrupt_path = sample_audio_dir / "corrupt.wav"
    assert corrupt_path.exists()

    content = corrupt_path.read_bytes()
    uploaded = SimpleUploadedFile("corrupt.wav", content, content_type="audio/wav")

    val_res = validate_audio_file(uploaded)
    assert not val_res["accepted"]
    assert "integrity check failed" in val_res["reason"].lower()

    uploaded.seek(0)
    response = client.post("/audio/upload/", {"audio_file": uploaded})

    assert response.status_code == 400
    data: Dict[str, Any] = response.json()
    assert data["accepted"] is False
    assert "integrity check failed" in data["reason"].lower()


def test_truncated_header_corrupted_file_rejection(client: Client) -> None:
    truncated_content = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    uploaded = SimpleUploadedFile("corrupt_trunc.wav", truncated_content, content_type="audio/wav")

    response = client.post("/audio/upload/", {"audio_file": uploaded})

    assert response.status_code == 400
    data: Dict[str, Any] = response.json()
    assert data["accepted"] is False
    assert "integrity check failed" in data["reason"].lower()


def test_unsupported_format_rejection(client: Client, sample_audio_dir: Path) -> None:
    txt_path = sample_audio_dir / "unsupported.txt"
    assert txt_path.exists()

    content = txt_path.read_bytes()
    uploaded = SimpleUploadedFile("unsupported.txt", content, content_type="text/plain")

    val_res = validate_audio_file(uploaded)
    assert not val_res["accepted"]
    assert "Unsupported audio format 'txt'" in val_res["reason"]

    uploaded.seek(0)
    response = client.post("/audio/upload/", {"audio_file": uploaded})

    assert response.status_code == 400
    data: Dict[str, Any] = response.json()
    assert data["accepted"] is False
    assert "Unsupported audio format 'txt'" in data["reason"]


@pytest.mark.parametrize("ext", ["xyz", "pdf", "exe", "doc"])
def test_unsupported_extensions_rejection(client: Client, ext: str) -> None:
    payload = b"Arbitrary payload testing unsupported extension rejection."
    uploaded = SimpleUploadedFile(f"file.{ext}", payload, content_type="application/octet-stream")

    response = client.post("/audio/upload/", {"audio_file": uploaded})

    assert response.status_code == 400
    data: Dict[str, Any] = response.json()
    assert data["accepted"] is False
    assert f"Unsupported audio format '{ext}'" in data["reason"]


def test_duplicate_upload_detection_and_flag(client: Client, sample_audio_dir: Path) -> None:
    valid_path = sample_audio_dir / "valid.wav"
    assert valid_path.exists()
    content = valid_path.read_bytes()

    upload_1 = SimpleUploadedFile("valid.wav", content, content_type="audio/wav")
    resp_1 = client.post("/audio/upload/", {"audio_file": upload_1})

    assert resp_1.status_code == 201
    data_1: Dict[str, Any] = resp_1.json()
    assert data_1["accepted"] is True
    original_audio_id = data_1["audio_id"]

    event_count_after_1 = AudioEvent.objects.filter(audio_id=original_audio_id).count()
    assert event_count_after_1 == 1

    upload_2 = SimpleUploadedFile("valid_duplicate.wav", content, content_type="audio/wav")
    resp_2 = client.post("/audio/upload/", {"audio_file": upload_2})

    assert resp_2.status_code == 200
    data_2: Dict[str, Any] = resp_2.json()
    assert data_2["accepted"] is True
    assert data_2.get("duplicate") is True
    assert data_2["audio_id"] == original_audio_id
    assert "Duplicate audio file content detected" in data_2["reason"]

    matching_hash_events = AudioEvent.objects.filter(hash=data_1["metadata"]["hash"])
    assert matching_hash_events.count() == 1


def test_near_duplicate_upload_detection_and_flag(client: Client) -> None:
    audio_bytes_1 = _create_synthetic_wav(duration=1.2, sample_rate=22050, frequency=440.0, amplitude=0.6)
    audio_bytes_2 = _create_synthetic_wav(duration=1.2, sample_rate=22050, frequency=440.0, amplitude=0.57)

    fp1 = compute_audio_fingerprint(audio_bytes_1)
    fp2 = compute_audio_fingerprint(audio_bytes_2)
    assert are_fingerprints_near_duplicate(fp1, fp2)

    upload_1 = SimpleUploadedFile("clip_baseline.wav", audio_bytes_1, content_type="audio/wav")
    resp_1 = client.post("/audio/upload/", {"audio_file": upload_1})

    assert resp_1.status_code == 201
    data_1 = resp_1.json()
    first_audio_id = data_1["audio_id"]
    event_1 = AudioEvent.objects.get(audio_id=first_audio_id)

    upload_2 = SimpleUploadedFile("clip_near_duplicate.wav", audio_bytes_2, content_type="audio/wav")
    resp_2 = client.post("/audio/upload/", {"audio_file": upload_2})

    assert resp_2.status_code == 201
    data_2 = resp_2.json()
    second_audio_id = data_2["audio_id"]
    assert second_audio_id != first_audio_id

    event_2 = AudioEvent.objects.get(audio_id=second_audio_id)
    assert event_2.suspected_duplicate_of_id == first_audio_id
    assert event_2.suspected_duplicate == event_1
    assert event_2.status == AudioEvent.STATUS_MANUAL_REVIEW
    assert data_2["prediction"]["status"] == AudioEvent.STATUS_MANUAL_REVIEW


def test_audio_quality_classification(client: Client) -> None:
    high_amp_bytes = _create_synthetic_wav(duration=1.0, sample_rate=22050, amplitude=0.8)
    up_good = SimpleUploadedFile("good_quality.wav", high_amp_bytes, content_type="audio/wav")
    resp_good = client.post("/audio/upload/", {"audio_file": up_good})

    assert resp_good.status_code == 201
    data_good = resp_good.json()
    assert data_good["metadata"]["quality"] == AudioEvent.QUALITY_GOOD

    low_amp_bytes = _create_synthetic_wav(duration=1.0, sample_rate=22050, amplitude=0.005)
    up_acceptable = SimpleUploadedFile("acceptable_quality.wav", low_amp_bytes, content_type="audio/wav")
    resp_acceptable = client.post("/audio/upload/", {"audio_file": up_acceptable})

    assert resp_acceptable.status_code == 201
    data_acceptable = resp_acceptable.json()
    assert data_acceptable["metadata"]["quality"] == AudioEvent.QUALITY_ACCEPTABLE
