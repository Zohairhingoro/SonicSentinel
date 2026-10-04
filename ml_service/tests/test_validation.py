import io
import math
import struct
import wave
from pathlib import Path
import pytest
try:
    from ml_service.audio.validation import (
        AudioMetadata,
        AudioValidationResult,
        load_audio_config,
        validate_audio_file,
    )
except ModuleNotFoundError:
    from audio.validation import (
        AudioMetadata,
        AudioValidationResult,
        load_audio_config,
        validate_audio_file,
    )



def create_wav_bytes(
    duration_seconds: float = 1.0,
    sample_rate: int = 22050,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    channels: int = 1,
) -> bytes:
    """Helper to generate in-memory PCM WAV file bytes."""
    num_samples = int(duration_seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        raw_data = bytearray()
        for i in range(num_samples):
            t = float(i) / sample_rate
            sample_val = int(amplitude * 32767.0 * math.sin(2.0 * math.pi * frequency * t))
            sample_bytes = struct.pack("<h", sample_val)
            for _ in range(channels):
                raw_data.extend(sample_bytes)

        wav_file.writeframes(raw_data)
    return buf.getvalue()


def test_load_audio_config():
    config = load_audio_config()
    assert isinstance(config, dict)
    assert "supported_formats" in config
    assert "wav" in config["supported_formats"]
    assert "max_duration_seconds" in config


def test_validate_valid_audio():
    wav_bytes = create_wav_bytes(duration_seconds=2.0, sample_rate=22050)
    result = validate_audio_file(wav_bytes, filename="test_clip.wav")

    assert result.is_valid is True
    assert result.error_code is None
    assert result.error_message is None
    assert result.metadata is not None
    assert result.metadata.format == "wav"
    assert result.metadata.sample_rate == 22050
    assert result.metadata.duration_seconds == 2.0
    assert result.metadata.is_silent is False


def test_validate_unsupported_format():
    result = validate_audio_file(b"fake data", filename="document.txt")
    assert result.is_valid is False
    assert result.error_code == "UNSUPPORTED_FORMAT"
    assert "Unsupported file format" in result.error_message


def test_validate_empty_file():
    result = validate_audio_file(b"", filename="empty.wav")
    assert result.is_valid is False
    assert result.error_code == "FILE_EMPTY"
    assert "empty" in result.error_message.lower()


def test_validate_corrupt_file():
    result = validate_audio_file(b"NOT_A_WAV_FILE_HEADER_CORRUPTED", filename="corrupt.wav")
    assert result.is_valid is False
    assert result.error_code == "FILE_CORRUPT"
    assert "corrupted" in result.error_message.lower()


def test_validate_too_short():
    wav_bytes = create_wav_bytes(duration_seconds=0.1)
    result = validate_audio_file(wav_bytes, filename="short.wav")
    assert result.is_valid is False
    assert result.error_code == "TOO_SHORT"
    assert "minimum" in result.error_message.lower()


def test_validate_too_long():
    config = load_audio_config()
    config["max_duration_seconds"] = 1.5
    wav_bytes = create_wav_bytes(duration_seconds=3.0)
    result = validate_audio_file(wav_bytes, filename="long.wav", config=config)
    assert result.is_valid is False
    assert result.error_code == "TOO_LONG"
    assert "exceeds" in result.error_message.lower()


def test_validate_silent_audio():
    silent_bytes = create_wav_bytes(duration_seconds=1.0, amplitude=0.0)
    result = validate_audio_file(silent_bytes, filename="silent.wav")
    assert result.is_valid is False
    assert result.error_code == "NO_USABLE_SIGNAL"
    assert "silence" in result.error_message.lower()


def test_validate_file_too_large():
    config = load_audio_config()
    config["max_file_size_mb"] = 0.00001
    wav_bytes = create_wav_bytes(duration_seconds=2.0)
    result = validate_audio_file(wav_bytes, filename="large.wav", config=config)
    assert result.is_valid is False
    assert result.error_code == "FILE_TOO_LARGE"


def test_result_to_dict():
    wav_bytes = create_wav_bytes(duration_seconds=1.0)
    result = validate_audio_file(wav_bytes, filename="test.wav")
    res_dict = result.to_dict()
    assert isinstance(res_dict, dict)
    assert res_dict["is_valid"] is True
    assert "metadata" in res_dict
    assert res_dict["metadata"]["duration_seconds"] == 1.0
