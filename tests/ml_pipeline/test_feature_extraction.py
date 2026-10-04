"""Unit tests for offline feature extraction in ml_pipeline."""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf

from ml_pipeline.features.extract_features import (
    FEATURE_COLUMNS,
    extract_features_for_row,
)
from ml_service.features.extract import extract_features

EXPECTED_FEATURE_DIM = 74


def _create_fixed_sine_wave(
    duration: float = 1.0,
    sample_rate: int = 22050,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate deterministic 1D sine wave numpy array."""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def test_feature_vector_shape_and_no_nan_inf_sine_wave() -> None:
    sr = 22050
    signal = _create_fixed_sine_wave(duration=1.0, sample_rate=sr, frequency=440.0)

    result = extract_features(signal, sr=sr)
    vec = result["features_vector"]

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape == (EXPECTED_FEATURE_DIM,)
    assert vec.dtype == np.float32

    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()
    assert np.isfinite(vec).all()


def test_feature_vector_shape_and_no_nan_inf_sample_audio(sample_audio_dir: Path) -> None:
    wav_path = sample_audio_dir / "valid.wav"
    assert wav_path.exists()

    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1).astype(np.float32)

    result = extract_features(data, sr=sr)
    vec = result["features_vector"]

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (EXPECTED_FEATURE_DIM,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()
    assert np.isfinite(vec).all()


def test_feature_vector_shape_and_no_nan_inf_multi_harmonic() -> None:
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    f1, f2, f3 = 440.0, 880.0, 1320.0
    signal = (
        0.3 * np.sin(2.0 * np.pi * f1 * t)
        + 0.2 * np.sin(2.0 * np.pi * f2 * t)
        + 0.1 * np.sin(2.0 * np.pi * f3 * t)
    ).astype(np.float32)

    result = extract_features(signal, sr=sr)
    vec = result["features_vector"]

    assert vec.shape == (EXPECTED_FEATURE_DIM,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()
    assert np.isfinite(vec).all()


def test_feature_vector_shape_and_no_nan_inf_noise() -> None:
    sr = 22050
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, 0.1, sr).astype(np.float32)

    result = extract_features(noise, sr=sr)
    vec = result["features_vector"]

    assert vec.shape == (EXPECTED_FEATURE_DIM,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()
    assert np.isfinite(vec).all()


def test_feature_vector_deterministic_reproducibility() -> None:
    sr = 22050
    signal = _create_fixed_sine_wave(duration=1.0, sample_rate=sr)

    res1 = extract_features(signal, sr=sr)
    res2 = extract_features(signal, sr=sr)

    vec1 = res1["features_vector"]
    vec2 = res2["features_vector"]

    assert np.allclose(vec1, vec2, atol=1e-6)


def test_feature_dict_structure_and_dimensions() -> None:
    sr = 22050
    signal = _create_fixed_sine_wave(duration=1.0, sample_rate=sr)

    result = extract_features(signal, sr=sr)
    feat_dict: Dict[str, Any] = result["feature_dict"]

    assert len(feat_dict["mfcc_mean"]) == 20
    assert len(feat_dict["mfcc_std"]) == 20
    assert len(feat_dict["chroma_mean"]) == 12
    assert len(feat_dict["chroma_std"]) == 12

    scalar_keys = [
        "zcr_mean",
        "zcr_std",
        "rms_mean",
        "rms_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
    ]
    for key in scalar_keys:
        assert key in feat_dict
        val = feat_dict[key]
        assert isinstance(val, (float, int))
        assert not np.isnan(val)
        assert not np.isinf(val)


def test_extract_features_for_row_pipeline_integration(sample_audio_dir: Path) -> None:
    sample_wav = sample_audio_dir / "valid.wav"
    assert sample_wav.exists()

    row_input = {
        "audio_id": "AUD-TEST-0001",
        "filename": "valid.wav",
        "class": "gunshot",
        "final_split": "train",
    }

    with patch("ml_pipeline.features.extract_features.resolve_audio_file", return_value=sample_wav):
        row_output = extract_features_for_row(row_input)

    assert row_output is not None
    assert row_output["audio_id"] == "AUD-TEST-0001"
    assert row_output["class"] == "gunshot"
    assert row_output["final_split"] == "train"

    feature_col_names = FEATURE_COLUMNS[3:]
    assert len(feature_col_names) == EXPECTED_FEATURE_DIM

    for col in feature_col_names:
        assert col in row_output
        val = row_output[col]
        assert isinstance(val, (float, int))
        assert not np.isnan(val)
        assert not np.isinf(val)


def test_extract_features_empty_signal_raises_error() -> None:
    empty_signal = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match="Cannot extract features from an empty audio signal array"):
        extract_features(empty_signal)
