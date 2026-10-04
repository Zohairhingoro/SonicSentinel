import numpy as np
import pytest

from features.extract import extract_features


def test_extract_features_synthetic_signal():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Generate 440 Hz sine wave tone
    y = 0.5 * np.sin(2 * np.pi * 440 * t)

    res = extract_features(y, sr=sr, n_mfcc=20, n_mels=128)

    assert "features_vector" in res
    assert "feature_dict" in res
    assert "raw_features" in res

    feat_vec = res["features_vector"]
    assert isinstance(feat_vec, np.ndarray)
    assert feat_vec.ndim == 1
    assert len(feat_vec) > 50

    feat_dict = res["feature_dict"]
    required_keys = [
        "mfcc_mean",
        "mfcc_std",
        "mel_spectrogram_mean",
        "chroma_mean",
        "chroma_std",
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
    for key in required_keys:
        assert key in feat_dict, f"Missing expected key: {key}"

    assert len(feat_dict["mfcc_mean"]) == 20
    assert len(feat_dict["chroma_mean"]) == 12
    assert feat_dict["rms_mean"] > 0


def test_extract_features_stereo_signal():
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    left = 0.3 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 880 * t)
    stereo_y = np.vstack([left, right])  # Shape (2, N)

    res = extract_features(stereo_y, sr=sr)
    assert isinstance(res["features_vector"], np.ndarray)


def test_extract_features_empty_signal():
    y = np.array([], dtype=np.float32)
    with pytest.raises(ValueError):
        extract_features(y)
