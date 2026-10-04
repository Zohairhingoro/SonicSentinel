import numpy as np
import pytest

from audio.visualization import (
    analyze_audio_visualization,
    classify_audio_quality,
    detect_clipping,
    estimate_background_noise,
    generate_spectrogram_data,
    generate_waveform_data,
)


def test_generate_waveform_data():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    waveform = generate_waveform_data(y, num_points=100)

    assert isinstance(waveform, list)
    assert len(waveform) == 100
    assert all(isinstance(v, float) for v in waveform)
    assert max(abs(v) for v in waveform) <= 1.001


def test_generate_spectrogram_data():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    spec = generate_spectrogram_data(y, sr=sr, n_mels=32, max_time_steps=50)

    assert isinstance(spec, list)
    assert len(spec) == 32
    assert all(isinstance(row, list) for row in spec)
    assert len(spec[0]) <= 50


def test_detect_clipping():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)

    # Clean signal (max 0.5)
    clean_y = 0.5 * np.sin(2 * np.pi * 440 * t)
    clean_info = detect_clipping(clean_y)
    assert clean_info["clipping_ratio"] == 0.0
    assert not clean_info["has_clipping"]

    # Clipped signal (constant 1.0)
    clipped_y = np.ones(sr, dtype=np.float32)
    clipped_info = detect_clipping(clipped_y)
    assert clipped_info["clipping_ratio"] > 0.9
    assert clipped_info["severe_clipping"]


def test_estimate_background_noise():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    noise_info = estimate_background_noise(y)

    assert "noise_floor_db" in noise_info
    assert "snr_db" in noise_info
    assert "noise_level" in noise_info
    assert noise_info["snr_db"] > 0


def test_classify_audio_quality():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)

    # Clean signal -> Good
    clean_y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    good_res = classify_audio_quality(clean_y, sr=sr)
    assert good_res["quality"] in ["Good", "Acceptable"]

    # Silent signal -> Unusable
    silent_y = np.zeros(sr, dtype=np.float32)
    silent_res = classify_audio_quality(silent_y, sr=sr)
    assert silent_res["quality"] == "Unusable"

    # Heavily clipped signal -> Unusable or Poor
    clipped_y = np.ones(sr, dtype=np.float32)
    clipped_res = classify_audio_quality(clipped_y, sr=sr)
    assert clipped_res["quality"] in ["Unusable", "Poor"]


def test_analyze_audio_visualization():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    res = analyze_audio_visualization(y, sr=sr)

    assert "waveform" in res
    assert "spectrogram" in res
    assert "quality" in res
    assert res["quality"] in ["Good", "Acceptable", "Poor", "Unusable"]
