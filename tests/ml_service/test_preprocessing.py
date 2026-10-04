"""Unit tests for ml_service audio preprocessing pipeline and transformations."""

import math
import sys
from pathlib import Path
from typing import List
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_SERVICE_DIR = REPO_ROOT / "ml_service"
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_service.audio.preprocessing import (
    AudioSegment,
    PreprocessedAudio,
    normalize_amplitude,
    pad_or_truncate,
    preprocess_audio,
    reduce_noise,
    resample_audio,
    segment_audio,
    to_mono,
    trim_silence,
)


def _generate_synthetic_tone(
    duration: float = 1.0,
    sample_rate: int = 22050,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate 1D synthetic sine wave float32 numpy array."""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)


def test_resample_audio_target_sample_rate() -> None:
    orig_sr = 44100
    target_sr = 22050
    duration = 1.0
    signal_44k = _generate_synthetic_tone(duration=duration, sample_rate=orig_sr)

    resampled = resample_audio(signal_44k, orig_sr=orig_sr, target_sr=target_sr)

    assert isinstance(resampled, np.ndarray)
    assert resampled.dtype == np.float32
    assert len(resampled) == int(duration * target_sr)


def test_resample_audio_upsampling() -> None:
    orig_sr = 16000
    target_sr = 22050
    duration = 0.5
    signal_16k = _generate_synthetic_tone(duration=duration, sample_rate=orig_sr)

    resampled = resample_audio(signal_16k, orig_sr=orig_sr, target_sr=target_sr)

    assert isinstance(resampled, np.ndarray)
    assert len(resampled) == int(duration * target_sr)


def test_resample_audio_identical_sample_rate() -> None:
    signal_22k = _generate_synthetic_tone(duration=1.0, sample_rate=22050)
    resampled = resample_audio(signal_22k, orig_sr=22050, target_sr=22050)

    assert len(resampled) == len(signal_22k)
    assert np.array_equal(resampled, signal_22k)


def test_resample_audio_empty_array() -> None:
    empty_arr = np.array([], dtype=np.float32)
    resampled = resample_audio(empty_arr, orig_sr=44100, target_sr=22050)

    assert len(resampled) == 0


def test_to_mono_single_channel_pass_through() -> None:
    mono_input = _generate_synthetic_tone(duration=0.5, sample_rate=22050)
    output = to_mono(mono_input)

    assert output.ndim == 1
    assert output.dtype == np.float32
    assert np.array_equal(output, mono_input)


def test_to_mono_collapses_stereo_channels() -> None:
    left = _generate_synthetic_tone(duration=0.5, sample_rate=22050, amplitude=0.8)
    right = _generate_synthetic_tone(duration=0.5, sample_rate=22050, amplitude=0.4)
    stereo_matrix = np.column_stack([left, right])

    mono_output = to_mono(stereo_matrix)

    assert mono_output.ndim == 1
    assert mono_output.dtype == np.float32
    assert len(mono_output) == len(left)
    expected_average = (left + right) / 2.0
    assert np.allclose(mono_output, expected_average, atol=1e-5)


def test_to_mono_collapses_transposed_stereo() -> None:
    left = _generate_synthetic_tone(duration=0.5, sample_rate=22050, amplitude=0.6)
    right = _generate_synthetic_tone(duration=0.5, sample_rate=22050, amplitude=0.2)
    transposed_stereo = np.vstack([left, right])

    mono_output = to_mono(transposed_stereo)

    assert mono_output.ndim == 1
    assert len(mono_output) == len(left)
    expected_average = (left + right) / 2.0
    assert np.allclose(mono_output, expected_average, atol=1e-5)


def test_to_mono_multi_channel_averaging() -> None:
    sample_rate = 22050
    ch1 = _generate_synthetic_tone(duration=0.2, sample_rate=sample_rate, amplitude=0.2)
    ch2 = _generate_synthetic_tone(duration=0.2, sample_rate=sample_rate, amplitude=0.4)
    ch3 = _generate_synthetic_tone(duration=0.2, sample_rate=sample_rate, amplitude=0.6)
    ch4 = _generate_synthetic_tone(duration=0.2, sample_rate=sample_rate, amplitude=0.8)
    quad_audio = np.column_stack([ch1, ch2, ch3, ch4])

    mono_output = to_mono(quad_audio)

    assert mono_output.ndim == 1
    expected_average = (ch1 + ch2 + ch3 + ch4) / 4.0
    assert np.allclose(mono_output, expected_average, atol=1e-5)


def test_trim_silence_removes_leading_and_trailing_silence() -> None:
    sample_rate = 22050
    leading_silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    active_tone = _generate_synthetic_tone(duration=1.0, sample_rate=sample_rate, amplitude=0.8)
    trailing_silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)

    signal_with_silence = np.concatenate([leading_silence, active_tone, trailing_silence])
    trimmed = trim_silence(signal_with_silence, sample_rate=sample_rate, threshold_db=25.0)

    assert len(trimmed) < len(signal_with_silence)
    assert len(trimmed) >= len(active_tone) - 1024
    assert len(trimmed) <= len(active_tone) + 2048


def test_trim_silence_all_silent_signal() -> None:
    sample_rate = 22050
    pure_silence = np.zeros(int(1.0 * sample_rate), dtype=np.float32)

    trimmed = trim_silence(pure_silence, sample_rate=sample_rate)

    assert len(trimmed) == len(pure_silence)


def test_trim_silence_short_signal() -> None:
    sample_rate = 22050
    short_signal = np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float32)

    trimmed = trim_silence(short_signal, sample_rate=sample_rate, frame_length=1024)

    assert len(trimmed) == len(short_signal)


def test_segment_audio_produces_configured_windows() -> None:
    sample_rate = 22050
    window_duration = 3.0
    window_overlap = 1.0
    audio_duration = 5.0

    audio = _generate_synthetic_tone(duration=audio_duration, sample_rate=sample_rate)
    segments: List[AudioSegment] = segment_audio(
        audio=audio,
        sample_rate=sample_rate,
        window_duration_seconds=window_duration,
        window_overlap_seconds=window_overlap,
    )

    expected_window_samples = int(window_duration * sample_rate)

    assert len(segments) == 3

    for idx, seg in enumerate(segments):
        assert isinstance(seg, AudioSegment)
        assert seg.sample_rate == sample_rate
        assert seg.segment_index == idx
        assert len(seg.data) == expected_window_samples

    assert pytest.approx(segments[0].start_time, abs=1e-3) == 0.0
    assert pytest.approx(segments[0].end_time, abs=1e-3) == 3.0
    assert pytest.approx(segments[1].start_time, abs=1e-3) == 2.0
    assert pytest.approx(segments[1].end_time, abs=1e-3) == 5.0
    assert pytest.approx(segments[2].start_time, abs=1e-3) == 4.0
    assert pytest.approx(segments[2].end_time, abs=1e-3) == 7.0


def test_segment_audio_non_overlapping_windows() -> None:
    sample_rate = 22050
    window_duration = 2.0
    window_overlap = 0.0
    audio_duration = 6.0

    audio = _generate_synthetic_tone(duration=audio_duration, sample_rate=sample_rate)
    segments = segment_audio(
        audio=audio,
        sample_rate=sample_rate,
        window_duration_seconds=window_duration,
        window_overlap_seconds=window_overlap,
    )

    expected_window_samples = int(window_duration * sample_rate)
    assert len(segments) == 3

    for idx, seg in enumerate(segments):
        assert len(seg.data) == expected_window_samples
        assert pytest.approx(seg.start_time, abs=1e-3) == float(idx * window_duration)
        assert pytest.approx(seg.end_time, abs=1e-3) == float((idx + 1) * window_duration)


def test_segment_audio_pads_trailing_incomplete_window() -> None:
    sample_rate = 22050
    window_duration = 3.0
    window_overlap = 1.0
    audio_duration = 5.5

    audio = _generate_synthetic_tone(duration=audio_duration, sample_rate=sample_rate)
    segments = segment_audio(
        audio=audio,
        sample_rate=sample_rate,
        window_duration_seconds=window_duration,
        window_overlap_seconds=window_overlap,
    )

    expected_window_samples = int(window_duration * sample_rate)
    assert len(segments) == 3

    for seg in segments:
        assert len(seg.data) == expected_window_samples

    last_seg = segments[-1]
    assert pytest.approx(last_seg.start_time, abs=1e-3) == 4.0
    assert pytest.approx(last_seg.end_time, abs=1e-3) == 7.0


def test_segment_audio_empty_input() -> None:
    empty_arr = np.array([], dtype=np.float32)
    segments = segment_audio(empty_arr, sample_rate=22050)

    assert len(segments) == 0


def test_segment_audio_to_dict_serialization() -> None:
    audio = _generate_synthetic_tone(duration=3.0, sample_rate=22050)
    segments = segment_audio(
        audio,
        sample_rate=22050,
        window_duration_seconds=3.0,
        window_overlap_seconds=0.0,
    )

    assert len(segments) == 1
    d = segments[0].to_dict()

    assert "sample_rate" in d
    assert "start_time" in d
    assert "end_time" in d
    assert "segment_index" in d
    assert "num_samples" in d
    assert "data" not in d


def test_normalize_amplitude_target_peak() -> None:
    raw_signal = _generate_synthetic_tone(duration=0.5, sample_rate=22050, amplitude=0.25)
    normalized = normalize_amplitude(raw_signal, target_peak=0.95)

    assert normalized.dtype == np.float32
    peak = float(np.max(np.abs(normalized)))
    assert pytest.approx(peak, abs=1e-3) == 0.95


def test_normalize_amplitude_all_zeros() -> None:
    zeros = np.zeros(1000, dtype=np.float32)
    normalized = normalize_amplitude(zeros, target_peak=0.95)

    assert np.all(normalized == 0.0)


def test_pad_or_truncate_padding() -> None:
    raw_signal = _generate_synthetic_tone(duration=0.5, sample_rate=1000)
    assert len(raw_signal) == 500

    padded = pad_or_truncate(raw_signal, target_length=1000)

    assert len(padded) == 1000
    assert np.array_equal(padded[:500], raw_signal)
    assert np.all(padded[500:] == 0.0)


def test_pad_or_truncate_truncation() -> None:
    raw_signal = _generate_synthetic_tone(duration=1.0, sample_rate=1000)
    assert len(raw_signal) == 1000

    truncated = pad_or_truncate(raw_signal, target_length=400)

    assert len(truncated) == 400
    assert np.array_equal(truncated, raw_signal[:400])


def test_pad_or_truncate_exact_length() -> None:
    raw_signal = _generate_synthetic_tone(duration=0.5, sample_rate=1000)
    processed = pad_or_truncate(raw_signal, target_length=500)

    assert len(processed) == 500
    assert np.array_equal(processed, raw_signal)


def test_reduce_noise_filters_dc_offset() -> None:
    sample_rate = 22050
    signal_with_dc = _generate_synthetic_tone(duration=0.5, sample_rate=sample_rate) + 0.3

    filtered = reduce_noise(signal_with_dc, sample_rate=sample_rate, cutoff_hz=50.0)

    assert len(filtered) == len(signal_with_dc)
    assert pytest.approx(float(np.mean(filtered)), abs=0.05) == 0.0


def test_preprocess_audio_end_to_end_pipeline() -> None:
    orig_sr = 44100
    target_sr = 22050

    left = _generate_synthetic_tone(duration=4.5, sample_rate=orig_sr, amplitude=0.4)
    right = _generate_synthetic_tone(duration=4.5, sample_rate=orig_sr, amplitude=0.6)
    stereo_input = np.column_stack([left, right])

    result: PreprocessedAudio = preprocess_audio(
        audio=stereo_input,
        sample_rate=orig_sr,
        target_sample_rate=target_sr,
        target_peak=0.95,
        window_duration_seconds=3.0,
        window_overlap_seconds=1.0,
    )

    assert isinstance(result, PreprocessedAudio)
    assert result.sample_rate == target_sr
    assert result.channels == 1
    assert result.data.ndim == 1
    assert len(result.segments) >= 1

    peak = float(np.max(np.abs(result.data)))
    assert pytest.approx(peak, abs=1e-2) == 0.95

    serialized = result.to_dict()
    assert serialized["sample_rate"] == target_sr
    assert serialized["channels"] == 1
    assert "segments" in serialized
    assert "metadata" in serialized
    assert serialized["metadata"]["original_sample_rate"] == orig_sr
    assert serialized["metadata"]["target_sample_rate"] == target_sr
