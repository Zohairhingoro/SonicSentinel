import math
import numpy as np
import pytest
try:
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
except ModuleNotFoundError:
    from audio.preprocessing import (
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


def generate_sine_wave(duration: float = 1.0, sr: int = 22050, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_to_mono():
    signal_1d = generate_sine_wave(1.0, 22050)
    mono_1d = to_mono(signal_1d)
    assert mono_1d.ndim == 1
    assert len(mono_1d) == len(signal_1d)

    stereo_2d = np.column_stack([signal_1d, signal_1d * 0.5])
    mono_2d = to_mono(stereo_2d)
    assert mono_2d.ndim == 1
    assert len(mono_2d) == len(signal_1d)
    assert np.allclose(mono_2d, signal_1d * 0.75, atol=1e-5)


def test_resample_audio():
    signal_44k = generate_sine_wave(1.0, 44100)
    resampled = resample_audio(signal_44k, orig_sr=44100, target_sr=22050)
    assert len(resampled) == 22050


def test_normalize_amplitude():
    signal_unnorm = generate_sine_wave(1.0, 22050, amp=0.2)
    normalized = normalize_amplitude(signal_unnorm, target_peak=0.95)
    assert pytest.approx(float(np.max(np.abs(normalized))), abs=1e-3) == 0.95


def test_trim_silence():
    sr = 22050
    silence = np.zeros(sr // 2, dtype=np.float32)
    tone = generate_sine_wave(1.0, sr, amp=0.8)
    audio_with_silence = np.concatenate([silence, tone, silence])

    trimmed = trim_silence(audio_with_silence, sample_rate=sr, threshold_db=20.0)
    assert len(trimmed) < len(audio_with_silence)
    assert len(trimmed) >= len(tone) - 1024


def test_reduce_noise():
    audio = generate_sine_wave(1.0, 22050, freq=440.0)
    reduced = reduce_noise(audio, sample_rate=22050, cutoff_hz=50.0)
    assert len(reduced) == len(audio)


def test_pad_or_truncate():
    audio = generate_sine_wave(1.0, 22050)
    padded = pad_or_truncate(audio, target_length=30000)
    assert len(padded) == 30000
    assert np.all(padded[len(audio):] == 0)

    truncated = pad_or_truncate(audio, target_length=10000)
    assert len(truncated) == 10000


def test_segment_audio():
    sr = 22050
    audio = generate_sine_wave(5.0, sr)
    segments = segment_audio(
        audio,
        sample_rate=sr,
        window_duration_seconds=3.0,
        window_overlap_seconds=1.0,
    )
    assert len(segments) > 0
    for i, seg in enumerate(segments):
        assert isinstance(seg, AudioSegment)
        assert seg.sample_rate == sr
        assert seg.segment_index == i
        assert len(seg.data) == int(3.0 * sr)
        assert seg.end_time > seg.start_time


def test_preprocess_audio_end_to_end():
    sr = 44100
    audio = generate_sine_wave(4.0, sr)
    result = preprocess_audio(
        audio,
        sample_rate=sr,
        target_sample_rate=22050,
        target_peak=0.95,
        window_duration_seconds=3.0,
        window_overlap_seconds=1.0,
    )
    assert isinstance(result, PreprocessedAudio)
    assert result.sample_rate == 22050
    assert len(result.segments) >= 1
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "segments" in d
