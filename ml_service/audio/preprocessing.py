import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import scipy.signal as signal
except ImportError:
    signal = None


@dataclass
class AudioSegment:
    """Represents a fixed-duration audio segment window with timestamp metadata."""

    data: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    segment_index: int

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary metadata representation excluding raw numpy array."""
        return {
            "sample_rate": self.sample_rate,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "segment_index": self.segment_index,
            "num_samples": len(self.data),
        }


@dataclass
class PreprocessedAudio:
    """Container holding preprocessed audio signal and windowed segments."""

    data: np.ndarray
    sample_rate: int
    duration_seconds: float
    channels: int
    segments: List[AudioSegment]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert preprocessed audio container summary to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 4),
            "channels": self.channels,
            "num_samples": len(self.data),
            "num_segments": len(self.segments),
            "segments": [seg.to_dict() for seg in self.segments],
            "metadata": self.metadata,
        }


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to single-channel 1D mono float32 array.

    Args:
        audio: Input audio numpy array (1D or 2D).

    Returns:
        1D float32 numpy array.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        if arr.shape[1] <= 8:
            return np.mean(arr, axis=1).astype(np.float32)
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            return np.mean(arr, axis=0).astype(np.float32)

    return arr.flatten().astype(np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 22050) -> np.ndarray:
    """Resample 1D audio array from orig_sr to target_sr.

    Args:
        audio: 1D audio float array.
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz (default 22050).

    Returns:
        Resampled 1D float32 numpy array.
    """
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)

    if signal is not None:
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        resampled = signal.resample_poly(audio, up, down)
        return resampled.astype(np.float32)

    num_samples = int(round(len(audio) * float(target_sr) / float(orig_sr)))
    old_indices = np.linspace(0, len(audio) - 1, len(audio))
    new_indices = np.linspace(0, len(audio) - 1, num_samples)
    return np.interp(new_indices, old_indices, audio).astype(np.float32)


def normalize_amplitude(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio amplitude so peak absolute value matches target_peak.

    Args:
        audio: 1D audio numpy array.
        target_peak: Target maximum absolute amplitude (default 0.95).

    Returns:
        Normalized 1D float32 numpy array.
    """
    arr = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if len(arr) > 0 else 0.0
    if peak > 0:
        return (arr / peak * target_peak).astype(np.float32)
    return arr


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = 30.0,
    frame_length: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """Trim leading and trailing silence from 1D audio array.

    Args:
        audio: 1D audio float array.
        sample_rate: Audio sample rate in Hz.
        threshold_db: Energy threshold below max frame energy in dB to trim.
        frame_length: Window frame size for energy calculation.
        hop_length: Hop step size between frames.

    Returns:
        Trimmed 1D float32 numpy array.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) <= frame_length:
        return arr

    n_frames = 1 + (len(arr) - frame_length) // hop_length
    energies = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = arr[start : start + frame_length]
        energies[i] = np.sqrt(np.mean(frame**2))

    max_energy = np.max(energies)
    if max_energy <= 1e-10:
        return arr

    db_energies = 20.0 * np.log10((energies + 1e-10) / max_energy)
    non_silent = np.where(db_energies >= -threshold_db)[0]

    if len(non_silent) == 0:
        return arr

    start_frame = non_silent[0]
    end_frame = non_silent[-1]

    start_idx = start_frame * hop_length
    end_idx = min(end_frame * hop_length + frame_length, len(arr))

    return arr[start_idx:end_idx]


def reduce_noise(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 50.0) -> np.ndarray:
    """Apply high-pass filtering to attenuate low-frequency rumble and background noise.

    Args:
        audio: 1D audio float array.
        sample_rate: Sample rate in Hz.
        cutoff_hz: High-pass filter cutoff frequency in Hz (default 50Hz).

    Returns:
        Noise-reduced 1D float32 numpy array.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == 0:
        return arr

    if signal is not None and sample_rate > 2 * cutoff_hz:
        nyquist = 0.5 * sample_rate
        norm_cutoff = cutoff_hz / nyquist
        b, a = signal.butter(4, norm_cutoff, btype="highpass")
        return signal.filtfilt(b, a, arr).astype(np.float32)

    return (arr - np.mean(arr)).astype(np.float32)


def pad_or_truncate(audio: np.ndarray, target_length: int, pad_mode: str = "constant") -> np.ndarray:
    """Pad audio with zeros or truncate to exact target sample length.

    Args:
        audio: 1D audio numpy array.
        target_length: Desired number of samples.
        pad_mode: Padding mode for np.pad (default 'constant').

    Returns:
        1D float32 numpy array of exact target_length.
    """
    arr = np.asarray(audio, dtype=np.float32)
    current_len = len(arr)

    if current_len == target_length:
        return arr

    if current_len > target_length:
        return arr[:target_length]

    pad_width = target_length - current_len
    return np.pad(arr, (0, pad_width), mode=pad_mode).astype(np.float32)


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    window_duration_seconds: float = 3.0,
    window_overlap_seconds: float = 1.0,
) -> List[AudioSegment]:
    """Segment audio into fixed-duration sliding window segments with start and end timestamps.

    Args:
        audio: 1D audio float array.
        sample_rate: Sample rate in Hz.
        window_duration_seconds: Duration of each window in seconds (default 3.0s).
        window_overlap_seconds: Overlap between consecutive windows in seconds (default 1.0s).

    Returns:
        List of AudioSegment objects.
    """
    arr = np.asarray(audio, dtype=np.float32)
    window_samples = int(round(window_duration_seconds * sample_rate))
    step_duration = window_duration_seconds - window_overlap_seconds
    step_samples = int(round(step_duration * sample_rate))

    if step_samples <= 0:
        step_samples = window_samples

    segments: List[AudioSegment] = []
    if len(arr) == 0:
        return segments

    start_idx = 0
    seg_idx = 0

    while start_idx < len(arr):
        end_idx = min(start_idx + window_samples, len(arr))
        chunk = arr[start_idx:end_idx]

        if len(chunk) < window_samples:
            chunk = pad_or_truncate(chunk, window_samples)

        start_time = float(start_idx) / float(sample_rate)
        end_time = start_time + window_duration_seconds

        segment = AudioSegment(
            data=chunk,
            sample_rate=sample_rate,
            start_time=start_time,
            end_time=end_time,
            segment_index=seg_idx,
        )
        segments.append(segment)

        seg_idx += 1
        start_idx += step_samples

        if end_idx >= len(arr) and start_idx >= len(arr):
            break

    return segments


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = 22050,
    target_peak: float = 0.95,
    trim_silence_flag: bool = True,
    reduce_noise_flag: bool = True,
    window_duration_seconds: float = 3.0,
    window_overlap_seconds: float = 1.0,
    target_duration_seconds: Optional[float] = None,
) -> PreprocessedAudio:
    """Execute complete audio preprocessing pipeline: mono conversion, resampling, noise reduction, silence trimming, amplitude normalization, optional length adjustment, and sliding-window segmentation.

    Args:
        audio: Input audio array (mono or multi-channel).
        sample_rate: Input audio sample rate in Hz.
        target_sample_rate: Target sample rate in Hz (default 22050).
        target_peak: Target peak amplitude for normalization (default 0.95).
        trim_silence_flag: Whether to trim leading/trailing silence.
        reduce_noise_flag: Whether to apply low-frequency noise reduction.
        window_duration_seconds: Window duration for segmentation in seconds (default 3.0).
        window_overlap_seconds: Window overlap for segmentation in seconds (default 1.0).
        target_duration_seconds: Optional target duration to pad/truncate overall signal.

    Returns:
        PreprocessedAudio object containing processed signal, segments, and metadata.
    """
    mono_audio = to_mono(audio)
    resampled_audio = resample_audio(mono_audio, orig_sr=sample_rate, target_sr=target_sample_rate)
    current_sr = target_sample_rate

    if reduce_noise_flag:
        cleaned_audio = reduce_noise(resampled_audio, sample_rate=current_sr)
    else:
        cleaned_audio = resampled_audio

    if trim_silence_flag:
        trimmed_audio = trim_silence(cleaned_audio, sample_rate=current_sr)
    else:
        trimmed_audio = cleaned_audio

    normalized_audio = normalize_amplitude(trimmed_audio, target_peak=target_peak)

    if target_duration_seconds is not None:
        target_len = int(round(target_duration_seconds * current_sr))
        final_audio = pad_or_truncate(normalized_audio, target_length=target_len)
    else:
        final_audio = normalized_audio

    duration_seconds = float(len(final_audio)) / float(current_sr) if len(final_audio) > 0 else 0.0

    segments = segment_audio(
        final_audio,
        sample_rate=current_sr,
        window_duration_seconds=window_duration_seconds,
        window_overlap_seconds=window_overlap_seconds,
    )

    metadata = {
        "original_sample_rate": sample_rate,
        "target_sample_rate": current_sr,
        "target_peak": target_peak,
        "trim_silence_applied": trim_silence_flag,
        "reduce_noise_applied": reduce_noise_flag,
        "window_duration_seconds": window_duration_seconds,
        "window_overlap_seconds": window_overlap_seconds,
    }

    return PreprocessedAudio(
        data=final_audio,
        sample_rate=current_sr,
        duration_seconds=duration_seconds,
        channels=1,
        segments=segments,
        metadata=metadata,
    )
