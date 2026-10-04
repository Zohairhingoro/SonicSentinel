"""Audio feature extraction module for SonicSentinel ML inference and pipeline.

Extracts MFCC, Mel Spectrogram, Chroma, Zero-Crossing Rate, RMS Energy,
Spectral Centroid, Spectral Bandwidth, and Spectral Roll-off.
"""

from typing import Any, Dict, Tuple
import numpy as np
from scipy import signal, fftpack

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def extract_features(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 20,
    n_mels: int = 128,
) -> Dict[str, Any]:
    """Extract 8 mandatory acoustic feature types from a 1D audio sample array.

    Args:
        y: 1D float array of preprocessed audio samples.
        sr: Sampling rate in Hz (default 22050).
        n_mfcc: Number of MFCC coefficients (default 20).
        n_mels: Number of Mel frequency bands (default 128).

    Returns:
        Dict containing:
        - 'features_vector': 1D float32 numpy array ready for ML model input.
        - 'feature_dict': Dictionary of scalar and list summary statistics.
        - 'raw_features': Dictionary of raw 2D feature matrices.
    """
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    if len(y) == 0:
        raise ValueError("Cannot extract features from an empty audio signal array.")

    y = y.astype(np.float32)

    if HAS_LIBROSA:
        return _extract_librosa(y, sr, n_mfcc, n_mels)
    return _extract_scipy(y, sr, n_mfcc, n_mels)


def _extract_librosa(
    y: np.ndarray, sr: int, n_mfcc: int, n_mels: int
) -> Dict[str, Any]:
    """Extract features using librosa."""
    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 2. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = np.mean(mel_spec_db, axis=1)
    mel_std = np.std(mel_spec_db, axis=1)

    # 3. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # 5. RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # 6. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = float(np.mean(spec_cent))
    cent_std = float(np.std(spec_cent))

    # 7. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bw_mean = float(np.mean(spec_bw))
    bw_std = float(np.std(spec_bw))

    # 8. Spectral Roll-off
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    roll_mean = float(np.mean(spec_roll))
    roll_std = float(np.std(spec_roll))

    # Flatten into single 1D feature vector
    feature_vector = np.hstack([
        mfcc_mean,
        mfcc_std,
        chroma_mean,
        chroma_std,
        np.array([zcr_mean, zcr_std, rms_mean, rms_std, cent_mean, cent_std, bw_mean, bw_std, roll_mean, roll_std], dtype=np.float32)
    ]).astype(np.float32)

    return {
        "features_vector": feature_vector,
        "feature_dict": {
            "mfcc_mean": mfcc_mean.tolist(),
            "mfcc_std": mfcc_std.tolist(),
            "mel_spectrogram_mean": mel_mean.tolist(),
            "chroma_mean": chroma_mean.tolist(),
            "chroma_std": chroma_std.tolist(),
            "zcr_mean": zcr_mean,
            "zcr_std": zcr_std,
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "spectral_centroid_mean": cent_mean,
            "spectral_centroid_std": cent_std,
            "spectral_bandwidth_mean": bw_mean,
            "spectral_bandwidth_std": bw_std,
            "spectral_rolloff_mean": roll_mean,
            "spectral_rolloff_std": roll_std,
        },
        "raw_features": {
            "mfcc": mfcc,
            "mel_spectrogram": mel_spec_db,
            "chroma": chroma,
            "zcr": zcr,
            "rms": rms,
            "spectral_centroid": spec_cent,
            "spectral_bandwidth": spec_bw,
            "spectral_rolloff": spec_roll,
        },
    }


def _extract_scipy(
    y: np.ndarray, sr: int, n_mfcc: int, n_mels: int
) -> Dict[str, Any]:
    """Pure scipy/numpy fallback for feature extraction if librosa is unavailable."""
    n_per_seg = min(512, max(64, len(y)))
    frequencies, times, stft_matrix = signal.stft(y, fs=sr, nperseg=n_per_seg)
    magnitude = np.abs(stft_matrix) + 1e-10

    # 1. Zero Crossing Rate
    zcr_frames = np.abs(np.diff(np.signbit(y))).astype(np.float32)
    zcr_mean = float(np.mean(zcr_frames))
    zcr_std = float(np.std(zcr_frames))

    # 2. RMS Energy
    rms_frames = np.sqrt(np.mean(magnitude**2, axis=0))
    rms_mean = float(np.mean(rms_frames))
    rms_std = float(np.std(rms_frames))

    # 3. Spectral Centroid
    freq_bins = frequencies[:, np.newaxis]
    cent_frames = np.sum(freq_bins * magnitude, axis=0) / np.sum(magnitude, axis=0)
    cent_mean = float(np.mean(cent_frames))
    cent_std = float(np.std(cent_frames))

    # 4. Spectral Bandwidth
    bw_frames = np.sqrt(
        np.sum(((freq_bins - cent_frames) ** 2) * magnitude, axis=0) / np.sum(magnitude, axis=0)
    )
    bw_mean = float(np.mean(bw_frames))
    bw_std = float(np.std(bw_frames))

    # 5. Spectral Roll-off (85% power cutoff)
    cum_energy = np.cumsum(magnitude**2, axis=0)
    total_energy = cum_energy[-1, :]
    threshold = 0.85 * total_energy
    rolloff_idx = np.argmax(cum_energy >= threshold, axis=0)
    roll_frames = frequencies[rolloff_idx]
    roll_mean = float(np.mean(roll_frames))
    roll_std = float(np.std(roll_frames))

    # 6. Mel Spectrogram (approximate log magnitude summary)
    mel_spec_db = 20 * np.log10(magnitude + 1e-10)
    mel_mean = np.mean(mel_spec_db, axis=1)
    if len(mel_mean) < n_mels:
        mel_mean = np.pad(mel_mean, (0, n_mels - len(mel_mean)))
    else:
        mel_mean = mel_mean[:n_mels]

    # 7. Chroma (12 semitone binning)
    chroma = np.zeros((12, magnitude.shape[1]), dtype=np.float32)
    for i, f in enumerate(frequencies):
        if f > 0:
            pitch_class = int(round(12 * np.log2(f / 440.0))) % 12
            chroma[pitch_class, :] += magnitude[i, :]
    chroma_norm = chroma / (np.max(chroma, axis=0, keepdims=True) + 1e-10)
    chroma_mean = np.mean(chroma_norm, axis=1)
    chroma_std = np.std(chroma_norm, axis=1)

    # 8. MFCC (DCT of log spectrum)
    mfcc = fftpack.dct(mel_spec_db, axis=0, type=2, norm="ortho")[:n_mfcc, :]
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    feature_vector = np.hstack([
        mfcc_mean,
        mfcc_std,
        chroma_mean,
        chroma_std,
        np.array([zcr_mean, zcr_std, rms_mean, rms_std, cent_mean, cent_std, bw_mean, bw_std, roll_mean, roll_std], dtype=np.float32)
    ]).astype(np.float32)

    return {
        "features_vector": feature_vector,
        "feature_dict": {
            "mfcc_mean": mfcc_mean.tolist(),
            "mfcc_std": mfcc_std.tolist(),
            "mel_spectrogram_mean": mel_mean.tolist(),
            "chroma_mean": chroma_mean.tolist(),
            "chroma_std": chroma_std.tolist(),
            "zcr_mean": zcr_mean,
            "zcr_std": zcr_std,
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "spectral_centroid_mean": cent_mean,
            "spectral_centroid_std": cent_std,
            "spectral_bandwidth_mean": bw_mean,
            "spectral_bandwidth_std": bw_std,
            "spectral_rolloff_mean": roll_mean,
            "spectral_rolloff_std": roll_std,
        },
        "raw_features": {
            "mfcc": mfcc,
            "mel_spectrogram": mel_spec_db,
            "chroma": chroma_norm,
            "zcr": zcr_frames[np.newaxis, :],
            "rms": rms_frames[np.newaxis, :],
            "spectral_centroid": cent_frames[np.newaxis, :],
            "spectral_bandwidth": bw_frames[np.newaxis, :],
            "spectral_rolloff": roll_frames[np.newaxis, :],
        },
    }
