"""
Audio Data Augmentation Module for SonicSentinel AI.

Applies controlled offline data augmentations ONLY to training split audio
files (never val/test split files) to increase model robustness against
environmental variations, channel differences, and noise.

Implemented Augmentations:
  1. Background Noise Addition (Gaussian / additive noise at target SNR)
  2. Time Shifting (horizontal roll with zero-padding)
  3. Pitch Shifting (semitone frequency shift via polyphase resampling)
  4. Time Stretching (tempo modification via signal resampling)
  5. Volume Adjustment (gain scaling in dB)
  6. Synthetic Reverberation (exponential decay impulse response convolution)

Outputs:
  - Augmented audio WAV files saved to audio_dataset/augmented/<class>/
  - Appends metadata rows to audio_dataset/metadata/dataset_index.csv with
    original_or_augmented="augmented" and source_split/final_split="train".

Usage:
    python ml_pipeline/augmentation/augment.py
    python ml_pipeline/augmentation/augment.py --num-aug-per-file 1 --limit 100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
import soundfile as sf

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_CSV = REPO_ROOT / "audio_dataset" / "metadata" / "dataset_index.csv"
DEFAULT_AUGMENTED_DIR = REPO_ROOT / "audio_dataset" / "augmented"

CSV_COLUMNS = [
    "audio_id",
    "filename",
    "class",
    "duration",
    "sampling_rate",
    "channels",
    "source",
    "environment",
    "original_or_augmented",
    "source_split",
    "final_split",
]

# ---------------------------------------------------------------------------
# Augmentation Core Functions
# ---------------------------------------------------------------------------

def add_noise(audio: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    """Add Gaussian background noise to audio at specified SNR dB.

    Args:
        audio: 1D float32 numpy array.
        snr_db: Target Signal-to-Noise Ratio in decibels.

    Returns:
        1D float32 numpy array with added noise.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == 0:
        return arr

    rms_signal = np.sqrt(np.mean(arr**2))
    if rms_signal <= 1e-10:
        return arr

    rms_noise = rms_signal / (10.0 ** (snr_db / 20.0))
    noise = np.random.normal(0.0, rms_noise, len(arr)).astype(np.float32)
    noisy_audio = arr + noise

    # Clip to [-1.0, 1.0] range to prevent clipping distortion
    return np.clip(noisy_audio, -1.0, 1.0).astype(np.float32)


def time_shift(audio: np.ndarray, shift_fraction: float = 0.15) -> np.ndarray:
    """Shift audio in time by a fraction of total duration (zero-padded).

    Args:
        audio: 1D float32 numpy array.
        shift_fraction: Fraction of array length to shift (-0.3 to 0.3).

    Returns:
        Shifted 1D float32 numpy array.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == 0:
        return arr

    shift_samples = int(round(shift_fraction * len(arr)))
    if shift_samples == 0:
        return arr

    res = np.zeros_like(arr)
    if shift_samples > 0:
        res[shift_samples:] = arr[:-shift_samples]
    else:
        abs_shift = abs(shift_samples)
        res[: len(arr) - abs_shift] = arr[abs_shift:]

    return res.astype(np.float32)


def pitch_shift(audio: np.ndarray, sample_rate: int, n_semitones: float = 2.0) -> np.ndarray:
    """Shift pitch up or down by n_semitones while preserving length.

    Args:
        audio: 1D float32 numpy array.
        sample_rate: Audio sampling rate in Hz.
        n_semitones: Pitch shift in semitones (e.g. -3.0 to +3.0).

    Returns:
        Pitch-shifted 1D float32 numpy array.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == 0 or n_semitones == 0:
        return arr

    factor = 2.0 ** (n_semitones / 12.0)
    new_len = max(1, int(round(len(arr) / factor)))

    # Resample to alter pitch
    resampled = signal.resample(arr, new_len)
    # Resample back to original length to maintain exact window duration
    output = signal.resample(resampled, len(arr))
    return np.clip(output, -1.0, 1.0).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Speed up or slow down audio duration by rate factor.

    Args:
        audio: 1D float32 numpy array.
        rate: Speed factor (e.g. 0.8 to 1.25).

    Returns:
        Time-stretched 1D float32 numpy array.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == 0 or rate == 1.0:
        return arr

    target_len = max(1, int(round(len(arr) / rate)))
    stretched = signal.resample(arr, target_len)
    return np.clip(stretched, -1.0, 1.0).astype(np.float32)


def adjust_volume(audio: np.ndarray, gain_db: float = 3.0) -> np.ndarray:
    """Scale audio amplitude by gain in decibels.

    Args:
        audio: 1D float32 numpy array.
        gain_db: Gain adjustment in dB (e.g. -6.0 to +6.0).

    Returns:
        Volume-adjusted 1D float32 numpy array.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == 0 or gain_db == 0:
        return arr

    scale = 10.0 ** (gain_db / 20.0)
    scaled = arr * scale
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def add_reverberation(
    audio: np.ndarray,
    sample_rate: int,
    delay_ms: float = 30.0,
    decay: float = 0.4,
) -> np.ndarray:
    """Apply synthetic room impulse response reverberation filter.

    Args:
        audio: 1D float32 numpy array.
        sample_rate: Audio sampling rate in Hz.
        delay_ms: Delay of primary reflection in milliseconds.
        decay: Decay attenuation factor of reflections (0.1 to 0.6).

    Returns:
        Reverberated 1D float32 numpy array.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == 0:
        return arr

    delay_samples = max(1, int(round((delay_ms / 1000.0) * sample_rate)))
    ir_len = delay_samples * 4
    ir = np.zeros(ir_len, dtype=np.float32)
    ir[0] = 1.0

    # Create primary and secondary reflections with exponential decay
    for tap in range(1, 4):
        idx = tap * delay_samples
        if idx < ir_len:
            ir[idx] = (decay ** tap) * random.choice([1.0, -1.0])

    reverb = signal.fftconvolve(arr, ir, mode="full")[: len(arr)]
    # Normalize peak amplitude after convolution
    peak = np.max(np.abs(reverb))
    if peak > 0:
        reverb = (reverb / peak) * max(0.9, np.max(np.abs(arr)))

    return np.clip(reverb, -1.0, 1.0).astype(np.float32)


# Available augmentation functions dictionary
AUGMENTATION_REGISTRY = {
    "noise": lambda a, sr: add_noise(a, snr_db=random.uniform(10.0, 20.0)),
    "shift": lambda a, sr: time_shift(a, shift_fraction=random.uniform(-0.2, 0.2)),
    "pitch": lambda a, sr: pitch_shift(a, sr, n_semitones=random.uniform(-2.5, 2.5)),
    "stretch": lambda a, sr: time_stretch(a, rate=random.uniform(0.85, 1.15)),
    "volume": lambda a, sr: adjust_volume(a, gain_db=random.uniform(-5.0, 5.0)),
    "reverb": lambda a, sr: add_reverberation(a, sr, delay_ms=random.uniform(20.0, 45.0)),
}


# ---------------------------------------------------------------------------
# Dataset Helper Functions
# ---------------------------------------------------------------------------

def resolve_training_source_file(row: dict) -> Optional[Path]:
    """Locate source audio file for a training split row."""
    cls_name = row.get("class", "")
    filename = row.get("filename", "")

    # Try train split directory first (audio_dataset/train/<class>/<filename>)
    p1 = REPO_ROOT / "audio_dataset" / "train" / cls_name / filename
    if p1.exists():
        return p1

    # Fallback to preprocessed directory (audio_dataset/preprocessed/train/<class>/<filename>)
    p2 = REPO_ROOT / "audio_dataset" / "preprocessed" / "train" / cls_name / filename
    if p2.exists():
        return p2

    # Fallback to raw train directory (audio_dataset/raw/train/<class>/<filename>)
    p3 = REPO_ROOT / "audio_dataset" / "raw" / "train" / cls_name / filename
    if p3.exists():
        return p3

    return None


def apply_augmentation(
    audio: np.ndarray,
    sample_rate: int,
    aug_type: str,
) -> Tuple[np.ndarray, str]:
    """Apply specific augmentation by name.

    Returns:
        (augmented_audio_array, aug_name_tag)
    """
    if aug_type in AUGMENTATION_REGISTRY:
        func = AUGMENTATION_REGISTRY[aug_type]
        return func(audio, sample_rate), aug_type

    # Default fallback to noise
    return add_noise(audio), "noise"


def augment_dataset(
    index_csv: Path = DEFAULT_INDEX_CSV,
    output_dir: Path = DEFAULT_AUGMENTED_DIR,
    num_aug_per_file: int = 1,
    limit: Optional[int] = None,
    seed: int = 42,
) -> int:
    """Apply data augmentations to training split audio files ONLY.

    Args:
        index_csv: Path to dataset_index.csv.
        output_dir: Path to write augmented audio WAV files.
        num_aug_per_file: Number of augmented variants per training file.
        limit: Optional maximum number of training files to process.
        seed: Random seed for reproducible augmentations.

    Returns:
        Total number of new augmented files created and appended.
    """
    random.seed(seed)
    np.random.seed(seed)

    if not index_csv.exists():
        print(f"ERROR: Dataset index CSV not found: {index_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading dataset index from {index_csv} ...")
    with index_csv.open(encoding="utf-8", newline="") as fh:
        all_rows = list(csv.DictReader(fh))

    # CRITICAL RULE: Filter ONLY training split files. NEVER touch val or test splits!
    train_rows = [
        r for r in all_rows
        if r.get("final_split") == "train" or (not r.get("final_split") and r.get("source_split") == "train")
    ]
    # Filter out any already-augmented files if re-running
    train_rows = [r for r in train_rows if r.get("original_or_augmented") != "augmented"]

    if limit is not None and limit > 0:
        train_rows = train_rows[:limit]

    total_train = len(train_rows)
    print(
        f"Found {total_train} original training files for augmentation "
        f"({num_aug_per_file} variant(s) each) ..."
    )

    aug_types = list(AUGMENTATION_REGISTRY.keys())
    new_rows: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for idx, row in enumerate(train_rows, 1):
        src_path = resolve_training_source_file(row)
        if src_path is None or not src_path.exists():
            fail_count += 1
            continue

        try:
            audio, sr = sf.read(str(src_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: Could not read {src_path.name}: {exc}", file=sys.stderr)
            fail_count += 1
            continue

        cls_name = row.get("class", "unknown")
        cls_out_dir = output_dir / cls_name
        cls_out_dir.mkdir(parents=True, exist_ok=True)

        original_stem = Path(row["filename"]).stem

        for aug_idx in range(num_aug_per_file):
            selected_aug = random.choice(aug_types)
            aug_audio, tag = apply_augmentation(audio, sr, selected_aug)

            aug_filename = f"aug_{tag}_{original_stem}_{aug_idx+1}.wav"
            dst_path = cls_out_dir / aug_filename

            try:
                sf.write(str(dst_path), aug_audio, sr)
                duration = round(float(len(aug_audio)) / float(sr), 6)

                new_row = {
                    "audio_id": str(uuid.uuid4()),
                    "filename": aug_filename,
                    "class": cls_name,
                    "duration": duration,
                    "sampling_rate": sr,
                    "channels": 1 if aug_audio.ndim == 1 else aug_audio.shape[1],
                    "source": row.get("filename", ""),
                    "environment": row.get("environment", ""),
                    "original_or_augmented": "augmented",
                    "source_split": "train",
                    "final_split": "train",
                }
                new_rows.append(new_row)
                success_count += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: Failed to write {aug_filename}: {exc}", file=sys.stderr)
                fail_count += 1

        if idx % 1000 == 0 or idx == total_train:
            print(
                f"  Progress: {idx}/{total_train} original files processed "
                f"({success_count} augmented files generated)",
                flush=True,
            )

    # Append new augmented rows to dataset_index.csv
    if new_rows:
        print(f"Appending {len(new_rows)} augmented metadata rows to {index_csv} ...")
        with index_csv.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writerows(new_rows)

    print(
        f"\nAugmentation Complete: {success_count} augmented files created, "
        f"{fail_count} failed."
    )
    print(f"Output directory: {output_dir}")
    return success_count


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply controlled data augmentations to training split audio files ONLY."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=DEFAULT_INDEX_CSV,
        help="Path to dataset_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AUGMENTED_DIR,
        help="Output directory for augmented WAV files",
    )
    parser.add_argument(
        "--num-aug-per-file",
        type=int,
        default=1,
        help="Number of augmented variants per training file (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of training files to augment (for testing/dry-run)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    augment_dataset(
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        num_aug_per_file=args.num_aug_per_file,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
