"""
Offline preprocessing pipeline script for SonicSentinel AI.

Imports and reuses the core audio preprocessing engine from
ml_service.audio.preprocessing (BE-M3) to eliminate logic duplication
between inference and training.

Reads audio_dataset/metadata/dataset_index.csv, applies audio preprocessing
(mono conversion, resampling to 22kHz, noise reduction, silence trimming,
amplitude normalization) to all training split audio files, and writes the
preprocessed output to audio_dataset/preprocessed/.

Usage:
    python ml_pipeline/preprocessing/preprocess.py
    python ml_pipeline/preprocessing/preprocess.py --split train --limit 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

import soundfile as sf

# ---------------------------------------------------------------------------
# Path setup & Module reuse from ml_service
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import shared preprocessing logic directly from ml_service (BE-M3)
from ml_service.audio.preprocessing import (  # noqa: E402
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

# Default path constants
DEFAULT_INDEX_CSV = REPO_ROOT / "audio_dataset" / "metadata" / "dataset_index.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "audio_dataset" / "preprocessed"


# ---------------------------------------------------------------------------
# Preprocessing Worker Functions
# ---------------------------------------------------------------------------

def process_single_file(
    src_path: Path,
    dst_path: Path,
    target_sample_rate: int = 22050,
    target_peak: float = 0.95,
    trim_silence_flag: bool = True,
    reduce_noise_flag: bool = True,
) -> Optional[PreprocessedAudio]:
    """Process a single audio file and write the result to dst_path.

    Args:
        src_path: Absolute path to the source audio file.
        dst_path: Absolute path to write the preprocessed audio WAV.
        target_sample_rate: Target sample rate in Hz (default 22050).
        target_peak: Peak amplitude scaling target (default 0.95).
        trim_silence_flag: Enable/disable silence trimming.
        reduce_noise_flag: Enable/disable noise reduction.

    Returns:
        PreprocessedAudio object if successful, None on error.
    """
    try:
        data, sample_rate = sf.read(str(src_path), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: Could not read {src_path.name}: {exc}", file=sys.stderr)
        return None

    try:
        processed = preprocess_audio(
            audio=data,
            sample_rate=sample_rate,
            target_sample_rate=target_sample_rate,
            target_peak=target_peak,
            trim_silence_flag=trim_silence_flag,
            reduce_noise_flag=reduce_noise_flag,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: Preprocessing failed for {src_path.name}: {exc}", file=sys.stderr)
        return None

    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst_path), processed.data, processed.sample_rate)
        return processed
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: Failed to write {dst_path.name}: {exc}", file=sys.stderr)
        return None


def resolve_source_file(row: dict) -> Path:
    """Locate source file on disk using dataset split layout."""
    split_name = row.get("final_split") or row.get("source_split") or "train"
    cls_name = row.get("class", "")
    filename = row.get("filename", "")

    # Try split directory first (e.g. audio_dataset/train/<class>/<filename>)
    split_path = REPO_ROOT / "audio_dataset" / split_name / cls_name / filename
    if split_path.exists():
        return split_path

    # Fallback to raw directory (e.g. audio_dataset/raw/train/<class>/<filename>)
    source_split = row.get("source_split", "train")
    raw_path = REPO_ROOT / "audio_dataset" / "raw" / source_split / cls_name / filename
    if raw_path.exists():
        return raw_path

    return split_path


def preprocess_dataset(
    index_csv: Path = DEFAULT_INDEX_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_split: str = "train",
    limit: Optional[int] = None,
) -> int:
    """Preprocess audio files from dataset_index.csv matching target_split.

    Args:
        index_csv: Path to dataset_index.csv.
        output_dir: Path to write output preprocessed files.
        target_split: Split to filter ('train', 'val', 'test', or 'all').
        limit: Optional maximum number of files to process.

    Returns:
        Number of files successfully preprocessed.
    """
    if not index_csv.exists():
        print(f"ERROR: Dataset index CSV not found: {index_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading dataset index from {index_csv} ...")
    with index_csv.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Filter rows based on target split
    if target_split.lower() != "all":
        rows = [
            r for r in rows
            if r.get("final_split", r.get("source_split")) == target_split
        ]

    if limit is not None and limit > 0:
        rows = rows[:limit]

    total = len(rows)
    print(f"Processing {total} file(s) for split '{target_split}' ...")

    success_count = 0
    fail_count = 0

    for idx, row in enumerate(rows, 1):
        src_path = resolve_source_file(row)
        cls_name = row.get("class", "unknown")
        split_name = row.get("final_split", row.get("source_split", "train"))
        filename = row.get("filename", "")

        # Write output to output_dir / split_name / cls_name / filename
        dst_path = output_dir / split_name / cls_name / filename

        res = process_single_file(src_path, dst_path)
        if res is not None:
            success_count += 1
        else:
            fail_count += 1

        if idx % 500 == 0 or idx == total:
            print(f"  Progress: {idx}/{total} processed ({success_count} succeeded, {fail_count} failed)", flush=True)

    print(
        f"\nPreprocessing Complete: {success_count} succeeded, "
        f"{fail_count} failed out of {total} total.",
        flush=True,
    )
    print(f"Output directory: {output_dir}", flush=True)
    return success_count


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess audio files listed in dataset_index.csv using shared ml_service logic."
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
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write preprocessed audio files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Dataset split to preprocess (default: train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing/dry-run)",
    )

    args = parser.parse_args()
    preprocess_dataset(
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        target_split=args.split,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
