"""
Offline Acoustic Feature Extraction Module for SonicSentinel AI.

Imports and reuses the core feature extraction logic directly from
ml_service.features.extract (BE-M4) to prevent duplication between
offline training and live online microservice inference.

Extracts 8 mandatory feature types:
  1. MFCC (20 means + 20 stds)
  2. Mel Spectrogram summary
  3. Chroma STFT (12 means + 12 stds)
  4. Zero-Crossing Rate (mean + std)
  5. RMS Energy (mean + std)
  6. Spectral Centroid (mean + std)
  7. Spectral Bandwidth (mean + std)
  8. Spectral Roll-off (mean + std)

Processes all audio files in the training and validation splits (original + augmented),
and writes the extracted 74-dimensional feature matrix to a single features.csv file
with audio_id as the primary join key back to dataset_index.csv.

Usage:
    python ml_pipeline/features/extract_features.py
    python ml_pipeline/features/extract_features.py --splits train val --limit 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Path setup & Module reuse from ml_service
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import shared feature extraction logic directly from ml_service (BE-M4)
from ml_service.features.extract import extract_features  # noqa: E402

# Default path constants
DEFAULT_INDEX_CSV = REPO_ROOT / "audio_dataset" / "metadata" / "dataset_index.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "audio_dataset" / "metadata" / "features.csv"

# Construct feature CSV column header list (Audio ID + metadata + 74 feature dimensions)
FEATURE_COLUMNS = [
    "audio_id",
    "class",
    "final_split",
] + [
    f"mfcc_mean_{i+1}" for i in range(20)
] + [
    f"mfcc_std_{i+1}" for i in range(20)
] + [
    f"chroma_mean_{i+1}" for i in range(12)
] + [
    f"chroma_std_{i+1}" for i in range(12)
] + [
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


# ---------------------------------------------------------------------------
# Worker Functions
# ---------------------------------------------------------------------------

def resolve_audio_file(row: dict) -> Optional[Path]:
    """Locate target audio file on disk based on metadata row attributes."""
    filename = row.get("filename", "")
    cls_name = row.get("class", "")
    final_split = row.get("final_split") or row.get("source_split") or "train"
    source_split = row.get("source_split") or "train"
    is_augmented = row.get("original_or_augmented") == "augmented"

    # 1. Augmented audio directory
    if is_augmented:
        aug_path = REPO_ROOT / "audio_dataset" / "augmented" / cls_name / filename
        if aug_path.exists():
            return aug_path

    # 2. Preprocessed directory
    prep_path = REPO_ROOT / "audio_dataset" / "preprocessed" / final_split / cls_name / filename
    if prep_path.exists():
        return prep_path

    # 3. Clean split directory (train / val / test)
    split_path = REPO_ROOT / "audio_dataset" / final_split / cls_name / filename
    if split_path.exists():
        return split_path

    # 4. Raw source directory fallback
    raw_path = REPO_ROOT / "audio_dataset" / "raw" / source_split / cls_name / filename
    if raw_path.exists():
        return raw_path

    return None


def extract_features_for_row(row: dict) -> Optional[dict]:
    """Load audio file and extract 74-dimensional acoustic feature vector.

    Args:
        row: Metadata dictionary row from dataset_index.csv.

    Returns:
        Dictionary row matching FEATURE_COLUMNS, or None on read/extraction failure.
    """
    src_path = resolve_audio_file(row)
    if src_path is None or not src_path.exists():
        return None

    try:
        data, sr = sf.read(str(src_path), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: Could not read {src_path.name}: {exc}", file=sys.stderr)
        return None

    if data.ndim > 1:
        data = np.mean(data, axis=1).astype(np.float32)

    if len(data) == 0:
        return None

    try:
        res = extract_features(data, sr=sr)
        vec = res["features_vector"]
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: Feature extraction failed for {src_path.name}: {exc}", file=sys.stderr)
        return None

    if len(vec) != 74:
        print(
            f"  WARNING: Unexpected feature vector length {len(vec)} for {src_path.name}",
            file=sys.stderr,
        )
        return None

    # Construct feature row mapping audio_id -> 74 numeric values
    feature_row = {
        "audio_id": row["audio_id"],
        "class": row.get("class", ""),
        "final_split": row.get("final_split") or row.get("source_split") or "train",
    }

    # Add feature values rounded for clean numeric CSV storage
    for col_name, val in zip(FEATURE_COLUMNS[3:], vec):
        feature_row[col_name] = round(float(val), 6)

    return feature_row


def extract_all_features(
    index_csv: Path = DEFAULT_INDEX_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    splits: Sequence[str] = ("train", "val"),
    limit: Optional[int] = None,
) -> int:
    """Extract acoustic features for all matching dataset rows and write features.csv.

    Args:
        index_csv: Path to dataset_index.csv.
        output_csv: Path to write output features.csv.
        splits: Sequence of split names to extract ('train', 'val', 'test').
        limit: Optional maximum number of rows to process.

    Returns:
        Number of successfully extracted feature rows.
    """
    if not index_csv.exists():
        print(f"ERROR: Dataset index CSV not found: {index_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading dataset index from {index_csv} ...")
    with index_csv.open(encoding="utf-8", newline="") as fh:
        all_rows = list(csv.DictReader(fh))

    split_set = {s.lower() for s in splits}
    target_rows = [
        r for r in all_rows
        if (r.get("final_split") or r.get("source_split") or "").lower() in split_set
    ]

    if limit is not None and limit > 0:
        target_rows = target_rows[:limit]

    total = len(target_rows)
    print(
        f"Extracting acoustic features for {total} audio file(s) across splits "
        f"{list(splits)} (original + augmented) ..."
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    extracted_rows: list[dict] = []
    success_count = 0
    fail_count = 0

    for idx, row in enumerate(target_rows, 1):
        f_row = extract_features_for_row(row)
        if f_row is not None:
            extracted_rows.append(f_row)
            success_count += 1
        else:
            fail_count += 1

        if idx % 1000 == 0 or idx == total:
            print(
                f"  Progress: {idx}/{total} processed ({success_count} succeeded, {fail_count} failed)",
                flush=True,
            )

    print(f"Writing {len(extracted_rows)} feature vectors to {output_csv} ...", flush=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(extracted_rows)

    # Export fast binary .npz file alongside CSV for rapid ML training loading
    output_npz = output_csv.with_suffix(".npz")
    print(f"Exporting compressed binary feature matrix to {output_npz} ...", flush=True)
    audio_ids = np.array([r["audio_id"] for r in extracted_rows], dtype=str)
    classes = np.array([r["class"] for r in extracted_rows], dtype=str)
    final_splits = np.array([r["final_split"] for r in extracted_rows], dtype=str)
    feature_names = np.array(FEATURE_COLUMNS[3:], dtype=str)
    
    # Extract 2D float32 feature matrix (N, 74)
    X = np.array(
        [[r[col] for col in FEATURE_COLUMNS[3:]] for r in extracted_rows],
        dtype=np.float32,
    )

    np.savez_compressed(
        output_npz,
        audio_ids=audio_ids,
        X=X,
        classes=classes,
        splits=final_splits,
        feature_names=feature_names,
    )

    print(
        f"\nFeature Extraction Complete: {success_count} succeeded, {fail_count} failed "
        f"out of {total} total."
    )
    print(f"Output CSV: {output_csv}")
    print(f"Output NPZ: {output_npz}")
    return success_count


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract acoustic features for train and val splits using shared ml_service logic."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=DEFAULT_INDEX_CSV,
        help="Path to dataset_index.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to write features.csv",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to process (default: train val)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing/dry-run)",
    )

    args = parser.parse_args()
    extract_all_features(
        index_csv=args.index_csv,
        output_csv=args.output_csv,
        splits=args.splits,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
