"""
Scan audio_dataset/raw/train/ and audio_dataset/raw/test/ and write
audio_dataset/metadata/dataset_index.csv.

Fail loudly if any class subfolder name does not match a canonical slug
defined in config/classes.json.  No silent skipping or guessing.

Run from the repository root:
    python ml_pipeline/data/build_dataset_index.py
"""

from __future__ import annotations

import csv
import json
import sys
import uuid
from pathlib import Path

import soundfile as sf

# ---------------------------------------------------------------------------
# Paths (all resolved relative to the repository root)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES_JSON = REPO_ROOT / "config" / "classes.json"
RAW_TRAIN_DIR = REPO_ROOT / "audio_dataset" / "raw" / "train"
RAW_TEST_DIR = REPO_ROOT / "audio_dataset" / "raw" / "test"
METADATA_DIR = REPO_ROOT / "audio_dataset" / "metadata"
OUTPUT_CSV = METADATA_DIR / "dataset_index.csv"

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

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
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category_to_slug(category: str) -> str:
    """
    Convert a human-readable SRS category name to its canonical slug.

    The mapping is the one declared in AGENTS.md section 2 and must stay in
    sync with the folder names used under audio_dataset/.

    'Aggression or Violent Conflict' is special-cased; the full phrase would
    be too long and the SRS itself uses 'aggression' as the folder slug.
    """
    special: dict[str, str] = {
        "Aggression or Violent Conflict": "aggression",
    }
    if category in special:
        return special[category]
    return category.lower().replace(" ", "_")


def _load_canonical_slugs() -> set[str]:
    with CLASSES_JSON.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return {_category_to_slug(cat) for cat in data["categories"]}


def _audio_metadata(path: Path) -> tuple[float, int, int] | None:
    """
    Return (duration_seconds, sample_rate, channels) for a supported audio
    file using soundfile.  Returns None if the file cannot be decoded.

    soundfile handles WAV/FLAC/OGG natively; MP3 and M4A require an ffmpeg
    backend (SoundFile >= 0.12 with libsndfile >= 1.1, or pysoundfile + ffmpeg).
    A clear warning is printed for files that cannot be read.
    """
    try:
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        return round(duration, 6), info.samplerate, info.channels
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not read '{path.name}': {exc}", file=sys.stderr)
        return None


def _infer_environment(folder_name: str) -> str:
    """
    Attempt to infer an environment tag from a class folder name.
    Returns a best-effort string or empty string for manual fill.

    Only slug substrings with an obvious environment implication are mapped;
    everything else is left blank for a human to fill in.
    """
    hints: dict[str, str] = {
        "machinery": "industrial",
        "vehicle": "outdoor/traffic",
        "animal": "outdoor/nature",
        "alarm": "indoor/outdoor",
        "glass": "indoor",
        "gunshot": "outdoor/indoor",
        "panic": "indoor/outdoor",
        "aggression": "indoor/outdoor",
        "help": "indoor/outdoor",
    }
    for keyword, env in hints.items():
        if keyword in folder_name:
            return env
    return ""


def _validate_folders(
    split_dir: Path, canonical_slugs: set[str]
) -> tuple[list[Path], list[str]]:
    """
    Return (valid_class_dirs, unmatched_folder_names).

    Only immediate child directories of split_dir are considered.
    """
    valid: list[Path] = []
    unmatched: list[str] = []

    if not split_dir.exists():
        print(
            f"WARNING: split directory does not exist, skipping: {split_dir}",
            file=sys.stderr,
        )
        return valid, unmatched

    for child in sorted(split_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in canonical_slugs:
            valid.append(child)
        else:
            unmatched.append(child.name)

    return valid, unmatched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_index() -> None:
    canonical_slugs = _load_canonical_slugs()

    valid_train, bad_train = _validate_folders(RAW_TRAIN_DIR, canonical_slugs)
    valid_test, bad_test = _validate_folders(RAW_TEST_DIR, canonical_slugs)

    all_bad: list[str] = []
    if bad_train:
        all_bad.extend(f"  raw/train/{name}" for name in bad_train)
    if bad_test:
        all_bad.extend(f"  raw/test/{name}" for name in bad_test)

    if all_bad:
        print(
            "ERROR: The following class subfolders do not match any canonical slug "
            "in config/classes.json.\n"
            "Rename the folders to match the table in AGENTS.md section 2 before "
            "re-running.\n"
            "\nUnmatched folders:",
            file=sys.stderr,
        )
        for entry in all_bad:
            print(entry, file=sys.stderr)
        print(
            f"\nCanonical slugs expected: {sorted(canonical_slugs)}",
            file=sys.stderr,
        )
        sys.exit(1)

    rows: list[dict] = []

    def _collect(class_dirs: list[Path], source_split: str) -> None:
        for class_dir in class_dirs:
            slug = class_dir.name
            environment = _infer_environment(slug)
            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                result = _audio_metadata(file_path)
                if result is None:
                    continue
                duration, sample_rate, channels = result
                rows.append(
                    {
                        "audio_id": str(uuid.uuid4()),
                        "filename": file_path.name,
                        "class": slug,
                        "duration": duration,
                        "sampling_rate": sample_rate,
                        "channels": channels,
                        "source": "",
                        "environment": environment,
                        "original_or_augmented": "original",
                        "source_split": source_split,
                    }
                )

    _collect(valid_train, "train")
    _collect(valid_test, "test")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Done. {len(rows)} audio files indexed -> {OUTPUT_CSV.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    build_index()
