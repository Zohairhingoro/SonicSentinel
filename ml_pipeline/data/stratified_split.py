"""
Read audio_dataset/metadata/dataset_index.csv and assign each row a
final_split column (train / val / test) per AGENTS.md section 2 split math:

  * source_split="test"  -> final_split="test"  (frozen, never reshuffled)
  * source_split="train" -> stratified carve per class:
        ~17.65% (15/85) -> final_split="val"
        remainder        -> final_split="train"

  Any rows that share a common parent-recording ID (inferred from the
  filename stem before the last numeric or segment suffix) are kept
  together in the same final split to avoid leakage.

After assigning splits the script:
  1. Writes final_split back to dataset_index.csv (in-place update).
  2. Physically copies files into audio_dataset/{train,val,test}/<class>/
     (test/ ends up an exact copy of raw/test/).
  3. Logs per-class train/val/test counts so an uneven carve is visible.

Run from the repository root:
    python ml_pipeline/data/stratified_split.py
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_TRAIN_DIR = REPO_ROOT / "audio_dataset" / "raw" / "train"
RAW_TEST_DIR = REPO_ROOT / "audio_dataset" / "raw" / "test"
SPLIT_DIRS = {
    "train": REPO_ROOT / "audio_dataset" / "train",
    "val":   REPO_ROOT / "audio_dataset" / "val",
    "test":  REPO_ROOT / "audio_dataset" / "test",
}
INDEX_CSV = REPO_ROOT / "audio_dataset" / "metadata" / "dataset_index.csv"

# Target fraction of source-train rows per class to carve into val.
# 15/85 ≈ 0.17647 delivers the SRS 70/15/15 overall.
VAL_FRACTION = 15 / 85

# Regex that strips a trailing numeric segment suffix from a stem so that
# files like "recording_001.wav" and "recording_002.wav" are grouped under
# the same parent ID "recording".  Plain hex-hash names (no numeric suffix)
# are their own group.
_SEGMENT_RE = re.compile(r"^(.+?)[-_](\d+)$")


def _recording_id(filename: str) -> str:
    """
    Infer a parent-recording ID from a filename stem.

    'siren_003.wav' -> 'siren'
    'a1b2c3d4.wav'  -> 'a1b2c3d4'  (no suffix stripped — own group)
    """
    stem = Path(filename).stem
    m = _SEGMENT_RE.match(stem)
    return m.group(1) if m else stem


# ---------------------------------------------------------------------------
# Stratified carve
# ---------------------------------------------------------------------------

def _assign_val_flags(
    rows: list[dict],
    val_fraction: float,
) -> list[dict]:
    """
    For a single class's source-train rows, group by recording ID then
    assign roughly val_fraction of rows to val, keeping all rows that
    share a recording ID together.

    Returns the same list with a '_final_split' key set on every row.
    """
    # Group rows by recording ID, preserving insertion order.
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_recording_id(row["filename"])].append(row)

    group_list = list(groups.values())
    target_val = round(len(rows) * val_fraction)

    # Greedily fill val until we reach or just pass the target.
    # Groups are iterated in a deterministic order (sorted by recording ID)
    # so results are reproducible without a fixed random seed.
    sorted_groups = sorted(group_list, key=lambda g: _recording_id(g[0]["filename"]))

    val_rows: list[dict] = []
    train_rows: list[dict] = []

    for grp in sorted_groups:
        if len(val_rows) < target_val:
            val_rows.extend(grp)
        else:
            train_rows.extend(grp)

    for row in val_rows:
        row["final_split"] = "val"
    for row in train_rows:
        row["final_split"] = "train"

    return val_rows + train_rows


def assign_splits(rows: list[dict]) -> list[dict]:
    """
    Assign final_split to every row and return the updated list.
    """
    # Freeze test rows immediately.
    test_rows = [r for r in rows if r["source_split"] == "test"]
    for r in test_rows:
        r["final_split"] = "test"

    # Group source-train rows by class and carve val per class.
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["source_split"] == "train":
            by_class[r["class"]].append(r)

    carved: list[dict] = []
    for cls, cls_rows in sorted(by_class.items()):
        carved.extend(_assign_val_flags(cls_rows, VAL_FRACTION))

    return test_rows + carved


# ---------------------------------------------------------------------------
# File copy
# ---------------------------------------------------------------------------

def _source_path(row: dict) -> Path:
    raw_base = RAW_TRAIN_DIR if row["source_split"] == "train" else RAW_TEST_DIR
    return raw_base / row["class"] / row["filename"]


def copy_files(rows: list[dict]) -> None:
    """
    Copy every file into audio_dataset/<final_split>/<class>/<filename>.
    Skips files that are already in place (idempotent).
    Prints a warning for any source file that does not exist.
    """
    for split_dir in SPLIT_DIRS.values():
        split_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    copied = 0
    skipped = 0

    for row in rows:
        src = _source_path(row)
        dst_dir = SPLIT_DIRS[row["final_split"]] / row["class"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / row["filename"]

        if not src.exists():
            missing.append(str(src))
            continue

        if dst.exists():
            skipped += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

    print(f"  Files copied : {copied}")
    print(f"  Files already present (skipped): {skipped}")

    if missing:
        print(
            f"\n  WARNING: {len(missing)} source file(s) not found on disk "
            f"(indexed but missing):",
            file=sys.stderr,
        )
        for p in missing[:20]:
            print(f"    {p}", file=sys.stderr)
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more", file=sys.stderr)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_counts(rows: list[dict]) -> None:
    """
    Print per-class train/val/test counts in a table.
    Flag any class whose val fraction deviates more than 3 pp from the target.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for row in rows:
        counts[row["class"]][row["final_split"]] += 1

    header = f"{'class':<30}  {'train':>6}  {'val':>6}  {'test':>6}  {'val%':>6}  {'note'}"
    print()
    print(header)
    print("-" * len(header))

    for cls in sorted(counts):
        tr = counts[cls]["train"]
        va = counts[cls]["val"]
        te = counts[cls]["test"]
        source_train_total = tr + va
        val_pct = (va / source_train_total * 100) if source_train_total else 0.0
        target_pct = VAL_FRACTION / (1 + VAL_FRACTION) * 100  # ≈ 15.0%
        flag = "  *** UNEVEN CARVE ***" if abs(val_pct - target_pct) > 3 else ""
        print(f"{cls:<30}  {tr:>6}  {va:>6}  {te:>6}  {val_pct:>5.1f}%{flag}")

    print()
    totals = {s: sum(counts[c][s] for c in counts) for s in ("train", "val", "test")}
    grand = sum(totals.values())
    print(
        f"{'TOTAL':<30}  {totals['train']:>6}  {totals['val']:>6}  "
        f"{totals['test']:>6}  "
        f"(train {totals['train']/grand*100:.1f}% / "
        f"val {totals['val']/grand*100:.1f}% / "
        f"test {totals['test']/grand*100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

ORIGINAL_COLUMNS = [
    "audio_id", "filename", "class", "duration", "sampling_rate",
    "channels", "source", "environment", "original_or_augmented", "source_split",
]
OUTPUT_COLUMNS = ORIGINAL_COLUMNS + ["final_split"]


def read_index() -> list[dict]:
    with INDEX_CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_index(rows: list[dict]) -> None:
    with INDEX_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INDEX_CSV.exists():
        print(
            f"ERROR: {INDEX_CSV} not found. "
            "Run ml_pipeline/data/build_dataset_index.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Reading {INDEX_CSV.relative_to(REPO_ROOT)} ...")
    rows = read_index()
    print(f"  {len(rows)} rows loaded.")

    # Re-run guard: if final_split already present, warn but continue
    # (idempotent — copies are skipped, CSV is overwritten with same values).
    if rows and "final_split" in rows[0]:
        print("  NOTE: final_split column already present — re-running (idempotent).")

    print("Assigning final splits ...")
    rows = assign_splits(rows)

    print("Writing updated dataset_index.csv ...")
    write_index(rows)

    print("Copying files into split directories ...")
    copy_files(rows)

    print("\nPer-class split counts:")
    log_counts(rows)

    print("Done.")


if __name__ == "__main__":
    main()
