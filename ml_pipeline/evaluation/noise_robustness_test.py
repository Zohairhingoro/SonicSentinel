"""
Noise Robustness Test for SonicSentinel AI.

Evaluates how the winning model's accuracy and per-class recall degrade when
synthetic background noise (from ml_pipeline/augmentation/augment.py add_noise())
is applied to a stratified sample of the held-out test split.

Noise severity levels tested (SNR in dB):
  - Clean       : no noise applied (baseline)
  - Mild noise  : SNR 20 dB  (light background activity)
  - Moderate    : SNR 10 dB  (busy environment)
  - Severe      : SNR  5 dB  (heavy noise masking)

Outputs:
  - Terminal    : per-condition accuracy / critical-class recall table
  - JSON        : docs/reports/noise_robustness_report.json
  - Markdown    : docs/reports/noise_robustness_report.md

Usage:
    python ml_pipeline/evaluation/noise_robustness_test.py
    python ml_pipeline/evaluation/noise_robustness_test.py --sample-size 200 --seed 7
    python ml_pipeline/evaluation/noise_robustness_test.py --model-path ml_service/model/artifacts/xgb_model.joblib
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score, classification_report

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_service.features.extract import extract_features          # noqa: E402
from ml_pipeline.augmentation.augment import add_noise            # noqa: E402

INDEX_CSV_PATH          = REPO_ROOT / "audio_dataset" / "metadata" / "dataset_index.csv"
ML_SERVICE_ARTIFACTS    = REPO_ROOT / "ml_service" / "model" / "artifacts"
COMPARISON_REPORT_PATH  = REPO_ROOT / "ml_pipeline" / "comparison_report" / "comparison_report.json"
DEFAULT_MODEL_PATH      = ML_SERVICE_ARTIFACTS / "xgb_model.joblib"
REPORTS_DIR             = REPO_ROOT / "docs" / "reports"

# Each tuple: (condition_label, snr_db | None)
# None means clean (base) no noise applied.
NOISE_CONDITIONS: List[Tuple[str, Optional[float]]] = [
    ("Clean (no noise)",  None),
    ("Mild  (SNR 20 dB)", 20.0),
    ("Moderate (SNR 10 dB)", 10.0),
    ("Severe (SNR  5 dB)",   5.0),
]

# Canonical 5 critical classes (AGENTS.md Sec.4, SRS Sec.1.7)
CRITICAL_CLASSES: List[Tuple[str, str]] = [
    ("gunshot",               "Gunshot"),
    ("glass_breaking",        "Glass Breaking"),
    ("panic_scream",          "Panic Scream"),
    ("aggression",            "Aggression / Violent Conflict"),
    ("person_asking_for_help","Person Asking for Help"),
]

RECALL_TARGET  = 0.85
DEFAULT_SAMPLE = 300   # stratified across all 10 classes

# ANSI colour helpers
_G = "\033[92m"
_R = "\033[91m"
_Y = "\033[93m"
_B = "\033[1m"
_X = "\033[0m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_X}" if sys.stdout.isatty() else text


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: Optional[Path] = None,
) -> Tuple[Any, str, str, List[str], Any]:
    """Load winning model artifact from disk.

    Returns:
        (model, model_name, model_version, classes, label_encoder)
    """
    path = model_path
    model_name    = "XGBoost"
    model_version = "xgb_v1.0"

    if path is None:
        if COMPARISON_REPORT_PATH.exists():
            try:
                with COMPARISON_REPORT_PATH.open(encoding="utf-8") as fh:
                    comp = json.load(fh)
                winning = comp.get("winning_model") or {}
                model_name    = winning.get("name",    model_name)
                model_version = winning.get("version", model_version)
            except Exception:
                pass

        slug = (
            "xgb" if "xgb" in model_name.lower() else
            "rf"  if ("rf" in model_name.lower() or "random" in model_name.lower()) else
            "svm"
        )
        candidates = [
            ML_SERVICE_ARTIFACTS / f"{slug}_model.joblib",
            DEFAULT_MODEL_PATH,
        ]
        for p in candidates:
            if p.exists():
                path = p
                break

    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found. Checked default path {DEFAULT_MODEL_PATH}"
        )

    print(f"Loading model artifact from {path.relative_to(REPO_ROOT)} ...")
    artifact      = joblib.load(path)
    model         = artifact.get("model")
    classes       = [str(c) for c in artifact.get("classes", [])]
    model_version = artifact.get("model_version", model_version)
    le            = artifact.get("label_encoder")
    return model, model_name, model_version, classes, le


# ---------------------------------------------------------------------------
# Test-split sampling
# ---------------------------------------------------------------------------

def resolve_audio_path(row: Dict[str, str]) -> Optional[Path]:
    """Return the first existing file path for a dataset_index row."""
    fn  = row.get("filename", "")
    cls = row.get("class", "")
    for base in (
        REPO_ROOT / "audio_dataset" / "test"  / cls / fn,
        REPO_ROOT / "audio_dataset" / "raw"   / "test" / cls / fn,
        REPO_ROOT / "audio_dataset" / "preprocessed" / "test" / cls / fn,
    ):
        if base.exists():
            return base
    return None


def load_stratified_sample(
    n_total: int,
    seed: int,
) -> Tuple[List[Path], List[str]]:
    """
    Draw a stratified random sample of n_total clips from the test split,
    keeping class proportions roughly equal.

    Returns:
        (file_paths, true_labels)  (base) parallel lists, only files that resolve.
    """
    if not INDEX_CSV_PATH.exists():
        raise FileNotFoundError(f"dataset_index.csv not found at {INDEX_CSV_PATH}")

    with INDEX_CSV_PATH.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("final_split") == "test"]

    if not rows:
        raise ValueError("No rows with final_split == 'test' in dataset_index.csv")

    rng = np.random.default_rng(seed)

    # Group by class
    by_class: Dict[str, List[Dict]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)

    n_classes   = len(by_class)
    per_class   = max(1, n_total // n_classes)
    file_paths: List[Path]  = []
    labels:     List[str]   = []

    for cls, cls_rows in sorted(by_class.items()):
        shuffled = rng.permutation(len(cls_rows)).tolist()
        drawn    = [cls_rows[i] for i in shuffled[:per_class]]
        for row in drawn:
            p = resolve_audio_path(row)
            if p is not None:
                file_paths.append(p)
                labels.append(cls)

    print(
        f"Stratified sample: {len(file_paths)} clips drawn from test split "
        f"({n_classes} classes, ~{per_class} per class, seed={seed})"
    )
    return file_paths, labels


# ---------------------------------------------------------------------------
# Per-condition feature extraction
# ---------------------------------------------------------------------------

def _extract_one(
    path: Path,
    snr_db: Optional[float],
) -> Optional[np.ndarray]:
    """Read audio, optionally apply noise, extract feature vector."""
    try:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        if len(audio) == 0:
            return None
        if snr_db is not None:
            audio = add_noise(audio, snr_db=snr_db)
        res = extract_features(audio, sr=sr)
        return res["features_vector"]
    except Exception:
        return None


def build_feature_matrix(
    file_paths: List[Path],
    labels:     List[str],
    snr_db:     Optional[float],
    n_workers:  int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply noise at snr_db (or none if None), extract features for every clip.

    Returns:
        (X, y_true) (base) arrays aligned to successful extractions only.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(lambda p: _extract_one(p, snr_db), file_paths))

    X_list, y_list = [], []
    for feat, label in zip(results, labels):
        if feat is not None:
            X_list.append(feat)
            y_list.append(label)

    return np.stack(X_list, axis=0).astype(np.float32), np.array(y_list, dtype=str)


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(
    model:   Any,
    le:      Any,
    classes: List[str],
    X:       np.ndarray,
    y_true:  np.ndarray,
    label:   str,
) -> Dict[str, Any]:
    """Run model inference and compute metrics for one noise condition."""
    t0        = time.time()
    preds_raw = model.predict(X)
    elapsed   = time.time() - t0

    if le is not None and isinstance(preds_raw[0], (int, np.integer)):
        y_pred = le.inverse_transform(preds_raw)
    else:
        y_pred = preds_raw

    acc      = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report   = classification_report(
        y_true, y_pred, labels=classes,
        output_dict=True, zero_division=0,
    )

    crit: Dict[str, Any] = {}
    for slug, human_label in CRITICAL_CLASSES:
        m      = report.get(slug, {})
        recall = float(m.get("recall", 0.0))
        crit[slug] = {
            "label":   human_label,
            "recall":  round(recall, 4),
            "passed":  recall >= RECALL_TARGET,
        }

    return {
        "condition":      label,
        "n_samples":      len(X),
        "accuracy":       round(acc, 4),
        "macro_f1":       round(macro_f1, 4),
        "inference_ms":   round(elapsed * 1000 / max(len(X), 1), 3),
        "critical_recall": crit,
        "per_class":      {
            cls: {
                "precision": round(float(report.get(cls, {}).get("precision", 0.0)), 4),
                "recall":    round(float(report.get(cls, {}).get("recall",    0.0)), 4),
                "f1":        round(float(report.get(cls, {}).get("f1-score",  0.0)), 4),
                "support":   int(report.get(cls, {}).get("support", 0)),
            }
            for cls in classes
        },
    }


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def render_terminal(
    results:      List[Dict[str, Any]],
    model_name:   str,
    model_version:str,
) -> None:
    print("\n" + "=" * 90)
    print(f"  {_c('SONICSENTINEL AI (base) NOISE ROBUSTNESS EVALUATION', _B)}")
    print(f"  Model: {model_name} ({model_version})")
    print("=" * 90)

    # -- Summary: accuracy / macro-F1 across conditions --
    print(f"\n  {'Condition':<26}{'Samples':>8}{'Accuracy':>10}{'Macro F1':>10}  "
          f"Accuracy delta vs Clean")
    print("  " + "-" * 72)

    clean_acc = results[0]["accuracy"] if results else 0.0
    for res in results:
        delta     = res["accuracy"] - clean_acc
        delta_str = f"{'(base)':>8}" if delta == 0 else f"{delta*100:>+8.2f}%"
        colour    = _G if delta >= 0 else _R
        print(
            f"  {res['condition']:<26}"
            f"{res['n_samples']:>8}"
            f"{res['accuracy']*100:>9.2f}%"
            f"{res['macro_f1']:>10.4f}"
            f"  {_c(delta_str, colour)}"
        )

    # -- Critical-class recall per condition --
    print(f"\n  {'Critical Class Recall by Noise Condition (Target >= 85%)'}")
    print("  " + "-" * 86)

    col_labels = [r["condition"] for r in results]
    hdr = f"  {'Sound Category':<32}" + "".join(f"{c:>17}" for c in col_labels)
    print(hdr)
    print("  " + "-" * 86)

    for slug, human_label in CRITICAL_CLASSES:
        row_str = f"  {human_label:<32}"
        for res in results:
            rec    = res["critical_recall"][slug]["recall"]
            passed = res["critical_recall"][slug]["passed"]
            cell   = f"{rec*100:.1f}%"
            row_str += f"{_c(cell, _G if passed else _R):>17}"
        print(row_str)

    print("  " + "-" * 86)
    print()

    # -- Per-class recall degradation table (clean -> severe) --
    if len(results) >= 2:
        clean  = results[0]
        severe = results[-1]
        print(f"  {'Recall Degradation: Clean -> Severe (SNR 5 dB)'}")
        print(f"  {'Sound Category':<32}{'Clean Recall':>14}{'Severe Recall':>14}{'delta Recall':>10}")
        print("  " + "-" * 72)
        all_classes = list(clean["per_class"].keys())
        for cls in sorted(all_classes):
            c_rec = clean["per_class"][cls]["recall"]
            s_rec = severe["per_class"][cls]["recall"]
            delta = s_rec - c_rec
            colour = _G if delta >= 0 else _R
            print(
                f"  {cls:<32}"
                f"{c_rec*100:>13.1f}%"
                f"{s_rec*100:>13.1f}%"
                f"  {_c(f'{delta*100:+.1f}%', colour)}"
            )
        print("  " + "-" * 72)

    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Save reports
# ---------------------------------------------------------------------------

def save_json(
    results:       List[Dict[str, Any]],
    model_name:    str,
    model_version: str,
    target_path:   Path,
) -> None:
    payload = {
        "report_type":    "noise_robustness_test",
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name":     model_name,
        "model_version":  model_version,
        "recall_target":  RECALL_TARGET,
        "noise_conditions": [
            {"label": label, "snr_db": snr}
            for label, snr in NOISE_CONDITIONS
        ],
        "results": results,
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_markdown(
    results:       List[Dict[str, Any]],
    model_name:    str,
    model_version: str,
    target_path:   Path,
) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines: List[str] = [
        "# SonicSentinel AI (base) Noise Robustness Evaluation",
        "",
        "> **Purpose**: Measure how accuracy and critical-class recall degrade "
        "when synthetic background noise is applied to the held-out test split.  ",
        f"> **Model**: `{model_name}` (`{model_version}`)  ",
        f"> **Report Generated**: {ts}  ",
        "",
        "---",
        "",
        "## 1. Overall Accuracy and Macro F1 by Noise Condition",
        "",
        "| Noise Condition | Samples | Accuracy | Macro F1 | delta Accuracy vs Clean |",
        "|:---|:---:|:---:|:---:|:---:|",
    ]

    clean_acc = results[0]["accuracy"] if results else 0.0
    for res in results:
        delta     = res["accuracy"] - clean_acc
        delta_str = "(base)" if delta == 0 else f"`{delta*100:+.2f}%`"
        lines.append(
            f"| {res['condition']} "
            f"| {res['n_samples']} "
            f"| **{res['accuracy']*100:.2f}%** "
            f"| {res['macro_f1']:.4f} "
            f"| {delta_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Critical-Class Recall by Noise Condition (Target >= 85%)",
        "",
    ])

    cond_cols = " | ".join(r["condition"] for r in results)
    sep_cols  = " | ".join(":---:" for _ in results)
    lines.append(f"| Sound Category | {cond_cols} |")
    lines.append(f"|:---|{sep_cols}|")

    for slug, human_label in CRITICAL_CLASSES:
        cells = []
        for res in results:
            rec    = res["critical_recall"][slug]["recall"]
            passed = res["critical_recall"][slug]["passed"]
            mark   = "OK" if passed else "FAIL"
            cells.append(f"{mark} {rec*100:.1f}%")
        lines.append(f"| **{human_label}** | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        f"> Recall target: **>= {RECALL_TARGET*100:.0f}%** per critical class "
        "(AGENTS.md Sec.4, SRS Sec.1.7)",
        "",
        "---",
        "",
        "## 3. Per-Class Recall: Clean vs Severe Noise",
        "",
        "| Sound Category | Clean Recall | Severe (5 dB) Recall | delta Recall |",
        "|:---|:---:|:---:|:---:|",
    ])

    if len(results) >= 2:
        clean  = results[0]
        severe = results[-1]
        for cls in sorted(clean["per_class"].keys()):
            c_rec = clean["per_class"][cls]["recall"]
            s_rec = severe["per_class"][cls]["recall"]
            delta = s_rec - c_rec
            lines.append(
                f"| `{cls}` | {c_rec*100:.1f}% | {s_rec*100:.1f}% | `{delta*100:+.1f}%` |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Interpretation",
        "",
        "- **Accuracy degradation** across noise levels indicates sensitivity to "
        "environmental noise; a large gap between Clean and Severe suggests the model "
        "relies heavily on spectral features that noise masks.",
        "- **Critical-class recall** may degrade non-uniformly (base) classes with "
        "distinctive impulsive signatures (Gunshot) tend to be more robust than "
        "sustained-sound classes (Panic Scream, Aggression).",
        "- Classes where recall falls below 85% under moderate noise should be "
        "targeted for noise-augmented retraining.",
        "",
        "---",
        "",
        "## 5. Related Reports",
        "",
        "- **Clean Test Evaluation**: "
        "[`docs/reports/test_evaluation_report.md`](file:///docs/reports/test_evaluation_report.md)",
        "- **Critical-Class Recall Audit**: "
        "[`docs/reports/critical_class_recall.md`](file:///docs/reports/critical_class_recall.md)",
        "- **JSON Machine-Readable**: "
        "[`docs/reports/noise_robustness_report.json`](file:///docs/reports/noise_robustness_report.json)",
        "",
    ])

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(
    model_path:  Optional[Path],
    sample_size: int,
    seed:        int,
) -> None:
    print("=" * 90)
    print("  SONICSENTINEL AI (base) NOISE ROBUSTNESS TEST PIPELINE")
    print("  Competition Theme: AcousticX Intelligence (TechWiz7 NextWave AI/ML)")
    print("=" * 90)

    model, model_name, model_version, classes, le = load_model(model_path)

    file_paths, labels = load_stratified_sample(sample_size, seed)

    n_workers = min(os.cpu_count() or 4, 8)
    results: List[Dict[str, Any]] = []

    for cond_label, snr_db in NOISE_CONDITIONS:
        noise_desc = "none" if snr_db is None else f"SNR {snr_db} dB"
        print(f"\n  Condition: {cond_label}  (noise: {noise_desc})")
        print(f"  Extracting features for {len(file_paths)} clips ...")

        t0 = time.time()
        X, y_true = build_feature_matrix(file_paths, labels, snr_db, n_workers)
        print(f"  Features ready ({len(X)} valid clips) in {time.time()-t0:.1f}s")

        res = evaluate_condition(model, le, classes, X, y_true, cond_label)
        results.append(res)

        crit_passing = sum(1 for m in res["critical_recall"].values() if m["passed"])
        print(
            f"  Accuracy: {res['accuracy']*100:.2f}%  |  "
            f"Macro F1: {res['macro_f1']:.4f}  |  "
            f"Critical classes passing: {crit_passing}/{len(CRITICAL_CLASSES)}"
        )

    render_terminal(results, model_name, model_version)

    json_path = REPORTS_DIR / "noise_robustness_report.json"
    save_json(results, model_name, model_version, json_path)
    print(f"Saved JSON report : {json_path.relative_to(REPO_ROOT)}")

    md_path = REPORTS_DIR / "noise_robustness_report.md"
    save_markdown(results, model_name, model_version, md_path)
    print(f"Saved Markdown    : {md_path.relative_to(REPO_ROOT)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply synthetic noise at multiple SNR levels to test-split clips, "
            "re-run the winning model, and report accuracy/recall degradation."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained model .joblib artifact (default: auto-detect from comparison report)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"Total number of test clips to sample (stratified, default: {DEFAULT_SAMPLE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42)",
    )
    args = parser.parse_args()
    run(
        model_path=args.model_path,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
