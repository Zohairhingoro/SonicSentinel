"""
Support Vector Machine (SVM) Training & Hyperparameter Tuning Engine for SonicSentinel AI.

Reads extracted features from audio_dataset/metadata/features.npz (or features.csv),
performs feature scaling on the training split, conducts systematic hyperparameter
tuning using the validation split to select optimal kernel and regularization parameters,
trains a calibrated probability-enabled SVM model, and exports model artifacts and evaluation metrics.

Outputs:
  - Model artifact: ml_pipeline/training/results/svm_model.joblib
  - Export copy: ml_service/model/artifacts/svm_model.joblib
  - Evaluation results: ml_pipeline/training/results/svm_results.json

Usage:
    python ml_pipeline/training/train_svm.py
    python ml_pipeline/training/train_svm.py --fast-tune
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "features.npz"
DEFAULT_CSV_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "features.csv"
RESULTS_DIR = REPO_ROOT / "ml_pipeline" / "training" / "results"
ML_SERVICE_ARTIFACTS_DIR = REPO_ROOT / "ml_service" / "model" / "artifacts"

# Mandatory 5 Critical Classes defined in SRS §1.6 & AGENTS.md
CRITICAL_CLASSES = [
    "gunshot",
    "glass_breaking",
    "panic_scream",
    "aggression",
    "person_asking_for_help",
]


# ---------------------------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------------------------

def load_feature_dataset(
    npz_path: Path = DEFAULT_NPZ_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Load acoustic features, splits, and class labels from NPZ or fallback CSV.

    Returns:
        X_train, y_train, X_val, y_val, audio_ids_train, audio_ids_val, classes_list, feature_names
    """
    if npz_path.exists():
        print(f"Loading feature matrix from compressed archive {npz_path.relative_to(REPO_ROOT)} ...")
        data = np.load(npz_path)
        X = data["X"]
        y = data["classes"]
        splits = data["splits"]
        audio_ids = data["audio_ids"]
        feature_names = data["feature_names"].tolist()
    elif csv_path.exists():
        print(f"Loading feature matrix from CSV file {csv_path.relative_to(REPO_ROOT)} ...")
        import csv
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        audio_ids = np.array([r["audio_id"] for r in rows], dtype=str)
        y = np.array([r["class"] for r in rows], dtype=str)
        splits = np.array([r["final_split"] for r in rows], dtype=str)
        feature_names = [k for k in rows[0].keys() if k not in ("audio_id", "class", "final_split")]
        X = np.array([[float(r[col]) for col in feature_names] for r in rows], dtype=np.float32)
    else:
        print(f"ERROR: Feature dataset not found at {npz_path} or {csv_path}", file=sys.stderr)
        sys.exit(1)

    train_mask = (splits == "train")
    val_mask = (splits == "val")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    audio_ids_train = audio_ids[train_mask]
    audio_ids_val = audio_ids[val_mask]

    classes_list = sorted(list(set(y_train) | set(y_val)))

    print(f"Dataset Loaded:")
    print(f"  Training split  : {len(X_train)} samples across {len(classes_list)} classes")
    print(f"  Validation split: {len(X_val)} samples across {len(classes_list)} classes")
    print(f"  Feature count   : {X.shape[1]} dimensions")

    return X_train, y_train, X_val, y_val, audio_ids_train, audio_ids_val, classes_list, feature_names


# ---------------------------------------------------------------------------
# Systematic Hyperparameter Tuning
# ---------------------------------------------------------------------------

def tune_svm_hyperparameters(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    fast_tune: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Perform grid search over SVM kernel and regularization C parameters on validation split.

    Returns:
        (best_params_dict, search_history_list)
    """
    if fast_tune:
        param_grid = [
            {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            {"C": 10.0, "kernel": "rbf", "gamma": "scale"},
        ]
    else:
        param_grid = [
            {"C": 0.1, "kernel": "rbf", "gamma": "scale"},
            {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            {"C": 10.0, "kernel": "rbf", "gamma": "scale"},
            {"C": 1.0, "kernel": "linear", "gamma": "scale"},
            {"C": 10.0, "kernel": "linear", "gamma": "scale"},
        ]

    print(f"\nEvaluating {len(param_grid)} SVM hyperparameter configurations on Validation split ...")
    search_history: List[Dict[str, Any]] = []

    best_score = -1.0
    best_params = param_grid[0]

    for idx, params in enumerate(param_grid, 1):
        t0 = time.time()
        candidate = SVC(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
            random_state=42,
        )
        candidate.fit(X_train_scaled, y_train)
        elapsed = time.time() - t0

        val_preds = candidate.predict(X_val_scaled)
        val_acc = float(accuracy_score(y_val, val_preds))
        val_f1 = float(f1_score(y_val, val_preds, average="macro"))

        record = {
            "trial": idx,
            "params": params,
            "val_accuracy": round(val_acc, 6),
            "val_macro_f1": round(val_f1, 6),
            "train_time_seconds": round(elapsed, 3),
        }
        search_history.append(record)

        print(
            f"  Trial {idx}/{len(param_grid)}: C={params['C']}, kernel='{params['kernel']}' -> "
            f"Val Acc: {val_acc*100:.2f}%, Val Macro F1: {val_f1:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

        if val_f1 > best_score:
            best_score = val_f1
            best_params = params

    print(
        f"\nOptimal SVM Parameters Selected: C={best_params['C']}, kernel='{best_params['kernel']}' "
        f"(Validation Macro F1: {best_score:.4f})"
    )
    return best_params, search_history


# ---------------------------------------------------------------------------
# Main Training & Evaluation Pipeline
# ---------------------------------------------------------------------------

def train_svm(fast_tune: bool = False) -> Dict[str, Any]:
    # 1. Load Data
    X_tr, y_tr, X_va, y_va, _, _, classes_list, feature_names = load_feature_dataset()

    # 2. Scale features (fit on train split only!)
    print("\nScaling acoustic features with StandardScaler (fit on train split) ...")
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    # 3. Systematic Hyperparameter Tuning
    best_params, search_history = tune_svm_hyperparameters(
        X_tr_scaled, y_tr, X_va_scaled, y_va, fast_tune=fast_tune
    )

    # 4. Train Final Calibrated Probability-Enabled Model
    print(f"\nTraining final calibrated SVM model with optimal parameters ...")
    t0 = time.time()
    base_svc = SVC(
        C=best_params["C"],
        kernel=best_params["kernel"],
        gamma=best_params.get("gamma", "scale"),
        random_state=42,
    )
    calibrated_clf = CalibratedClassifierCV(estimator=base_svc, cv=3)
    calibrated_clf.fit(X_tr_scaled, y_tr)
    train_time = time.time() - t0
    print(f"Final Model Training Complete in {train_time:.2f} seconds.")

    # 5. Evaluate Model
    train_preds = calibrated_clf.predict(X_tr_scaled)
    val_preds = calibrated_clf.predict(X_va_scaled)

    train_acc = float(accuracy_score(y_tr, train_preds))
    val_acc = float(accuracy_score(y_va, val_preds))

    train_f1 = float(f1_score(y_tr, train_preds, average="macro"))
    val_f1 = float(f1_score(y_va, val_preds, average="macro"))

    # Detailed Classification Report
    report_dict = classification_report(
        y_va, val_preds, target_names=classes_list, output_dict=True
    )
    conf_mat = confusion_matrix(y_va, val_preds, labels=classes_list).tolist()

    # Extract SRS Critical Class Recalls
    critical_recalls: Dict[str, float] = {}
    for crit_cls in CRITICAL_CLASSES:
        if crit_cls in report_dict:
            rec = float(report_dict[crit_cls]["recall"])
            critical_recalls[crit_cls] = round(rec, 4)
            status = "PASSED (>= 85%)" if rec >= 0.85 else "FAILED (< 85%)"
            print(f"  SRS Critical Recall [{crit_cls}]: {rec*100:.2f}% -> {status}")
        else:
            critical_recalls[crit_cls] = 0.0

    print(f"\nSVM Final Performance:")
    print(f"  Training Accuracy   : {train_acc*100:.2f}% | Macro F1: {train_f1:.4f}")
    print(f"  Validation Accuracy : {val_acc*100:.2f}% | Macro F1: {val_f1:.4f}")

    # 6. Save Artifacts & Results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_SERVICE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": calibrated_clf,
        "scaler": scaler,
        "classes": classes_list,
        "feature_names": feature_names,
        "best_params": best_params,
        "model_version": "svm_v1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    model_path = RESULTS_DIR / "svm_model.joblib"
    joblib.dump(model_artifact, model_path)
    print(f"\nSaved model artifact to {model_path.relative_to(REPO_ROOT)}")

    # Copy to ml_service/model/artifacts/
    service_model_path = ML_SERVICE_ARTIFACTS_DIR / "svm_model.joblib"
    shutil.copy2(model_path, service_model_path)
    print(f"Exported model copy to {service_model_path.relative_to(REPO_ROOT)}")

    results_payload = {
        "model_name": "Support Vector Machine (SVM)",
        "model_version": "svm_v1.0",
        "best_hyperparameters": best_params,
        "training_time_seconds": round(train_time, 3),
        "train_accuracy": round(train_acc, 6),
        "val_accuracy": round(val_acc, 6),
        "train_macro_f1": round(train_f1, 6),
        "val_macro_f1": round(val_f1, 6),
        "critical_class_recalls": critical_recalls,
        "hyperparameter_search": search_history,
        "classification_report": report_dict,
        "confusion_matrix": conf_mat,
        "classes": classes_list,
    }

    results_path = RESULTS_DIR / "svm_results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)

    print(f"Saved evaluation metrics to {results_path.relative_to(REPO_ROOT)}")
    return results_payload


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train & tune SVM classifier on extracted acoustic features."
    )
    parser.add_argument(
        "--fast-tune",
        action="store_true",
        help="Run fast hyperparameter search for quick testing",
    )

    args = parser.parse_args()
    train_svm(fast_tune=args.fast_tune)


if __name__ == "__main__":
    main()
