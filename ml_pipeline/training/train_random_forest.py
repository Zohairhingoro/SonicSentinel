"""
Random Forest Training & Hyperparameter Tuning Engine for SonicSentinel AI.

Reads extracted features from audio_dataset/metadata/features.npz (or features.csv),
conducts systematic hyperparameter tuning using the validation split to select
optimal n_estimators and max_depth settings, trains a final Random Forest model
with class_weight=''balanced'' to compensate for class imbalance, and exports
model artifacts and evaluation metrics.

Outputs:
  - Model artifact: ml_pipeline/training/results/rf_model.joblib
  - Export copy:    ml_service/model/artifacts/rf_model.joblib
  - Evaluation:     ml_pipeline/training/results/rf_results.json

Usage:
    python ml_pipeline/training/train_random_forest.py
    python ml_pipeline/training/train_random_forest.py --fast-tune
    python ml_pipeline/training/train_random_forest.py --n-jobs 4
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "features.npz"
DEFAULT_CSV_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "features.csv"
RESULTS_DIR = REPO_ROOT / "ml_pipeline" / "training" / "results"
ML_SERVICE_ARTIFACTS_DIR = REPO_ROOT / "ml_service" / "model" / "artifacts"

CRITICAL_CLASSES = [
    "gunshot",
    "glass_breaking",
    "panic_scream",
    "aggression",
    "person_asking_for_help",
]


def load_feature_dataset(
    npz_path: Path = DEFAULT_NPZ_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Load acoustic feature matrix and split masks from NPZ or fallback CSV.

    Returns:
        X_train, y_train, X_val, y_val, classes_list, feature_names
    """
    if npz_path.exists():
        print(f"Loading feature matrix from {npz_path.relative_to(REPO_ROOT)} ...")
        data = np.load(npz_path)
        X = data["X"]
        y = data["classes"]
        splits = data["splits"]
        feature_names = data["feature_names"].tolist()
    elif csv_path.exists():
        print(f"Loading feature matrix from {csv_path.relative_to(REPO_ROOT)} ...")
        import csv
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        y = np.array([r["class"] for r in rows], dtype=str)
        splits = np.array([r["final_split"] for r in rows], dtype=str)
        feature_names = [k for k in rows[0].keys() if k not in ("audio_id", "class", "final_split")]
        X = np.array([[float(r[col]) for col in feature_names] for r in rows], dtype=np.float32)
    else:
        print(f"ERROR: Feature dataset not found at {npz_path} or {csv_path}", file=sys.stderr)
        sys.exit(1)

    train_mask = splits == "train"
    val_mask = splits == "val"
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    classes_list = sorted(set(y_train) | set(y_val))

    print(f"Dataset Loaded:")
    print(f"  Training split  : {len(X_train)} samples across {len(classes_list)} classes")
    print(f"  Validation split: {len(X_val)} samples across {len(classes_list)} classes")
    print(f"  Feature count   : {X.shape[1]} dimensions")

    return X_train, y_train, X_val, y_val, classes_list, feature_names


def tune_rf_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fast_tune: bool = False,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Grid search over n_estimators and max_depth evaluated on validation split.

    Returns:
        (best_params_dict, search_history_list)
    """
    if fast_tune:
        param_grid = [
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
            {"n_estimators": 300, "max_depth": 30,   "min_samples_leaf": 2},
        ]
    else:
        param_grid = [
            {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
            {"n_estimators": 300, "max_depth": 30,   "min_samples_leaf": 2},
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 300, "max_depth": 20,   "min_samples_leaf": 4},
        ]

    print(f"\nEvaluating {len(param_grid)} RF hyperparameter configurations on validation split ...")
    search_history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_params = param_grid[0]

    for idx, params in enumerate(param_grid, 1):
        t0 = time.time()
        candidate = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            random_state=42,
            n_jobs=n_jobs,
        )
        candidate.fit(X_train, y_train)
        elapsed = time.time() - t0

        val_preds = candidate.predict(X_val)
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
            f"  Trial {idx}/{len(param_grid)}: n_est={params['n_estimators']}, "
            f"max_depth={params['max_depth']}, min_leaf={params['min_samples_leaf']} "
            f"-> Val Acc: {val_acc*100:.2f}%, Val Macro F1: {val_f1:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

        if val_f1 > best_score:
            best_score = val_f1
            best_params = params

    print(
        f"\nOptimal RF Parameters: n_estimators={best_params['n_estimators']}, "
        f"max_depth={best_params['max_depth']}, min_leaf={best_params['min_samples_leaf']} "
        f"(Validation Macro F1: {best_score:.4f})"
    )
    return best_params, search_history


def train_random_forest(fast_tune: bool = False, n_jobs: int = -1) -> Dict[str, Any]:
    X_tr, y_tr, X_va, y_va, classes_list, feature_names = load_feature_dataset()

    best_params, search_history = tune_rf_hyperparameters(
        X_tr, y_tr, X_va, y_va, fast_tune=fast_tune, n_jobs=n_jobs
    )

    print(f"\nTraining final Random Forest model with optimal parameters ...")
    t0 = time.time()
    final_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        class_weight="balanced",
        random_state=42,
        n_jobs=n_jobs,
    )
    final_model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    print(f"Final Model Training Complete in {train_time:.2f} seconds.")

    train_preds = final_model.predict(X_tr)
    val_preds = final_model.predict(X_va)

    train_acc = float(accuracy_score(y_tr, train_preds))
    val_acc = float(accuracy_score(y_va, val_preds))
    train_f1 = float(f1_score(y_tr, train_preds, average="macro"))
    val_f1 = float(f1_score(y_va, val_preds, average="macro"))

    report_dict = classification_report(y_va, val_preds, target_names=classes_list, output_dict=True)
    conf_mat = confusion_matrix(y_va, val_preds, labels=classes_list).tolist()

    feature_importances = dict(zip(feature_names, final_model.feature_importances_.tolist()))
    top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]

    critical_recalls: Dict[str, float] = {}
    for crit_cls in CRITICAL_CLASSES:
        if crit_cls in report_dict:
            rec = float(report_dict[crit_cls]["recall"])
            critical_recalls[crit_cls] = round(rec, 4)
            status = "PASSED (>= 85%)" if rec >= 0.85 else "FAILED (< 85%)"
            print(f"  SRS Critical Recall [{crit_cls}]: {rec*100:.2f}% -> {status}")
        else:
            critical_recalls[crit_cls] = 0.0

    print(f"\nRandom Forest Final Performance:")
    print(f"  Training Accuracy   : {train_acc*100:.2f}% | Macro F1: {train_f1:.4f}")
    print(f"  Validation Accuracy : {val_acc*100:.2f}% | Macro F1: {val_f1:.4f}")
    print(f"\nTop-10 Most Important Features:")
    for feat_name, importance in top_features:
        print(f"  {feat_name:<35s}: {importance:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_SERVICE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": final_model,
        "classes": classes_list,
        "feature_names": feature_names,
        "best_params": best_params,
        "feature_importances": feature_importances,
        "model_version": "rf_v1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    model_path = RESULTS_DIR / "rf_model.joblib"
    joblib.dump(model_artifact, model_path)
    print(f"\nSaved model artifact to {model_path.relative_to(REPO_ROOT)}")

    service_model_path = ML_SERVICE_ARTIFACTS_DIR / "rf_model.joblib"
    shutil.copy2(model_path, service_model_path)
    print(f"Exported model copy to {service_model_path.relative_to(REPO_ROOT)}")

    results_payload = {
        "model_name": "Random Forest",
        "model_version": "rf_v1.0",
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
        "top10_feature_importances": dict(top_features),
    }

    results_path = RESULTS_DIR / "rf_results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)

    print(f"Saved evaluation metrics to {results_path.relative_to(REPO_ROOT)}")
    return results_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train & tune Random Forest classifier on extracted acoustic features."
    )
    parser.add_argument(
        "--fast-tune",
        action="store_true",
        help="Run 2-candidate fast hyperparameter search",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for sklearn (-1 = all cores, default: -1)",
    )
    args = parser.parse_args()
    train_random_forest(fast_tune=args.fast_tune, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
