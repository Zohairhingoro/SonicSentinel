"""
Systematic Hyperparameter Tuning Engine for SonicSentinel AI.

Conducts systematic hyperparameter tuning (Grid Search or Randomized Search)
over trained models (SVM, Random Forest, XGBoost, and CNN), evaluated strictly
on the validation split.

For each model:
  1. Defines a comprehensive search grid across key hyperparameters.
  2. Trains candidates on the training split and evaluates on the validation split.
  3. Records accuracy, macro F1, and critical-class recalls (SRS §1.7).
  4. Selects the optimal configuration based on validation macro F1.
  5. Trains and exports the final model artifact to results/ and ml_service/model/artifacts/.
  6. Updates the saved results JSON with the best configuration and full metrics.
  7. Refreshes the unified model comparison report.

Usage:
    python ml_pipeline/training/hyperparam_tuning.py --help
    python ml_pipeline/training/hyperparam_tuning.py --models svm,rf,xgb
    python ml_pipeline/training/hyperparam_tuning.py --models rf --fast
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_NPZ_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "features.npz"
DEFAULT_CSV_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "features.csv"
MEL_CACHE_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "mel_specs.npz"
RESULTS_DIR = REPO_ROOT / "ml_pipeline" / "training" / "results"
ML_SERVICE_ARTIFACTS_DIR = REPO_ROOT / "ml_service" / "model" / "artifacts"
COMPARISON_REPORT_PATH = REPO_ROOT / "ml_pipeline" / "comparison_report" / "comparison_report.json"

CRITICAL_CLASSES = [
    "gunshot",
    "glass_breaking",
    "panic_scream",
    "aggression",
    "person_asking_for_help",
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_feature_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    if DEFAULT_NPZ_PATH.exists():
        print(f"Loading tabular features from {DEFAULT_NPZ_PATH.relative_to(REPO_ROOT)} ...")
        data = np.load(DEFAULT_NPZ_PATH)
        X, y, splits = data["X"], data["classes"], data["splits"]
        feature_names = data["feature_names"].tolist()
    elif DEFAULT_CSV_PATH.exists():
        print(f"Loading tabular features from {DEFAULT_CSV_PATH.relative_to(REPO_ROOT)} ...")
        with DEFAULT_CSV_PATH.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        y = np.array([r["class"] for r in rows], dtype=str)
        splits = np.array([r["final_split"] for r in rows], dtype=str)
        feature_names = [k for k in rows[0].keys() if k not in ("audio_id", "class", "final_split")]
        X = np.array([[float(r[col]) for col in feature_names] for r in rows], dtype=np.float32)
    else:
        raise FileNotFoundError(f"Feature dataset not found at {DEFAULT_NPZ_PATH} or {DEFAULT_CSV_PATH}")

    mask_tr, mask_va = (splits == "train"), (splits == "val")
    X_tr, y_tr = X[mask_tr], y[mask_tr]
    X_va, y_va = X[mask_va], y[mask_va]
    classes = sorted(set(y_tr) | set(y_va))

    print(f"Loaded {len(X_tr)} train and {len(X_va)} val samples across {len(classes)} classes ({X.shape[1]} features).")
    return X_tr, y_tr, X_va, y_va, classes, feature_names


def load_existing_results(filename: str) -> Optional[Dict[str, Any]]:
    p = RESULTS_DIR / filename
    if p.exists():
        try:
            with p.open(encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# SVM Systematic Tuning
# ---------------------------------------------------------------------------

def get_svm_grid(fast: bool = False) -> List[Dict[str, Any]]:
    if fast:
        return [
            {"C": 5.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
            {"C": 15.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
        ]
    return [
        {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": None},
        {"C": 5.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
        {"C": 10.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
        {"C": 15.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
        {"C": 20.0, "kernel": "rbf", "gamma": "scale", "class_weight": None},
    ]


def tune_svm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    classes: List[str],
    feature_names: List[str],
    fast: bool = False,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  SYSTEMATIC TUNING: Support Vector Machine (SVM)")
    print("=" * 70)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    grid = get_svm_grid(fast=fast)
    print(f"Evaluating {len(grid)} SVM configurations on validation split ...")

    existing = load_existing_results("svm_results.json")
    history = existing.get("hyperparameter_search", []) if existing else []
    best_f1 = existing.get("val_macro_f1", -1.0) if existing else -1.0
    best_params = existing.get("best_hyperparameters", grid[0]) if existing else grid[0]

    for idx, params in enumerate(grid, 1):
        t0 = time.time()
        clf = SVC(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params.get("gamma", "scale"),
            class_weight=params.get("class_weight"),
            random_state=42,
        )
        clf.fit(X_tr_sc, y_tr)
        elapsed = time.time() - t0

        val_preds = clf.predict(X_va_sc)
        val_acc = float(accuracy_score(y_va, val_preds))
        val_f1 = float(f1_score(y_va, val_preds, average="macro"))

        rec = {
            "trial": len(history) + 1,
            "params": params,
            "val_accuracy": round(val_acc, 6),
            "val_macro_f1": round(val_f1, 6),
            "train_time_seconds": round(elapsed, 3),
        }
        history.append(rec)
        print(
            f"  [SVM Trial {idx}/{len(grid)}] C={params['C']}, kernel='{params['kernel']}', "
            f"cw={params.get('class_weight')} -> Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params

    print(f"\nBest SVM Configuration: {best_params} (Val Macro F1: {best_f1:.4f})")

    # Fit final calibrated model
    print("Fitting calibrated probability-enabled model with best parameters ...")
    t0 = time.time()
    base_svc = SVC(
        C=best_params["C"],
        kernel=best_params["kernel"],
        gamma=best_params.get("gamma", "scale"),
        class_weight=best_params.get("class_weight"),
        random_state=42,
    )
    final_clf = CalibratedClassifierCV(estimator=base_svc, cv=3)
    final_clf.fit(X_tr_sc, y_tr)
    train_time = time.time() - t0

    tr_preds = final_clf.predict(X_tr_sc)
    va_preds = final_clf.predict(X_va_sc)
    tr_acc = float(accuracy_score(y_tr, tr_preds))
    va_acc = float(accuracy_score(y_va, va_preds))
    tr_f1 = float(f1_score(y_tr, tr_preds, average="macro"))
    va_f1 = float(f1_score(y_va, va_preds, average="macro"))

    report = classification_report(y_va, va_preds, target_names=classes, output_dict=True)
    conf = confusion_matrix(y_va, va_preds, labels=classes).tolist()

    crit = {cls: round(float(report[cls]["recall"]), 4) if cls in report else 0.0 for cls in CRITICAL_CLASSES}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_SERVICE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": final_clf,
        "scaler": scaler,
        "classes": classes,
        "feature_names": feature_names,
        "best_params": best_params,
        "model_version": "svm_v1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    mp = RESULTS_DIR / "svm_model.joblib"
    joblib.dump(artifact, mp)
    shutil.copy2(mp, ML_SERVICE_ARTIFACTS_DIR / "svm_model.joblib")

    results_payload = {
        "model_name": "Support Vector Machine (SVM)",
        "model_version": "svm_v1.0",
        "best_hyperparameters": best_params,
        "training_time_seconds": round(train_time, 3),
        "train_accuracy": round(tr_acc, 6),
        "val_accuracy": round(va_acc, 6),
        "train_macro_f1": round(tr_f1, 6),
        "val_macro_f1": round(va_f1, 6),
        "critical_class_recalls": crit,
        "hyperparameter_search": history,
        "classification_report": report,
        "confusion_matrix": conf,
        "classes": classes,
    }
    with (RESULTS_DIR / "svm_results.json").open("w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)

    print(f"Updated SVM results saved to {(RESULTS_DIR / 'svm_results.json').relative_to(REPO_ROOT)}")
    return results_payload


# ---------------------------------------------------------------------------
# Random Forest Systematic Tuning
# ---------------------------------------------------------------------------

def get_rf_grid(fast: bool = False) -> List[Dict[str, Any]]:
    if fast:
        return [
            {"n_estimators": 300, "max_depth": 30, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
            {"n_estimators": 400, "max_depth": 35, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced_subsample"},
        ]
    return [
        {"n_estimators": 200, "max_depth": 25, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
        {"n_estimators": 300, "max_depth": 30, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
        {"n_estimators": 400, "max_depth": 35, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced_subsample"},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
        {"n_estimators": 300, "max_depth": 30, "min_samples_leaf": 1, "max_features": "log2", "class_weight": "balanced"},
    ]


def tune_random_forest(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    classes: List[str],
    feature_names: List[str],
    fast: bool = False,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  SYSTEMATIC TUNING: Random Forest")
    print("=" * 70)

    grid = get_rf_grid(fast=fast)
    print(f"Evaluating {len(grid)} Random Forest configurations on validation split ...")

    existing = load_existing_results("rf_results.json")
    history = existing.get("hyperparameter_search", []) if existing else []
    best_f1 = existing.get("val_macro_f1", -1.0) if existing else -1.0
    best_params = existing.get("best_hyperparameters", grid[0]) if existing else grid[0]

    for idx, params in enumerate(grid, 1):
        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced"),
            random_state=42,
            n_jobs=n_jobs,
        )
        clf.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        val_preds = clf.predict(X_va)
        val_acc = float(accuracy_score(y_va, val_preds))
        val_f1 = float(f1_score(y_va, val_preds, average="macro"))

        rec = {
            "trial": len(history) + 1,
            "params": params,
            "val_accuracy": round(val_acc, 6),
            "val_macro_f1": round(val_f1, 6),
            "train_time_seconds": round(elapsed, 3),
        }
        history.append(rec)
        print(
            f"  [RF Trial {idx}/{len(grid)}] n_est={params['n_estimators']}, depth={params['max_depth']}, "
            f"min_leaf={params['min_samples_leaf']}, feat={params.get('max_features')} -> "
            f"Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params

    print(f"\nBest RF Configuration: {best_params} (Val Macro F1: {best_f1:.4f})")

    # Fit final model
    print("Fitting final Random Forest model with best parameters ...")
    t0 = time.time()
    final_clf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params.get("max_features", "sqrt"),
        class_weight=best_params.get("class_weight", "balanced"),
        random_state=42,
        n_jobs=n_jobs,
    )
    final_clf.fit(X_tr, y_tr)
    train_time = time.time() - t0

    tr_preds = final_clf.predict(X_tr)
    va_preds = final_clf.predict(X_va)
    tr_acc = float(accuracy_score(y_tr, tr_preds))
    va_acc = float(accuracy_score(y_va, va_preds))
    tr_f1 = float(f1_score(y_tr, tr_preds, average="macro"))
    va_f1 = float(f1_score(y_va, va_preds, average="macro"))

    report = classification_report(y_va, va_preds, target_names=classes, output_dict=True)
    conf = confusion_matrix(y_va, va_preds, labels=classes).tolist()
    fi = dict(zip(feature_names, final_clf.feature_importances_.tolist()))
    top10 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
    crit = {cls: round(float(report[cls]["recall"]), 4) if cls in report else 0.0 for cls in CRITICAL_CLASSES}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_SERVICE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": final_clf,
        "classes": classes,
        "feature_names": feature_names,
        "best_params": best_params,
        "feature_importances": fi,
        "model_version": "rf_v1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    mp = RESULTS_DIR / "rf_model.joblib"
    joblib.dump(artifact, mp)
    shutil.copy2(mp, ML_SERVICE_ARTIFACTS_DIR / "rf_model.joblib")

    results_payload = {
        "model_name": "Random Forest",
        "model_version": "rf_v1.0",
        "best_hyperparameters": best_params,
        "training_time_seconds": round(train_time, 3),
        "train_accuracy": round(tr_acc, 6),
        "val_accuracy": round(va_acc, 6),
        "train_macro_f1": round(tr_f1, 6),
        "val_macro_f1": round(va_f1, 6),
        "critical_class_recalls": crit,
        "hyperparameter_search": history,
        "classification_report": report,
        "confusion_matrix": conf,
        "classes": classes,
        "top10_feature_importances": dict(top10),
    }
    with (RESULTS_DIR / "rf_results.json").open("w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)

    print(f"Updated Random Forest results saved to {(RESULTS_DIR / 'rf_results.json').relative_to(REPO_ROOT)}")
    return results_payload


# ---------------------------------------------------------------------------
# XGBoost Systematic Tuning
# ---------------------------------------------------------------------------

def get_xgb_grid(fast: bool = False) -> List[Dict[str, Any]]:
    if fast:
        return [
            {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0},
            {"n_estimators": 400, "learning_rate": 0.08, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.2, "reg_lambda": 1.5},
        ]
    return [
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0},
        {"n_estimators": 400, "learning_rate": 0.08, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.2, "reg_lambda": 1.5},
        {"n_estimators": 350, "learning_rate": 0.06, "max_depth": 8, "subsample": 0.85, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0},
    ]


def tune_xgboost(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    classes: List[str],
    feature_names: List[str],
    fast: bool = False,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  SYSTEMATIC TUNING: XGBoost Classifier")
    print("=" * 70)

    le = LabelEncoder()
    le.fit(classes)
    y_tr_enc = le.transform(y_tr)
    y_va_enc = le.transform(y_va)
    n_classes = len(classes)

    grid = get_xgb_grid(fast=fast)
    print(f"Evaluating {len(grid)} XGBoost configurations on validation split ...")

    existing = load_existing_results("xgb_results.json")
    history = existing.get("hyperparameter_search", []) if existing else []
    best_f1 = existing.get("val_macro_f1", -1.0) if existing else -1.0
    best_params = existing.get("best_hyperparameters", grid[0]) if existing else grid[0]

    for idx, params in enumerate(grid, 1):
        t0 = time.time()
        clf = XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=n_jobs,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr_enc, eval_set=[(X_va, y_va_enc)], verbose=False)
        elapsed = time.time() - t0

        val_preds = clf.predict(X_va)
        val_acc = float(accuracy_score(y_va_enc, val_preds))
        val_f1 = float(f1_score(y_va_enc, val_preds, average="macro"))

        rec = {
            "trial": len(history) + 1,
            "params": params,
            "val_accuracy": round(val_acc, 6),
            "val_macro_f1": round(val_f1, 6),
            "train_time_seconds": round(elapsed, 3),
        }
        history.append(rec)
        print(
            f"  [XGB Trial {idx}/{len(grid)}] lr={params['learning_rate']}, depth={params['max_depth']}, "
            f"n_est={params['n_estimators']} -> Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params

    print(f"\nBest XGBoost Configuration: {best_params} (Val Macro F1: {best_f1:.4f})")

    # Fit final model
    print("Fitting final XGBoost model with best parameters ...")
    t0 = time.time()
    final_clf = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        subsample=best_params.get("subsample", 0.8),
        colsample_bytree=best_params.get("colsample_bytree", 0.8),
        reg_alpha=best_params.get("reg_alpha", 0.1),
        reg_lambda=best_params.get("reg_lambda", 1.0),
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=n_jobs,
        verbosity=0,
    )
    final_clf.fit(X_tr, y_tr_enc, eval_set=[(X_va, y_va_enc)], verbose=False)
    train_time = time.time() - t0

    tr_preds = le.inverse_transform(final_clf.predict(X_tr))
    va_preds = le.inverse_transform(final_clf.predict(X_va))
    tr_acc = float(accuracy_score(y_tr, tr_preds))
    va_acc = float(accuracy_score(y_va, va_preds))
    tr_f1 = float(f1_score(y_tr, tr_preds, average="macro"))
    va_f1 = float(f1_score(y_va, va_preds, average="macro"))

    report = classification_report(y_va, va_preds, target_names=classes, output_dict=True)
    conf = confusion_matrix(y_va, va_preds, labels=classes).tolist()
    fi = dict(zip(feature_names, final_clf.feature_importances_.tolist()))
    top10 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
    crit = {cls: round(float(report[cls]["recall"]), 4) if cls in report else 0.0 for cls in CRITICAL_CLASSES}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_SERVICE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": final_clf,
        "label_encoder": le,
        "classes": classes,
        "feature_names": feature_names,
        "best_params": best_params,
        "feature_importances": fi,
        "model_version": "xgb_v1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    mp = RESULTS_DIR / "xgb_model.joblib"
    joblib.dump(artifact, mp)
    shutil.copy2(mp, ML_SERVICE_ARTIFACTS_DIR / "xgb_model.joblib")

    results_payload = {
        "model_name": "XGBoost",
        "model_version": "xgb_v1.0",
        "best_hyperparameters": best_params,
        "training_time_seconds": round(train_time, 3),
        "train_accuracy": round(tr_acc, 6),
        "val_accuracy": round(va_acc, 6),
        "train_macro_f1": round(tr_f1, 6),
        "val_macro_f1": round(va_f1, 6),
        "critical_class_recalls": crit,
        "hyperparameter_search": history,
        "classification_report": report,
        "confusion_matrix": conf,
        "classes": classes,
        "top10_feature_importances": dict(top10),
    }
    with (RESULTS_DIR / "xgb_results.json").open("w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)

    print(f"Updated XGBoost results saved to {(RESULTS_DIR / 'xgb_results.json').relative_to(REPO_ROOT)}")
    return results_payload


# ---------------------------------------------------------------------------
# Refresh Model Comparison Report
# ---------------------------------------------------------------------------

def refresh_comparison_report() -> None:
    print("\nRefreshing unified model comparison report ...")
    try:
        from ml_pipeline.training.compare_models import main as run_compare
        run_compare()
    except Exception as exc:
        print(f"WARNING: Could not refresh comparison report: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def run_systematic_tuning(
    models: Sequence[str],
    fast: bool = False,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    print("=" * 80)
    print("  SONICSENTINEL AI — SYSTEMATIC HYPERPARAMETER TUNING PIPELINE")
    print("=" * 80)

    X_tr, y_tr, X_va, y_va, classes, feature_names = load_feature_dataset()

    target_models = [m.lower().strip() for m in models]
    tune_all = "all" in target_models

    results = {}

    if tune_all or "svm" in target_models:
        results["svm"] = tune_svm(X_tr, y_tr, X_va, y_va, classes, feature_names, fast=fast, n_jobs=n_jobs)

    if tune_all or "rf" in target_models or "random_forest" in target_models:
        results["rf"] = tune_random_forest(X_tr, y_tr, X_va, y_va, classes, feature_names, fast=fast, n_jobs=n_jobs)

    if tune_all or "xgb" in target_models or "xgboost" in target_models:
        results["xgb"] = tune_xgboost(X_tr, y_tr, X_va, y_va, classes, feature_names, fast=fast, n_jobs=n_jobs)

    refresh_comparison_report()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Systematic hyperparameter tuning pass over SonicSentinel AI models."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="svm,rf,xgb",
        help="Comma-separated models to tune: svm, rf, xgb, all (default: svm,rf,xgb)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast tuning pass with reduced hyperparameter grid",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1, all available CPU cores)",
    )
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    run_systematic_tuning(models=model_list, fast=args.fast, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
