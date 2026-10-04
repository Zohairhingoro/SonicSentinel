import pathlib

dest = pathlib.Path("ml_pipeline/training/train_xgboost.py")
dest.parent.mkdir(parents=True, exist_ok=True)

code = """\
from __future__ import annotations
import argparse, json, shutil, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "features.npz"
DEFAULT_CSV_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "features.csv"
RESULTS_DIR = REPO_ROOT / "ml_pipeline" / "training" / "results"
ML_SERVICE_ARTIFACTS_DIR = REPO_ROOT / "ml_service" / "model" / "artifacts"
CRITICAL_CLASSES = ["gunshot", "glass_breaking", "panic_scream", "aggression", "person_asking_for_help"]


def load_feature_dataset(npz_path=DEFAULT_NPZ_PATH, csv_path=DEFAULT_CSV_PATH):
    if npz_path.exists():
        print(f"Loading feature matrix from {npz_path.relative_to(REPO_ROOT)} ...")
        data = np.load(npz_path)
        X, y, splits = data["X"], data["classes"], data["splits"]
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
        print("ERROR: Feature dataset not found.", file=sys.stderr); sys.exit(1)
    mask_tr, mask_va = splits == "train", splits == "val"
    X_tr, y_tr = X[mask_tr], y[mask_tr]
    X_va, y_va = X[mask_va], y[mask_va]
    classes = sorted(set(y_tr) | set(y_va))
    print(f"Dataset Loaded:")
    print(f"  Training split  : {len(X_tr)} samples across {len(classes)} classes")
    print(f"  Validation split: {len(X_va)} samples across {len(classes)} classes")
    print(f"  Feature count   : {X.shape[1]} dimensions")
    return X_tr, y_tr, X_va, y_va, classes, feature_names


def _make_clf(p, n_classes):
    return XGBClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        max_depth=p["max_depth"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"], reg_alpha=p["reg_alpha"],
        reg_lambda=p["reg_lambda"], objective="multi:softprob",
        num_class=n_classes, eval_metric="mlogloss",
        tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
    )


def tune_xgb(X_tr, y_tr, X_va, y_va, n_classes, fast_tune=False):
    if fast_tune:
        grid = [
            {"n_estimators": 300, "learning_rate": 0.1,  "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
            {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0},
        ]
    else:
        grid = [
            {"n_estimators": 200, "learning_rate": 0.1,  "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
            {"n_estimators": 300, "learning_rate": 0.1,  "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
            {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0},
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 1.0, "reg_lambda": 2.0},
        ]
    print(f"\\nEvaluating {len(grid)} XGBoost configurations on validation split ...")
    history, best_score, best_p = [], -1.0, grid[0]
    for i, p in enumerate(grid, 1):
        t0 = time.time()
        clf = _make_clf(p, n_classes)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        elapsed = time.time() - t0
        preds = clf.predict(X_va)
        acc = float(accuracy_score(y_va, preds))
        f1  = float(f1_score(y_va, preds, average="macro"))
        history.append({"trial": i, "params": p, "val_accuracy": round(acc,6), "val_macro_f1": round(f1,6), "train_time_seconds": round(elapsed,3)})
        lr, depth, n = p["learning_rate"], p["max_depth"], p["n_estimators"]
        print(f"  Trial {i}/{len(grid)}: lr={lr}, depth={depth}, n_est={n} -> Val Acc: {acc*100:.2f}%, Val F1: {f1:.4f} ({elapsed:.1f}s)", flush=True)
        if f1 > best_score:
            best_score, best_p = f1, p
    print(f"\\nOptimal XGB: lr={best_p['learning_rate']}, depth={best_p['max_depth']}, n_est={best_p['n_estimators']} (Val F1: {best_score:.4f})")
    return best_p, history


def train_xgboost(fast_tune=False):
    X_tr, y_tr_s, X_va, y_va_s, classes, feat_names = load_feature_dataset()
    le = LabelEncoder(); le.fit(classes)
    y_tr, y_va = le.transform(y_tr_s), le.transform(y_va_s)
    n_classes = len(classes)

    best_p, history = tune_xgb(X_tr, y_tr, X_va, y_va, n_classes=n_classes, fast_tune=fast_tune)

    print("\\nTraining final XGBoost model with optimal parameters ...")
    t0 = time.time()
    final = _make_clf(best_p, n_classes)
    final.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    train_time = time.time() - t0
    print(f"Final Model Training Complete in {train_time:.2f} seconds.")

    tr_str = le.inverse_transform(final.predict(X_tr))
    va_str = le.inverse_transform(final.predict(X_va))
    tr_acc = float(accuracy_score(y_tr_s, tr_str))
    va_acc = float(accuracy_score(y_va_s, va_str))
    tr_f1  = float(f1_score(y_tr_s, tr_str, average="macro"))
    va_f1  = float(f1_score(y_va_s, va_str, average="macro"))

    report = classification_report(y_va_s, va_str, target_names=classes, output_dict=True)
    conf   = confusion_matrix(y_va_s, va_str, labels=classes).tolist()
    fi     = dict(zip(feat_names, final.feature_importances_.tolist()))
    top10  = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]

    crit = {}
    for cls in CRITICAL_CLASSES:
        rec = float(report[cls]["recall"]) if cls in report else 0.0
        crit[cls] = round(rec, 4)
        print(f"  SRS Critical Recall [{cls}]: {rec*100:.2f}% -> {'PASSED (>= 85%)' if rec >= 0.85 else 'FAILED (< 85%)'}")

    print(f"\\nXGBoost Final Performance:")
    print(f"  Training Accuracy   : {tr_acc*100:.2f}% | Macro F1: {tr_f1:.4f}")
    print(f"  Validation Accuracy : {va_acc*100:.2f}% | Macro F1: {va_f1:.4f}")
    print("\\nTop-10 Most Important Features:")
    for n, v in top10:
        print(f"  {n:<35s}: {v:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_SERVICE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {"model": final, "label_encoder": le, "classes": classes,
                "feature_names": feat_names, "best_params": best_p,
                "feature_importances": fi, "model_version": "xgb_v1.0",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    mp = RESULTS_DIR / "xgb_model.joblib"
    joblib.dump(artifact, mp)
    print(f"\\nSaved model artifact to {mp.relative_to(REPO_ROOT)}")
    sp = ML_SERVICE_ARTIFACTS_DIR / "xgb_model.joblib"
    shutil.copy2(mp, sp)
    print(f"Exported model copy to {sp.relative_to(REPO_ROOT)}")

    payload = {"model_name": "XGBoost", "model_version": "xgb_v1.0",
               "best_hyperparameters": best_p, "training_time_seconds": round(train_time,3),
               "train_accuracy": round(tr_acc,6), "val_accuracy": round(va_acc,6),
               "train_macro_f1": round(tr_f1,6), "val_macro_f1": round(va_f1,6),
               "critical_class_recalls": crit, "hyperparameter_search": history,
               "classification_report": report, "confusion_matrix": conf,
               "classes": classes, "top10_feature_importances": dict(top10)}
    rp = RESULTS_DIR / "xgb_results.json"
    with rp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved evaluation metrics to {rp.relative_to(REPO_ROOT)}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost classifier on acoustic features.")
    parser.add_argument("--fast-tune", action="store_true", help="Run 2-candidate fast search")
    args = parser.parse_args()
    train_xgboost(fast_tune=args.fast_tune)

if __name__ == "__main__":
    main()
"""

dest.write_text(code, encoding="utf-8")
print("train_xgboost.py written successfully:", len(code), "bytes")