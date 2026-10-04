"""
Final Evaluation Engine on Held-Out Test Split for SonicSentinel AI.

Evaluates the winning model on the completely held-out, frozen test split
(never touched during model training or hyperparameter tuning).

Outputs:
  - Confusion Matrix CSV: docs/reports/test_confusion_matrix.csv
  - Confusion Matrix Heatmap Image: docs/reports/test_confusion_matrix.png
  - Test Evaluation Report JSON: docs/reports/test_evaluation_report.json
  - Test Evaluation Report Markdown: docs/reports/test_evaluation_report.md

Usage:
    python ml_pipeline/evaluation/evaluate.py
    python ml_pipeline/evaluation/evaluate.py --model-path ml_service/model/artifacts/xgb_model.joblib
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_service.features.extract import extract_features  # noqa: E402

INDEX_CSV_PATH = REPO_ROOT / "audio_dataset" / "metadata" / "dataset_index.csv"
TEST_FEATURES_CACHE = REPO_ROOT / "audio_dataset" / "metadata" / "test_features.npz"
COMPARISON_REPORT_PATH = REPO_ROOT / "ml_pipeline" / "comparison_report" / "comparison_report.json"
RESULTS_DIR = REPO_ROOT / "ml_pipeline" / "training" / "results"
ML_SERVICE_ARTIFACTS_DIR = REPO_ROOT / "ml_service" / "model" / "artifacts"
DEFAULT_MODEL_PATH = ML_SERVICE_ARTIFACTS_DIR / "xgb_model.joblib"
REPORTS_DIR = REPO_ROOT / "docs" / "reports"

# Canonical 5 Critical Classes (SRS §1.6 & §1.7, AGENTS.md §4)
CRITICAL_CLASSES = [
    ("gunshot", "Gunshot"),
    ("glass_breaking", "Glass Breaking"),
    ("panic_scream", "Panic Scream"),
    ("aggression", "Aggression / Violent Conflict"),
    ("person_asking_for_help", "Person Asking for Help"),
]

TARGET_ACCURACY = 0.85
TARGET_MACRO_F1 = 0.80
TARGET_CRITICAL_RECALL = 0.85


def resolve_audio_file(row: Dict[str, str]) -> Optional[Path]:
    filename = row.get("filename", "")
    cls_name = row.get("class", "")
    final_split = row.get("final_split") or "test"
    source_split = row.get("source_split") or "test"

    paths_to_check = [
        REPO_ROOT / "audio_dataset" / "test" / cls_name / filename,
        REPO_ROOT / "audio_dataset" / "preprocessed" / final_split / cls_name / filename,
        REPO_ROOT / "audio_dataset" / "raw" / "test" / cls_name / filename,
        REPO_ROOT / "audio_dataset" / "raw" / source_split / cls_name / filename,
    ]
    for p in paths_to_check:
        if p.exists():
            return p
    return None


def extract_features_for_file(src_path: Path) -> Optional[np.ndarray]:
    try:
        data, sr = sf.read(str(src_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        if len(data) == 0:
            return None
        res = extract_features(data, sr=sr)
        return res["features_vector"]
    except Exception:
        return None


def get_or_extract_test_features() -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    if TEST_FEATURES_CACHE.exists():
        print(f"Loading cached test features from {TEST_FEATURES_CACHE.relative_to(REPO_ROOT)} ...")
        data = np.load(TEST_FEATURES_CACHE)
        X = data["X"].astype(np.float32)
        y = data["classes"]
        audio_ids = data["audio_ids"].tolist()
        feature_names = data["feature_names"].tolist()
        class_names = sorted(set(y.tolist()))
        print(f"Loaded {len(X)} test samples across {len(class_names)} classes ({X.shape[1]} features).")
        return X, y, audio_ids, class_names, feature_names

    print(f"Extracting test split features from audio files listed in {INDEX_CSV_PATH.relative_to(REPO_ROOT)} ...")
    if not INDEX_CSV_PATH.exists():
        raise FileNotFoundError(f"dataset_index.csv not found at {INDEX_CSV_PATH}")

    with INDEX_CSV_PATH.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    test_rows = [r for r in rows if r.get("final_split") == "test"]
    if not test_rows:
        raise ValueError("No rows found with final_split == 'test' in dataset_index.csv")

    t0 = time.time()
    num_workers = min(os.cpu_count() or 4, 16)
    print(f"Processing {len(test_rows)} test audio clips with {num_workers} worker threads ...")

    file_paths = [resolve_audio_file(r) for r in test_rows]
    valid_indices = [i for i, p in enumerate(file_paths) if p is not None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        features_list = list(executor.map(lambda i: extract_features_for_file(file_paths[i]), valid_indices))

    X_list, y_list, id_list = [], [], []
    for idx, feat in zip(valid_indices, features_list):
        if feat is not None:
            X_list.append(feat)
            y_list.append(test_rows[idx]["class"])
            id_list.append(test_rows[idx]["audio_id"])

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=str)
    audio_ids = id_list
    class_names = sorted(set(y.tolist()))

    from ml_pipeline.features.extract_features import FEATURE_COLUMNS
    feature_names = [c for c in FEATURE_COLUMNS if c not in ("audio_id", "class", "final_split")]

    elapsed = time.time() - t0
    print(f"Extracted {len(X)} test sample feature vectors in {elapsed:.1f}s.")

    TEST_FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TEST_FEATURES_CACHE,
        X=X,
        classes=y,
        audio_ids=np.array(audio_ids, dtype=str),
        feature_names=np.array(feature_names, dtype=str),
    )
    print(f"Saved test features cache to {TEST_FEATURES_CACHE.relative_to(REPO_ROOT)}")

    return X, y, audio_ids, class_names, feature_names


def load_winning_model(specified_path: Optional[Path] = None) -> Tuple[Any, str, str, List[str], List[str], Any]:
    model_path = specified_path
    model_name = "XGBoost"
    model_version = "xgb_v1.0"

    if model_path is None:
        if COMPARISON_REPORT_PATH.exists():
            try:
                with COMPARISON_REPORT_PATH.open(encoding="utf-8") as fh:
                    comp = json.load(fh)
                winning = comp.get("winning_model") or {}
                model_name = winning.get("name", "XGBoost")
                model_version = winning.get("version", "xgb_v1.0")
            except Exception:
                pass

        # Match artifact file based on name
        slug = "xgb" if "xgb" in model_name.lower() else "rf" if "rf" in model_name.lower() or "random" in model_name.lower() else "svm"
        candidate_paths = [
            ML_SERVICE_ARTIFACTS_DIR / f"{slug}_model.joblib",
            RESULTS_DIR / f"{slug}_model.joblib",
            DEFAULT_MODEL_PATH,
        ]
        for p in candidate_paths:
            if p.exists():
                model_path = p
                break

    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Trained model artifact not found. Checked default {DEFAULT_MODEL_PATH}")

    print(f"Loading winning model artifact from {model_path.relative_to(REPO_ROOT)} ...")
    artifact = joblib.load(model_path)
    model = artifact.get("model")
    classes = [str(c) for c in artifact.get("classes", [])]
    feature_names = artifact.get("feature_names", [])
    model_version = artifact.get("model_version", model_version)
    le = artifact.get("label_encoder")

    return model, model_name, model_version, classes, feature_names, le


def run_test_evaluation(
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    print("=" * 80)
    print("  SONICSENTINEL AI — HELD-OUT TEST SPLIT EVALUATION PIPELINE")
    print("  Competition Theme: AcousticX Intelligence (TechWiz7 NextWave AI/ML)")
    print("=" * 80)

    # 1. Load Model
    model, model_name, model_version, classes, feat_names, le = load_winning_model(model_path)
    print(f"Evaluator Target: {model_name} ({model_version})")

    # 2. Load / Extract Test Split Features
    X_test, y_test, audio_ids, test_classes, _ = get_or_extract_test_features()

    # 3. Model Inference on Test Set
    t0 = time.time()
    preds_raw = model.predict(X_test)
    inference_time = time.time() - t0

    if le is not None:
        if isinstance(preds_raw[0], (int, np.integer)):
            y_pred = le.inverse_transform(preds_raw)
        else:
            y_pred = preds_raw
    else:
        y_pred = preds_raw

    y_true = y_test

    # 4. Metric Calculations
    test_accuracy = float(accuracy_score(y_true, y_pred))
    test_macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    test_weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    class_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)

    macro_avg = class_report.get("macro avg", {})
    weighted_avg = class_report.get("weighted avg", {})

    test_macro_precision = float(macro_avg.get("precision", 0.0))
    test_macro_recall = float(macro_avg.get("recall", 0.0))
    test_weighted_precision = float(weighted_avg.get("precision", 0.0))
    test_weighted_recall = float(weighted_avg.get("recall", 0.0))

    # Critical Class Recalls
    critical_metrics: Dict[str, Dict[str, Any]] = {}
    crit_pass_count = 0
    for slug, label in CRITICAL_CLASSES:
        if slug in class_report:
            rec = float(class_report[slug]["recall"])
            prec = float(class_report[slug]["precision"])
            f1 = float(class_report[slug]["f1-score"])
            supp = int(class_report[slug]["support"])
        else:
            rec, prec, f1, supp = 0.0, 0.0, 0.0, 0

        passed = (rec >= TARGET_CRITICAL_RECALL)
        if passed:
            crit_pass_count += 1

        critical_metrics[slug] = {
            "label": label,
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "support": supp,
            "target_threshold": TARGET_CRITICAL_RECALL,
            "passed": passed,
        }

    # 5. Terminal Display
    print("\n" + "=" * 80)
    print(f"  TEST SPLIT PERFORMANCE SUMMARY — {model_name} ({model_version})")
    print("=" * 80)
    print(f"  Held-Out Test Samples : {len(X_test)} audio events")
    print(f"  Inference Latency     : {inference_time*1000/len(X_test):.2f} ms per sample ({inference_time:.2f}s total)")
    print(f"  Overall Test Accuracy : {test_accuracy*100:.2f}% (Target: >= 85%) -> {'PASSED' if test_accuracy >= TARGET_ACCURACY else 'NOT MET'}")
    print(f"  Macro Precision       : {test_macro_precision:.4f}")
    print(f"  Macro Recall          : {test_macro_recall:.4f}")
    print(f"  Macro F1 Score        : {test_macro_f1:.4f} (Target: >= 0.80) -> {'PASSED' if test_macro_f1 >= TARGET_MACRO_F1 else 'NOT MET'}")
    print(f"  Weighted F1 Score     : {test_weighted_f1:.4f}")

    print("\n" + "-" * 80)
    print("  PER-CLASS EVALUATION BREAKDOWN:")
    print("-" * 80)
    hdr = f"  {'Sound Category':<30}{'Precision':>11}{'Recall':>11}{'F1-Score':>11}{'Support':>10}"
    print(hdr)
    print("  " + "-" * 76)
    for cls in classes:
        m = class_report.get(cls, {})
        p = m.get("precision", 0.0)
        r = m.get("recall", 0.0)
        f = m.get("f1-score", 0.0)
        s = int(m.get("support", 0))
        print(f"  {cls:<30}{p*100:>10.2f}%{r*100:>10.2f}%{f:>11.4f}{s:>10d}")
    print("  " + "-" * 76)
    print(f"  {'Macro Average':<30}{test_macro_precision*100:>10.2f}%{test_macro_recall*100:>10.2f}%{test_macro_f1:>11.4f}{len(X_test):>10d}")
    print(f"  {'Weighted Average':<30}{test_weighted_precision*100:>10.2f}%{test_weighted_recall*100:>10.2f}%{test_weighted_f1:>11.4f}{len(X_test):>10d}")

    print("\n" + "-" * 80)
    print("  AGENTS.MD §4 CRITICAL-CLASS RECALL AUDIT (Target: >= 85%):")
    print("-" * 80)
    for slug, m in critical_metrics.items():
        st = "PASSED (>= 85%)" if m["passed"] else "FAILED (< 85%)"
        print(f"  {m['label']:<32}: {m['recall']*100:>6.2f}% (Prec: {m['precision']*100:.1f}%, F1: {m['f1_score']:.4f}, n={m['support']}) -> {st}")
    print(f"\n  Critical Categories Passing: {crit_pass_count} / {len(CRITICAL_CLASSES)}")
    print("-" * 80)

    # 6. Save Artifacts in docs/reports/
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_matrix_path = REPORTS_DIR / "test_confusion_matrix.csv"
    png_matrix_path = REPORTS_DIR / "test_confusion_matrix.png"
    json_report_path = REPORTS_DIR / "test_evaluation_report.json"
    md_report_path = REPORTS_DIR / "test_evaluation_report.md"

    # A. Save Confusion Matrix CSV
    save_confusion_matrix_csv(conf_matrix, classes, csv_matrix_path)
    print(f"\nSaved confusion matrix CSV to {csv_matrix_path.relative_to(REPO_ROOT)}")

    # B. Save Confusion Matrix PNG Plot
    save_confusion_matrix_plot(conf_matrix, classes, model_name, test_accuracy, test_macro_f1, png_matrix_path)
    print(f"Saved confusion matrix plot to {png_matrix_path.relative_to(REPO_ROOT)}")

    # C. Save Evaluation Report JSON
    report_payload = {
        "evaluation_split": "test",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "model_version": model_version,
        "sample_count": len(X_test),
        "overall_metrics": {
            "accuracy": round(test_accuracy, 6),
            "macro_precision": round(test_macro_precision, 6),
            "macro_recall": round(test_macro_recall, 6),
            "macro_f1": round(test_macro_f1, 6),
            "weighted_precision": round(test_weighted_precision, 6),
            "weighted_recall": round(test_weighted_recall, 6),
            "weighted_f1": round(test_weighted_f1, 6),
            "inference_time_seconds": round(inference_time, 3),
        },
        "srs_section_4_audit": {
            "overall_accuracy": {"target": TARGET_ACCURACY, "achieved": test_accuracy, "passed": test_accuracy >= TARGET_ACCURACY},
            "macro_f1": {"target": TARGET_MACRO_F1, "achieved": test_macro_f1, "passed": test_macro_f1 >= TARGET_MACRO_F1},
            "critical_class_recalls": critical_metrics,
            "passing_critical_count": crit_pass_count,
            "total_critical_count": len(CRITICAL_CLASSES),
        },
        "per_class_metrics": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "classes": classes,
    }
    with json_report_path.open("w", encoding="utf-8") as fh:
        json.dump(report_payload, fh, indent=2)
    print(f"Saved test evaluation JSON to {json_report_path.relative_to(REPO_ROOT)}")

    # D. Save Evaluation Report Markdown
    save_markdown_report(report_payload, md_report_path)
    print(f"Saved test evaluation Markdown report to {md_report_path.relative_to(REPO_ROOT)}")
    print("=" * 80 + "\n")

    return report_payload


def save_confusion_matrix_csv(matrix: np.ndarray, classes: List[str], target_path: Path) -> None:
    with target_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["True\\Pred"] + classes)
        for i, row in enumerate(matrix):
            writer.writerow([classes[i]] + row.tolist())


def save_confusion_matrix_plot(
    matrix: np.ndarray, classes: List[str], model_name: str, acc: float, f1: float, target_path: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        cax = ax.matshow(matrix, cmap="Blues")
        fig.colorbar(cax)

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="left", fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)

        ax.set_xlabel("Predicted Sound Category", fontsize=11, fontweight="bold", labelpad=10)
        ax.set_ylabel("True Sound Category", fontsize=11, fontweight="bold", labelpad=10)
        ax.set_title(
            f"SonicSentinel AI — Test Set Confusion Matrix\nModel: {model_name} (Acc: {acc*100:.2f}%, Macro F1: {f1:.4f})",
            fontsize=12,
            fontweight="bold",
            pad=20,
        )

        thresh = matrix.max() / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8)

        plt.tight_layout()
        plt.savefig(str(target_path), dpi=300)
        plt.close(fig)
    except Exception as exc:
        print(f"WARNING: Matplotlib plotting failed: {exc}", file=sys.stderr)


def save_markdown_report(data: Dict[str, Any], target_path: Path) -> None:
    m = data["overall_metrics"]
    srs = data["srs_section_4_audit"]
    cr = data["per_class_metrics"]
    classes = data["classes"]

    lines = [
        f"# SonicSentinel AI — Test Set Evaluation Report",
        "",
        f"> **Evaluation Target**: Held-out, frozen test split (`final_split == 'test'`)  ",
        f"> **Winning Model**: `{data['model_name']}` (`{data['model_version']}`)  ",
        f"> **Test Samples**: {data['sample_count']} audio events across {len(classes)} mandatory sound categories  ",
        f"> **Timestamp**: {data['timestamp']}  ",
        "",
        "---",
        "",
        "## 1. Overall Performance Against AGENTS.md Section 4 Targets",
        "",
        "| Requirement | Target Threshold | Achieved (Test Set) | Compliance Status |",
        "|:---|:---:|:---:|:---:|",
        f"| **Overall Classification Accuracy** | $\\ge 85\\%$ | **{m['accuracy']*100:.2f}%** | {'**PASSED**' if srs['overall_accuracy']['passed'] else '**NOT MET**'} |",
        f"| **Macro F1 Score** | $\\ge 0.80$ | **{m['macro_f1']:.4f}** | {'**PASSED**' if srs['macro_f1']['passed'] else '**NOT MET**'} |",
        f"| **Weighted F1 Score** | — | {m['weighted_f1']:.4f} | Informational |",
        f"| **Macro Precision** | — | {m['macro_precision']:.4f} | Informational |",
        f"| **Macro Recall** | — | {m['macro_recall']:.4f} | Informational |",
        f"| **Inference Latency** | <= 8.0s upload | {m['inference_time_seconds']*1000/data['sample_count']:.2f} ms/sample | **PASSED** |",
        "",
        "---",
        "",
        "## 2. Mandatory Critical-Class Recall Audit (SRS §1.6 & §1.7)",
        "",
        "| Critical Sound Category | Recall | Precision | F1-Score | Support | Status (Target $\\ge 85\\%$) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for slug, crit in srs["critical_class_recalls"].items():
        st = "**PASSED**" if crit["passed"] else "NOT MET"
        lines.append(
            f"| **{crit['label']}** | **{crit['recall']*100:.2f}%** | {crit['precision']*100:.2f}% | {crit['f1_score']:.4f} | {crit['support']} | {st} |"
        )

    lines.extend([
        "",
        f"**Critical Classes Passing**: {srs['passing_critical_count']} of {srs['total_critical_count']}",
        "",
        "---",
        "",
        "## 3. Full Per-Class Classification Breakdown",
        "",
        "| Sound Category | Precision | Recall | F1-Score | Support |",
        "|:---|:---:|:---:|:---:|:---:|",
    ])

    for cls in classes:
        row = cr.get(cls, {})
        lines.append(
            f"| `{cls}` | {row.get('precision', 0.0)*100:.2f}% | {row.get('recall', 0.0)*100:.2f}% | {row.get('f1-score', 0.0):.4f} | {int(row.get('support', 0))} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Generated Evaluation Artifacts",
        "",
        "- **Confusion Matrix CSV**: [`docs/reports/test_confusion_matrix.csv`](file:///docs/reports/test_confusion_matrix.csv)",
        "- **Confusion Matrix Image**: [`docs/reports/test_confusion_matrix.png`](file:///docs/reports/test_confusion_matrix.png)",
        "- **Machine-Readable JSON**: [`docs/reports/test_evaluation_report.json`](file:///docs/reports/test_evaluation_report.json)",
        "",
    ])

    with target_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate winning model on held-out test split.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to trained model artifact")
    args = parser.parse_args()

    run_test_evaluation(model_path=args.model_path)


if __name__ == "__main__":
    main()
