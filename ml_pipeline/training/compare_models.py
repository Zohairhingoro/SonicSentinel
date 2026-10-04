"""
Model Comparison & Selection Engine for SonicSentinel AI.

Loads validation results from every trained model (SVM, Random Forest, XGBoost, CNN),
systematically compares:
  - Accuracy (Validation & Training, Overfit Gap)
  - Macro Precision, Macro Recall, Macro F1
  - Weighted F1
  - Critical-Class Recalls (Gunshot, Glass Breaking, Panic Scream, Aggression, Person Asking for Help)

Evaluates each model strictly against the hard targets in AGENTS.md Section 4:
  - Overall Accuracy >= 85%
  - Macro F1 >= 0.80
  - Critical-Class Recall >= 85% for each mandatory critical category

Selects and logs the winning model and exports a structured comparison report to
ml_pipeline/comparison_report/comparison_report.json.

Usage:
    python ml_pipeline/training/compare_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "ml_pipeline" / "training" / "results"
COMPARISON_DIR = REPO_ROOT / "ml_pipeline" / "comparison_report"

# Canonical critical classes per SRS §1.6 & §1.7 and AGENTS.md
CRITICAL_CLASSES: List[Tuple[str, str]] = [
    ("gunshot", "Gunshot"),
    ("glass_breaking", "Glass Breaking"),
    ("panic_scream", "Panic Scream"),
    ("aggression", "Aggression / Violent Conflict"),
    ("person_asking_for_help", "Person Asking for Help"),
]

# Section 4 SRS Hard Targets
TARGET_ACCURACY = 0.85
TARGET_MACRO_F1 = 0.80
TARGET_CRITICAL_RECALL = 0.85

MODEL_REGISTRY: Dict[str, str] = {
    "XGBoost": "xgb_results.json",
    "Random Forest": "rf_results.json",
    "SVM": "svm_results.json",
    "CNN": "cnn_results.json",
}


def load_results() -> Dict[str, Any]:
    loaded: Dict[str, Any] = {}
    missing: List[str] = []

    for name, filename in MODEL_REGISTRY.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            missing.append(f"{name} ({filename})")
        else:
            try:
                with path.open(encoding="utf-8") as fh:
                    loaded[name] = json.load(fh)
            except Exception as exc:
                print(f"WARNING: Could not parse {path.name}: {exc}", file=sys.stderr)

    if missing:
        print(f"INFO: Result files not found for: {', '.join(missing)}", file=sys.stderr)
    return loaded


def extract_metrics(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    cr = data.get("classification_report", {})
    macro_avg = cr.get("macro avg", {})
    weighted_avg = cr.get("weighted avg", {})

    val_acc = float(data.get("val_accuracy", 0.0))
    train_acc = float(data.get("train_accuracy", 0.0))
    macro_f1 = float(data.get("val_macro_f1") or macro_avg.get("f1-score") or 0.0)
    macro_prec = float(macro_avg.get("precision", 0.0))
    macro_rec = float(macro_avg.get("recall", 0.0))
    weighted_f1 = float(weighted_avg.get("f1-score", 0.0))
    training_time = float(data.get("training_time_seconds", 0.0))

    crit_recalls: Dict[str, float] = {}
    crit_precisions: Dict[str, float] = {}
    crit_f1s: Dict[str, float] = {}
    saved_crit = data.get("critical_class_recalls", {})

    for slug, label in CRITICAL_CLASSES:
        if slug in cr:
            rec = float(cr[slug].get("recall", 0.0))
            prec = float(cr[slug].get("precision", 0.0))
            f1 = float(cr[slug].get("f1-score", 0.0))
        elif slug in saved_crit:
            rec = float(saved_crit[slug])
            prec = 0.0
            f1 = 0.0
        else:
            rec, prec, f1 = 0.0, 0.0, 0.0

        crit_recalls[slug] = rec
        crit_precisions[slug] = prec
        crit_f1s[slug] = f1

    avg_crit_recall = float(np_mean(list(crit_recalls.values()))) if crit_recalls else 0.0
    crit_pass_count = sum(1 for r in crit_recalls.values() if r >= TARGET_CRITICAL_RECALL)

    return {
        "model_name": name,
        "model_version": data.get("model_version", "v1.0"),
        "best_hyperparameters": data.get("best_hyperparameters", {}),
        "val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "overfit_gap": round(train_acc - val_acc, 6),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_time_seconds": training_time,
        "critical_recalls": crit_recalls,
        "critical_precisions": crit_precisions,
        "critical_f1s": crit_f1s,
        "avg_critical_recall": round(avg_crit_recall, 4),
        "critical_pass_count": crit_pass_count,
        "srs_accuracy_pass": val_acc >= TARGET_ACCURACY,
        "srs_f1_pass": macro_f1 >= TARGET_MACRO_F1,
        "srs_critical_pass": crit_pass_count == len(CRITICAL_CLASSES),
    }


def np_mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def rank_models(metrics_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Rank primarily by validation macro F1, secondarily by validation accuracy
    sorted_items = sorted(
        metrics_map.values(),
        key=lambda m: (m["macro_f1"], m["val_accuracy"], m["avg_critical_recall"]),
        reverse=True,
    )
    for rank, item in enumerate(sorted_items, 1):
        item["rank"] = rank
    return sorted_items


def print_comparison_tables(table: List[Dict[str, Any]]) -> None:
    width = 100
    print("=" * width)
    print("  SONICSENTINEL AI — COMPREHENSIVE MODEL COMPARISON & SELECTION")
    print("  TechWiz7 NextWave AI/ML | AcousticX Intelligence")
    print("=" * width)

    # 1. Main Overall Metrics Table
    print("\n1. OVERALL VALIDATION METRICS COMPARISON:")
    print("-" * width)
    hdr = f"{'Rank':<5}{'Model':<16}{'Val Acc':>9}{'Macro Prec':>12}{'Macro Rec':>11}{'Macro F1':>10}{'Wtd F1':>9}{'Train Acc':>11}{'Gap':>8}{'Time(s)':>8}"
    print(hdr)
    print("-" * width)

    for r in table:
        print(
            f"  {r['rank']:<3}{r['model_name']:<16}"
            f"{r['val_accuracy']*100:>8.2f}%"
            f"{r['macro_precision']:>12.4f}"
            f"{r['macro_recall']:>11.4f}"
            f"{r['macro_f1']:>10.4f}"
            f"{r['weighted_f1']:>9.4f}"
            f"{r['train_accuracy']*100:>10.2f}%"
            f"{r['overfit_gap']*100:>+7.2f}%"
            f"{r['training_time_seconds']:>7.1f}s"
        )
    print("-" * width)

    # 2. Critical-Class Recall Table
    print(f"\n2. SRS §1.6 & §1.7 CRITICAL-CLASS RECALL BREAKDOWN (Target: >= {TARGET_CRITICAL_RECALL*100:.0f}%):")
    print("-" * width)
    crit_hdr = f"{'Critical Category':<32}" + "".join(f"{r['model_name']:>16}" for r in table)
    print(crit_hdr)
    print("-" * width)

    for slug, label in CRITICAL_CLASSES:
        row_str = f"  {label:<30}"
        for r in table:
            rec = r["critical_recalls"].get(slug, 0.0)
            status = "PASS" if rec >= TARGET_CRITICAL_RECALL else "FAIL"
            row_str += f"{rec*100:>10.1f}% [{status}]"
        print(row_str)

    print("-" * width)
    avg_row = f"  {'Average Critical Recall':<30}" + "".join(f"{r['avg_critical_recall']*100:>14.1f}%" for r in table)
    print(avg_row)
    passed_row = f"  {'Passing Critical Classes (/5)':<30}" + "".join(f"{r['critical_pass_count']:>14d}/5" for r in table)
    print(passed_row)
    print("-" * width)

    # 3. Section 4 Hard Targets Evaluation
    winner = table[0]
    print("\n3. AGENTS.MD SECTION 4 HARD-TARGETS COMPLIANCE AUDIT:")
    print("-" * width)
    print(f"{'Target Requirement':<42}{'Target Threshold':<18}{'Winning Model (' + winner['model_name'] + ')':<22}{'Compliance':<10}")
    print("-" * width)

    acc_status = "PASSED" if winner["srs_accuracy_pass"] else "NOT MET"
    print(f"  {'Overall Validation Accuracy':<40}{'>= ' + str(int(TARGET_ACCURACY*100)) + '%':<18}{winner['val_accuracy']*100:.2f}%{'':<15}{acc_status}")

    f1_status = "PASSED" if winner["srs_f1_pass"] else "NOT MET"
    print(f"  {'Overall Macro F1 Score':<40}{'>= ' + f'{TARGET_MACRO_F1:.2f}':<18}{winner['macro_f1']:.4f}{'':<15}{f1_status}")

    for slug, label in CRITICAL_CLASSES:
        rec = winner["critical_recalls"].get(slug, 0.0)
        c_status = "PASSED" if rec >= TARGET_CRITICAL_RECALL else "NOT MET"
        print(f"  {'Recall: ' + label:<40}{'>= ' + str(int(TARGET_CRITICAL_RECALL*100)) + '%':<18}{rec*100:.2f}%{'':<15}{c_status}")

    print("-" * width)

    # 4. Winning Model Announcement
    print("\n" + "=" * width)
    print(f"  WINNING MODEL SELECTED: {winner['model_name']} ({winner['model_version']})")
    print("=" * width)
    print(f"  Rank               : #1 of {len(table)} evaluated models")
    print(f"  Validation Accuracy: {winner['val_accuracy']*100:.2f}%")
    print(f"  Validation Macro F1: {winner['macro_f1']:.4f} (Lead over #2: +{(winner['macro_f1'] - table[1]['macro_f1']):.4f})")
    print(f"  Validation Precision: {winner['macro_precision']:.4f}")
    print(f"  Validation Recall  : {winner['macro_recall']:.4f}")
    print(f"  Weighted F1        : {winner['weighted_f1']:.4f}")
    print(f"  Best Parameters    : {winner['best_hyperparameters']}")
    print(f"  Deployment Artifact: ml_service/model/artifacts/xgb_model.joblib (Active)")
    print("=" * width + "\n")


def export_report(table: List[Dict[str, Any]], raw_results: Dict[str, Any]) -> Path:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    report_path = COMPARISON_DIR / "comparison_report.json"

    winner = table[0]
    payload = {
        "report_title": "SonicSentinel AI Model Comparison Report",
        "winning_model": {
            "name": winner["model_name"],
            "version": winner["model_version"],
            "val_accuracy": winner["val_accuracy"],
            "val_macro_f1": winner["macro_f1"],
            "macro_precision": winner["macro_precision"],
            "macro_recall": winner["macro_recall"],
            "weighted_f1": winner["weighted_f1"],
            "critical_recalls": winner["critical_recalls"],
            "best_hyperparameters": winner["best_hyperparameters"],
        },
        "srs_section_4_targets": {
            "overall_accuracy": {"target": TARGET_ACCURACY, "achieved": winner["val_accuracy"], "passed": winner["srs_accuracy_pass"]},
            "macro_f1": {"target": TARGET_MACRO_F1, "achieved": winner["macro_f1"], "passed": winner["srs_f1_pass"]},
            "critical_recalls": {
                slug: {
                    "label": label,
                    "target": TARGET_CRITICAL_RECALL,
                    "achieved": winner["critical_recalls"].get(slug, 0.0),
                    "passed": winner["critical_recalls"].get(slug, 0.0) >= TARGET_CRITICAL_RECALL,
                }
                for slug, label in CRITICAL_CLASSES
            },
        },
        "comparison_table": table,
        "models_evaluated": [r["model_name"] for r in table],
    }

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return report_path


def main() -> None:
    results = load_results()
    if not results:
        print("ERROR: No model results found in ml_pipeline/training/results/", file=sys.stderr)
        sys.exit(1)

    metrics_map = {name: extract_metrics(name, data) for name, data in results.items()}
    table = rank_models(metrics_map)

    print_comparison_tables(table)
    report_file = export_report(table, results)
    print(f"Structured comparison report exported to {report_file.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
