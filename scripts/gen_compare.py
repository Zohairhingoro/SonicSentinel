import pathlib

dest = pathlib.Path("ml_pipeline/training/compare_models.py")

code = """\
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "ml_pipeline" / "training" / "results"
COMPARISON_DIR = REPO_ROOT / "ml_pipeline" / "comparison_report"
CRITICAL_CLASSES = ["gunshot", "glass_breaking", "panic_scream", "aggression", "person_asking_for_help"]

MODEL_FILES = {
    "SVM": "svm_results.json",
    "Random Forest": "rf_results.json",
    "XGBoost": "xgb_results.json",
}


def load_results() -> Dict[str, Any]:
    loaded = {}
    missing = []
    for name, fname in MODEL_FILES.items():
        p = RESULTS_DIR / fname
        if not p.exists():
            missing.append(f"{name} ({fname})")
        else:
            with p.open(encoding="utf-8") as fh:
                loaded[name] = json.load(fh)
    if missing:
        print(f"WARNING: Missing result files for: {', '.join(missing)}", file=sys.stderr)
    return loaded


def rank_models(results: Dict[str, Any]) -> List[str]:
    return sorted(results.keys(), key=lambda n: results[n]["val_macro_f1"], reverse=True)


def build_comparison_table(results: Dict[str, Any], ranking: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for rank, name in enumerate(ranking, 1):
        r = results[name]
        crit = r.get("critical_class_recalls", {})
        row = {
            "rank": rank,
            "model_name": name,
            "model_version": r.get("model_version", ""),
            "train_accuracy": r["train_accuracy"],
            "val_accuracy": r["val_accuracy"],
            "train_macro_f1": r["train_macro_f1"],
            "val_macro_f1": r["val_macro_f1"],
            "overfit_gap_accuracy": round(r["train_accuracy"] - r["val_accuracy"], 6),
            "overfit_gap_f1": round(r["train_macro_f1"] - r["val_macro_f1"], 6),
            "training_time_seconds": r.get("training_time_seconds", 0),
            "critical_class_recalls": crit,
            "srs_accuracy_pass": r["val_accuracy"] >= 0.85,
            "srs_f1_pass": r["val_macro_f1"] >= 0.80,
            "critical_recalls_passing": {cls: (rec >= 0.85) for cls, rec in crit.items()},
        }
        rows.append(row)
    return rows


def print_report(table: List[Dict[str, Any]]) -> None:
    sep = "-" * 80
    print(sep)
    print("  MODEL COMPARISON REPORT  —  SonicSentinel AI")
    print(sep)
    header = f"{'Rank':<5}{'Model':<22}{'Val Acc':>10}{'Val F1':>10}{'Train Acc':>11}{'Gap':>8}{'Time(s)':>9}"
    print(header)
    print("-" * len(header))
    for row in table:
        acc_flag = " PASS" if row["srs_accuracy_pass"] else " FAIL"
        f1_flag  = " PASS" if row["srs_f1_pass"] else " FAIL"
        print(
            f"  {row['rank']:<4}{row['model_name']:<22}"
            f"{row['val_accuracy']*100:>8.2f}%{f1_flag}"
            f"  {row['val_macro_f1']:>6.4f}{f1_flag}"
            f"  {row['train_accuracy']*100:>8.2f}%"
            f"  {row['overfit_gap_accuracy']*100:>+6.2f}%"
            f"  {row['training_time_seconds']:>7.1f}s"
        )
    print()
    print("SRS Targets: Val Accuracy >= 85% | Val Macro F1 >= 0.80 | Critical Recall >= 85%")
    print()
    print("Critical-Class Recall per Model (SRS §1.6 threshold: 85%):")
    print(f"  {'Class':<30}", end="")
    for row in table:
        print(f"{row['model_name']:>16}", end="")
    print()
    for cls in CRITICAL_CLASSES:
        print(f"  {cls:<30}", end="")
        for row in table:
            rec = row["critical_class_recalls"].get(cls, 0.0)
            passed = "PASS" if rec >= 0.85 else "FAIL"
            print(f"  {rec*100:>5.1f}% {passed:>4}", end="")
        print()
    print()
    best = table[0]
    print(f"Best Model (by Val Macro F1): {best['model_name']} — {best['model_version']}")
    print(f"  Val Accuracy : {best['val_accuracy']*100:.2f}%  (SRS target >=85%: {'MET' if best['srs_accuracy_pass'] else 'NOT MET'})")
    print(f"  Val Macro F1 : {best['val_macro_f1']:.4f}     (SRS target >=0.80: {'MET' if best['srs_f1_pass'] else 'NOT MET'})")
    print(sep)


def main() -> None:
    results = load_results()
    if not results:
        print("ERROR: No model results found. Run the training scripts first.", file=sys.stderr)
        sys.exit(1)

    ranking = rank_models(results)
    table = build_comparison_table(results, ranking)
    print_report(table)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    report_path = COMPARISON_DIR / "comparison_report.json"
    payload = {
        "comparison_table": table,
        "best_model": table[0]["model_name"],
        "best_model_version": table[0]["model_version"],
        "srs_targets": {
            "val_accuracy": 0.85,
            "val_macro_f1": 0.80,
            "critical_recall": 0.85,
        },
        "model_results": {name: results[name] for name in ranking},
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"Comparison report saved to {report_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
"""

dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(code, encoding="utf-8")
print("compare_models.py written:", len(code), "bytes")