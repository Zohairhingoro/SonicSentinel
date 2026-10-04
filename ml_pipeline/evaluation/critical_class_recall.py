"""
Critical-Class Recall Reporter for SonicSentinel AI.

Isolates and reports test-set recall for the five mandatory critical sound
categories defined in AGENTS.md §4:
    Gunshot, Glass Breaking, Panic Scream, Aggression, Person Asking for Help

Each class is evaluated against the 85% recall target from AGENTS.md §4.
The script reads results from the existing test-evaluation JSON produced by
evaluate.py (no re-inference needed by default) and emits:

  - Terminal: rich, color-coded compliance table
  - JSON     : docs/reports/critical_class_recall.json
  - Markdown : docs/reports/critical_class_recall.md

If --rerun is supplied the model is re-evaluated on the held-out test split
first (delegates to evaluate.py's run_test_evaluation()) before reading the
fresh results.

Usage:
    python ml_pipeline/evaluation/critical_class_recall.py
    python ml_pipeline/evaluation/critical_class_recall.py --rerun
    python ml_pipeline/evaluation/critical_class_recall.py --source-json path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_REPORT_JSON = REPO_ROOT / "docs" / "reports" / "test_evaluation_report.json"
REPORTS_DIR = REPO_ROOT / "docs" / "reports"

# Canonical 5 critical classes — slugs must match dataset folder names (AGENTS.md §2)
CRITICAL_CLASSES: List[Tuple[str, str]] = [
    ("gunshot",              "Gunshot"),
    ("glass_breaking",       "Glass Breaking"),
    ("panic_scream",         "Panic Scream"),
    ("aggression",           "Aggression / Violent Conflict"),
    ("person_asking_for_help", "Person Asking for Help"),
]

# Hard target from AGENTS.md §4
RECALL_TARGET = 0.85

# ANSI colour codes — degrade gracefully on terminals that strip them
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _colourise(text: str, colour: str) -> str:
    """Wrap text in ANSI colour only when stdout is a real TTY."""
    if sys.stdout.isatty():
        return f"{colour}{text}{_RESET}"
    return text


def load_eval_report(source_path: Path) -> Dict[str, Any]:
    """
    Load the JSON evaluation report produced by evaluate.py.

    Raises FileNotFoundError when the report is absent so callers can direct
    the user to run evaluate.py or pass --rerun.
    """
    if not source_path.exists():
        raise FileNotFoundError(
            f"Test evaluation report not found at {source_path}.\n"
            "Run  python ml_pipeline/evaluation/evaluate.py  first, or pass "
            "--rerun to have this script trigger it automatically."
        )
    with source_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def extract_critical_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull recall, precision, F1, support, and compliance flag for each of the
    five critical classes out of the top-level evaluation report.

    The report produced by evaluate.py already contains a
    srs_section_4_audit.critical_class_recalls block; this function uses that
    data and supplements it with anything missing (e.g. if a report from an
    older run lacks a field).
    """
    per_class: Dict[str, Any] = report.get("per_class_metrics", {})
    existing_audit: Dict[str, Any] = (
        report.get("srs_section_4_audit", {}).get("critical_class_recalls", {})
    )

    result: Dict[str, Any] = {}
    for slug, label in CRITICAL_CLASSES:
        if slug in existing_audit:
            entry = existing_audit[slug]
            recall    = float(entry.get("recall", 0.0))
            precision = float(entry.get("precision", 0.0))
            f1        = float(entry.get("f1_score",   0.0))
            support   = int(entry.get("support",      0))
        elif slug in per_class:
            m         = per_class[slug]
            recall    = float(m.get("recall",    0.0))
            precision = float(m.get("precision", 0.0))
            f1        = float(m.get("f1-score",  0.0))
            support   = int(m.get("support",     0))
        else:
            recall = precision = f1 = 0.0
            support = 0

        gap    = recall - RECALL_TARGET
        passed = recall >= RECALL_TARGET

        result[slug] = {
            "label":            label,
            "recall":           round(recall,    4),
            "precision":        round(precision, 4),
            "f1_score":         round(f1,        4),
            "support":          support,
            "target_threshold": RECALL_TARGET,
            "gap_from_target":  round(gap,       4),
            "passed":           passed,
        }
    return result


def render_terminal(
    critical: Dict[str, Any],
    report_meta: Dict[str, Any],
) -> None:
    """Print the focused critical-class recall audit to stdout."""
    model_name    = report_meta.get("model_name",    "—")
    model_version = report_meta.get("model_version", "—")
    ts            = report_meta.get("timestamp",     "—")
    sample_count  = report_meta.get("sample_count",  "—")

    header = "SONICSENTINEL AI — CRITICAL-CLASS RECALL AUDIT (AGENTS.md §4)"
    print("\n" + "=" * 80)
    print(f"  {_colourise(header, _BOLD)}")
    print("  Competition Theme: AcousticX Intelligence (TechWiz7 NextWave AI/ML)")
    print("=" * 80)
    print(f"  Source Model   : {model_name} ({model_version})")
    print(f"  Evaluation Set : Held-out test split ({sample_count} samples)")
    print(f"  Report Date    : {ts}")
    print(f"  Recall Target  : >= {RECALL_TARGET * 100:.0f}%  (AGENTS.md §4, SRS §1.7)")
    print("-" * 80)

    col_w = {
        "category": 34, "recall": 8, "prec": 10,
        "f1": 8, "n": 7, "gap": 10,
    }
    hdr = (
        f"  {'Sound Category':<{col_w['category']}}"
        f"{'Recall':>{col_w['recall']}}"
        f"{'Precision':>{col_w['prec']}}"
        f"{'F1':>{col_w['f1']}}"
        f"{'n':>{col_w['n']}}"
        f"{'Gap':>{col_w['gap']}}"
        f"  Status"
    )
    print(hdr)
    print("  " + "-" * 76)

    passing = 0
    for slug, m in critical.items():
        gap_str = f"{m['gap_from_target'] * 100:+.2f}%"
        status  = "PASSED (>= 85%)" if m["passed"] else "NOT MET (< 85%)"
        colour  = _GREEN if m["passed"] else _RED

        row = (
            f"  {m['label']:<{col_w['category']}}"
            f"{m['recall'] * 100:>{col_w['recall']}.2f}%"
            f"{m['precision'] * 100:>{col_w['prec']}.2f}%"
            f"{m['f1_score']:>{col_w['f1']}.4f}"
            f"{m['support']:>{col_w['n']}}"
            f"{gap_str:>{col_w['gap']}}"
            f"  {_colourise(status, colour)}"
        )
        print(row)
        if m["passed"]:
            passing += 1

    total   = len(CRITICAL_CLASSES)
    failing = total - passing

    print("  " + "-" * 76)
    print(
        f"\n  Critical Classes Passing : "
        f"{_colourise(str(passing), _GREEN)} / {total}"
        f"  |  Failing : "
        f"{_colourise(str(failing), _RED if failing else _GREEN)} / {total}"
    )

    if failing:
        print(
            f"\n  {_colourise('ACTION REQUIRED', _YELLOW)}: "
            f"{failing} critical class(es) do not meet the >= 85% recall target."
        )
        print("  Refer to the Gap column to prioritise which classes need improvement.")
    else:
        print(
            f"\n  {_colourise('ALL TARGETS MET', _GREEN)}: "
            "Every critical class exceeds the >= 85% recall threshold."
        )

    print("=" * 80 + "\n")


def save_json_report(
    critical: Dict[str, Any],
    report_meta: Dict[str, Any],
    target_path: Path,
) -> None:
    """Persist focused critical-class recall data as JSON."""
    passing = sum(1 for m in critical.values() if m["passed"])
    payload = {
        "report_type":           "critical_class_recall_audit",
        "timestamp":             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_model":          report_meta.get("model_name",    "—"),
        "source_model_version":  report_meta.get("model_version", "—"),
        "source_eval_timestamp": report_meta.get("timestamp",     "—"),
        "evaluation_split":      "test",
        "sample_count":          report_meta.get("sample_count",  0),
        "recall_target":         RECALL_TARGET,
        "passing_count":         passing,
        "failing_count":         len(CRITICAL_CLASSES) - passing,
        "total_critical_count":  len(CRITICAL_CLASSES),
        "all_targets_met":       passing == len(CRITICAL_CLASSES),
        "critical_classes":      critical,
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_markdown_report(
    critical: Dict[str, Any],
    report_meta: Dict[str, Any],
    target_path: Path,
) -> None:
    """Persist focused critical-class recall data as a Markdown report."""
    model_name    = report_meta.get("model_name",    "—")
    model_version = report_meta.get("model_version", "—")
    ts            = report_meta.get("timestamp",     "—")
    sample_count  = report_meta.get("sample_count",  "—")
    passing       = sum(1 for m in critical.values() if m["passed"])
    total         = len(CRITICAL_CLASSES)

    lines: List[str] = [
        "# SonicSentinel AI — Critical-Class Recall Audit",
        "",
        "> **Report Type**: Focused critical-class recall (AGENTS.md §4, SRS §1.7)  ",
        f"> **Source Model**: `{model_name}` (`{model_version}`)  ",
        f"> **Evaluation Split**: Held-out, frozen test split (`{sample_count}` samples)  ",
        f"> **Source Evaluation Timestamp**: {ts}  ",
        f"> **Report Generated**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  ",
        "",
        "---",
        "",
        "## Mandatory Critical-Class Recall Targets (AGENTS.md §4)",
        "",
        (
            "The five classes below carry the highest safety stakes and are subject to an "
            "**individual >= 85% recall** requirement defined in AGENTS.md §4 and SRS §1.7. "
            "A model that fails any single class does not meet the competition specification "
            "for that category, regardless of overall accuracy."
        ),
        "",
        "| # | Critical Sound Category | Recall | Precision | F1-Score | Support | "
        "Gap from 85% | Status |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for i, (slug, m) in enumerate(critical.items(), start=1):
        gap_str = f"{m['gap_from_target'] * 100:+.2f}%"
        status  = "PASSED" if m["passed"] else "NOT MET"
        lines.append(
            f"| {i} | **{m['label']}** "
            f"| **{m['recall'] * 100:.2f}%** "
            f"| {m['precision'] * 100:.2f}% "
            f"| {m['f1_score']:.4f} "
            f"| {m['support']} "
            f"| `{gap_str}` "
            f"| {status} |"
        )

    lines.extend([
        "",
        f"**Critical Classes Passing**: {passing} of {total}  ",
        f"**Recall Target**: >= {RECALL_TARGET * 100:.0f}% per class (AGENTS.md §4)  ",
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ])

    if passing == total:
        lines.append(
            "> All five critical classes exceed the 85% recall threshold. "
            "The model satisfies AGENTS.md §4 critical-class requirements."
        )
    else:
        failing_labels = [m["label"] for m in critical.values() if not m["passed"]]
        lines.append(
            f"> **{total - passing} of {total} critical classes do not meet the target.** "
            "Classes requiring attention:"
        )
        for label in failing_labels:
            lines.append(f">   - {label}")

        lines.extend([
            "",
            "### Recommended Next Steps",
            "",
            "| Priority | Action |",
            "|:---:|:---|",
        ])

        # Order worst-to-best recall so the most urgent gap is addressed first
        sorted_failing = sorted(
            [(slug, m) for slug, m in critical.items() if not m["passed"]],
            key=lambda x: x[1]["recall"],
        )
        priority_labels = ["High", "High", "Medium", "Medium", "Low"]
        for rank, (slug, m) in enumerate(sorted_failing):
            pri = priority_labels[min(rank, len(priority_labels) - 1)]
            gap_abs = abs(m["gap_from_target"])
            hint = (
                f"Boost **{m['label']}** recall — currently {m['recall'] * 100:.2f}%, "
                f"needs +{gap_abs * 100:.2f} pp. "
                "Consider targeted data augmentation, class-weighted loss, or "
                "decision-threshold calibration."
            )
            lines.append(f"| {pri} | {hint} |")

    lines.extend([
        "",
        "---",
        "",
        "## Related Reports",
        "",
        "- **Full Test Evaluation**: "
        "[`docs/reports/test_evaluation_report.md`]"
        "(file:///docs/reports/test_evaluation_report.md)",
        "- **JSON Machine-Readable**: "
        "[`docs/reports/critical_class_recall.json`]"
        "(file:///docs/reports/critical_class_recall.json)",
        "- **Confusion Matrix (PNG)**: "
        "[`docs/reports/test_confusion_matrix.png`]"
        "(file:///docs/reports/test_confusion_matrix.png)",
        "",
    ])

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def run(source_json: Path, rerun: bool) -> None:
    """Load results, render terminal output, and save both report formats."""
    if rerun:
        print("--rerun requested: triggering full test evaluation first ...\n")
        from ml_pipeline.evaluation.evaluate import run_test_evaluation
        run_test_evaluation()

    report = load_eval_report(source_json)

    critical = extract_critical_metrics(report)
    report_meta = {
        "model_name":    report.get("model_name",    "—"),
        "model_version": report.get("model_version", "—"),
        "timestamp":     report.get("timestamp",     "—"),
        "sample_count":  report.get("sample_count",  0),
    }

    render_terminal(critical, report_meta)

    json_out = REPORTS_DIR / "critical_class_recall.json"
    save_json_report(critical, report_meta, json_out)
    print(f"Saved JSON report : {json_out.relative_to(REPO_ROOT)}")

    md_out = REPORTS_DIR / "critical_class_recall.md"
    save_markdown_report(critical, report_meta, md_out)
    print(f"Saved Markdown    : {md_out.relative_to(REPO_ROOT)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report test-set recall for the five mandatory critical sound classes "
            "and flag each against the 85% target from AGENTS.md §4."
        )
    )
    parser.add_argument(
        "--source-json",
        type=Path,
        default=EVAL_REPORT_JSON,
        help="Path to an existing test_evaluation_report.json (default: docs/reports/)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run full model evaluation on the test split before reporting",
    )
    args = parser.parse_args()
    run(source_json=args.source_json, rerun=args.rerun)


if __name__ == "__main__":
    main()
