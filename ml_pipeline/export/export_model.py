"""
Model Export Script for SonicSentinel AI.

Reads the winning model selected by ml_pipeline/training/compare_models.py,
copies its .joblib binary into ml_service/model/artifacts/, and writes a fresh
metadata.json that matches exactly what ml_service/model/loader.py expects.

Loader contract (ml_service/model/loader.py):
    metadata.json must contain:
        version      str     -- semver string (e.g. "1.2.0")
        trained_on   str     -- ISO-8601 date of the training run
        metrics      dict    -- at minimum: accuracy, macro_f1, precision, recall
        classes      list    -- display labels matching config/classes.json order
        feature_type str     -- descriptor for feature set used
        sample_rate  int     -- audio sample rate the feature extractor was called with

Extended fields written by this script (consumed by Django webapp, not the loader):
        model_type           str
        model_version        str
        best_hyperparameters dict
        validation_metrics   dict   -- val-split metrics from training results JSON
        test_metrics         dict   -- held-out test-split metrics (if available)
        critical_class_recalls dict -- per-class recall on critical classes (val + test)
        noise_robustness     dict   -- accuracy degradation per SNR condition (if available)
        export_timestamp     str

Outputs:
    ml_service/model/artifacts/<slug>_model.joblib  -- symlinked or copied model binary
    ml_service/model/artifacts/metadata.json        -- overwritten with fresh content

Usage:
    python ml_pipeline/export/export_model.py
    python ml_pipeline/export/export_model.py --model xgb
    python ml_pipeline/export/export_model.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMPARISON_REPORT   = REPO_ROOT / "ml_pipeline" / "comparison_report" / "comparison_report.json"
TRAINING_RESULTS    = REPO_ROOT / "ml_pipeline" / "training" / "results"
TEST_EVAL_REPORT    = REPO_ROOT / "docs" / "reports" / "test_evaluation_report.json"
NOISE_REPORT        = REPO_ROOT / "docs" / "reports" / "noise_robustness_report.json"
ARTIFACTS_DIR       = REPO_ROOT / "ml_service" / "model" / "artifacts"
CLASSES_JSON        = REPO_ROOT / "config" / "classes.json"

# Display-name class labels in canonical order (AGENTS.md Sec.2)
CANONICAL_DISPLAY_LABELS: List[str] = [
    "Machinery Fault",
    "Glass Breaking",
    "Alarm or Siren",
    "Vehicle Horn",
    "Animal Sound",
    "Gunshot",
    "Panic Scream",
    "Aggression or Violent Conflict",
    "Person Asking for Help",
    "Background Noise",
]

# Slug -> display label mapping used for critical-recall reporting
CRITICAL_SLUG_LABELS: Dict[str, str] = {
    "gunshot":               "Gunshot",
    "glass_breaking":        "Glass Breaking",
    "panic_scream":          "Panic Scream",
    "aggression":            "Aggression / Violent Conflict",
    "person_asking_for_help":"Person Asking for Help",
}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON from path, return None if missing or malformed."""
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"  WARNING: Could not parse {path.relative_to(REPO_ROOT)}: {exc}", file=sys.stderr)
        return None


def resolve_class_labels() -> List[str]:
    """Return display class labels from config/classes.json, falling back to CANONICAL_DISPLAY_LABELS."""
    data = load_json(CLASSES_JSON)
    if data and isinstance(data.get("categories"), list):
        return data["categories"]
    return CANONICAL_DISPLAY_LABELS


def pick_winning_model(override_slug: Optional[str]) -> Dict[str, Any]:
    """
    Determine which model to export.

    Priority:
      1. --model CLI override (slug: xgb | rf | svm | cnn)
      2. comparison_report.json winning_model block
      3. Default: xgb

    Returns a dict with keys: slug, name, version, results_json, artifact_joblib
    """
    slug_map = {
        "xgb": ("XGBoost",       "xgb_v1.0"),
        "rf":  ("Random Forest",  "rf_v1.0"),
        "svm": ("SVM",            "svm_v1.0"),
        "cnn": ("CNN",            "cnn_v1.0"),
    }

    if override_slug:
        slug = override_slug.lower().strip()
        name, version = slug_map.get(slug, ("XGBoost", "xgb_v1.0"))
        print(f"  Model override: --model {slug} -> {name} ({version})")
    else:
        comp = load_json(COMPARISON_REPORT)
        if comp and comp.get("winning_model"):
            w    = comp["winning_model"]
            name = w.get("name", "XGBoost")
            version = w.get("version", "xgb_v1.0")
            slug = (
                "xgb" if "xgb" in name.lower() else
                "rf"  if ("rf" in name.lower() or "random" in name.lower()) else
                "svm" if "svm" in name.lower() else
                "cnn"
            )
            print(f"  Winning model from comparison_report.json: {name} ({version})")
        else:
            slug, name, version = "xgb", "XGBoost", "xgb_v1.0"
            print(f"  comparison_report.json not found; defaulting to {name}")

    results_json = TRAINING_RESULTS / f"{slug}_results.json"
    artifact     = ARTIFACTS_DIR    / f"{slug}_model.joblib"

    if not artifact.exists():
        # Also check training results dir
        fallback = TRAINING_RESULTS / f"{slug}_model.joblib"
        if fallback.exists():
            artifact = fallback

    return {
        "slug":             slug,
        "name":             name,
        "version":          version,
        "results_json":     results_json,
        "artifact_joblib":  artifact,
    }


def build_validation_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract validation metrics from training results JSON."""
    cr = results.get("classification_report", {})
    macro = cr.get("macro avg", {})
    return {
        "accuracy":         round(float(results.get("val_accuracy",  0.0)), 6),
        "macro_f1":         round(float(results.get("val_macro_f1",  0.0)), 6),
        "macro_precision":  round(float(macro.get("precision",       0.0)), 6),
        "macro_recall":     round(float(macro.get("recall",          0.0)), 6),
    }


def build_test_metrics(test_report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract test-split metrics from the evaluation report, if available."""
    if not test_report:
        return None
    m = test_report.get("overall_metrics", {})
    return {
        "accuracy":         round(float(m.get("accuracy",        0.0)), 6),
        "macro_f1":         round(float(m.get("macro_f1",        0.0)), 6),
        "macro_precision":  round(float(m.get("macro_precision", 0.0)), 6),
        "macro_recall":     round(float(m.get("macro_recall",    0.0)), 6),
        "weighted_f1":      round(float(m.get("weighted_f1",     0.0)), 6),
        "sample_count":     int(test_report.get("sample_count",  0)),
    }


def build_critical_recalls(
    results: Dict[str, Any],
    test_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble per-critical-class recall from both val and test sources."""
    val_crit  = results.get("critical_class_recalls", {})
    test_crit = (test_report or {}).get("srs_section_4_audit", {}).get("critical_class_recalls", {})

    output: Dict[str, Any] = {}
    for slug, label in CRITICAL_SLUG_LABELS.items():
        entry: Dict[str, Any] = {"label": label}
        if slug in val_crit:
            entry["val_recall"]  = round(float(val_crit[slug]),  4)
        if slug in test_crit:
            entry["test_recall"] = round(float(test_crit[slug].get("recall", 0.0)), 4)
            entry["test_precision"] = round(float(test_crit[slug].get("precision", 0.0)), 4)
            entry["test_f1"]     = round(float(test_crit[slug].get("f1_score", 0.0)), 4)
        entry["target"] = 0.85
        output[slug] = entry
    return output


def build_noise_robustness_summary(noise_report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract a compact noise robustness summary from noise_robustness_report.json."""
    if not noise_report:
        return None
    results = noise_report.get("results", [])
    if not results:
        return None

    clean = next((r for r in results if r.get("condition", "").lower().startswith("clean")), None)
    summary: Dict[str, Any] = {"conditions": []}

    for res in results:
        cond: Dict[str, Any] = {
            "label":    res.get("condition", ""),
            "accuracy": res.get("accuracy", 0.0),
            "macro_f1": res.get("macro_f1", 0.0),
        }
        if clean and res is not clean:
            cond["accuracy_delta"] = round(res["accuracy"] - clean["accuracy"], 4)
        summary["conditions"].append(cond)

    if clean:
        summary["clean_accuracy"] = clean.get("accuracy", 0.0)
    return summary


def render_terminal_summary(
    model_info:    Dict[str, Any],
    val_metrics:   Dict[str, Any],
    test_metrics:  Optional[Dict[str, Any]],
    critical:      Dict[str, Any],
    artifact_dest: Path,
    metadata_dest: Path,
    dry_run:       bool,
) -> None:
    label = "[DRY RUN] " if dry_run else ""
    print("\n" + "=" * 80)
    print(f"  SONICSENTINEL AI -- MODEL EXPORT SUMMARY  {label}")
    print("=" * 80)
    print(f"  Model        : {model_info['name']} ({model_info['version']})")
    print(f"  Source binary: {model_info['artifact_joblib'].relative_to(REPO_ROOT)}")
    print(f"  Dest binary  : {artifact_dest.relative_to(REPO_ROOT)}")
    print(f"  metadata.json: {metadata_dest.relative_to(REPO_ROOT)}")
    print()
    print(f"  Validation metrics (val split):")
    print(f"    Accuracy  : {val_metrics['accuracy']*100:.2f}%")
    print(f"    Macro F1  : {val_metrics['macro_f1']:.4f}")
    if test_metrics:
        print(f"  Test metrics (held-out test split, {test_metrics['sample_count']} samples):")
        print(f"    Accuracy  : {test_metrics['accuracy']*100:.2f}%")
        print(f"    Macro F1  : {test_metrics['macro_f1']:.4f}")
    print()
    print(f"  Critical-class recalls (target >= 85%):")
    for slug, m in critical.items():
        val_r  = m.get("val_recall",  None)
        test_r = m.get("test_recall", None)
        val_s  = f"val={val_r*100:.1f}%"  if val_r  is not None else "val=n/a"
        test_s = f"test={test_r*100:.1f}%" if test_r is not None else "test=n/a"
        passed = (test_r or val_r or 0.0) >= 0.85
        flag   = "PASSED" if passed else "NOT MET"
        print(f"    {m['label']:<34}: {val_s:<14} {test_s:<16} {flag}")
    print("=" * 80 + "\n")


def export(
    model_slug: Optional[str],
    dry_run:    bool,
) -> None:
    print("=" * 80)
    print("  SONICSENTINEL AI -- MODEL EXPORT PIPELINE")
    print("  Writes to ml_service/model/artifacts/")
    print("=" * 80 + "\n")

    # 1. Resolve winning model
    model_info = pick_winning_model(model_slug)

    src_artifact = model_info["artifact_joblib"]
    if not src_artifact.exists():
        raise FileNotFoundError(
            f"Model binary not found: {src_artifact}\n"
            f"Run the training scripts first (ml_pipeline/training/train_{model_info['slug']}.py)"
        )
    print(f"  Source artifact : {src_artifact.relative_to(REPO_ROOT)} ({src_artifact.stat().st_size//1024} KB)")

    # 2. Load supplementary data
    results_data = load_json(model_info["results_json"]) or {}
    test_report  = load_json(TEST_EVAL_REPORT)
    noise_report = load_json(NOISE_REPORT)

    if not results_data:
        print(f"  WARNING: Training results JSON not found at {model_info['results_json'].relative_to(REPO_ROOT)} -- metrics will be empty", file=sys.stderr)

    # 3. Build metric blocks
    val_metrics  = build_validation_metrics(results_data)
    test_metrics = build_test_metrics(test_report)
    critical     = build_critical_recalls(results_data, test_report)
    noise_summary= build_noise_robustness_summary(noise_report)

    # 4. Resolve canonical class labels
    classes = resolve_class_labels()

    # 5. Semver: bump patch from existing metadata if present
    existing_meta = load_json(ARTIFACTS_DIR / "metadata.json") or {}
    existing_ver  = existing_meta.get("version", "1.0.0")
    try:
        major, minor, patch = [int(x) for x in existing_ver.split(".")]
        new_version = f"{major}.{minor}.{patch + 1}"
    except Exception:
        new_version = "1.0.0"

    # 6. Assemble metadata.json payload
    # Fields consumed by loader.py are at the top level; extended fields follow.
    metadata: Dict[str, Any] = {
        # ── loader.py required fields ──────────────────────────────────────
        "version":      new_version,
        "trained_on":   time.strftime("%Y-%m-%d", time.gmtime()),
        "metrics": {
            # Prefer test metrics if available; fall back to val
            "accuracy":  (test_metrics or val_metrics)["accuracy"],
            "macro_f1":  (test_metrics or val_metrics)["macro_f1"],
            "precision": (test_metrics or val_metrics)["macro_precision"],
            "recall":    (test_metrics or val_metrics)["macro_recall"],
        },
        "classes":      classes,
        "feature_type": "mfcc_mel_chroma_zcr_rms_spectral",
        "sample_rate":  22050,

        # ── extended provenance fields ──────────────────────────────────────
        "model_type":           model_info["name"],
        "model_version":        model_info["version"],
        "best_hyperparameters": results_data.get("best_hyperparameters", {}),
        "validation_metrics":   val_metrics,
        "test_metrics":         test_metrics,
        "critical_class_recalls": critical,
        "noise_robustness":     noise_summary,
        "export_timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "srs_targets": {
            "accuracy_target": 0.85,
            "macro_f1_target": 0.80,
            "critical_recall_target": 0.85,
        },
    }

    # 7. Determine destination paths
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dest_slug     = model_info["slug"]
    artifact_dest = ARTIFACTS_DIR / f"{dest_slug}_model.joblib"
    metadata_dest = ARTIFACTS_DIR / "metadata.json"

    render_terminal_summary(
        model_info, val_metrics, test_metrics,
        critical, artifact_dest, metadata_dest, dry_run,
    )

    if dry_run:
        print("  DRY RUN: no files written.\n")
        return

    # 8a. Copy model binary (skip if source == dest)
    if src_artifact.resolve() != artifact_dest.resolve():
        shutil.copy2(src_artifact, artifact_dest)
        print(f"  Copied  {src_artifact.name} -> {artifact_dest.relative_to(REPO_ROOT)}")
    else:
        print(f"  Model binary already at destination -- no copy needed")

    # 8b. Write metadata.json
    with metadata_dest.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"  Written {metadata_dest.relative_to(REPO_ROOT)}")

    # 9. Sanity-check: verify loader can read the artifact
    print("\n  Verifying loader can read the exported artifact ...")
    try:
        from ml_service.model.loader import load_model_artifact
        art = load_model_artifact(artifacts_dir=ARTIFACTS_DIR)
        assert art.version == new_version, f"version mismatch: {art.version}"
        assert hasattr(art.model, "predict"), "loaded object has no predict()"
        print(f"  Loader verification PASSED  (version={art.version}, model={type(art.model).__name__})")
    except Exception as exc:
        print(f"  WARNING: Loader verification raised an exception: {exc}", file=sys.stderr)
        print("  The files were still written; check ml_service/model/loader.py manually.", file=sys.stderr)

    print(f"\n  Export complete -> {ARTIFACTS_DIR.relative_to(REPO_ROOT)}/\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Serialize the winning model and write metadata.json into "
            "ml_service/model/artifacts/ for the FastAPI loader."
        )
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "rf", "svm", "cnn"],
        default=None,
        help="Override which model to export (default: read from comparison_report.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be exported without writing any files",
    )
    args = parser.parse_args()
    export(model_slug=args.model, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
