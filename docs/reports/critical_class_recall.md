# SonicSentinel AI — Critical-Class Recall Audit

> **Report Type**: Focused critical-class recall (AGENTS.md §4, SRS §1.7)  
> **Source Model**: `XGBoost` (`xgb_v1.0`)  
> **Evaluation Split**: Held-out, frozen test split (`2143` samples)  
> **Source Evaluation Timestamp**: 2026-09-27T12:31:59Z  
> **Report Generated**: 2026-09-27T12:54:31Z  

---

## Mandatory Critical-Class Recall Targets (AGENTS.md §4)

The five classes below carry the highest safety stakes and are subject to an **individual >= 85% recall** requirement defined in AGENTS.md §4 and SRS §1.7. A model that fails any single class does not meet the competition specification for that category, regardless of overall accuracy.

| # | Critical Sound Category | Recall | Precision | F1-Score | Support | Gap from 85% | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Gunshot** | **96.72%** | 52.06% | 0.6769 | 183 | `+11.72%` | PASSED |
| 2 | **Glass Breaking** | **19.01%** | 42.59% | 0.2629 | 121 | `-65.99%` | NOT MET |
| 3 | **Panic Scream** | **12.07%** | 36.84% | 0.1818 | 58 | `-72.93%` | NOT MET |
| 4 | **Aggression / Violent Conflict** | **7.96%** | 32.14% | 0.1277 | 113 | `-77.04%` | NOT MET |
| 5 | **Person Asking for Help** | **78.70%** | 77.98% | 0.7834 | 108 | `-6.30%` | NOT MET |

**Critical Classes Passing**: 1 of 5  
**Recall Target**: >= 85% per class (AGENTS.md §4)  

---

## Interpretation

> **4 of 5 critical classes do not meet the target.** Classes requiring attention:
>   - Glass Breaking
>   - Panic Scream
>   - Aggression / Violent Conflict
>   - Person Asking for Help

### Recommended Next Steps

| Priority | Action |
|:---:|:---|
| High | Boost **Aggression / Violent Conflict** recall — currently 7.96%, needs +77.04 pp. Consider targeted data augmentation, class-weighted loss, or decision-threshold calibration. |
| High | Boost **Panic Scream** recall — currently 12.07%, needs +72.93 pp. Consider targeted data augmentation, class-weighted loss, or decision-threshold calibration. |
| Medium | Boost **Glass Breaking** recall — currently 19.01%, needs +65.99 pp. Consider targeted data augmentation, class-weighted loss, or decision-threshold calibration. |
| Medium | Boost **Person Asking for Help** recall — currently 78.70%, needs +6.30 pp. Consider targeted data augmentation, class-weighted loss, or decision-threshold calibration. |

---

## Related Reports

- **Full Test Evaluation**: [`docs/reports/test_evaluation_report.md`](file:///docs/reports/test_evaluation_report.md)
- **JSON Machine-Readable**: [`docs/reports/critical_class_recall.json`](file:///docs/reports/critical_class_recall.json)
- **Confusion Matrix (PNG)**: [`docs/reports/test_confusion_matrix.png`](file:///docs/reports/test_confusion_matrix.png)
