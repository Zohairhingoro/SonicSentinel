# SonicSentinel AI — Test Set Evaluation Report

> **Evaluation Target**: Held-out, frozen test split (`final_split == 'test'`)  
> **Winning Model**: `XGBoost` (`xgb_v1.0`)  
> **Test Samples**: 2143 audio events across 10 mandatory sound categories  
> **Timestamp**: 2026-09-27T12:31:59Z  

---

## 1. Overall Performance Against AGENTS.md Section 4 Targets

| Requirement | Target Threshold | Achieved (Test Set) | Compliance Status |
|:---|:---:|:---:|:---:|
| **Overall Classification Accuracy** | $\ge 85\%$ | **52.40%** | **NOT MET** |
| **Macro F1 Score** | $\ge 0.80$ | **0.4276** | **NOT MET** |
| **Weighted F1 Score** | — | 0.4672 | Informational |
| **Macro Precision** | — | 0.4856 | Informational |
| **Macro Recall** | — | 0.4555 | Informational |
| **Inference Latency** | <= 8.0s upload | 0.08 ms/sample | **PASSED** |

---

## 2. Mandatory Critical-Class Recall Audit (SRS §1.6 & §1.7)

| Critical Sound Category | Recall | Precision | F1-Score | Support | Status (Target $\ge 85\%$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Gunshot** | **96.72%** | 52.06% | 0.6769 | 183 | **PASSED** |
| **Glass Breaking** | **19.01%** | 42.59% | 0.2629 | 121 | NOT MET |
| **Panic Scream** | **12.07%** | 36.84% | 0.1818 | 58 | NOT MET |
| **Aggression / Violent Conflict** | **7.96%** | 32.14% | 0.1277 | 113 | NOT MET |
| **Person Asking for Help** | **78.70%** | 77.98% | 0.7834 | 108 | NOT MET |

**Critical Classes Passing**: 1 of 5

---

## 3. Full Per-Class Classification Breakdown

| Sound Category | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| `aggression` | 32.14% | 7.96% | 0.1277 | 113 |
| `alarm_or_siren` | 51.61% | 9.97% | 0.1671 | 321 |
| `animal_sound` | 47.14% | 72.10% | 0.5700 | 491 |
| `background_noise` | 52.98% | 71.27% | 0.6078 | 362 |
| `glass_breaking` | 42.59% | 19.01% | 0.2629 | 121 |
| `gunshot` | 52.06% | 96.72% | 0.6769 | 183 |
| `machinery_fault` | 1.01% | 0.55% | 0.0071 | 183 |
| `panic_scream` | 36.84% | 12.07% | 0.1818 | 58 |
| `person_asking_for_help` | 77.98% | 78.70% | 0.7834 | 108 |
| `vehicle_horn` | 91.24% | 87.19% | 0.8917 | 203 |

---

## 4. Generated Evaluation Artifacts

- **Confusion Matrix CSV**: [`docs/reports/test_confusion_matrix.csv`](file:///docs/reports/test_confusion_matrix.csv)
- **Confusion Matrix Image**: [`docs/reports/test_confusion_matrix.png`](file:///docs/reports/test_confusion_matrix.png)
- **Machine-Readable JSON**: [`docs/reports/test_evaluation_report.json`](file:///docs/reports/test_evaluation_report.json)
