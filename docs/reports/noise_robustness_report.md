# SonicSentinel AI (base) Noise Robustness Evaluation

> **Purpose**: Measure how accuracy and critical-class recall degrade when synthetic background noise is applied to the held-out test split.  
> **Model**: `XGBoost` (`xgb_v1.0`)  
> **Report Generated**: 2026-09-27T13:03:45Z  

---

## 1. Overall Accuracy and Macro F1 by Noise Condition

| Noise Condition | Samples | Accuracy | Macro F1 | delta Accuracy vs Clean |
|:---|:---:|:---:|:---:|:---:|
| Clean (no noise) | 300 | **44.33%** | 0.3863 | (base) |
| Mild  (SNR 20 dB) | 300 | **45.00%** | 0.4008 | `+0.67%` |
| Moderate (SNR 10 dB) | 300 | **42.33%** | 0.3697 | `-2.00%` |
| Severe (SNR  5 dB) | 300 | **30.00%** | 0.2382 | `-14.33%` |

---

## 2. Critical-Class Recall by Noise Condition (Target >= 85%)

| Sound Category | Clean (no noise) | Mild  (SNR 20 dB) | Moderate (SNR 10 dB) | Severe (SNR  5 dB) |
|:---|:---: | :---: | :---: | :---:|
| **Gunshot** | OK 100.0% | OK 100.0% | OK 93.3% | OK 90.0% |
| **Glass Breaking** | FAIL 10.0% | FAIL 13.3% | FAIL 16.7% | FAIL 3.3% |
| **Panic Scream** | FAIL 6.7% | FAIL 3.3% | FAIL 3.3% | FAIL 0.0% |
| **Aggression / Violent Conflict** | FAIL 6.7% | FAIL 6.7% | FAIL 3.3% | FAIL 0.0% |
| **Person Asking for Help** | OK 93.3% | FAIL 66.7% | FAIL 76.7% | FAIL 6.7% |

> Recall target: **>= 85%** per critical class (AGENTS.md Sec.4, SRS Sec.1.7)

---

## 3. Per-Class Recall: Clean vs Severe Noise

| Sound Category | Clean Recall | Severe (5 dB) Recall | delta Recall |
|:---|:---:|:---:|:---:|
| `aggression` | 6.7% | 0.0% | `-6.7%` |
| `alarm_or_siren` | 6.7% | 0.0% | `-6.7%` |
| `animal_sound` | 76.7% | 73.3% | `-3.3%` |
| `background_noise` | 60.0% | 63.3% | `+3.3%` |
| `glass_breaking` | 10.0% | 3.3% | `-6.7%` |
| `gunshot` | 100.0% | 90.0% | `-10.0%` |
| `machinery_fault` | 0.0% | 13.3% | `+13.3%` |
| `panic_scream` | 6.7% | 0.0% | `-6.7%` |
| `person_asking_for_help` | 93.3% | 6.7% | `-86.7%` |
| `vehicle_horn` | 83.3% | 50.0% | `-33.3%` |

---

## 4. Interpretation

- **Accuracy degradation** across noise levels indicates sensitivity to environmental noise; a large gap between Clean and Severe suggests the model relies heavily on spectral features that noise masks.
- **Critical-class recall** may degrade non-uniformly (base) classes with distinctive impulsive signatures (Gunshot) tend to be more robust than sustained-sound classes (Panic Scream, Aggression).
- Classes where recall falls below 85% under moderate noise should be targeted for noise-augmented retraining.

---

## 5. Related Reports

- **Clean Test Evaluation**: [`docs/reports/test_evaluation_report.md`](file:///docs/reports/test_evaluation_report.md)
- **Critical-Class Recall Audit**: [`docs/reports/critical_class_recall.md`](file:///docs/reports/critical_class_recall.md)
- **JSON Machine-Readable**: [`docs/reports/noise_robustness_report.json`](file:///docs/reports/noise_robustness_report.json)
