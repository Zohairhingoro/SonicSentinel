# SonicSentinel AI — Model Prediction and Confidence Comparison Report

> **Deliverable Requirement:** SRS §1.6 & §1.7 — Dual Independent Model Verification
> Independent verification between trained Python model and client-side GTM TensorFlow.js model
> **Date Generated:** 2026-09-27T13:38:57Z | **Total Test Clips:** 100 (10 per class)

---

## 1. Executive Summary

| Metric | Value | Reference / Notes |
|---|---|---|
| **Test Samples Evaluated** | 100 | 10 unseen clips per class across all 10 mandatory categories |
| **Python Model Accuracy** | **42.00%** | XGBoost (xgb_v1.0) |
| **GTM Model Accuracy** | **17.00%** | TensorFlow.js (client-side export) |
| **Model Agreement Rate** | **37.00%** | Percentage of clips where both models predicted the identical class |
| **Mean Confidence Difference** | **0.3289** | Absolute delta between Python top confidence and GTM top confidence |
| **Manual Review Queue Rate** | **100.00%** | 100 clips flagged for human review |
| **Alerts Triggered** | **18** | Passed confidence, margin, and audio quality rule thresholds |

---

## 2. Match Status & System Health Breakdown

| Match Status Category | Count | Percentage | Operational Significance |
|---|---|---|---|
| **Uncertain Result** | 37 | 37.0% | Low confidence, low margin, unusable quality, or overlapping acoustic signal |
| **Model Disagreement** | 63 | 63.0% | Models predicted different sound categories; sent to manual review queue |

---

## 3. Model Prediction and Confidence Comparison Table (Full Unseen Test Set)

| Audio ID | Filename | Actual Class | Python Prediction (Conf) | GTM Prediction (Conf) | Match Status | Conf Diff | Quality | Severity | Alert Status | Manual Review | Final Decision | Correct? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `e8af3cc7` | `1 (18).wav` | Machinery Fault | Gunshot (0.9768) | Gunshot (0.5248) | Uncertain Result | 0.4520 | Acceptable | Critical | Alert Generated | Required | Gunshot (Pending Review) | **Incorrect** |
| `a8b2a698` | `1 (39).wav` | Machinery Fault | Gunshot (0.9722) | Gunshot (0.5248) | Uncertain Result | 0.4474 | Poor | Critical | No Alert | Required | Gunshot (Pending Review) | **Incorrect** |
| `867dd392` | `2 (68).wav` | Machinery Fault | Gunshot (0.9787) | Animal Sound (0.3447) | Model Disagreement | 0.6340 | Good | Critical | No Alert | Required | Gunshot (Pending Review) | **Incorrect** |
| `07aed790` | `2 (28).wav` | Machinery Fault | Gunshot (0.9878) | Machinery Fault (0.4254) | Model Disagreement | 0.5624 | Acceptable | Critical | Alert Generated | Required | Gunshot (Pending Review) | **Incorrect** |
| `2c1892e3` | `2 (60).wav` | Machinery Fault | Gunshot (0.9153) | Animal Sound (0.3447) | Model Disagreement | 0.5706 | Acceptable | Critical | No Alert | Required | Gunshot (Pending Review) | **Incorrect** |
| `6774b7e7` | `1 (14).wav` | Machinery Fault | Gunshot (0.9574) | Animal Sound (0.3447) | Model Disagreement | 0.6127 | Acceptable | Critical | No Alert | Required | Gunshot (Pending Review) | **Incorrect** |
| `f6eb5ca7` | `1 (22).wav` | Machinery Fault | Gunshot (0.5133) | Machinery Fault (0.4254) | Model Disagreement | 0.0879 | Poor | Critical | No Alert | Required | Gunshot (Pending Review) | **Incorrect** |
| `179324f3` | `1 (62).wav` | Machinery Fault | Animal Sound (0.4691) | Animal Sound (0.3447) | Uncertain Result | 0.1244 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `c8eb670a` | `2 (33).wav` | Machinery Fault | Gunshot (0.9937) | Machinery Fault (0.4254) | Model Disagreement | 0.5683 | Good | Critical | Alert Generated | Required | Gunshot (Pending Review) | **Incorrect** |
| `fb9f5f07` | `2 (16).wav` | Machinery Fault | Gunshot (0.9015) | Animal Sound (0.3447) | Model Disagreement | 0.5568 | Good | Critical | No Alert | Required | Gunshot (Pending Review) | **Incorrect** |
| `a1b1e2f5` | `1-84536-A-39.wav` | Glass Breaking | Background Noise (0.7830) | Machinery Fault (0.4254) | Model Disagreement | 0.3576 | Good | Informational | Alert Generated | Required | Background Noise (Pending Review) | **Incorrect** |
| `c492700f` | `0baae49b.wav` | Glass Breaking | Glass Breaking (0.4515) | Glass Breaking (0.4644) | Uncertain Result | 0.0129 | Good | High | No Alert | Required | Glass Breaking (Pending Review) | **Correct** |
| `516e3528` | `1b897583.wav` | Glass Breaking | Animal Sound (0.6575) | Gunshot (0.5248) | Model Disagreement | 0.1327 | Poor | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `80968858` | `0c4daa47.wav` | Glass Breaking | Glass Breaking (0.5617) | Gunshot (0.5248) | Model Disagreement | 0.0369 | Good | High | No Alert | Required | Glass Breaking (Pending Review) | **Correct** |
| `a347f7f6` | `0e7b1902.wav` | Glass Breaking | Background Noise (0.6826) | Animal Sound (0.3447) | Model Disagreement | 0.3379 | Good | Informational | No Alert | Required | Background Noise (Pending Review) | **Incorrect** |
| `0d97c799` | `1-17742-A-12.wav` | Glass Breaking | Animal Sound (0.4610) | Gunshot (0.5248) | Model Disagreement | 0.0638 | Poor | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `aeced51c` | `4-201800-A-31.wav` | Glass Breaking | Background Noise (0.6092) | Animal Sound (0.3447) | Model Disagreement | 0.2645 | Good | Informational | No Alert | Required | Background Noise (Pending Review) | **Incorrect** |
| `9511f03d` | `1ec61c9b.wav` | Glass Breaking | Animal Sound (0.6060) | Animal Sound (0.3447) | Uncertain Result | 0.2613 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `1078a62f` | `4-204115-A-39.wav` | Glass Breaking | Background Noise (0.9339) | Animal Sound (0.3447) | Model Disagreement | 0.5892 | Good | Informational | No Alert | Required | Background Noise (Pending Review) | **Incorrect** |
| `351e6fd7` | `1b3eca9e.wav` | Glass Breaking | Animal Sound (0.8028) | Animal Sound (0.3447) | Uncertain Result | 0.4581 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `88a2b599` | `2-70052-A-42.wav` | Alarm or Siren | Alarm or Siren (0.8024) | Alarm or Siren (0.4783) | Uncertain Result | 0.3241 | Poor | High | No Alert | Required | Alarm or Siren (Pending Review) | **Correct** |
| `ce6e79c5` | `5-117118-A-42.wav` | Alarm or Siren | Animal Sound (0.3210) | Alarm or Siren (0.4783) | Model Disagreement | 0.1573 | Acceptable | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `3c48676d` | `5cf3f9f2.wav` | Alarm or Siren | Animal Sound (0.3870) | Animal Sound (0.3447) | Uncertain Result | 0.0423 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `54f027a7` | `4-204684-A-37.wav` | Alarm or Siren | Glass Breaking (0.4409) | Glass Breaking (0.4644) | Uncertain Result | 0.0235 | Poor | High | No Alert | Required | Glass Breaking (Pending Review) | **Incorrect** |
| `eb01331f` | `26173-8-0-0.wav` | Alarm or Siren | Animal Sound (0.4394) | Animal Sound (0.3447) | Uncertain Result | 0.0947 | Poor | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `6a6adde8` | `3df2fc0e.wav` | Alarm or Siren | Vehicle Horn (0.6320) | Animal Sound (0.3447) | Model Disagreement | 0.2873 | Good | Low | No Alert | Required | Vehicle Horn (Pending Review) | **Incorrect** |
| `6e118792` | `1c6d56db.wav` | Alarm or Siren | Animal Sound (0.6596) | Machinery Fault (0.4254) | Model Disagreement | 0.2342 | Good | Informational | Alert Generated | Required | Animal Sound (Pending Review) | **Incorrect** |
| `2b5f3d49` | `50ebc0a7.wav` | Alarm or Siren | Animal Sound (0.7006) | Animal Sound (0.3447) | Uncertain Result | 0.3559 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `2d6a1d6e` | `3-51731-A-42.wav` | Alarm or Siren | Machinery Fault (0.3726) | Animal Sound (0.3447) | Model Disagreement | 0.0279 | Poor | Medium | No Alert | Required | Machinery Fault (Pending Review) | **Incorrect** |
| `13d0b906` | `5-179863-A-43.wav` | Alarm or Siren | Background Noise (0.8970) | Animal Sound (0.3447) | Model Disagreement | 0.5523 | Good | Informational | No Alert | Required | Background Noise (Pending Review) | **Incorrect** |
| `0cdbdc37` | `_J_8_S_V.wav` | Vehicle Horn | Vehicle Horn (0.9952) | Animal Sound (0.3447) | Model Disagreement | 0.6505 | Acceptable | Low | No Alert | Required | Vehicle Horn (Pending Review) | **Correct** |
| `b1981ef5` | `_C_7_S_L.wav` | Vehicle Horn | Vehicle Horn (0.9737) | Animal Sound (0.3447) | Model Disagreement | 0.6290 | Acceptable | Low | No Alert | Required | Vehicle Horn (Pending Review) | **Correct** |
| `7584ba39` | `_J_4_A_L.wav` | Vehicle Horn | Vehicle Horn (0.9971) | Animal Sound (0.3447) | Model Disagreement | 0.6524 | Poor | Low | No Alert | Required | Vehicle Horn (Pending Review) | **Correct** |
| `21411273` | `_J_7_A_V.wav` | Vehicle Horn | Vehicle Horn (0.9904) | Animal Sound (0.3447) | Model Disagreement | 0.6457 | Poor | Low | No Alert | Required | Vehicle Horn (Pending Review) | **Correct** |
| `77d28902` | `_C_9_A_V.wav` | Vehicle Horn | Vehicle Horn (0.9622) | Animal Sound (0.3447) | Model Disagreement | 0.6175 | Good | Low | No Alert | Required | Vehicle Horn (Pending Review) | **Correct** |
| `c890c7a0` | `_J_2_S_L.wav` | Vehicle Horn | Vehicle Horn (0.9929) | Animal Sound (0.3447) | Model Disagreement | 0.6482 | Poor | Low | No Alert | Required | Vehicle Horn (Pending Review) | **Correct** |
| `76049eae` | `_C_8_A_S.wav` | Vehicle Horn | Vehicle Horn (0.9918) | Animal Sound (0.3447) | Model Disagreement | 0.6471 | Poor | Low | No Alert | Required | Vehicle Horn (Pending Review) | **Correct** |
| `4d032244` | `1-17124-A-43.wav` | Vehicle Horn | Background Noise (0.9704) | Animal Sound (0.3447) | Model Disagreement | 0.6257 | Good | Informational | No Alert | Required | Background Noise (Pending Review) | **Incorrect** |
| `bffa3334` | `_C_9_B_V.wav` | Vehicle Horn | Vehicle Horn (0.9640) | Animal Sound (0.3447) | Model Disagreement | 0.6193 | Poor | Low | No Alert | Required | Vehicle Horn (Pending Review) | **Correct** |
| `fa77f371` | `_C_1_B_L.wav` | Vehicle Horn | Vehicle Horn (0.7320) | Background Noise (0.5505) | Model Disagreement | 0.1815 | Good | Low | Alert Generated | Required | Vehicle Horn (Pending Review) | **Correct** |
| `03ae10be` | `0b593941.wav` | Animal Sound | Animal Sound (0.5446) | Animal Sound (0.3447) | Uncertain Result | 0.1999 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Correct** |
| `5fbc9e24` | `2b826103.wav` | Animal Sound | Animal Sound (0.6862) | Animal Sound (0.3447) | Uncertain Result | 0.3415 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Correct** |
| `44a3e06c` | `2c2a3c77.wav` | Animal Sound | Animal Sound (0.6632) | Background Noise (0.5505) | Model Disagreement | 0.1127 | Acceptable | Informational | Alert Generated | Required | Animal Sound (Pending Review) | **Correct** |
| `5449f17b` | `3-105236-A-7.wav` | Animal Sound | Animal Sound (0.7863) | Animal Sound (0.3447) | Uncertain Result | 0.4416 | Acceptable | Informational | No Alert | Required | Animal Sound (Pending Review) | **Correct** |
| `6cf43ade` | `1-81851-A-31.wav` | Animal Sound | Background Noise (0.6537) | Animal Sound (0.3447) | Model Disagreement | 0.3090 | Poor | Informational | No Alert | Required | Background Noise (Pending Review) | **Incorrect** |
| `f09f6141` | `1-5-16_Historic...` | Animal Sound | Animal Sound (0.2936) | Animal Sound (0.3447) | Uncertain Result | 0.0511 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Correct** |
| `70b33d26` | `2-108760-B-14.wav` | Animal Sound | Animal Sound (0.7988) | Animal Sound (0.3447) | Uncertain Result | 0.4541 | Poor | Informational | No Alert | Required | Animal Sound (Pending Review) | **Correct** |
| `12db67bc` | `3-92637-A-18.wav` | Animal Sound | Animal Sound (0.7354) | Animal Sound (0.3447) | Uncertain Result | 0.3907 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Correct** |
| `fe1929bc` | `2-182508-A-8.wav` | Animal Sound | Animal Sound (0.9652) | Animal Sound (0.3447) | Uncertain Result | 0.6205 | Acceptable | Informational | No Alert | Required | Animal Sound (Pending Review) | **Correct** |
| `67ad1ea7` | `3-118069-B-27.wav` | Animal Sound | Animal Sound (0.6634) | Animal Sound (0.3447) | Uncertain Result | 0.3187 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Correct** |
| `4b2b1a93` | `5 (30).wav` | Gunshot | Gunshot (0.9982) | Machinery Fault (0.4254) | Model Disagreement | 0.5728 | Good | Critical | Alert Generated | Required | Gunshot (Pending Review) | **Correct** |
| `43c10ff8` | `5 (41).wav` | Gunshot | Gunshot (0.9965) | Animal Sound (0.3447) | Model Disagreement | 0.6518 | Good | Critical | No Alert | Required | Gunshot (Pending Review) | **Correct** |
| `903e214d` | `3 (8).wav` | Gunshot | Gunshot (0.8299) | Machinery Fault (0.4254) | Model Disagreement | 0.4045 | Acceptable | Critical | No Alert | Required | Gunshot (Pending Review) | **Correct** |
| `c9479f39` | `4 (14).wav` | Gunshot | Gunshot (0.9990) | Machinery Fault (0.4254) | Model Disagreement | 0.5736 | Acceptable | Critical | Alert Generated | Required | Gunshot (Pending Review) | **Correct** |
| `034263b4` | `4 (85).wav` | Gunshot | Gunshot (0.9973) | Animal Sound (0.3447) | Model Disagreement | 0.6526 | Good | Critical | No Alert | Required | Gunshot (Pending Review) | **Correct** |
| `4f6542c4` | `3a8bcc90.wav` | Gunshot | Animal Sound (0.4945) | Animal Sound (0.3447) | Uncertain Result | 0.1498 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `ceeb8320` | `5 (28).wav` | Gunshot | Gunshot (0.9919) | Alarm or Siren (0.4783) | Model Disagreement | 0.5136 | Acceptable | Critical | Alert Generated | Required | Gunshot (Pending Review) | **Correct** |
| `b7f1cabe` | `4 (25).wav` | Gunshot | Gunshot (0.9880) | Machinery Fault (0.4254) | Model Disagreement | 0.5626 | Good | Critical | Alert Generated | Required | Gunshot (Pending Review) | **Correct** |
| `177aec7d` | `5 (18).wav` | Gunshot | Gunshot (0.9978) | Animal Sound (0.3447) | Model Disagreement | 0.6531 | Good | Critical | No Alert | Required | Gunshot (Pending Review) | **Correct** |
| `811c270e` | `4 (88).wav` | Gunshot | Gunshot (0.9989) | Animal Sound (0.3447) | Model Disagreement | 0.6542 | Good | Critical | No Alert | Required | Gunshot (Pending Review) | **Correct** |
| `70d1fbd4` | `9s5V2kMiCpw_out...` | Panic Scream | Animal Sound (0.5754) | Machinery Fault (0.4254) | Model Disagreement | 0.1500 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `2e84cb8c` | `9-5c4-1T_v0_out...` | Panic Scream | Animal Sound (0.4248) | Animal Sound (0.3447) | Uncertain Result | 0.0801 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `eec18a65` | `82V80XZKV8M_out...` | Panic Scream | Animal Sound (0.5305) | Machinery Fault (0.4254) | Model Disagreement | 0.1051 | Acceptable | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `73c15af0` | `8qsjlzvgaZM_out...` | Panic Scream | Animal Sound (0.3844) | Machinery Fault (0.4254) | Model Disagreement | 0.0410 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `cecdf39f` | `9l9TdwUim2E_out...` | Panic Scream | Animal Sound (0.7533) | Machinery Fault (0.4254) | Model Disagreement | 0.3279 | Good | Informational | Alert Generated | Required | Animal Sound (Pending Review) | **Incorrect** |
| `429f94ae` | `5c6wTpeOqpg_out...` | Panic Scream | Panic Scream (0.3970) | Alarm or Siren (0.4783) | Model Disagreement | 0.0813 | Poor | Critical | No Alert | Required | Panic Scream (Pending Review) | **Correct** |
| `513dd2c8` | `-3bGlOhRkAo_out...` | Panic Scream | Animal Sound (0.4224) | Machinery Fault (0.4254) | Model Disagreement | 0.0030 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `da0fbed7` | `3PGk_NqInRs_out...` | Panic Scream | Panic Scream (0.3880) | Animal Sound (0.3447) | Model Disagreement | 0.0433 | Good | Critical | No Alert | Required | Panic Scream (Pending Review) | **Correct** |
| `9ad01003` | `083cc652.wav` | Panic Scream | Animal Sound (0.3327) | Machinery Fault (0.4254) | Model Disagreement | 0.0927 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `8ed4a849` | `5dYlvKV6mrc_out...` | Panic Scream | Animal Sound (0.7458) | Alarm or Siren (0.4783) | Model Disagreement | 0.2675 | Acceptable | Informational | Alert Generated | Required | Animal Sound (Pending Review) | **Incorrect** |
| `26483725` | `0cbc10f0.wav` | Aggression or Violent Conflict | Animal Sound (0.4330) | Animal Sound (0.3447) | Uncertain Result | 0.0883 | Poor | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `09d50a38` | `5ec6f01a.wav` | Aggression or Violent Conflict | Animal Sound (0.6421) | Animal Sound (0.3447) | Uncertain Result | 0.2974 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `16ab8d85` | `1-84393-A-32.wav` | Aggression or Violent Conflict | Background Noise (0.8104) | Animal Sound (0.3447) | Model Disagreement | 0.4657 | Good | Informational | No Alert | Required | Background Noise (Pending Review) | **Incorrect** |
| `0f7b247a` | `4yBWQ0Z5c1I_out...` | Aggression or Violent Conflict | Alarm or Siren (0.8044) | Machinery Fault (0.4254) | Model Disagreement | 0.3790 | Unusable | High | No Alert | Required | Alarm or Siren (Pending Review) | **Incorrect** |
| `86d5b480` | `4a19253c.wav` | Aggression or Violent Conflict | Gunshot (0.2673) | Machinery Fault (0.4254) | Model Disagreement | 0.1581 | Good | Critical | No Alert | Required | Gunshot (Pending Review) | **Incorrect** |
| `3e1bd5ed` | `2-76408-B-22.wav` | Aggression or Violent Conflict | Animal Sound (0.3977) | Animal Sound (0.3447) | Uncertain Result | 0.0530 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `429dbd9f` | `1a1edda6.wav` | Aggression or Violent Conflict | Animal Sound (0.5299) | Animal Sound (0.3447) | Uncertain Result | 0.1852 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `20094c98` | `4-155650-A-24.wav` | Aggression or Violent Conflict | Animal Sound (0.5744) | Animal Sound (0.3447) | Uncertain Result | 0.2297 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `9dcde1b6` | `3-112557-A-23.wav` | Aggression or Violent Conflict | Animal Sound (0.4960) | Animal Sound (0.3447) | Uncertain Result | 0.1513 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `078b7e12` | `5a8910e1.wav` | Aggression or Violent Conflict | Animal Sound (0.3377) | Machinery Fault (0.4254) | Model Disagreement | 0.0877 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `38792acd` | `01_02_02_01_08.wav` | Person Asking for Help | Person Asking for Help (0.8482) | Machinery Fault (0.4254) | Model Disagreement | 0.4228 | Good | Critical | Alert Generated | Required | Person Asking for Help (Pending Review) | **Correct** |
| `3d8e2dab` | `1-211527-A-20.wav` | Person Asking for Help | Animal Sound (0.8291) | Animal Sound (0.3447) | Uncertain Result | 0.4844 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `7cee0c98` | `01_02_01_01_06.wav` | Person Asking for Help | Person Asking for Help (0.8649) | Animal Sound (0.3447) | Model Disagreement | 0.5202 | Good | Critical | No Alert | Required | Person Asking for Help (Pending Review) | **Correct** |
| `6952b75a` | `01_02_02_01_12.wav` | Person Asking for Help | Person Asking for Help (0.8701) | Animal Sound (0.3447) | Model Disagreement | 0.5254 | Good | Critical | No Alert | Required | Person Asking for Help (Pending Review) | **Correct** |
| `e73c9a4b` | `01_04_00_02_09.wav` | Person Asking for Help | Person Asking for Help (0.3020) | Machinery Fault (0.4254) | Model Disagreement | 0.1234 | Good | Critical | No Alert | Required | Person Asking for Help (Pending Review) | **Correct** |
| `ff77c126` | `01_01_01_01_06.wav` | Person Asking for Help | Person Asking for Help (0.8649) | Machinery Fault (0.4254) | Model Disagreement | 0.4395 | Acceptable | Critical | Alert Generated | Required | Person Asking for Help (Pending Review) | **Correct** |
| `037c935c` | `1DHcsuOeMgE_out...` | Person Asking for Help | Animal Sound (0.6029) | Alarm or Siren (0.4783) | Model Disagreement | 0.1246 | Good | Informational | Alert Generated | Required | Animal Sound (Pending Review) | **Incorrect** |
| `7cd4159e` | `01_04_01_01_18.wav` | Person Asking for Help | Animal Sound (0.3747) | Animal Sound (0.3447) | Uncertain Result | 0.0300 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `a5e31a47` | `0qC99Z0YeMY_out...` | Person Asking for Help | Animal Sound (0.7481) | Animal Sound (0.3447) | Uncertain Result | 0.4034 | Acceptable | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `5bad22ff` | `1e8fe3d2.wav` | Person Asking for Help | Animal Sound (0.8093) | Animal Sound (0.3447) | Uncertain Result | 0.4646 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `63bab1bb` | `2eb3002c.wav` | Background Noise | Animal Sound (0.5355) | Animal Sound (0.3447) | Uncertain Result | 0.1908 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `14e45f45` | `3-156907-A-15.wav` | Background Noise | Background Noise (0.8875) | Animal Sound (0.3447) | Model Disagreement | 0.5428 | Good | Informational | No Alert | Required | Background Noise (Pending Review) | **Correct** |
| `44a08948` | `3-157615-A-10.wav` | Background Noise | Animal Sound (0.5426) | Glass Breaking (0.4644) | Model Disagreement | 0.0782 | Unusable | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `cbe31939` | `2a0260fd.wav` | Background Noise | Animal Sound (0.3881) | Animal Sound (0.3447) | Uncertain Result | 0.0434 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `16b500e9` | `0b4aa3a9.wav` | Background Noise | Animal Sound (0.4342) | Animal Sound (0.3447) | Uncertain Result | 0.0895 | Good | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `79d2298d` | `2bab5b09.wav` | Background Noise | Background Noise (0.9135) | Background Noise (0.5505) | Uncertain Result | 0.3630 | Good | Informational | Alert Generated | Required | Background Noise (Pending Review) | **Correct** |
| `9cfc206a` | `2d4961a8.wav` | Background Noise | Background Noise (0.8887) | Background Noise (0.5505) | Uncertain Result | 0.3382 | Acceptable | Informational | Alert Generated | Required | Background Noise (Pending Review) | **Correct** |
| `ee571546` | `2ae04dd7.wav` | Background Noise | Animal Sound (0.4023) | Animal Sound (0.3447) | Uncertain Result | 0.0576 | Acceptable | Informational | No Alert | Required | Animal Sound (Pending Review) | **Incorrect** |
| `16bec7c1` | `2da75f59.wav` | Background Noise | Background Noise (0.5495) | Animal Sound (0.3447) | Model Disagreement | 0.2048 | Good | Informational | No Alert | Required | Background Noise (Pending Review) | **Correct** |
| `0617c750` | `3-143119-A-21.wav` | Background Noise | Background Noise (0.7234) | Animal Sound (0.3447) | Model Disagreement | 0.3787 | Good | Informational | No Alert | Required | Background Noise (Pending Review) | **Correct** |

---

## 4. Architectural Integrity Declaration

- **Independent Execution:** The Python ML service and client-side GTM TensorFlow.js model operated in strictly separated execution environments. No prediction or confidence score was shared or fed between pipelines.
- **No Hardcoded Values:** All predictions, confidence scores, quality levels, match statuses, and alert decisions were computed dynamically from actual audio data and configuration files (`config/alert_rules.json`, `config/classes.json`).
