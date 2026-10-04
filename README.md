# SonicSentinel AI

> **AcousticX Intelligence - Sound-Event Detection System**
> Built for the **TechWiz7 (Aptech) NextWave AI and ML Competition**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Django](https://img.shields.io/badge/Django-5.x-green?logo=django)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-teal?logo=fastapi)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-GTM-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-Competition%20Use-lightgrey)

---

## Team - ML Masterz

| # | Team Member | Role | Modules / Tasks | Days |
|---|---|---|---|---|
| 1 | **Muhammad Suhaib Raza** *(Team Leader)* | ML Engineer & Project Lead | Project coordination, Python model training (SVM / Random Forest / XGBoost), alert rule engine design & configuration | 6 |
| 2 | **Shehbaz Ali** | Data Engineer & ML Engineer | Dataset collection & curation, dataset sorting & cleaning, Google Teachable Machine (GTM) training & TF.js integration | 6 |
| 3 | **Zohair Hingoro** | Full-Stack Developer | Front-End UI (Django templates, dashboards, live monitor), Back-End (Django views, APIs, RBAC, audit system), Dataset collection support | 6 |
| 4 | **Abdullah Faisal** | QA Engineer & Technical Writer | Project documentation (SRS, dev log, reports), quality assurance & testing, general team support | 6 |

---

## Competition Info

| Field | Details |
|---|---|
| **Competition** | TechWiz7 - Aptech NextWave AI and ML |
| **Team Name** | ML Masterz |
| **Project Name** | SonicSentinel AI (AcousticX Intelligence) |
| **Team Leader** | Muhammad Suhaib Raza |
| **Members** | Muhammad Suhaib Raza, Shehbaz Ali, Zohair Hingoro, Abdullah Faisal |

---

## 0. Overview

**SonicSentinel AI** is an AI-powered, web-based, real-time sound-event detection and classification platform. The system analyzes uploaded audio files as well as live microphone streams, classifying audio into ten mandatory sound categories using **two independently trained models**:

1. **Python Primary ML Model**: Hosted on a stateless FastAPI microservice (SVM / Random Forest / XGBoost / CNN).
2. **Client-Side GTM Classifier**: Exported TensorFlow.js model executing directly in the user's web browser via Google Teachable Machine.

The platform automatically computes prediction confidence, evaluates audio signal quality, measures inter-model agreement, applies a configurable critical-event rule engine, raises severity-rated alerts, and supports human-in-the-loop manual review with complete audit tracking.

> *Disclaimer: SonicSentinel AI is a competition prototype developed for TechWiz7 NextWave AI & ML. It is not a certified emergency-response or law-enforcement system.*

---

### Mandatory Sound Categories

SonicSentinel AI classifies sound events into 10 canonical categories defined in `config/classes.json`:

| Category Key | Display Name | Default Severity | Operational Behavior |
|---|---|---|---|
| `machinery_fault` | Machinery Fault | Medium | Triggers equipment maintenance alert & recommended inspection |
| `glass_breaking` | Glass Breaking | High | Triggers high-severity security alert |
| `alarm_or_siren` | Alarm or Siren | High | Triggers attention alert for emergency siren/alarm |
| `vehicle_horn` | Vehicle Horn | Low | Environmental / traffic sound event |
| `animal_sound` | Animal Sound | Informational | Ambient acoustic event |
| `gunshot` | Gunshot | Critical | Triggers critical security alert (requires confirmation & confidence check) |
| `panic_scream` | Panic Scream | Critical | Triggers critical safety alert |
| `aggression` | Aggression or Violent Conflict | High / Critical | Security conflict event based on confidence & repetition |
| `person_asking_for_help` | Person Asking for Help | Critical | Critical safety alert for emergency help call |
| `background_noise` | Background Noise | Informational | Ambient baseline noise |

---

## 1. System Architecture

The architecture enforces strict decoupling between the web platform, the ML inference engine, and client-side browser classification:

```
              +-------------------------------------------------+
              |                User Web Browser                 |
              |                                                 |
              |  +--------------------+  +-------------------+  |
              |  |  Django Web UI     |  | Teachable Machine |  |
              |  |  (Dashboards/Mic)  |  | TF.js Client Model|  |
              |  +---------+----------+  +---------+---------+  |
              +------------|-----------------------|------------+
                           | (HTTP REST)           | (In-Browser Inference)
                           v                       v
              +-----------------------+  +-------------------+
              |  Django Server        |  | Client Browser    |
              |  (Python 3.11/SQLite) |  | Local Classification|
              |  System of Record &   |  +-------------------+
              |  Alert Engine         |
              +-----------+-----------+
                          |
                          | POST /predict (REST)
                          v
              +-----------------------+
              |  FastAPI ML Service   |
              |  Stateless Inference  |
              |  (No DB Access)       |
              +-----------------------+
```

### Decoupled Layers

| Layer | Technology | Responsibility |
|---|---|---|
| **UI & System of Record** | Django 5 (Python) | User authentication, RBAC, SQLite database (events, predictions, alerts, reviews, audit logs), dashboards, analytics, downloadable PDF/CSV reports. |
| **ML Inference Service** | FastAPI (Python) | Stateless microservice. Validates audio, resamples/normalizes signal, extracts spectral features (MFCC, Mel Spectrogram, Chroma, ZCR, RMS), runs trained Python ML model. **Zero database access.** |
| **GTM Classifier** | TensorFlow.js (Browser) | Runs in-browser via TF.js on the client side. Classifies raw audio clips independently without sharing state or model inputs with the server-side Python model. |

---

## 2. Directory Layout

```
SonicSentinel/
|-- config/                      # Configuration files
|   |-- alert_rules.json         # Rule engine thresholds & severities
|   |-- audio_config.json        # Audio validation & sample rate constraints
|   +-- classes.json             # 10 canonical sound category definitions
|-- webapp/                      # Django Web Application (System of Record)
|   |-- manage.py
|   |-- accounts/                # User authentication & RBAC roles
|   |-- audio_events/            # Audio ingestion, metadata, duplication check
|   |-- predictions/             # Model comparison engine & consistency check
|   |-- alerts/                  # Rule engine, alert management & seeding
|   |-- reviews/                 # Manual review queue & non-destructive overrides
|   |-- dashboard/               # Analytical dashboards, event search & reports
|   |-- live_monitor/            # Real-time microphone monitoring UI
|   |-- audit/                   # System-wide audit log tracking
|   +-- core/                    # Storage, hash check & global helpers
|-- ml_service/                  # FastAPI ML Inference Microservice
|   |-- main.py                  # Service entrypoint
|   |-- api/                     # REST route handlers (/predict, /health)
|   |-- audio/                   # Preprocessing, validation, waveform/spectrogram
|   |-- features/                # Feature extraction (MFCC, Mel, Chroma, ZCR, RMS)
|   +-- model/artifacts/         # Exported ML model binaries (.pkl, .joblib, .json)
|-- ml_pipeline/                 # Offline ML Training Pipeline
|   |-- data/                    # Dataset indexing & stratified splitting
|   |-- features/                # Offline feature extraction
|   |-- training/                # SVM, Random Forest, XGBoost model training
|   |-- evaluation/              # Confusion matrix & classification reports
|   +-- export/                  # Artifact packaging for ml_service
|-- gtm_model/                   # Client-Side TensorFlow.js Teachable Machine Model
|   |-- tfjs_model/              # model.json and weights.bin
|   +-- metadata.json
|-- tests/                       # Automated Test Suites
|   |-- django/                  # RBAC, views, and database tests
|   |-- ml_service/              # FastAPI contract & API tests
|   |-- ml_pipeline/             # Feature extraction & pipeline tests
|   +-- e2e/                     # Full system end-to-end integration tests
|-- scripts/                     # Helper & evaluation scripts
|-- docker-compose.yml           # Container orchestration
|-- mypy.ini / pyrightconfig.json
+-- requirements.txt             # Python dependencies
```

---

## 3. Installation & Quick Start

### Prerequisites
- **Python 3.10+** (Python 3.11 or 3.14 supported)
- **Virtual Environment** (`venv`)
- **FFmpeg & libsndfile** (recommended for full audio format support like MP3/FLAC/OGG/M4A)

### 1. Clone & Environment Setup

```bash
# Clone the repository
git clone https://github.com/Zohairhingoro/SonicSentinel.git
cd SonicSentinel

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Django Web Application (Port 8000)

```bash
cd webapp
python manage.py migrate
python manage.py seed_alert_rules
python manage.py createsuperuser
python manage.py runserver 0.0.0.0:8000
```

- **Web App Dashboard**: http://localhost:8000
- **Admin Panel**: http://localhost:8000/admin

### 3. Run FastAPI ML Microservice (Port 8001)

```bash
cd ml_service
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

- **Service Health**: http://localhost:8001/health
- **Swagger API Docs**: http://localhost:8001/docs

### 4. Running via Docker Compose

```bash
docker-compose up --build
```

---

## 4. Offline ML Pipeline & Model Training

```bash
cd ml_pipeline
python data/build_dataset_index.py
python data/stratified_split.py
python features/extract_features.py
python training/train_svm.py
python training/train_random_forest.py
python training/train_xgboost.py
python training/compare_models.py
python export/export_model.py
```

---

## 5. API Specification: POST /predict

- **Endpoint**: POST http://localhost:8001/predict
- **Content-Type**: multipart/form-data
- **Body**: File field `file` or `audio_file` (WAV, MP3, FLAC, OGG, M4A)

Example Response (200 OK):

```json
{
  "predicted_class": "Panic Scream",
  "confidence": {
    "background_noise": 0.01,
    "panic_scream": 0.85,
    "gunshot": 0.03
  },
  "top3": [
    {"category": "Panic Scream", "confidence": 0.85},
    {"category": "Gunshot", "confidence": 0.03},
    {"category": "Aggression or Violent Conflict", "confidence": 0.03}
  ],
  "quality": "Good",
  "model_version": "svm-1.0.0"
}
```

---

## 6. Role-Based Access Control (RBAC)

SonicSentinel enforces 5 granular user roles managed through Django permissions:

| Role | Key Capabilities |
|---|---|
| **Normal User** | Audio file upload, view own uploaded events, audio preview player. |
| **Audio Reviewer** | Access manual review queue, confirm or override classifications (non-destructive), add review notes. |
| **Security Operator** | Live microphone monitoring, view real-time security alerts, batch upload audio files. |
| **Maintenance Operator** | Machinery fault alerts, equipment inspection logs, operational health warnings. |
| **Administrator** | Full system access, alert rule configuration, audit log inspection, user role assignment, CSV/Excel export, data retention management. |

---

## 7. Running Test Suites

```bash
# Django Web Application Tests
cd webapp
python manage.py test

# FastAPI ML Inference Microservice Tests
python -m pytest tests/ml_service -v

# ML Feature Pipeline Tests
python -m pytest tests/ml_pipeline -v

# Integration & End-to-End Tests
python -m pytest tests/e2e -v
```

### Test Results Summary

| Test Suite | Tests | Result | Duration | Scope |
|---|---|---|---|---|
| `tests/e2e/` | 15 | 15/15 PASSED | 13.5s | Full e2e flow (register, login, upload, poll, alerts, manual review) |
| `tests/ml_service/` | 49 | 49/49 PASSED | 0.6s | FastAPI /predict contract, preprocessing, audio validation |
| `tests/ml_pipeline/` | 8 | 8/8 PASSED | 1.6s | Feature vector shapes, NaN/Inf bounds, multi-harmonic extraction |
| `tests/django/` | 8 | 8/8 PASSED | 62.1s | Comprehensive RBAC unauthorized access matrix across all 5 roles |

---

## 8. Key Features Checklist

- [x] **Dual-Model Inference**: Server-side Python model + client-side browser TensorFlow.js model.
- [x] **Model Comparison Engine**: Calculates confidence delta, top-3 overlap, and inter-model agreement status (Acceptable Match, Weak Match, Model Disagreement, Uncertain Result).
- [x] **Audio Quality Check**: Detects audio clipping, silence ratio, and background noise level (Good, Acceptable, Poor, Unusable).
- [x] **Critical-Event Rule Engine**: Automatically evaluates detections against active alert rules to assign severities (Informational, Low, Medium, High, Critical).
- [x] **Duplicate Detection**: Computes SHA-256 file hash to reject identical uploads and detects near-duplicate audio clips.
- [x] **Live Microphone Stream**: Real-time audio capture, window segmentation, browser permission state tracking, and live detection visualizer.
- [x] **Human-in-the-Loop Manual Review**: Review queue with non-destructive classification overrides.
- [x] **Audit Trail**: Logs all user authentication events, uploads, mic sessions, predictions, rule updates, and manual reviews.
- [x] **Analytical Dashboard & Reporting**: Event filtering, trend analysis, per-event downloadable analysis reports, and admin data exports.

---

## 9. AI Usage Declaration

All AI tools used during development are documented in `AI_USAGE.md` in compliance with TechWiz7 competition transparency rules.

---

*SonicSentinel AI - ML Masterz Team 2026*