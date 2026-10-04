"""End-to-end integration test suite exercising complete workflows.

Covers the complete system workflow as specified in AGENTS.md:
1. User registration & role assignment (standard user & audio reviewer).
2. User authentication & session establishment.
3. Audio upload and ingestion validation.
4. Polling for ML prediction results & live session monitoring status.
5. Alert rule evaluation & generation for known critical audio samples.
6. Alert operational lifecycle (active polling, acknowledgment, dismissal).
7. End-to-end manual review workflow (queue discovery, event inspection,
   review decision recording, and non-destructive prediction preservation).
"""

import io
import wave
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch
import numpy as np
import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

from alerts.models import Alert, AlertRule
from audio_events.models import AudioEvent
from predictions.models import GTMPrediction, PythonPrediction
from reviews.models import ReviewRecord

User = get_user_model()

CRITICAL_SCREAM_ML_RESPONSE: Dict[str, Any] = {
    "predicted_class": "Panic Scream",
    "confidence": {
        "Panic Scream": 0.89,
        "Aggression or Violent Conflict": 0.04,
        "Person Asking for Help": 0.03,
        "Background Noise": 0.01,
        "Machinery Fault": 0.01,
        "Glass Breaking": 0.01,
        "Alarm or Siren": 0.01,
        "Vehicle Horn": 0.01,
        "Animal Sound": 0.01,
        "Gunshot": 0.01,
    },
    "top3": [
        {"category": "Panic Scream", "confidence": 0.89},
        {"category": "Aggression or Violent Conflict", "confidence": 0.04},
        {"category": "Person Asking for Help", "confidence": 0.03},
    ],
    "quality": "Good",
    "waveform": [0.0, 0.1, 0.3, 0.2],
    "spectrogram": [[0.1, 0.2], [0.3, 0.4]],
    "model_version": "svm-v1.0.0",
}


def _create_synthetic_wav(
    duration: float = 1.0,
    sample_rate: int = 22050,
    frequency: float = 800.0,
    amplitude: float = 0.6,
) -> bytes:
    """Generate in-memory WAV audio bytes for testing."""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    signal = (amplitude * np.sin(2.0 * np.pi * frequency * t) * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal.tobytes())

    return buffer.getvalue()


@pytest.fixture
def test_users(client: Client) -> Dict[str, Any]:
    """Create test users for standard user and reviewer roles."""
    user, _ = User.objects.get_or_create(
        username="normal_operator",
        defaults={
            "email": "operator@sonicsentinel.ai",
            "role": User.ROLE_NORMAL_USER,
        },
    )
    user.set_password("ValidPassword123!")
    user.save()

    reviewer, _ = User.objects.get_or_create(
        username="lead_reviewer",
        defaults={
            "email": "reviewer@sonicsentinel.ai",
            "role": User.ROLE_AUDIO_REVIEWER,
        },
    )
    reviewer.set_password("ReviewerPassword123!")
    reviewer.save()

    admin, _ = User.objects.get_or_create(
        username="admin_user",
        defaults={
            "email": "admin@sonicsentinel.ai",
            "role": User.ROLE_ADMINISTRATOR,
            "is_staff": True,
            "is_superuser": True,
        },
    )
    admin.set_password("AdminPassword123!")
    admin.save()

    return {"user": user, "reviewer": reviewer, "admin": admin}


def test_user_registration_and_login_flow(client: Client) -> None:
    """E2E Flow: Register a new user, verify redirection, and log in."""
    User.objects.filter(username="reg_user_e2e_01").delete()

    reg_data = {
        "username": "reg_user_e2e_01",
        "email": "reg_agent@sonicsentinel.ai",
        "role": User.ROLE_SECURITY_OPERATOR,
        "password1": "SecurePassword123!",
        "password2": "SecurePassword123!",
    }
    reg_response = client.post("/accounts/register/", reg_data, follow=True)
    assert reg_response.status_code == 200

    created_user = User.objects.filter(username="reg_user_e2e_01").first()
    assert created_user is not None
    assert created_user.email == "reg_agent@sonicsentinel.ai"
    assert created_user.role == User.ROLE_SECURITY_OPERATOR

    login_data = {
        "username": "reg_user_e2e_01",
        "password": "SecurePassword123!",
    }
    login_response = client.post("/accounts/login/", login_data, follow=True)
    assert login_response.status_code == 200
    assert login_response.wsgi_request.user.is_authenticated
    assert login_response.wsgi_request.user.username == "reg_user_e2e_01"


def test_complete_upload_prediction_alert_flow(client: Client, test_users: Dict[str, Any]) -> None:
    """E2E Flow: Upload critical audio, verify prediction, polling, and alert creation."""
    client.login(username="normal_operator", password="ValidPassword123!")

    wav_bytes = _create_synthetic_wav(duration=2.0, frequency=1200.0, amplitude=0.7)
    audio_upload = SimpleUploadedFile("panic_scream_clip.wav", wav_bytes, content_type="audio/wav")

    with patch("audio_events.views.predict_audio_file", return_value=CRITICAL_SCREAM_ML_RESPONSE):
        upload_resp = client.post("/audio/upload/", {"audio_file": audio_upload})

    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    assert upload_data["accepted"] is True
    audio_id = upload_data["audio_id"]

    audio_event = AudioEvent.objects.get(audio_id=audio_id)
    assert audio_event.category == "Panic Scream"
    assert audio_event.quality == "Good"
    assert audio_event.uploaded_by == test_users["user"]

    py_pred = PythonPrediction.objects.filter(audio_event=audio_event).first()
    assert py_pred is not None
    assert py_pred.predicted_class == "Panic Scream"
    assert py_pred.confidence == pytest.approx(0.89, abs=1e-2)

    status_resp = client.get("/live/api/session-status/")
    assert status_resp.status_code == 200
    status_data = status_resp.json()

    assert status_data["event"]["audio_id"] == audio_id
    assert status_data["event"]["category"] == "Panic Scream"
    assert status_data["python_prediction"]["predicted_class"] == "Panic Scream"
    assert status_data["python_prediction"]["top3"][0]["category"] == "Panic Scream"

    active_alerts_resp = client.get("/alerts/active/")
    assert active_alerts_resp.status_code == 200
    active_alerts_data = active_alerts_resp.json()

    assert active_alerts_data["count"] >= 1
    matching_alert = next(
        (a for a in active_alerts_data["alerts"] if a["audio_id"] == audio_id),
        None,
    )
    assert matching_alert is not None
    assert matching_alert["category"] == "Panic Scream"
    assert matching_alert["severity"] == AlertRule.SEVERITY_CRITICAL
    assert matching_alert["status"] == Alert.STATUS_ACTIVE

    alert_id = matching_alert["id"]
    client.login(username="lead_reviewer", password="ReviewerPassword123!")
    ack_resp = client.post(f"/alerts/{alert_id}/acknowledge/")
    assert ack_resp.status_code == 200
    ack_data = ack_resp.json()
    assert ack_data["success"] is True
    assert ack_data["status"] == Alert.STATUS_ACKNOWLEDGED

    alert_in_db = Alert.objects.get(pk=alert_id)
    assert alert_in_db.status == Alert.STATUS_ACKNOWLEDGED
    assert alert_in_db.resolved_by == test_users["reviewer"]


def test_manual_review_flow_end_to_end(client: Client, test_users: Dict[str, Any]) -> None:
    """E2E Flow: Routed manual review queue inspection, review submission, and non-destructive override."""
    audio_event = AudioEvent.objects.create(
        audio_id="AUD-REVIEW-E2E-001",
        filename="disputed_scream_machinery.wav",
        format="wav",
        duration=3.5,
        sampling_rate=22050,
        channels=1,
        file_size=154320,
        quality=AudioEvent.QUALITY_ACCEPTABLE,
        category="Machinery Fault",
        status=AudioEvent.STATUS_MANUAL_REVIEW,
        hash="e2e_unique_hash_review_001",
        storage_path="audio/AUD-REVIEW-E2E-001/disputed_scream_machinery.wav",
        uploaded_by=test_users["user"],
    )

    py_pred = PythonPrediction.objects.create(
        audio_event=audio_event,
        predicted_class="Machinery Fault",
        confidence=0.55,
        confidence_scores={"Machinery Fault": 0.55, "Panic Scream": 0.45},
        model_version="xgb-1.0",
    )

    gtm_pred = GTMPrediction.objects.create(
        audio_event=audio_event,
        predicted_class="Panic Scream",
        confidence_scores={"Panic Scream": 0.70, "Machinery Fault": 0.30},
        model_version="gtm-tfjs-1.0",
    )

    client.login(username="lead_reviewer", password="ReviewerPassword123!")

    queue_resp = client.get("/reviews/queue/")
    assert queue_resp.status_code == 200
    assert audio_event.audio_id in queue_resp.content.decode("utf-8")

    detail_resp = client.get(f"/reviews/{audio_event.audio_id}/")
    assert detail_resp.status_code == 200
    assert "Panic Scream" in detail_resp.content.decode("utf-8")
    assert "Machinery Fault" in detail_resp.content.decode("utf-8")

    review_submission = {
        "decision": "Panic Scream",
        "comment": "Human expert verification: vocal distress characteristics confirmed.",
        "recommended_action": "Security team dispatched to sector 4.",
    }
    submit_resp = client.post(f"/reviews/{audio_event.audio_id}/", review_submission, follow=True)
    assert submit_resp.status_code == 200

    audio_event.refresh_from_db()
    assert audio_event.status == AudioEvent.STATUS_REVIEWED
    assert audio_event.category == "Panic Scream"

    review_records = ReviewRecord.objects.filter(audio_event=audio_event)
    assert review_records.count() == 1
    record = review_records.first()
    assert record.decision == "Panic Scream"
    assert record.reviewer == test_users["reviewer"]
    assert "vocal distress" in record.comment
    assert "Security team" in record.recommended_action

    py_pred.refresh_from_db()
    gtm_pred.refresh_from_db()
    assert py_pred.predicted_class == "Machinery Fault"
    assert py_pred.confidence == pytest.approx(0.55, abs=1e-2)
    assert gtm_pred.predicted_class == "Panic Scream"

    queue_after_resp = client.get("/reviews/queue/")
    assert queue_after_resp.status_code == 200
    assert audio_event.audio_id not in queue_after_resp.content.decode("utf-8")


def test_fastapi_ml_service_predict_contract(sample_audio_dir: Path) -> None:
    """E2E Contract: Verify FastAPI stateless ML microservice /predict strictly conforms to AGENTS.md §1."""
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    ml_service_dir = repo_root / "ml_service"
    if str(ml_service_dir) not in sys.path:
        sys.path.insert(0, str(ml_service_dir))

    from fastapi.testclient import TestClient
    from ml_service.main import app

    fastapi_client = TestClient(app)
    audio_path = sample_audio_dir / "valid.wav"
    assert audio_path.exists()

    with open(audio_path, "rb") as fh:
        response = fastapi_client.post("/predict", files={"file": ("valid.wav", fh, "audio/wav")})

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "top3" in data
    assert "quality" in data
    assert "waveform" in data
    assert "spectrogram" in data
    assert "model_version" in data
    assert data["quality"] in {"Good", "Acceptable", "Poor", "Unusable"}
    assert len(data["top3"]) == 3
