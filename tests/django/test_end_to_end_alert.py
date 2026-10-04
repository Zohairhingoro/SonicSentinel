"""
tests/django/test_end_to_end_alert.py - Integration test for the critical-alert pipeline.

Exercises the full stack using Django's test database:
  upload -> validation -> storage -> ML client call (mocked) -> PythonPrediction saved
  -> comparison triggered -> alert rule evaluated -> Alert row asserted.

The ML inference service is mocked so no real FastAPI/Flask process is needed.
A GTMPrediction row is created directly in the test to satisfy the comparison
engine (GTM inference is client-side/browser-side in production and never part of
the Django upload path), completing the comparison step and enabling rule evaluation.
"""

import io
import sys
import wave
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = REPO_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import django
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings.dev")
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse

from accounts.models import User
from alerts.models import Alert, AlertRule
from alerts.rule_engine import evaluate_and_trigger_alert
from audio_events.models import AudioEvent
from predictions.models import ComparisonResult, GTMPrediction, PythonPrediction


SAMPLE_AUDIO_DIR = REPO_ROOT / "sample_audio"

_GUNSHOT_CONFIDENCE = {
    "Gunshot": 0.93,
    "Glass Breaking": 0.03,
    "Alarm or Siren": 0.01,
    "Vehicle Horn": 0.01,
    "Animal Sound": 0.005,
    "Machinery Fault": 0.005,
    "Panic Scream": 0.005,
    "Aggression or Violent Conflict": 0.005,
    "Person Asking for Help": 0.005,
    "Background Noise": 0.005,
}

_MOCK_ML_RESPONSE = {
    "predicted_class": "Gunshot",
    "confidence": _GUNSHOT_CONFIDENCE,
    "top3": [
        {"category": "Gunshot", "confidence": 0.93},
        {"category": "Glass Breaking", "confidence": 0.03},
        {"category": "Alarm or Siren", "confidence": 0.01},
    ],
    "quality": "Good",
    "waveform": [],
    "spectrogram": [],
    "model_version": "test-1.0.0",
}

_GTM_CONFIDENCE = {
    "Gunshot": 0.91,
    "Glass Breaking": 0.04,
    "Alarm or Siren": 0.01,
    "Vehicle Horn": 0.01,
    "Animal Sound": 0.005,
    "Machinery Fault": 0.005,
    "Panic Scream": 0.005,
    "Aggression or Violent Conflict": 0.005,
    "Person Asking for Help": 0.005,
    "Background Noise": 0.005,
}


def _make_wav_bytes(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Return minimal PCM WAV bytes with a low-amplitude signal to pass the silence check."""
    n_samples = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        import struct
        frames = struct.pack("<" + "h" * n_samples, *([64] * n_samples))
        wf.writeframes(frames)
    return buf.getvalue()


class CriticalAlertEndToEndTest(TestCase):
    """Integration test: upload -> validation -> storage -> ML (mocked) -> comparison -> Alert."""

    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            username="testoperator",
            password="S3cur3Pass!",
            role=User.ROLE_SECURITY_OPERATOR,
        )

        # Gunshot AlertRule with controlled thresholds, independent of config/alert_rules.json.
        cls.gunshot_rule = AlertRule.objects.create(
            category="Gunshot",
            min_confidence=0.85,
            top_two_margin=0.10,
            required_consecutive_detections=1,
            requires_model_agreement=True,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            severity=AlertRule.SEVERITY_CRITICAL,
            recommended_action="Critical security alert: verified gunshot detected.",
        )

    def setUp(self):
        self.client = Client()
        self.client.login(username="testoperator", password="S3cur3Pass!")

    def _upload_audio(self, wav_bytes: bytes):
        """POST wav_bytes to the upload endpoint and return (response, json body)."""
        upload_file = SimpleUploadedFile(
            name="gunshot_sample.wav",
            content=wav_bytes,
            content_type="audio/wav",
        )
        response = self.client.post(
            reverse("audio_events:upload_audio"),
            data={"audio_file": upload_file},
            format="multipart",
        )
        return response, response.json()

    @patch("audio_events.views.predict_audio_file", return_value=_MOCK_ML_RESPONSE)
    def test_gunshot_upload_generates_critical_alert(self, mock_predict):
        """
        Full pipeline:
          1. Upload a valid WAV clip.
          2. Validation passes; AudioEvent is stored.
          3. ML client (mocked) returns high-confidence Gunshot prediction.
          4. PythonPrediction row is created via upload view.
          5. GTMPrediction row created directly (browser-side in prod).
          6. Comparison engine fires; ComparisonResult row created.
          7. Rule engine evaluation produces a Critical Alert row.
        """
        wav_bytes = _make_wav_bytes()
        response, body = self._upload_audio(wav_bytes)

        self.assertEqual(response.status_code, 201, msg=f"Upload failed: {body}")
        self.assertTrue(body.get("accepted"), msg=f"Validation rejected: {body}")

        audio_id = body["audio_id"]
        audio_event = AudioEvent.objects.get(audio_id=audio_id)

        mock_predict.assert_called_once()

        py_pred = PythonPrediction.objects.filter(audio_event=audio_event).first()
        self.assertIsNotNone(py_pred, "PythonPrediction was not created by the upload view")
        self.assertEqual(py_pred.predicted_class, "Gunshot")
        self.assertGreaterEqual(py_pred.confidence, 0.85)

        # Simulate browser-side GTM prediction arriving and being stored.
        GTMPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Gunshot",
            confidence_scores=_GTM_CONFIDENCE,
            model_version="gtm-test-1.0.0",
        )

        comp_result = ComparisonResult.objects.filter(audio_event=audio_event).first()
        self.assertIsNotNone(comp_result, "ComparisonResult was not created after GTMPrediction")
        self.assertEqual(comp_result.match_status, ComparisonResult.MATCH_ACCEPTABLE)

        alert = evaluate_and_trigger_alert(
            audio_event=audio_event,
            comparison_result=comp_result,
            prediction=py_pred,
        )

        self.assertIsNotNone(alert, "evaluate_and_trigger_alert returned None; expected a Critical Alert")
        self.assertEqual(alert.severity, AlertRule.SEVERITY_CRITICAL)
        self.assertEqual(alert.status, Alert.STATUS_ACTIVE)
        self.assertEqual(alert.audio_event, audio_event)

        db_alert = Alert.objects.filter(audio_event=audio_event).first()
        self.assertIsNotNone(db_alert, "Alert row does not exist in the database")
        self.assertEqual(db_alert.severity, AlertRule.SEVERITY_CRITICAL)

        audio_event.refresh_from_db()
        self.assertEqual(audio_event.status, AudioEvent.STATUS_ALERT_GENERATED)
        self.assertEqual(audio_event.severity, AlertRule.SEVERITY_CRITICAL)

    @patch("audio_events.views.predict_audio_file", return_value=_MOCK_ML_RESPONSE)
    def test_duplicate_alert_not_created_on_repeat_evaluation(self, mock_predict):
        """Rule engine must return the existing Alert without creating a second row."""
        wav_bytes = _make_wav_bytes()
        response, body = self._upload_audio(wav_bytes)
        self.assertEqual(response.status_code, 201)

        audio_event = AudioEvent.objects.get(audio_id=body["audio_id"])
        py_pred = PythonPrediction.objects.filter(audio_event=audio_event).first()

        GTMPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Gunshot",
            confidence_scores=_GTM_CONFIDENCE,
            model_version="gtm-test-1.0.0",
        )
        comp_result = ComparisonResult.objects.filter(audio_event=audio_event).first()

        alert_first = evaluate_and_trigger_alert(
            audio_event=audio_event,
            comparison_result=comp_result,
            prediction=py_pred,
        )
        alert_second = evaluate_and_trigger_alert(
            audio_event=audio_event,
            comparison_result=comp_result,
            prediction=py_pred,
        )

        self.assertEqual(alert_first.pk, alert_second.pk, "Rule engine created a duplicate Alert row")
        self.assertEqual(Alert.objects.filter(audio_event=audio_event).count(), 1)

    @patch(
        "audio_events.views.predict_audio_file",
        return_value={
            "predicted_class": "Gunshot",
            "confidence": {
                "Gunshot": 0.72,
                "Glass Breaking": 0.133,
                "Alarm or Siren": 0.133,
                "Vehicle Horn": 0.01,
                "Animal Sound": 0.001,
                "Machinery Fault": 0.001,
                "Panic Scream": 0.001,
                "Aggression or Violent Conflict": 0.001,
                "Person Asking for Help": 0.001,
                "Background Noise": 0.001,
            },
            "top3": [],
            "quality": "Good",
            "waveform": [],
            "spectrogram": [],
            "model_version": "test-1.0.0",
        },
    )
    def test_low_confidence_gunshot_does_not_trigger_alert(self, mock_predict):
        """Confidence 0.72 < rule threshold 0.85 -> alert must NOT be created."""
        wav_bytes = _make_wav_bytes()
        response, body = self._upload_audio(wav_bytes)
        self.assertEqual(response.status_code, 201)

        audio_event = AudioEvent.objects.get(audio_id=body["audio_id"])
        py_pred = PythonPrediction.objects.filter(audio_event=audio_event).first()
        self.assertIsNotNone(py_pred)

        GTMPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Gunshot",
            confidence_scores={
                "Gunshot": 0.70,
                "Glass Breaking": 0.145,
                "Alarm or Siren": 0.145,
                "Vehicle Horn": 0.003,
                "Animal Sound": 0.001,
                "Machinery Fault": 0.001,
                "Panic Scream": 0.001,
                "Aggression or Violent Conflict": 0.001,
                "Person Asking for Help": 0.001,
                "Background Noise": 0.001,
            },
            model_version="gtm-test-1.0.0",
        )
        comp_result = ComparisonResult.objects.filter(audio_event=audio_event).first()

        alert = evaluate_and_trigger_alert(
            audio_event=audio_event,
            comparison_result=comp_result,
            prediction=py_pred,
        )

        self.assertIsNone(alert, "Alert was incorrectly triggered despite confidence below threshold")
        self.assertEqual(Alert.objects.filter(audio_event=audio_event).count(), 0)

    @patch("audio_events.views.predict_audio_file", return_value=_MOCK_ML_RESPONSE)
    def test_model_disagreement_blocks_critical_alert(self, mock_predict):
        """GTM predicts a different class -> ComparisonResult is Model Disagreement -> no Alert."""
        wav_bytes = _make_wav_bytes()
        response, body = self._upload_audio(wav_bytes)
        self.assertEqual(response.status_code, 201)

        audio_event = AudioEvent.objects.get(audio_id=body["audio_id"])
        Alert.objects.filter(audio_event=audio_event).delete()
        py_pred = PythonPrediction.objects.filter(audio_event=audio_event).first()

        disagreeing_gtm_conf = {
            "Glass Breaking": 0.88,
            "Gunshot": 0.05,
            "Alarm or Siren": 0.02,
            "Vehicle Horn": 0.01,
            "Animal Sound": 0.01,
            "Machinery Fault": 0.005,
            "Panic Scream": 0.005,
            "Aggression or Violent Conflict": 0.005,
            "Person Asking for Help": 0.005,
            "Background Noise": 0.005,
        }
        GTMPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Glass Breaking",
            confidence_scores=disagreeing_gtm_conf,
            model_version="gtm-test-1.0.0",
        )
        comp_result = ComparisonResult.objects.filter(audio_event=audio_event).first()

        self.assertIsNotNone(comp_result)
        self.assertEqual(comp_result.match_status, ComparisonResult.MATCH_DISAGREEMENT)

        alert = evaluate_and_trigger_alert(
            audio_event=audio_event,
            comparison_result=comp_result,
            prediction=py_pred,
        )

        self.assertIsNone(alert, "Alert was incorrectly triggered despite model disagreement")
        self.assertEqual(Alert.objects.filter(audio_event=audio_event).count(), 0)
