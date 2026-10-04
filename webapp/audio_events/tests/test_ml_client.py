import io
import math
import struct
import wave
from unittest.mock import MagicMock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from audio_events.models import AudioEvent
from audio_events.services.ml_client import (
    MLServiceError,
    MLServiceUnavailable,
    predict_audio_file,
)
from predictions.models import PythonPrediction


def create_synthetic_wav() -> bytes:
    """Generate minimal valid WAV header and samples in bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        samples = [struct.pack("<h", int(0.5 * 32767.0 * math.sin(2.0 * math.pi * 440.0 * (i / 16000)))) for i in range(16000)]
        wav_file.writeframes(b"".join(samples))
    buffer.seek(0)
    return buffer.read()


class MLClientTest(TestCase):
    @patch("audio_events.services.ml_client.httpx.post")
    def test_predict_audio_file_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predicted_class": "Gunshot",
            "confidence": {"Gunshot": 0.95, "Background Noise": 0.05},
            "top3": [{"category": "Gunshot", "confidence": 0.95}],
            "quality": "Good",
            "waveform": [0.1, 0.2],
            "spectrogram": [[0.5]],
            "model_version": "1.0.0",
        }
        mock_post.return_value = mock_response

        wav_bytes = create_synthetic_wav()
        res = predict_audio_file(wav_bytes, filename="test.wav", service_url="http://127.0.0.1:8001/predict")

        self.assertEqual(res["predicted_class"], "Gunshot")
        self.assertEqual(res["confidence"]["Gunshot"], 0.95)

    @patch("audio_events.services.ml_client.httpx.post")
    def test_predict_audio_file_connection_error(self, mock_post):
        import httpx
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        wav_bytes = create_synthetic_wav()
        with self.assertRaises(MLServiceUnavailable):
            predict_audio_file(wav_bytes, service_url="http://127.0.0.1:8001/predict")

    @patch("audio_events.services.ml_client.httpx.post")
    def test_predict_audio_file_non_200_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Validation failed"
        mock_response.json.return_value = {"detail": "Audio file contains silence"}
        mock_post.return_value = mock_response

        wav_bytes = create_synthetic_wav()
        with self.assertRaises(MLServiceError) as ctx:
            predict_audio_file(wav_bytes, service_url="http://127.0.0.1:8001/predict")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("silence", str(ctx.exception))


class UploadAudioIntegrationTest(TestCase):
    @patch("audio_events.views.predict_audio_file")
    def test_upload_audio_creates_python_prediction(self, mock_predict):
        mock_predict.return_value = {
            "predicted_class": "Glass Breaking",
            "confidence": {"Glass Breaking": 0.88, "Background Noise": 0.12},
            "quality": "Good",
            "model_version": "1.0.0",
        }

        wav_bytes = create_synthetic_wav()
        uploaded = SimpleUploadedFile("glass.wav", wav_bytes, content_type="audio/wav")

        response = self.client.post("/audio/upload/api/", {"audio_file": uploaded})
        self.assertEqual(response.status_code, 201)

        data = response.json()
        self.assertTrue(data["accepted"])
        audio_id = data["audio_id"]

        evt = AudioEvent.objects.get(audio_id=audio_id)
        self.assertEqual(evt.category, "Glass Breaking")
        self.assertIn(evt.status, [AudioEvent.STATUS_CLASSIFIED, AudioEvent.STATUS_ALERT_GENERATED])

        preds = PythonPrediction.objects.filter(audio_event=evt)
        self.assertEqual(preds.count(), 1)
        pred = preds.first()
        self.assertEqual(pred.predicted_class, "Glass Breaking")
        self.assertEqual(pred.confidence, 0.88)

    @patch("audio_events.views.predict_audio_file")
    def test_upload_audio_duplicate_returns_existing(self, mock_predict):
        mock_predict.return_value = {
            "predicted_class": "Glass Breaking",
            "confidence": {"Glass Breaking": 0.88},
            "quality": "Good",
            "model_version": "1.0.0",
        }

        wav_bytes = create_synthetic_wav()
        uploaded1 = SimpleUploadedFile("clip.wav", wav_bytes, content_type="audio/wav")
        res1 = self.client.post("/audio/upload/", {"audio_file": uploaded1})
        self.assertEqual(res1.status_code, 201)
        first_audio_id = res1.json()["audio_id"]

        uploaded2 = SimpleUploadedFile("copy.wav", wav_bytes, content_type="audio/wav")
        res2 = self.client.post("/audio/upload/", {"audio_file": uploaded2})
        self.assertEqual(res2.status_code, 200)

        data2 = res2.json()
        self.assertTrue(data2["accepted"])
        self.assertTrue(data2["duplicate"])
        self.assertEqual(data2["audio_id"], first_audio_id)
        self.assertEqual(AudioEvent.objects.count(), 1)

