from django.contrib.auth import get_user_model
from django.test import TestCase

from audio_events.models import AudioEvent
from predictions.models import PythonPrediction, GTMPrediction, ComparisonResult

User = get_user_model()


class PredictionsModelsTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="pred_user", password="password123")
        self.audio_event = AudioEvent.objects.create(
            filename="test_audio.wav",
            format="WAV",
            duration=4.0,
            sampling_rate=22050,
            file_size=100000,
            uploaded_by=self.user,
            category="Gunshot",
            storage_path="/media/test_audio.wav",
            hash="pred_test_hash_1",
        )

    def test_python_prediction_creation(self):
        scores = {"Gunshot": 0.92, "Background Noise": 0.08}
        pred = PythonPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Gunshot",
            confidence_scores=scores,
            model_version="v1.0.0",
        )
        self.assertEqual(pred.predicted_class, "Gunshot")
        self.assertEqual(pred.confidence_scores["Gunshot"], 0.92)
        self.assertEqual(self.audio_event.python_predictions.first(), pred)
        self.assertIn("Gunshot", str(pred))

    def test_gtm_prediction_creation(self):
        scores = {"Gunshot": 0.88, "Background Noise": 0.12}
        gtm_pred = GTMPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Gunshot",
            confidence_scores=scores,
            model_version="gtm-tfjs-v1",
        )
        self.assertEqual(gtm_pred.predicted_class, "Gunshot")
        self.assertEqual(self.audio_event.gtm_predictions.first(), gtm_pred)
        self.assertIn("Gunshot", str(gtm_pred))

    def test_comparison_result_creation(self):
        comp = ComparisonResult.objects.create(
            audio_event=self.audio_event,
            confidence_difference=0.04,
            top_two_margin=0.80,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            is_uncertain=False,
            is_overlapping=False,
        )
        self.assertEqual(comp.match_status, ComparisonResult.MATCH_ACCEPTABLE)
        self.assertFalse(comp.is_uncertain)
        self.assertFalse(comp.is_overlapping)
        self.assertEqual(self.audio_event.comparison_results.first(), comp)
        self.assertIn("Acceptable Match", str(comp))
