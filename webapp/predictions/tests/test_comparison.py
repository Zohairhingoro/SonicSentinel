"""Unit tests for pure comparison math and database ComparisonResult creation logic."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from audio_events.models import AudioEvent
from predictions.comparison import (
    calculate_top_two_margin,
    compare_predictions_pure,
    compute_confidence_difference,
    create_comparison_if_ready,
)
from predictions.models import ComparisonResult, GTMPrediction, PythonPrediction

User = get_user_model()


class TestPureComparisonLogic(TestCase):
    """Unit tests for pure math comparison functions, independent of the database."""

    def test_calculate_top_two_margin(self):
        self.assertEqual(calculate_top_two_margin({}), 0.0)
        self.assertEqual(calculate_top_two_margin({"Gunshot": 0.85}), 0.85)

        scores = {"Gunshot": 0.85, "Panic Scream": 0.10, "Background Noise": 0.05}
        self.assertAlmostEqual(calculate_top_two_margin(scores), 0.75)

    def test_compute_confidence_difference(self):
        self.assertAlmostEqual(compute_confidence_difference(0.90, 0.80), 0.10)
        self.assertAlmostEqual(compute_confidence_difference(0.75, 0.75), 0.0)

    def test_acceptable_match(self):
        py_scores = {"Gunshot": 0.90, "Background Noise": 0.10}
        gtm_scores = {"Gunshot": 0.88, "Background Noise": 0.12}

        metrics = compare_predictions_pure(
            py_predicted_class="Gunshot",
            py_confidence_scores=py_scores,
            gtm_predicted_class="Gunshot",
            gtm_confidence_scores=gtm_scores,
            audio_quality="Good",
        )

        self.assertEqual(metrics["match_status"], ComparisonResult.MATCH_ACCEPTABLE)
        self.assertFalse(metrics["is_uncertain"])
        self.assertAlmostEqual(metrics["confidence_difference"], 0.02)
        self.assertAlmostEqual(metrics["top_two_margin"], 0.76)

    def test_model_disagreement(self):
        py_scores = {"Gunshot": 0.85, "Background Noise": 0.15}
        gtm_scores = {"Background Noise": 0.70, "Gunshot": 0.30}

        metrics = compare_predictions_pure(
            py_predicted_class="Gunshot",
            py_confidence_scores=py_scores,
            gtm_predicted_class="Background Noise",
            gtm_confidence_scores=gtm_scores,
        )

        self.assertEqual(metrics["match_status"], ComparisonResult.MATCH_DISAGREEMENT)
        self.assertTrue(metrics["is_uncertain"])

    def test_weak_match_due_to_confidence_difference(self):
        py_scores = {"Gunshot": 0.95, "Background Noise": 0.05}
        gtm_scores = {"Gunshot": 0.70, "Background Noise": 0.30}

        metrics = compare_predictions_pure(
            py_predicted_class="Gunshot",
            py_confidence_scores=py_scores,
            gtm_predicted_class="Gunshot",
            gtm_confidence_scores=gtm_scores,
        )

        self.assertEqual(metrics["match_status"], ComparisonResult.MATCH_WEAK)
        self.assertAlmostEqual(metrics["confidence_difference"], 0.25)

    def test_uncertain_result_due_to_low_confidence(self):
        py_scores = {"Gunshot": 0.45, "Background Noise": 0.55}
        gtm_scores = {"Gunshot": 0.40, "Background Noise": 0.60}

        metrics = compare_predictions_pure(
            py_predicted_class="Gunshot",
            py_confidence_scores=py_scores,
            gtm_predicted_class="Gunshot",
            gtm_confidence_scores=gtm_scores,
        )

        self.assertEqual(metrics["match_status"], ComparisonResult.MATCH_UNCERTAIN)
        self.assertTrue(metrics["is_uncertain"])

    def test_overlapping_sound_detection(self):
        py_scores = {"Gunshot": 0.48, "Panic Scream": 0.46, "Background Noise": 0.06}
        gtm_scores = {"Gunshot": 0.50, "Panic Scream": 0.45, "Background Noise": 0.05}

        metrics = compare_predictions_pure(
            py_predicted_class="Gunshot",
            py_confidence_scores=py_scores,
            gtm_predicted_class="Gunshot",
            gtm_confidence_scores=gtm_scores,
        )

        self.assertTrue(metrics["is_overlapping"])


class TestDatabaseComparisonIntegration(TestCase):
    """Integration tests for auto-creating ComparisonResult when predictions exist."""

    def setUp(self):
        self.user = User.objects.create_user(username="comp_test_user", password="password123")
        self.audio_event = AudioEvent.objects.create(
            filename="comp_audio.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=22050,
            file_size=50000,
            uploaded_by=self.user,
            storage_path="/media/comp_audio.wav",
            hash="comp_test_hash_123",
            quality="Good",
        )

    def test_comparison_result_created_when_both_predictions_exist(self):
        self.assertIsNone(ComparisonResult.objects.filter(audio_event=self.audio_event).first())

        # Step 1: Create PythonPrediction
        PythonPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Glass Breaking",
            confidence_scores={"Glass Breaking": 0.92, "Background Noise": 0.08},
            model_version="py-v1",
        )

        # Still no ComparisonResult because GTMPrediction is missing
        self.assertIsNone(ComparisonResult.objects.filter(audio_event=self.audio_event).first())

        # Step 2: Create GTMPrediction
        GTMPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Glass Breaking",
            confidence_scores={"Glass Breaking": 0.88, "Background Noise": 0.12},
            model_version="gtm-tfjs-v1",
        )

        # Now ComparisonResult MUST exist automatically
        comp = ComparisonResult.objects.filter(audio_event=self.audio_event).first()
        self.assertIsNotNone(comp)
        self.assertEqual(comp.match_status, ComparisonResult.MATCH_ACCEPTABLE)
        self.assertAlmostEqual(comp.confidence_difference, 0.04)
        self.assertFalse(comp.is_uncertain)

    def test_create_comparison_if_ready_direct_call(self):
        PythonPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Gunshot",
            confidence_scores={"Gunshot": 0.80, "Alarm or Siren": 0.20},
            model_version="py-v1",
        )
        GTMPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Alarm or Siren",
            confidence_scores={"Alarm or Siren": 0.75, "Gunshot": 0.25},
            model_version="gtm-tfjs-v1",
        )

        comp = create_comparison_if_ready(self.audio_event)
        self.assertIsNotNone(comp)
        self.assertEqual(comp.match_status, ComparisonResult.MATCH_DISAGREEMENT)
        self.assertTrue(comp.is_uncertain)
