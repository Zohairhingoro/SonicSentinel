from django.test import TestCase
from django.template.loader import render_to_string

from audio_events.models import AudioEvent
from predictions.models import ComparisonResult, GTMPrediction, PythonPrediction


class ModelComparisonComponentTestCase(TestCase):
    def setUp(self):
        self.audio_event = AudioEvent.objects.create(
            audio_id="AUD-COMP1001",
            filename="gunshot_test.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=22050,
            file_size=100000,
            category="Gunshot",
            quality=AudioEvent.QUALITY_GOOD,
            storage_path="audio/AUD-COMP1001/gunshot_test.wav",
            hash="comp_test_hash_1",
        )

        self.py_pred = PythonPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Gunshot",
            confidence=0.92,
            confidence_scores={
                "Gunshot": 0.92,
                "Background Noise": 0.05,
                "Panic Scream": 0.03,
            },
            model_version="1.0.0",
        )

        self.gtm_pred = GTMPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Gunshot",
            confidence_scores={
                "Gunshot": 0.88,
                "Background Noise": 0.07,
                "Panic Scream": 0.05,
            },
            model_version="1.0.0",
        )

        self.comparison = ComparisonResult.objects.create(
            audio_event=self.audio_event,
            confidence_difference=0.04,
            top_two_margin=0.87,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
        )

    def test_component_rendering_acceptable_match(self):
        html = render_to_string("components/model_comparison.html", {"comparison": self.comparison})

        self.assertIn("AUD-COMP1001", html)
        self.assertIn("Acceptable Match", html)
        self.assertIn("bg-emerald-500", html)  # Distinct color badge for Acceptable Match
        self.assertIn("Python ML Model", html)
        self.assertIn("GTM TF.js Model", html)
        self.assertIn("Gunshot", html)
        self.assertIn("0.0400", html)  # Confidence difference
        self.assertIn("0.9200", html)  # Top confidence for Python
        self.assertIn("0.8800", html)  # Top confidence for GTM

    def test_component_rendering_model_disagreement_badge(self):
        self.comparison.match_status = ComparisonResult.MATCH_DISAGREEMENT
        self.comparison.save()

        html = render_to_string("components/model_comparison.html", {"comparison": self.comparison})

        self.assertIn("Model Disagreement", html)
        self.assertIn("bg-rose-500", html)  # Distinct color badge for Model Disagreement

    def test_component_rendering_weak_match_badge(self):
        self.comparison.match_status = ComparisonResult.MATCH_WEAK
        self.comparison.save()

        html = render_to_string("components/model_comparison.html", {"comparison": self.comparison})

        self.assertIn("Weak Match", html)
        self.assertIn("bg-amber-500", html)  # Distinct color badge for Weak Match

    def test_component_rendering_uncertain_result_badge(self):
        self.comparison.match_status = ComparisonResult.MATCH_UNCERTAIN
        self.comparison.save()

        html = render_to_string("components/model_comparison.html", {"comparison": self.comparison})

        self.assertIn("Uncertain Result", html)
        self.assertIn("bg-purple-500", html)  # Distinct color badge for Uncertain Result
