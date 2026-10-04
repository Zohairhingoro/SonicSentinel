from datetime import timedelta
from django.utils import timezone
from django.test import TestCase

from alerts.models import Alert, AlertRule
from audio_events.models import AudioEvent
from dashboard.analytics import (
    get_admin_summary_stats,
    get_all_analytics_summary,
    get_audio_quality_breakdown,
    get_average_alert_response_time,
    get_average_confidence,
    get_category_frequency,
    get_confidence_distribution,
    get_critical_alerts_count,
    get_critical_event_frequency,
    get_detection_trends,
    get_false_positive_negative_counts,
    get_model_disagreement_count,
    get_model_disagreement_rate,
    get_poor_quality_count,
    get_total_audio_events,
)
from predictions.models import ComparisonResult, PythonPrediction
from reviews.models import ReviewRecord


class AnalyticsAggregationTest(TestCase):
    def setUp(self):
        # Create audio events
        self.evt1 = AudioEvent.objects.create(
            filename="gunshot.wav",
            format="wav",
            duration=3.0,
            sampling_rate=22050,
            file_size=1000,
            quality=AudioEvent.QUALITY_GOOD,
            category="Gunshot",
            severity="Critical",
            storage_path="/media/audio/gunshot.wav",
            hash="hash1",
        )
        self.evt2 = AudioEvent.objects.create(
            filename="noise.wav",
            format="wav",
            duration=3.0,
            sampling_rate=22050,
            file_size=1000,
            quality=AudioEvent.QUALITY_POOR,
            category="Background Noise",
            severity="Informational",
            storage_path="/media/audio/noise.wav",
            hash="hash2",
        )

        # Create Python predictions
        PythonPrediction.objects.create(
            audio_event=self.evt1,
            predicted_class="Gunshot",
            confidence=0.92,
            confidence_scores={"Gunshot": 0.92},
            model_version="1.0.0",
        )
        PythonPrediction.objects.create(
            audio_event=self.evt2,
            predicted_class="Background Noise",
            confidence=0.45,
            confidence_scores={"Background Noise": 0.45},
            model_version="1.0.0",
        )

        # Create Comparison results
        ComparisonResult.objects.create(
            audio_event=self.evt1,
            confidence_difference=0.05,
            top_two_margin=0.4,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
        )
        ComparisonResult.objects.create(
            audio_event=self.evt2,
            confidence_difference=0.35,
            top_two_margin=0.05,
            match_status=ComparisonResult.MATCH_DISAGREEMENT,
            is_uncertain=True,
        )

        # Create Alert and AlertRule
        self.rule = AlertRule.objects.create(
            category="Gunshot",
            severity=AlertRule.SEVERITY_CRITICAL,
            recommended_action="Dispatch security",
        )
        now = timezone.now()
        alert = Alert.objects.create(
            audio_event=self.evt1,
            alert_rule=self.rule,
            severity=AlertRule.SEVERITY_CRITICAL,
            status=Alert.STATUS_ACKNOWLEDGED,
            resolved_at=now,
        )
        Alert.objects.filter(pk=alert.pk).update(created_at=now - timedelta(minutes=10))

        # Create Review record
        ReviewRecord.objects.create(
            audio_event=self.evt2,
            decision="Gunshot",
            comment="Reclassified noise as gunshot sound",
        )

    def test_get_total_audio_events(self):
        self.assertEqual(get_total_audio_events(), 2)

    def test_get_critical_alerts_count(self):
        self.assertEqual(get_critical_alerts_count(), 1)

    def test_get_average_confidence(self):
        self.assertEqual(get_average_confidence(), 0.685)

    def test_get_model_disagreement_count(self):
        self.assertEqual(get_model_disagreement_count(), 1)

    def test_get_poor_quality_count(self):
        self.assertEqual(get_poor_quality_count(), 1)

    def test_get_admin_summary_stats(self):
        stats = get_admin_summary_stats()
        self.assertEqual(stats["total_audio_events"], 2)
        self.assertEqual(stats["critical_alerts_count"], 1)
        self.assertEqual(stats["average_confidence"], 0.685)
        self.assertEqual(stats["model_disagreement_count"], 1)
        self.assertEqual(stats["poor_quality_count"], 1)

    def test_get_category_frequency(self):
        data = get_category_frequency()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("Gunshot"), 1)
        self.assertEqual(data.get("Background Noise"), 1)

    def test_get_critical_event_frequency(self):
        data = get_critical_event_frequency()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("total_critical"), 1)
        self.assertIn("Gunshot", data.get("by_category", {}))

    def test_get_confidence_distribution(self):
        data = get_confidence_distribution()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("< 0.50"), 1)
        self.assertEqual(data.get("0.85 - 1.00"), 1)

    def test_get_false_positive_negative_counts(self):
        data = get_false_positive_negative_counts()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("total_reviews"), 1)
        self.assertEqual(data.get("total_overrides"), 1)
        self.assertEqual(data.get("false_negatives"), 1)
        self.assertEqual(data.get("false_positives"), 0)

    def test_get_model_disagreement_rate(self):
        data = get_model_disagreement_rate()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("total_comparisons"), 2)
        self.assertEqual(data.get("disagreements"), 1)
        self.assertEqual(data.get("disagreement_rate"), 0.5)

    def test_get_audio_quality_breakdown(self):
        data = get_audio_quality_breakdown()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("Good"), 1)
        self.assertEqual(data.get("Poor"), 1)

    def test_get_average_alert_response_time(self):
        data = get_average_alert_response_time()
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("resolved_count"), 1)
        resp_sec = float(data.get("avg_response_seconds") or 0.0)
        resp_min = float(data.get("avg_response_minutes") or 0.0)
        self.assertGreater(resp_sec, 500.0)
        self.assertAlmostEqual(resp_min, 10.0, delta=0.5)

    def test_get_detection_trends(self):
        data = get_detection_trends(days=30)
        self.assertIsInstance(data, dict)
        self.assertIn("dates", data)
        self.assertIn("categories", data)
        self.assertIn("datasets", data)
        self.assertEqual(len(data["dates"]), 30)
        self.assertIn("Gunshot", data["datasets"])
        self.assertIn("Background Noise", data["datasets"])
        # Both created audio events are from today so today's bucket count should be >= 1
        today_str = timezone.now().date().strftime("%Y-%m-%d")
        if today_str in data["dates"]:
            idx = data["dates"].index(today_str)
            self.assertGreaterEqual(data["datasets"]["Gunshot"][idx], 1)

    def test_get_all_analytics_summary(self):
        data = get_all_analytics_summary()
        self.assertIsInstance(data, dict)
        self.assertIn("category_frequency", data)
        self.assertIn("critical_event_frequency", data)
        self.assertIn("confidence_distribution", data)
        self.assertIn("false_positive_negative", data)
        self.assertIn("model_disagreement", data)
        self.assertIn("audio_quality_breakdown", data)
        self.assertIn("alert_response_time", data)
        self.assertIn("admin_summary_stats", data)
        self.assertIn("detection_trends", data)

