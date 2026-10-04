from django.contrib.auth import get_user_model
from django.test import TestCase

from alerts.models import Alert, AlertRule
from alerts.rule_engine import (
    evaluate_alert_conditions,
    evaluate_and_trigger_alert,
    evaluate_rule_against_metrics,
    get_consecutive_detections_count,
)
from audio_events.models import AudioEvent, Segment
from predictions.models import ComparisonResult, PythonPrediction

User = get_user_model()


class PureRuleEngineUnitTestCase(TestCase):
    """Unit tests for pure rule evaluation functions without database access."""

    def test_evaluate_alert_conditions_confidence_threshold(self):
        # Passes when confidence >= min_confidence
        passes = evaluate_alert_conditions(
            confidence=0.85,
            top_two_margin=0.20,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
            min_confidence=0.80,
            required_top_two_margin=0.15,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=1,
        )
        self.assertTrue(passes)

        # Fails when confidence < min_confidence
        fails = evaluate_alert_conditions(
            confidence=0.75,
            top_two_margin=0.20,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
            min_confidence=0.80,
            required_top_two_margin=0.15,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=1,
        )
        self.assertFalse(fails)

    def test_evaluate_alert_conditions_top_two_margin(self):
        # Fails when top_two_margin < required_top_two_margin
        fails = evaluate_alert_conditions(
            confidence=0.90,
            top_two_margin=0.05,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
            min_confidence=0.80,
            required_top_two_margin=0.15,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=1,
        )
        self.assertFalse(fails)

    def test_evaluate_alert_conditions_model_agreement(self):
        # Requires model agreement = True
        # Fails when model disagreement
        fails = evaluate_alert_conditions(
            confidence=0.90,
            top_two_margin=0.25,
            match_status=ComparisonResult.MATCH_DISAGREEMENT,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
            min_confidence=0.80,
            required_top_two_margin=0.15,
            requires_model_agreement=True,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=1,
        )
        self.assertFalse(fails)

        # Passes when acceptable match
        passes = evaluate_alert_conditions(
            confidence=0.90,
            top_two_margin=0.25,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
            min_confidence=0.80,
            required_top_two_margin=0.15,
            requires_model_agreement=True,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=1,
        )
        self.assertTrue(passes)

    def test_evaluate_alert_conditions_audio_quality(self):
        # Fails when audio quality ('Poor') is less than minimum required ('Acceptable')
        fails = evaluate_alert_conditions(
            confidence=0.90,
            top_two_margin=0.25,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_POOR,
            consecutive_count=1,
            min_confidence=0.80,
            required_top_two_margin=0.15,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=1,
        )
        self.assertFalse(fails)

    def test_evaluate_alert_conditions_consecutive_detections(self):
        # Fails when consecutive_count (1) < required (2)
        fails = evaluate_alert_conditions(
            confidence=0.90,
            top_two_margin=0.25,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
            min_confidence=0.80,
            required_top_two_margin=0.15,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=2,
        )
        self.assertFalse(fails)

        # Passes when consecutive_count (2) >= required (2)
        passes = evaluate_alert_conditions(
            confidence=0.90,
            top_two_margin=0.25,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=2,
            min_confidence=0.80,
            required_top_two_margin=0.15,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=2,
        )
        self.assertTrue(passes)


class RuleEngineDBTestCase(TestCase):
    """Integration tests for rule engine database queries and alert creation."""

    def setUp(self):
        self.user = User.objects.create_user(username="rule_operator", password="password123")
        self.rule = AlertRule.objects.create(
            category="Gunshot",
            min_confidence=0.85,
            top_two_margin=0.15,
            required_consecutive_detections=2,
            requires_model_agreement=True,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            severity=AlertRule.SEVERITY_CRITICAL,
            recommended_action="Critical security alert - verified gunshot detected",
        )

        self.audio_event = AudioEvent.objects.create(
            filename="gunshot_clip.wav",
            format="WAV",
            duration=2.5,
            sampling_rate=22050,
            file_size=110000,
            uploaded_by=self.user,
            category="Gunshot",
            quality=AudioEvent.QUALITY_GOOD,
            storage_path="/media/gunshot_clip.wav",
            hash="gunshot_hash_123",
        )

        self.prediction = PythonPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Gunshot",
            confidence=0.90,
            confidence_scores={"Gunshot": 0.90, "Background Noise": 0.05},
            model_version="1.0.0",
        )

        self.comparison = ComparisonResult.objects.create(
            audio_event=self.audio_event,
            confidence_difference=0.05,
            top_two_margin=0.85,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
        )

    def test_evaluate_and_trigger_alert_requires_consecutive_fail(self):
        # Single event -> consecutive count = 1 < required (2), so no alert created
        alert = evaluate_and_trigger_alert(self.audio_event, self.comparison, self.prediction)
        self.assertIsNone(alert)
        self.assertEqual(Alert.objects.count(), 0)

    def test_evaluate_and_trigger_alert_success(self):
        # Create a second prediction for the same category to reach consecutive_count = 2
        event2 = AudioEvent.objects.create(
            filename="gunshot_clip2.wav",
            format="WAV",
            duration=2.5,
            sampling_rate=22050,
            file_size=110000,
            uploaded_by=self.user,
            category="Gunshot",
            quality=AudioEvent.QUALITY_GOOD,
            storage_path="/media/gunshot_clip2.wav",
            hash="gunshot_hash_456",
        )
        pred2 = PythonPrediction.objects.create(
            audio_event=event2,
            predicted_class="Gunshot",
            confidence=0.92,
            confidence_scores={"Gunshot": 0.92, "Background Noise": 0.04},
            model_version="1.0.0",
        )
        comp2 = ComparisonResult.objects.create(
            audio_event=event2,
            confidence_difference=0.04,
            top_two_margin=0.88,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
        )

        alert = evaluate_and_trigger_alert(event2, comp2, pred2)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, AlertRule.SEVERITY_CRITICAL)
        self.assertEqual(alert.alert_rule, self.rule)

        event2.refresh_from_db()
        self.assertEqual(event2.status, AudioEvent.STATUS_ALERT_GENERATED)
        self.assertEqual(event2.severity, AlertRule.SEVERITY_CRITICAL)
