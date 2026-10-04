"""
tests/django/test_alert_rule_engine.py — Unit tests for alerts/rule_engine.py.

Tests the alert rule evaluation engine against mock AlertRule configurations:
  - An event that should generate a critical alert (all criteria pass)
  - An event blocked because confidence is below the rule threshold
  - An event blocked by insufficient model agreement (Model Disagreement when agreement is required)
  - An event blocked by poor audio quality (below required minimum quality level)
  - Pure evaluate_alert_conditions function coverage for all threshold parameters
  - evaluate_rule_against_metrics using mock rule objects
  - Repeated alert trigger idempotency (returning existing alert without duplicates)
  - Fallback to default rule when no category-specific rule is found
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = REPO_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from django.test import TestCase

from alerts.models import Alert, AlertRule
from alerts.rule_engine import (
    evaluate_alert_conditions,
    evaluate_and_trigger_alert,
    evaluate_rule_against_metrics,
)
from audio_events.models import AudioEvent
from predictions.models import ComparisonResult, PythonPrediction


class AlertRuleEngineTestCase(TestCase):
    """Verifies rule engine evaluation logic against mocked/isolated AlertRule configurations."""

    def setUp(self):
        # Explicit mock AlertRule configurations (independent of config/alert_rules.json)
        self.critical_rule = AlertRule.objects.create(
            category="Gunshot",
            min_confidence=0.85,
            top_two_margin=0.15,
            required_consecutive_detections=1,
            requires_model_agreement=True,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            severity=AlertRule.SEVERITY_CRITICAL,
            recommended_action="Critical security alert - verified gunshot detected",
        )

        self.panic_rule = AlertRule.objects.create(
            category="Panic Scream",
            min_confidence=0.80,
            top_two_margin=0.15,
            required_consecutive_detections=1,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            severity=AlertRule.SEVERITY_CRITICAL,
            recommended_action="Critical safety alert - panic scream detected",
        )

        self.medium_rule = AlertRule.objects.create(
            category="Machinery Fault",
            min_confidence=0.70,
            top_two_margin=0.15,
            required_consecutive_detections=1,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            severity=AlertRule.SEVERITY_MEDIUM,
            recommended_action="Maintenance alert + recommended inspection",
        )

        self.default_bg_rule = AlertRule.objects.create(
            category="Background Noise",
            min_confidence=0.50,
            top_two_margin=0.10,
            required_consecutive_detections=1,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_UNUSABLE,
            severity=AlertRule.SEVERITY_INFORMATIONAL,
            recommended_action="Non-critical ambient noise",
        )

    def _create_audio_event(self, audio_id: str, category: str, quality: str) -> AudioEvent:
        """Helper to create an AudioEvent instance."""
        return AudioEvent.objects.create(
            audio_id=audio_id,
            filename=f"{audio_id}.wav",
            format="wav",
            duration=2.5,
            sampling_rate=22050,
            channels=1,
            bit_depth=16,
            file_size=110250,
            storage_path=f"audio/{audio_id}/{audio_id}.wav",
            hash=f"hash_{audio_id}",
            quality=quality,
            category=category,
            status=AudioEvent.STATUS_CLASSIFIED,
        )

    # -------------------------------------------------------------------------
    # 1. Event that SHOULD generate a critical alert
    # -------------------------------------------------------------------------
    def test_critical_alert_generated_when_all_conditions_pass(self):
        """High-confidence gunshot event with acceptable model agreement and good quality triggers Critical alert."""
        audio_event = self._create_audio_event("evt-crit-001", "Gunshot", AudioEvent.QUALITY_GOOD)

        pred = PythonPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Gunshot",
            confidence=0.92,
            confidence_scores={"Gunshot": 0.92, "Background Noise": 0.08},
            model_version="xgb_v1.0",
        )

        comp = ComparisonResult.objects.create(
            audio_event=audio_event,
            confidence_difference=0.08,
            top_two_margin=0.84,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            is_uncertain=False,
            is_overlapping=False,
        )

        alert = evaluate_and_trigger_alert(audio_event, comparison_result=comp, prediction=pred)

        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, AlertRule.SEVERITY_CRITICAL)
        self.assertEqual(alert.status, Alert.STATUS_ACTIVE)
        self.assertEqual(alert.alert_rule, self.critical_rule)
        self.assertEqual(alert.audio_event, audio_event)

        # Event status and severity updated in database
        audio_event.refresh_from_db()
        self.assertEqual(audio_event.status, AudioEvent.STATUS_ALERT_GENERATED)
        self.assertEqual(audio_event.severity, AlertRule.SEVERITY_CRITICAL)
        self.assertEqual(Alert.objects.filter(audio_event=audio_event).count(), 1)

    # -------------------------------------------------------------------------
    # 2. Event BLOCKED: Confidence below threshold
    # -------------------------------------------------------------------------
    def test_alert_blocked_when_confidence_below_threshold(self):
        """Gunshot prediction with confidence 0.70 (< 0.85 rule threshold) must NOT generate alert."""
        audio_event = self._create_audio_event("evt-lowconf-002", "Gunshot", AudioEvent.QUALITY_GOOD)

        pred = PythonPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Gunshot",
            confidence=0.70,
            confidence_scores={"Gunshot": 0.70, "Background Noise": 0.30},
            model_version="xgb_v1.0",
        )

        comp = ComparisonResult.objects.create(
            audio_event=audio_event,
            confidence_difference=0.05,
            top_two_margin=0.40,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            is_uncertain=False,
            is_overlapping=False,
        )

        alert = evaluate_and_trigger_alert(audio_event, comparison_result=comp, prediction=pred)

        self.assertIsNone(alert)
        self.assertEqual(Alert.objects.filter(audio_event=audio_event).count(), 0)

        # AudioEvent status remains unmodified
        audio_event.refresh_from_db()
        self.assertEqual(audio_event.status, AudioEvent.STATUS_CLASSIFIED)

    # -------------------------------------------------------------------------
    # 3. Event BLOCKED: Insufficient model agreement
    # -------------------------------------------------------------------------
    def test_alert_blocked_by_insufficient_model_agreement(self):
        """When requires_model_agreement=True, Model Disagreement must block alert generation."""
        audio_event = self._create_audio_event("evt-disagree-003", "Gunshot", AudioEvent.QUALITY_GOOD)

        pred = PythonPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Gunshot",
            confidence=0.95,  # High confidence passes confidence check
            confidence_scores={"Gunshot": 0.95, "Background Noise": 0.05},
            model_version="xgb_v1.0",
        )

        comp = ComparisonResult.objects.create(
            audio_event=audio_event,
            confidence_difference=0.55,
            top_two_margin=0.90,
            match_status=ComparisonResult.MATCH_DISAGREEMENT,  # Disagreement blocks alert
            is_uncertain=True,
            is_overlapping=False,
        )

        alert = evaluate_and_trigger_alert(audio_event, comparison_result=comp, prediction=pred)

        self.assertIsNone(alert)
        self.assertEqual(Alert.objects.filter(audio_event=audio_event).count(), 0)

        audio_event.refresh_from_db()
        self.assertEqual(audio_event.status, AudioEvent.STATUS_CLASSIFIED)

    def test_alert_blocked_by_uncertain_result_when_agreement_required(self):
        """When requires_model_agreement=True, Uncertain Result must block alert generation."""
        audio_event = self._create_audio_event("evt-uncertain-004", "Gunshot", AudioEvent.QUALITY_GOOD)

        pred = PythonPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Gunshot",
            confidence=0.88,
            confidence_scores={"Gunshot": 0.88, "Background Noise": 0.12},
            model_version="xgb_v1.0",
        )

        comp = ComparisonResult.objects.create(
            audio_event=audio_event,
            confidence_difference=0.20,
            top_two_margin=0.76,
            match_status=ComparisonResult.MATCH_UNCERTAIN,
            is_uncertain=True,
            is_overlapping=False,
        )

        alert = evaluate_and_trigger_alert(audio_event, comparison_result=comp, prediction=pred)
        self.assertIsNone(alert)

    # -------------------------------------------------------------------------
    # 4. Event BLOCKED: Poor audio quality
    # -------------------------------------------------------------------------
    def test_alert_blocked_by_poor_audio_quality(self):
        """Audio quality 'Poor' (rank 2) below rule requirement 'Acceptable' (rank 3) blocks alert."""
        audio_event = self._create_audio_event("evt-poor-005", "Gunshot", AudioEvent.QUALITY_POOR)

        pred = PythonPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Gunshot",
            confidence=0.95,
            confidence_scores={"Gunshot": 0.95, "Background Noise": 0.05},
            model_version="xgb_v1.0",
        )

        comp = ComparisonResult.objects.create(
            audio_event=audio_event,
            confidence_difference=0.05,
            top_two_margin=0.90,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            is_uncertain=False,
            is_overlapping=False,
        )

        alert = evaluate_and_trigger_alert(audio_event, comparison_result=comp, prediction=pred)

        self.assertIsNone(alert)
        self.assertEqual(Alert.objects.filter(audio_event=audio_event).count(), 0)

    def test_alert_blocked_by_unusable_audio_quality(self):
        """Audio quality 'Unusable' (rank 1) blocks alert."""
        audio_event = self._create_audio_event("evt-unusable-006", "Gunshot", AudioEvent.QUALITY_UNUSABLE)

        pred = PythonPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Gunshot",
            confidence=0.99,
            confidence_scores={"Gunshot": 0.99, "Background Noise": 0.01},
            model_version="xgb_v1.0",
        )

        comp = ComparisonResult.objects.create(
            audio_event=audio_event,
            confidence_difference=0.01,
            top_two_margin=0.98,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            is_uncertain=False,
            is_overlapping=False,
        )

        alert = evaluate_and_trigger_alert(audio_event, comparison_result=comp, prediction=pred)
        self.assertIsNone(alert)

    # -------------------------------------------------------------------------
    # 5. Pure Function evaluate_alert_conditions Unit Tests
    # -------------------------------------------------------------------------
    def test_pure_evaluate_alert_conditions_pass(self):
        """Pure evaluation function returns True when all conditions are satisfied."""
        result = evaluate_alert_conditions(
            confidence=0.88,
            top_two_margin=0.25,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
            min_confidence=0.85,
            required_top_two_margin=0.15,
            requires_model_agreement=True,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=1,
        )
        self.assertTrue(result)

    def test_pure_evaluate_alert_conditions_fails_low_margin(self):
        """Pure evaluation returns False when top-two margin is below required threshold."""
        result = evaluate_alert_conditions(
            confidence=0.90,
            top_two_margin=0.05,  # Below 0.15
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
            min_confidence=0.85,
            required_top_two_margin=0.15,
            requires_model_agreement=True,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=1,
        )
        self.assertFalse(result)

    def test_pure_evaluate_alert_conditions_fails_consecutive_detections(self):
        """Pure evaluation returns False when consecutive count < required count."""
        result = evaluate_alert_conditions(
            confidence=0.90,
            top_two_margin=0.30,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
            min_confidence=0.85,
            required_top_two_margin=0.15,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=2,  # Requires 2, given 1
        )
        self.assertFalse(result)

    def test_evaluate_rule_against_metrics_with_duck_typed_rule(self):
        """evaluate_rule_against_metrics accepts a mock duck-typed rule object."""
        mock_rule = MagicMock(
            min_confidence=0.75,
            top_two_margin=0.10,
            requires_model_agreement=False,
            requires_min_quality=AlertRule.QUALITY_ACCEPTABLE,
            required_consecutive_detections=1,
        )

        passes = evaluate_rule_against_metrics(
            rule=mock_rule,
            confidence=0.80,
            top_two_margin=0.20,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
        )
        self.assertTrue(passes)

        fails = evaluate_rule_against_metrics(
            rule=mock_rule,
            confidence=0.60,  # Below 0.75
            top_two_margin=0.20,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            quality=AlertRule.QUALITY_GOOD,
            consecutive_count=1,
        )
        self.assertFalse(fails)

    # -------------------------------------------------------------------------
    # 6. Idempotency & Fallback Behavior
    # -------------------------------------------------------------------------
    def test_alert_idempotency_returns_existing_alert(self):
        """Re-evaluating an event that already has an active alert returns the existing alert without creating a duplicate."""
        audio_event = self._create_audio_event("evt-idemp-007", "Gunshot", AudioEvent.QUALITY_GOOD)

        pred = PythonPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Gunshot",
            confidence=0.95,
            confidence_scores={"Gunshot": 0.95, "Background Noise": 0.05},
            model_version="xgb_v1.0",
        )

        comp = ComparisonResult.objects.create(
            audio_event=audio_event,
            confidence_difference=0.05,
            top_two_margin=0.90,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            is_uncertain=False,
            is_overlapping=False,
        )

        alert1 = evaluate_and_trigger_alert(audio_event, comparison_result=comp, prediction=pred)
        alert2 = evaluate_and_trigger_alert(audio_event, comparison_result=comp, prediction=pred)

        self.assertEqual(alert1.id, alert2.id)
        self.assertEqual(Alert.objects.filter(audio_event=audio_event).count(), 1)

    def test_fallback_to_default_rule_when_category_unmatched(self):
        """When event category has no explicit AlertRule, engine falls back to 'Background Noise' rule."""
        audio_event = self._create_audio_event("evt-unmatched-008", "Unknown Category", AudioEvent.QUALITY_GOOD)

        pred = PythonPrediction.objects.create(
            audio_event=audio_event,
            predicted_class="Unknown Category",
            confidence=0.65,
            confidence_scores={"Unknown Category": 0.65},
            model_version="xgb_v1.0",
        )

        comp = ComparisonResult.objects.create(
            audio_event=audio_event,
            confidence_difference=0.05,
            top_two_margin=0.55,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            is_uncertain=False,
            is_overlapping=False,
        )

        # Passes default_bg_rule (min_confidence=0.50, severity=Informational)
        alert = evaluate_and_trigger_alert(audio_event, comparison_result=comp, prediction=pred)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.alert_rule, self.default_bg_rule)
        self.assertEqual(alert.severity, AlertRule.SEVERITY_INFORMATIONAL)
