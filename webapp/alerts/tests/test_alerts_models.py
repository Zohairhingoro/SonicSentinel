from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from alerts.models import Alert, AlertRule
from audio_events.models import AudioEvent

User = get_user_model()


class AlertsModelsTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="alert_operator", password="password123")
        self.audio_event = AudioEvent.objects.create(
            filename="gunshot.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=22050,
            file_size=120000,
            uploaded_by=self.user,
            category="Gunshot",
            storage_path="/media/gunshot.wav",
            hash="alert_test_hash_1",
        )

    def test_seed_alert_rules_command(self):
        call_command("seed_alert_rules")
        self.assertEqual(AlertRule.objects.count(), 10)

        gunshot_rule = AlertRule.objects.get(category="Gunshot")
        self.assertEqual(gunshot_rule.severity, AlertRule.SEVERITY_CRITICAL)
        self.assertEqual(gunshot_rule.min_confidence, 0.85)
        self.assertEqual(gunshot_rule.required_consecutive_detections, 2)
        self.assertIn("verified gunshot detected", gunshot_rule.recommended_action)

    def test_alert_creation_and_status(self):
        rule = AlertRule.objects.create(
            category="Glass Breaking",
            min_confidence=0.75,
            severity=AlertRule.SEVERITY_HIGH,
            recommended_action="High-severity security alert",
        )

        alert = Alert.objects.create(
            audio_event=self.audio_event,
            alert_rule=rule,
            severity=AlertRule.SEVERITY_HIGH,
            status=Alert.STATUS_ACTIVE,
        )

        self.assertEqual(alert.status, Alert.STATUS_ACTIVE)
        self.assertEqual(alert.severity, AlertRule.SEVERITY_HIGH)
        self.assertEqual(self.audio_event.alerts.first(), alert)
        self.assertIn("High", str(alert))

        # Acknowledge alert
        alert.status = Alert.STATUS_ACKNOWLEDGED
        alert.resolved_by = self.user
        alert.save()
        self.assertEqual(alert.status, Alert.STATUS_ACKNOWLEDGED)
        self.assertEqual(alert.resolved_by, self.user)
