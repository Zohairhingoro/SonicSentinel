from django.contrib.auth import get_user_model
from django.test import TestCase
from django.template.loader import render_to_string
from django.urls import reverse

from alerts.models import Alert, AlertRule
from audio_events.models import AudioEvent

User = get_user_model()


class ActiveAlertsWidgetTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="widget_user", password="password123")
        self.rule_critical = AlertRule.objects.create(
            category="Gunshot",
            severity=AlertRule.SEVERITY_CRITICAL,
            recommended_action="Dispatch security immediately",
        )
        self.rule_high = AlertRule.objects.create(
            category="Glass Breaking",
            severity=AlertRule.SEVERITY_HIGH,
            recommended_action="Inspect security perimeter",
        )

        self.audio1 = AudioEvent.objects.create(
            audio_id="AUD-WIDGET01",
            filename="gunshot_clip.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=22050,
            file_size=100000,
            category="Gunshot",
            storage_path="/media/audio/gunshot_clip.wav",
            hash="widget_hash_1",
        )

        self.audio2 = AudioEvent.objects.create(
            audio_id="AUD-WIDGET02",
            filename="glass_clip.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=22050,
            file_size=100000,
            category="Glass Breaking",
            storage_path="/media/audio/glass_clip.wav",
            hash="widget_hash_2",
        )

        self.audio3 = AudioEvent.objects.create(
            audio_id="AUD-WIDGET03",
            filename="dismissed_clip.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=22050,
            file_size=100000,
            category="Gunshot",
            storage_path="/media/audio/dismissed_clip.wav",
            hash="widget_hash_3",
        )

        # Active alert
        self.alert_active = Alert.objects.create(
            audio_event=self.audio1,
            alert_rule=self.rule_critical,
            severity=AlertRule.SEVERITY_CRITICAL,
            status=Alert.STATUS_ACTIVE,
        )

        # Escalated alert (active / unacknowledged)
        self.alert_escalated = Alert.objects.create(
            audio_event=self.audio2,
            alert_rule=self.rule_high,
            severity=AlertRule.SEVERITY_HIGH,
            status=Alert.STATUS_ESCALATED,
        )

        # Dismissed alert (should NOT appear in active polling API)
        self.alert_dismissed = Alert.objects.create(
            audio_event=self.audio3,
            alert_rule=self.rule_critical,
            severity=AlertRule.SEVERITY_CRITICAL,
            status=Alert.STATUS_DISMISSED,
        )

    def test_active_alerts_api_returns_unacknowledged_alerts(self):
        response = self.client.get(reverse("alerts:active_api"))
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("count", data)
        self.assertIn("alerts", data)

        # Only 2 alerts are active/escalated (dismissed alert is excluded)
        self.assertEqual(data["count"], 2)
        alert_ids = [a["id"] for a in data["alerts"]]
        self.assertIn(self.alert_active.id, alert_ids)
        self.assertIn(self.alert_escalated.id, alert_ids)
        self.assertNotIn(self.alert_dismissed.id, alert_ids)

        # Verify alert fields in JSON payload
        first_alert = data["alerts"][0]
        self.assertIn("audio_id", first_alert)
        self.assertIn("category", first_alert)
        self.assertIn("severity", first_alert)
        self.assertIn("time_ago", first_alert)
        self.assertIn("event_detail_url", first_alert)
        self.assertIn("/audio/events/?search=", first_alert["event_detail_url"])

    def test_active_alerts_widget_template_rendering(self):
        html = render_to_string("components/active_alerts_widget.html")
        self.assertIn('id="active-alerts-widget-card"', html)
        self.assertIn('id="active-alerts-list"', html)
        self.assertIn('id="active-alert-count"', html)
        self.assertIn("Active Sound Event Alerts", html)
        self.assertIn("pollActiveAlerts", html)
