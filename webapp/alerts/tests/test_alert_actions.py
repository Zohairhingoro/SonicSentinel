from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from alerts.models import Alert, AlertRule
from audio_events.models import AudioEvent

User = get_user_model()


class AlertActionsTestCase(TestCase):
    def setUp(self):
        # Create users for each role
        self.normal_user = User.objects.create_user(
            username="user_normal", password="password123", role=User.ROLE_NORMAL_USER
        )
        self.reviewer_user = User.objects.create_user(
            username="user_reviewer", password="password123", role=User.ROLE_AUDIO_REVIEWER
        )
        self.security_user = User.objects.create_user(
            username="user_security", password="password123", role=User.ROLE_SECURITY_OPERATOR
        )
        self.maint_user = User.objects.create_user(
            username="user_maint", password="password123", role=User.ROLE_MAINTENANCE_OPERATOR
        )
        self.admin_user = User.objects.create_user(
            username="user_admin", password="password123", role=User.ROLE_ADMINISTRATOR
        )

        # Create alert rules
        self.rule_gunshot = AlertRule.objects.create(
            category="Gunshot",
            severity=AlertRule.SEVERITY_CRITICAL,
            recommended_action="Emergency response",
        )
        self.rule_machinery = AlertRule.objects.create(
            category="Machinery Fault",
            severity=AlertRule.SEVERITY_MEDIUM,
            recommended_action="Inspect bearings",
        )

        # Create audio events
        self.audio_gunshot = AudioEvent.objects.create(
            audio_id="AUD-GUNSHOT01",
            filename="gunshot.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=22050,
            file_size=50000,
            category="Gunshot",
            storage_path="/media/gunshot.wav",
            hash="hash_gunshot_1",
        )

        self.audio_machinery = AudioEvent.objects.create(
            audio_id="AUD-MACHINERY01",
            filename="machinery.wav",
            format="WAV",
            duration=4.0,
            sampling_rate=22050,
            file_size=60000,
            category="Machinery Fault",
            storage_path="/media/machinery.wav",
            hash="hash_machinery_1",
        )

        # Create alerts
        self.alert_security = Alert.objects.create(
            audio_event=self.audio_gunshot,
            alert_rule=self.rule_gunshot,
            severity=AlertRule.SEVERITY_CRITICAL,
            status=Alert.STATUS_ACTIVE,
        )

        self.alert_maintenance = Alert.objects.create(
            audio_event=self.audio_machinery,
            alert_rule=self.rule_machinery,
            severity=AlertRule.SEVERITY_MEDIUM,
            status=Alert.STATUS_ACTIVE,
        )

    def test_unauthenticated_user_cannot_action_alert(self):
        url = reverse("alerts:acknowledge", kwargs={"alert_id": self.alert_security.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 403)

    def test_normal_user_cannot_action_alert(self):
        self.client.login(username=self.normal_user.username, password="password123")
        url = reverse("alerts:acknowledge", kwargs={"alert_id": self.alert_security.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 403)

    def test_security_operator_can_acknowledge_security_alert(self):
        self.client.login(username=self.security_user.username, password="password123")
        url = reverse("alerts:acknowledge", kwargs={"alert_id": self.alert_security.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], Alert.STATUS_ACKNOWLEDGED)

        self.alert_security.refresh_from_db()
        self.assertEqual(self.alert_security.status, Alert.STATUS_ACKNOWLEDGED)
        self.assertIsNotNone(self.alert_security.resolved_at)
        self.assertEqual(self.alert_security.resolved_by, self.security_user)

    def test_maintenance_operator_can_dismiss_machinery_alert(self):
        self.client.login(username=self.maint_user.username, password="password123")
        url = reverse("alerts:dismiss", kwargs={"alert_id": self.alert_maintenance.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], Alert.STATUS_DISMISSED)

        self.alert_maintenance.refresh_from_db()
        self.assertEqual(self.alert_maintenance.status, Alert.STATUS_DISMISSED)
        self.assertIsNotNone(self.alert_maintenance.resolved_at)
        self.assertEqual(self.alert_maintenance.resolved_by, self.maint_user)

    def test_admin_can_escalate_alert(self):
        self.client.login(username=self.admin_user.username, password="password123")
        url = reverse("alerts:escalate", kwargs={"alert_id": self.alert_maintenance.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], Alert.STATUS_ESCALATED)
        self.assertEqual(data["severity"], AlertRule.SEVERITY_CRITICAL)

        self.alert_maintenance.refresh_from_db()
        self.assertEqual(self.alert_maintenance.status, Alert.STATUS_ESCALATED)
        self.assertEqual(self.alert_maintenance.severity, AlertRule.SEVERITY_CRITICAL)

    def test_maintenance_operator_cannot_escalate_security_alert(self):
        self.client.login(username=self.maint_user.username, password="password123")
        url = reverse("alerts:escalate", kwargs={"alert_id": self.alert_security.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 403)
