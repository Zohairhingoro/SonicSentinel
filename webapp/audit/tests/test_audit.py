from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase

from audit.models import AuditLog
from audit.services import log_action

User = get_user_model()


class AuditLogModelTest(TestCase):
    def test_action_choices_complete(self):
        expected = {
            AuditLog.ACTION_LOGIN,
            AuditLog.ACTION_UPLOAD,
            AuditLog.ACTION_MIC_SESSION,
            AuditLog.ACTION_PREDICTION,
            AuditLog.ACTION_ALERT,
            AuditLog.ACTION_REVIEW,
            AuditLog.ACTION_OVERRIDE,
            AuditLog.ACTION_EXPORT,
            AuditLog.ACTION_MODEL_UPDATE,
        }
        defined = {code for code, _ in AuditLog.ACTION_CHOICES}
        self.assertEqual(defined, expected)


class LogActionServiceTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="auditor", password="password123")

    def test_log_action_creates_row(self):
        entry = log_action(self.user, AuditLog.ACTION_UPLOAD, detail={"filename": "test.wav"})
        self.assertIsNotNone(entry.pk)
        self.assertEqual(entry.action, AuditLog.ACTION_UPLOAD)
        self.assertEqual(entry.user, self.user)
        self.assertEqual(entry.detail["filename"], "test.wav")

    def test_log_action_anonymous_user(self):
        anon = MagicMock()
        anon.is_authenticated = False
        entry = log_action(anon, AuditLog.ACTION_LOGIN, detail={"failed": True})
        self.assertIsNone(entry.user)
        self.assertTrue(entry.detail["failed"])

    def test_log_action_none_user(self):
        entry = log_action(None, AuditLog.ACTION_MODEL_UPDATE, detail={"version": "2.0.0"})
        self.assertIsNone(entry.user)
        self.assertEqual(entry.detail["version"], "2.0.0")

    def test_log_action_default_empty_detail(self):
        entry = log_action(self.user, AuditLog.ACTION_EXPORT)
        self.assertEqual(entry.detail, {})

    def test_audit_log_ordering_latest_first(self):
        log_action(self.user, AuditLog.ACTION_LOGIN)
        log_action(self.user, AuditLog.ACTION_UPLOAD)
        entries = list(AuditLog.objects.all())
        self.assertEqual(entries[0].action, AuditLog.ACTION_UPLOAD)
        self.assertEqual(entries[1].action, AuditLog.ACTION_LOGIN)


class LoginSignalTest(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user(username="signaluser", password="password123")

    def test_successful_login_creates_audit_entry(self):
        from django.contrib.auth.signals import user_logged_in
        request = self.factory.get("/")
        request.META["REMOTE_ADDR"] = "127.0.0.1"
        request.META["HTTP_USER_AGENT"] = "TestBrowser/1.0"

        before_count = AuditLog.objects.filter(action=AuditLog.ACTION_LOGIN).count()
        user_logged_in.send(sender=self.user.__class__, request=request, user=self.user)
        after_count = AuditLog.objects.filter(action=AuditLog.ACTION_LOGIN).count()

        self.assertEqual(after_count, before_count + 1)
        entry = AuditLog.objects.filter(action=AuditLog.ACTION_LOGIN).order_by("-timestamp").first()
        self.assertEqual(entry.detail["ip"], "127.0.0.1")

    def test_failed_login_creates_anonymous_audit_entry(self):
        from django.contrib.auth.signals import user_login_failed
        request = self.factory.post("/login/")
        request.META["REMOTE_ADDR"] = "192.168.1.1"

        before_count = AuditLog.objects.filter(action=AuditLog.ACTION_LOGIN).count()
        user_login_failed.send(
            sender=User,
            credentials={"username": "baduser"},
            request=request,
        )
        after_count = AuditLog.objects.filter(action=AuditLog.ACTION_LOGIN).count()

        self.assertEqual(after_count, before_count + 1)
        entry = AuditLog.objects.filter(action=AuditLog.ACTION_LOGIN).order_by("-timestamp").first()
        self.assertIsNone(entry.user)
        self.assertTrue(entry.detail.get("failed"))
