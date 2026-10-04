from datetime import timedelta
from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse
from django.utils.timezone import now

from audit.models import AuditLog
from audit.services import list_audit_logs, log_action

User = get_user_model()


class AuditViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()

        # Users
        self.admin = User.objects.create_user(
            username="admin_user",
            password="password123",
            role=User.ROLE_ADMINISTRATOR,
            is_staff=True,
            is_superuser=True,
        )
        self.normal_user = User.objects.create_user(
            username="normal_user",
            password="password123",
            role=User.ROLE_NORMAL_USER,
        )

        # Create audit logs
        self.log1 = log_action(self.admin, AuditLog.ACTION_LOGIN, {"ip": "127.0.0.1"})
        self.log2 = log_action(self.normal_user, AuditLog.ACTION_UPLOAD, {"filename": "sample.wav"})
        self.log3 = log_action(self.normal_user, AuditLog.ACTION_REVIEW, {"decision": "Glass Breaking"})

    def test_anonymous_access_redirects(self):
        response = self.client.get(reverse("audit:log_list"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_normal_user_access_denied(self):
        self.client.login(username="normal_user", password="password123")
        response = self.client.get(reverse("audit:log_list"))
        self.assertEqual(response.status_code, 403)

    def test_admin_user_access_granted(self):
        self.client.login(username="admin_user", password="password123")
        response = self.client.get(reverse("audit:log_list"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "audit/log_list.html")
        self.assertIn("logs_page", response.context)
        # Login signal also creates an AuditLog row, so total count is at least 3
        self.assertGreaterEqual(len(response.context["logs_page"].object_list), 3)

    def test_filter_by_action(self):
        qs = list_audit_logs({"action": AuditLog.ACTION_UPLOAD})
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs.first(), self.log2)

    def test_filter_by_user(self):
        qs = list_audit_logs({"user": self.normal_user})
        self.assertEqual(qs.count(), 2)
        self.assertIn(self.log2, qs)
        self.assertIn(self.log3, qs)

    def test_filter_by_date_range(self):
        time_now = now()
        start = time_now - timedelta(days=1)
        end = time_now + timedelta(days=1)

        qs = list_audit_logs({"start_date": start, "end_date": end})
        self.assertGreaterEqual(qs.count(), 3)
