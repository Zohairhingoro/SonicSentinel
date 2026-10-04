from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from audio_events.models import AudioEvent

User = get_user_model()


class DashboardViewsTestCase(TestCase):
    def setUp(self):
        self.admin_user = User.objects.create_superuser(
            username="admin_dash",
            email="admin@sentinel.ai",
            password="adminpassword123",
        )
        self.normal_user = User.objects.create_user(
            username="normal_dash",
            email="user@sentinel.ai",
            password="userpassword123",
        )

        self.evt1 = AudioEvent.objects.create(
            audio_id="AUD-DASH0001",
            filename="glass_break.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=16000,
            file_size=96000,
            category="Glass Breaking",
            quality=AudioEvent.QUALITY_GOOD,
            severity="High",
            storage_path="audio/AUD-DASH0001/glass_break.wav",
            hash="hash_dash_1",
        )

    def test_home_page_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "SonicSentinel AI")

    def test_dashboard_index_view_returns_200(self):
        response = self.client.get(reverse("dashboard:index"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "home.html")
        self.assertIn("events_page", response.context)

    def test_user_dashboard_view_returns_200(self):
        self.client.login(username="normal_dash", password="userpassword123")
        response = self.client.get(reverse("dashboard:user"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "dashboard/user.html")
        self.assertEqual(response.context.get("dashboard_type"), "user")
        self.assertIn("recent_uploads", response.context)
        self.assertIn("current_detections", response.context)
        self.assertIn("critical_events", response.context)
        self.assertIn("quality_warnings", response.context)
        self.assertIn("review_records", response.context)

    def test_admin_dashboard_view_returns_200(self):
        self.client.login(username="admin_dash", password="adminpassword123")
        response = self.client.get(reverse("dashboard:admin"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "dashboard/admin.html")
        self.assertEqual(response.context.get("dashboard_type"), "admin")
        self.assertIn("summary_stats", response.context)
        self.assertIn("total_audio_events", response.context["summary_stats"])


    def test_analytics_dashboard_view_returns_200(self):
        response = self.client.get(reverse("dashboard:analytics"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "dashboard/analytics.html")
        self.assertIn("analytics", response.context)
        self.assertIn("events_page", response.context)

    def test_analytics_dashboard_view_with_filters(self):
        response = self.client.get(reverse("dashboard:analytics"), {"category": "Glass Breaking"})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "dashboard/analytics.html")
        self.assertContains(response, "glass_break.wav")

    def test_export_view_requires_authentication(self):
        response = self.client.get(reverse("dashboard:export"))
        self.assertEqual(response.status_code, 302)

    def test_export_view_requires_admin_role(self):
        self.client.login(username="normal_dash", password="userpassword123")
        response = self.client.get(reverse("dashboard:export"))
        self.assertEqual(response.status_code, 403)

    def test_export_view_succeeds_for_admin(self):
        self.client.login(username="admin_dash", password="adminpassword123")
        response = self.client.get(reverse("dashboard:export"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("attachment;", response["Content-Disposition"])
