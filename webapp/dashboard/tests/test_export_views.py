from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from audio_events.models import AudioEvent

User = get_user_model()


class ExportViewsTestCase(TestCase):
    def setUp(self):
        self.admin_user = User.objects.create_superuser(
            username="admin_exporter",
            email="admin@sentinel.ai",
            password="adminpassword123",
        )
        self.normal_user = User.objects.create_user(
            username="normal_user",
            email="user@sentinel.ai",
            password="userpassword123",
        )

        self.evt1 = AudioEvent.objects.create(
            audio_id="AUD-EXP0001",
            filename="gunshot_test.wav",
            format="WAV",
            duration=2.5,
            sampling_rate=16000,
            file_size=80000,
            category="Gunshot",
            quality=AudioEvent.QUALITY_GOOD,
            severity="Critical",
            storage_path="audio/AUD-EXP0001/gunshot_test.wav",
            hash="hash_exp_1",
        )

        self.evt2 = AudioEvent.objects.create(
            audio_id="AUD-EXP0002",
            filename="siren_test.wav",
            format="WAV",
            duration=4.1,
            sampling_rate=22050,
            file_size=120000,
            category="Alarm or Siren",
            quality=AudioEvent.QUALITY_POOR,
            severity="High",
            storage_path="audio/AUD-EXP0002/siren_test.wav",
            hash="hash_exp_2",
        )

    def test_export_view_anonymous_redirects_to_login(self):
        response = self.client.get(reverse("dashboard:export"))
        self.assertEqual(response.status_code, 302)

    def test_export_view_non_admin_forbidden(self):
        self.client.login(username="normal_user", password="userpassword123")
        response = self.client.get(reverse("dashboard:export"))
        self.assertEqual(response.status_code, 403)

    def test_export_view_admin_csv_success(self):
        self.client.login(username="admin_exporter", password="adminpassword123")
        response = self.client.get(reverse("dashboard:export"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("attachment;", response["Content-Disposition"])
        self.assertIn("gunshot_test.wav", response.content.decode("utf-8"))
        self.assertIn("siren_test.wav", response.content.decode("utf-8"))

    def test_export_view_admin_excel_format_success(self):
        self.client.login(username="admin_exporter", password="adminpassword123")
        response = self.client.get(reverse("dashboard:export"), {"format": "excel"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("application/vnd.ms-excel", response["Content-Type"])
        self.assertIn(".xlsx", response["Content-Disposition"])

    def test_export_view_preserves_query_filters(self):
        self.client.login(username="admin_exporter", password="adminpassword123")
        # Filter by category="Gunshot"
        response = self.client.get(reverse("dashboard:export"), {"category": "Gunshot", "format": "csv"})

        self.assertEqual(response.status_code, 200)
        csv_text = response.content.decode("utf-8")

        # Verify only Gunshot record is present, Siren record is excluded
        self.assertIn("gunshot_test.wav", csv_text)
        self.assertNotIn("siren_test.wav", csv_text)

    def test_event_list_page_renders_export_buttons_for_admin(self):
        self.client.login(username="admin_exporter", password="adminpassword123")
        response = self.client.get(reverse("audio_events:event_list"), {"category": "Gunshot"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="export-csv-btn"')
        self.assertContains(response, 'id="export-excel-btn"')
        self.assertContains(response, "category=Gunshot")

    def test_analytics_page_renders_export_buttons_for_admin(self):
        self.client.login(username="admin_exporter", password="adminpassword123")
        response = self.client.get(reverse("dashboard:analytics"), {"quality": "Good"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="analytics-export-csv-btn"')
        self.assertContains(response, 'id="analytics-export-excel-btn"')
        self.assertContains(response, "quality=Good")
