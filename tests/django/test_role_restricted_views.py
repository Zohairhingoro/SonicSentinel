"""Unit tests for role-based access control across all protected Django views."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = REPO_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from audio_events.models import AudioEvent

User = get_user_model()


class RoleRestrictedViewsTestCase(TestCase):
    """Verifies that every role-restricted view enforces strict access control."""

    @classmethod
    def setUpTestData(cls) -> None:
        cls.passwords: Dict[str, str] = {
            "normal_user": "NormalPass123!",
            "audio_reviewer": "ReviewerPass123!",
            "security_operator": "SecurityPass123!",
            "maintenance_operator": "MaintenancePass123!",
            "administrator": "AdminPass123!",
        }

        cls.users: Dict[str, Any] = {}
        for role_name, password in cls.passwords.items():
            user = User.objects.create_user(
                username=f"test_{role_name}",
                email=f"{role_name}@sentinel.ai",
                password=password,
                role=role_name,
            )
            cls.users[role_name] = user

        cls.review_event = AudioEvent.objects.create(
            audio_id="AUD-RBAC-REV01",
            filename="screaming_sample.wav",
            format="wav",
            duration=2.5,
            sampling_rate=22050,
            channels=1,
            bit_depth=16,
            file_size=110250,
            status=AudioEvent.STATUS_MANUAL_REVIEW,
            category="Panic Scream",
            quality=AudioEvent.QUALITY_GOOD,
            severity="Critical",
            storage_path="audio/AUD-RBAC-REV01/screaming_sample.wav",
            hash="hash_rbac_rev_01",
        )

    def setUp(self) -> None:
        self.client = Client()

    def _login_as(self, role_name: str) -> None:
        username = self.users[role_name].username
        password = self.passwords[role_name]
        logged_in = self.client.login(username=username, password=password)
        self.assertTrue(logged_in, f"Failed to login user {username}")

    def test_admin_dashboard_access_control(self) -> None:
        url = reverse("dashboard:admin")

        resp_anon = self.client.get(url)
        self.assertEqual(resp_anon.status_code, 302)

        unauthorized_roles = [
            "normal_user",
            "audio_reviewer",
            "security_operator",
            "maintenance_operator",
        ]
        for role in unauthorized_roles:
            self._login_as(role)
            resp = self.client.get(url)
            self.assertEqual(
                resp.status_code,
                403,
                f"Role '{role}' should be denied (403) from admin dashboard",
            )

        self._login_as("administrator")
        resp_admin = self.client.get(url)
        self.assertEqual(resp_admin.status_code, 200)

    def test_batch_upload_access_control(self) -> None:
        url = reverse("audio_events:batch_upload")

        resp_anon = self.client.get(url)
        self.assertEqual(resp_anon.status_code, 302)

        unauthorized_roles = [
            "normal_user",
            "audio_reviewer",
            "maintenance_operator",
        ]
        for role in unauthorized_roles:
            self._login_as(role)
            resp = self.client.get(url)
            self.assertEqual(
                resp.status_code,
                403,
                f"Role '{role}' should be denied (403) from batch upload",
            )

        authorized_roles = ["security_operator", "administrator"]
        for role in authorized_roles:
            self._login_as(role)
            resp = self.client.get(url)
            self.assertEqual(
                resp.status_code,
                200,
                f"Role '{role}' should have access (200) to batch upload",
            )

    def test_review_queue_access_control(self) -> None:
        url = reverse("reviews:queue")

        resp_anon = self.client.get(url)
        self.assertEqual(resp_anon.status_code, 302)

        unauthorized_roles = [
            "normal_user",
            "security_operator",
            "maintenance_operator",
        ]
        for role in unauthorized_roles:
            self._login_as(role)
            resp = self.client.get(url)
            self.assertEqual(
                resp.status_code,
                403,
                f"Role '{role}' should be denied (403) from review queue",
            )

        authorized_roles = ["audio_reviewer", "administrator"]
        for role in authorized_roles:
            self._login_as(role)
            resp = self.client.get(url)
            self.assertEqual(
                resp.status_code,
                200,
                f"Role '{role}' should have access (200) to review queue",
            )

    def test_review_detail_access_control(self) -> None:
        url = reverse("reviews:detail", kwargs={"audio_id": self.review_event.audio_id})

        resp_anon = self.client.get(url)
        self.assertEqual(resp_anon.status_code, 302)

        unauthorized_roles = [
            "normal_user",
            "security_operator",
            "maintenance_operator",
        ]
        for role in unauthorized_roles:
            self._login_as(role)
            resp = self.client.get(url)
            self.assertEqual(
                resp.status_code,
                403,
                f"Role '{role}' should be denied (403) from review detail",
            )

        authorized_roles = ["audio_reviewer", "administrator"]
        for role in authorized_roles:
            self._login_as(role)
            resp = self.client.get(url)
            self.assertEqual(
                resp.status_code,
                200,
                f"Role '{role}' should have access (200) to review detail",
            )

    def test_review_decision_submission_access_control(self) -> None:
        url = reverse("reviews:detail", kwargs={"audio_id": self.review_event.audio_id})
        # The decision must be a valid category display name from config/classes.json
        # (as returned by get_categories() — title-cased, not slugs).
        post_data = {
            "decision": "Panic Scream",
            "comment": "Confirmed by reviewer",
            "recommended_action": "No further action",
        }

        unauthorized_roles = [
            "normal_user",
            "security_operator",
            "maintenance_operator",
        ]
        for role in unauthorized_roles:
            self._login_as(role)
            resp = self.client.post(url, post_data)
            self.assertEqual(
                resp.status_code,
                403,
                f"Role '{role}' should be denied (403) from submitting review decision",
            )

        self._login_as("audio_reviewer")
        resp_reviewer = self.client.post(url, post_data)
        self.assertEqual(resp_reviewer.status_code, 302)

    def test_exports_access_control(self) -> None:
        url = reverse("dashboard:export")

        resp_anon = self.client.get(url)
        self.assertEqual(resp_anon.status_code, 302)

        unauthorized_roles = [
            "normal_user",
            "audio_reviewer",
            "security_operator",
            "maintenance_operator",
        ]
        for role in unauthorized_roles:
            self._login_as(role)
            resp = self.client.get(url)
            self.assertEqual(
                resp.status_code,
                403,
                f"Role '{role}' should be denied (403) from exports",
            )

        self._login_as("administrator")
        resp_admin = self.client.get(url)
        self.assertEqual(resp_admin.status_code, 200)
        self.assertIn("attachment;", resp_admin["Content-Disposition"])

    def test_audit_log_access_control(self) -> None:
        urls = [reverse("audit:index"), reverse("audit:log_list")]

        for url in urls:
            self.client.logout()
            resp_anon = self.client.get(url)
            self.assertEqual(resp_anon.status_code, 302)

            unauthorized_roles = [
                "normal_user",
                "audio_reviewer",
                "security_operator",
                "maintenance_operator",
            ]
            for role in unauthorized_roles:
                self._login_as(role)
                resp = self.client.get(url)
                self.assertEqual(
                    resp.status_code,
                    403,
                    f"Role '{role}' should be denied (403) from audit logs ({url})",
                )

            self._login_as("administrator")
            resp_admin = self.client.get(url)
            self.assertEqual(resp_admin.status_code, 200)

    def test_comprehensive_matrix_unauthorized_cases(self) -> None:
        matrix: List[Tuple[str, str, List[str], List[str]]] = [
            (
                "Admin Dashboard",
                reverse("dashboard:admin"),
                ["administrator"],
                ["normal_user", "audio_reviewer", "security_operator", "maintenance_operator"],
            ),
            (
                "Batch Upload",
                reverse("audio_events:batch_upload"),
                ["administrator", "security_operator"],
                ["normal_user", "audio_reviewer", "maintenance_operator"],
            ),
            (
                "Review Queue",
                reverse("reviews:queue"),
                ["administrator", "audio_reviewer"],
                ["normal_user", "security_operator", "maintenance_operator"],
            ),
            (
                "Review Detail",
                reverse("reviews:detail", kwargs={"audio_id": self.review_event.audio_id}),
                ["administrator", "audio_reviewer"],
                ["normal_user", "security_operator", "maintenance_operator"],
            ),
            (
                "Exports",
                reverse("dashboard:export"),
                ["administrator"],
                ["normal_user", "audio_reviewer", "security_operator", "maintenance_operator"],
            ),
            (
                "Audit Log",
                reverse("audit:index"),
                ["administrator"],
                ["normal_user", "audio_reviewer", "security_operator", "maintenance_operator"],
            ),
        ]

        for view_name, url, authorized_roles, unauthorized_roles in matrix:
            self.client.logout()
            resp_anon = self.client.get(url)
            self.assertEqual(
                resp_anon.status_code,
                302,
                f"Anonymous user must be redirected (302) for {view_name}",
            )

            for unauth_role in unauthorized_roles:
                self._login_as(unauth_role)
                resp_unauth = self.client.get(url)
                self.assertEqual(
                    resp_unauth.status_code,
                    403,
                    f"Role '{unauth_role}' must receive 403 for {view_name}",
                )

            for auth_role in authorized_roles:
                self._login_as(auth_role)
                resp_auth = self.client.get(url)
                self.assertEqual(
                    resp_auth.status_code,
                    200,
                    f"Role '{auth_role}' must receive 200 for {view_name}",
                )
