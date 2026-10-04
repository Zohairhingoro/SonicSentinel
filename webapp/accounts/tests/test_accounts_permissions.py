from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory, TestCase
from django.core.exceptions import PermissionDenied

from accounts.permissions import (
    is_normal_user,
    is_audio_reviewer,
    is_security_operator,
    is_maintenance_operator,
    is_administrator,
    is_staff_user,
    NormalUserRequiredMixin,
    AudioReviewerRequiredMixin,
    SecurityOperatorRequiredMixin,
    MaintenanceOperatorRequiredMixin,
    AdministratorRequiredMixin,
    StaffRequiredMixin,
    require_normal_user,
    require_audio_reviewer,
    require_security_operator,
    require_maintenance_operator,
    require_administrator,
    require_staff,
)

User = get_user_model()


class AccountsPermissionsTestCase(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.normal_user = User.objects.create_user(
            username="normal", password="password", role=User.ROLE_NORMAL_USER
        )
        self.reviewer = User.objects.create_user(
            username="reviewer", password="password", role=User.ROLE_AUDIO_REVIEWER
        )
        self.sec_op = User.objects.create_user(
            username="sec_op", password="password", role=User.ROLE_SECURITY_OPERATOR
        )
        self.maint_op = User.objects.create_user(
            username="maint_op", password="password", role=User.ROLE_MAINTENANCE_OPERATOR
        )
        self.admin = User.objects.create_user(
            username="admin", password="password", role=User.ROLE_ADMINISTRATOR
        )
        self.superuser = User.objects.create_superuser(
            username="super", password="password", role=User.ROLE_ADMINISTRATOR
        )

    def test_role_helper_functions(self):
        anon = AnonymousUser()

        # Unauthenticated returns False for all
        self.assertFalse(is_normal_user(anon))
        self.assertFalse(is_audio_reviewer(anon))
        self.assertFalse(is_security_operator(anon))
        self.assertFalse(is_maintenance_operator(anon))
        self.assertFalse(is_administrator(anon))
        self.assertFalse(is_staff_user(anon))

        # Check normal_user
        self.assertTrue(is_normal_user(self.normal_user))
        self.assertFalse(is_audio_reviewer(self.normal_user))
        self.assertFalse(is_staff_user(self.normal_user))

        # Check audio_reviewer
        self.assertTrue(is_audio_reviewer(self.reviewer))
        self.assertTrue(is_staff_user(self.reviewer))
        self.assertFalse(is_normal_user(self.reviewer))

        # Check security_operator
        self.assertTrue(is_security_operator(self.sec_op))
        self.assertTrue(is_staff_user(self.sec_op))

        # Check maintenance_operator
        self.assertTrue(is_maintenance_operator(self.maint_op))
        self.assertTrue(is_staff_user(self.maint_op))

        # Check administrator and superuser
        self.assertTrue(is_administrator(self.admin))
        self.assertTrue(is_administrator(self.superuser))
        self.assertTrue(is_staff_user(self.admin))
        self.assertTrue(is_staff_user(self.superuser))

    def test_view_permissions_with_test_client(self):
        # Unauthenticated access to /accounts/ returns 302 redirect to login
        response = self.client.get("/accounts/")
        self.assertEqual(response.status_code, 302)

        # Unauthenticated access to /accounts/profile/ returns 302
        response = self.client.get("/accounts/profile/")
        self.assertEqual(response.status_code, 302)

        # Normal user accessing /accounts/ (staff only) returns 403
        self.client.force_login(self.normal_user)
        response = self.client.get("/accounts/")
        self.assertEqual(response.status_code, 403)

        # Normal user accessing /accounts/profile/ (normal user mixin) returns 200
        response = self.client.get("/accounts/profile/")
        self.assertEqual(response.status_code, 200)

        # Reviewer accessing /accounts/ (staff only) returns 200
        self.client.force_login(self.reviewer)
        response = self.client.get("/accounts/")
        self.assertEqual(response.status_code, 200)

        # Admin accessing /accounts/ returns 200
        self.client.force_login(self.admin)
        response = self.client.get("/accounts/")
        self.assertEqual(response.status_code, 200)
