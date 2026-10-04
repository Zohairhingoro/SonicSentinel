"""
tests/django/test_permission_mixins.py — Unit tests for accounts/permissions.py.

Verifies that each permission mixin:
  - NormalUserRequiredMixin
  - AudioReviewerRequiredMixin
  - SecurityOperatorRequiredMixin
  - MaintenanceOperatorRequiredMixin
  - AdministratorRequiredMixin
  - StaffRequiredMixin
as well as the corresponding view decorators (require_normal_user, require_audio_reviewer, etc.)
correctly allow access to authorized roles and deny access (HTTP 403 / PermissionDenied or HTTP 302 redirect)
to unauthorized roles and unauthenticated users across representative protected views.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.core.exceptions import PermissionDenied
from django.http import HttpResponse
from django.test import RequestFactory, TestCase, override_settings
from django.urls import path
from django.views import View

from accounts.permissions import (
    AdministratorRequiredMixin,
    AudioReviewerRequiredMixin,
    MaintenanceOperatorRequiredMixin,
    NormalUserRequiredMixin,
    SecurityOperatorRequiredMixin,
    StaffRequiredMixin,
    is_administrator,
    is_audio_reviewer,
    is_maintenance_operator,
    is_normal_user,
    is_security_operator,
    is_staff_user,
    require_administrator,
    require_audio_reviewer,
    require_maintenance_operator,
    require_normal_user,
    require_security_operator,
    require_staff,
)

User = get_user_model()


# Representative Class-Based Views for each Permission Mixin
class NormalUserProtectedView(NormalUserRequiredMixin, View):
    def get(self, request):
        return HttpResponse("normal_user_granted")


class AudioReviewerProtectedView(AudioReviewerRequiredMixin, View):
    def get(self, request):
        return HttpResponse("audio_reviewer_granted")


class SecurityOperatorProtectedView(SecurityOperatorRequiredMixin, View):
    def get(self, request):
        return HttpResponse("security_operator_granted")


class MaintenanceOperatorProtectedView(MaintenanceOperatorRequiredMixin, View):
    def get(self, request):
        return HttpResponse("maintenance_operator_granted")


class AdministratorProtectedView(AdministratorRequiredMixin, View):
    def get(self, request):
        return HttpResponse("administrator_granted")


class StaffProtectedView(StaffRequiredMixin, View):
    def get(self, request):
        return HttpResponse("staff_granted")


# Representative Function-Based Views for each Decorator
@require_normal_user
def normal_user_fbv(request):
    return HttpResponse("fbv_normal_user_granted")


@require_audio_reviewer
def audio_reviewer_fbv(request):
    return HttpResponse("fbv_audio_reviewer_granted")


@require_security_operator
def security_operator_fbv(request):
    return HttpResponse("fbv_security_operator_granted")


@require_maintenance_operator
def maintenance_operator_fbv(request):
    return HttpResponse("fbv_maintenance_operator_granted")


@require_administrator
def administrator_fbv(request):
    return HttpResponse("fbv_administrator_granted")


@require_staff
def staff_fbv(request):
    return HttpResponse("fbv_staff_granted")


# Test URL configuration
urlpatterns = [
    path("test/cbv/normal/", NormalUserProtectedView.as_view(), name="test_cbv_normal"),
    path("test/cbv/reviewer/", AudioReviewerProtectedView.as_view(), name="test_cbv_reviewer"),
    path("test/cbv/security/", SecurityOperatorProtectedView.as_view(), name="test_cbv_security"),
    path("test/cbv/maintenance/", MaintenanceOperatorProtectedView.as_view(), name="test_cbv_maintenance"),
    path("test/cbv/admin/", AdministratorProtectedView.as_view(), name="test_cbv_admin"),
    path("test/cbv/staff/", StaffProtectedView.as_view(), name="test_cbv_staff"),
    path("test/fbv/normal/", normal_user_fbv, name="test_fbv_normal"),
    path("test/fbv/reviewer/", audio_reviewer_fbv, name="test_fbv_reviewer"),
    path("test/fbv/security/", security_operator_fbv, name="test_fbv_security"),
    path("test/fbv/maintenance/", maintenance_operator_fbv, name="test_fbv_maintenance"),
    path("test/fbv/admin/", administrator_fbv, name="test_fbv_admin"),
    path("test/fbv/staff/", staff_fbv, name="test_fbv_staff"),
]


@override_settings(ROOT_URLCONF=__name__)
class PermissionMixinsTestCase(TestCase):
    """Verifies that each permission mixin permits only authorized roles and denies all other roles."""

    def setUp(self):
        self.factory = RequestFactory()

        self.normal_user = User.objects.create_user(
            username="test_normal", password="password123", role=User.ROLE_NORMAL_USER
        )
        self.audio_reviewer = User.objects.create_user(
            username="test_reviewer", password="password123", role=User.ROLE_AUDIO_REVIEWER
        )
        self.security_operator = User.objects.create_user(
            username="test_sec_op", password="password123", role=User.ROLE_SECURITY_OPERATOR
        )
        self.maintenance_operator = User.objects.create_user(
            username="test_maint_op", password="password123", role=User.ROLE_MAINTENANCE_OPERATOR
        )
        self.administrator = User.objects.create_user(
            username="test_admin", password="password123", role=User.ROLE_ADMINISTRATOR
        )
        self.superuser = User.objects.create_superuser(
            username="test_super", password="password123", role=User.ROLE_NORMAL_USER
        )

    # -------------------------------------------------------------------------
    # Helper Function Unit Tests
    # -------------------------------------------------------------------------
    def test_role_helper_functions_with_anonymous_user(self):
        """Anonymous / unauthenticated users must return False for all role helpers."""
        anon = AnonymousUser()
        self.assertFalse(is_normal_user(anon))
        self.assertFalse(is_audio_reviewer(anon))
        self.assertFalse(is_security_operator(anon))
        self.assertFalse(is_maintenance_operator(anon))
        self.assertFalse(is_administrator(anon))
        self.assertFalse(is_staff_user(anon))

    def test_role_helper_functions_with_none(self):
        """None passed to role helpers must return False."""
        self.assertFalse(is_normal_user(None))
        self.assertFalse(is_audio_reviewer(None))
        self.assertFalse(is_security_operator(None))
        self.assertFalse(is_maintenance_operator(None))
        self.assertFalse(is_administrator(None))
        self.assertFalse(is_staff_user(None))

    def test_role_helper_functions_role_matrix(self):
        """Verify role helpers against all authenticated user roles."""
        # Normal User
        self.assertTrue(is_normal_user(self.normal_user))
        self.assertFalse(is_audio_reviewer(self.normal_user))
        self.assertFalse(is_security_operator(self.normal_user))
        self.assertFalse(is_maintenance_operator(self.normal_user))
        self.assertFalse(is_administrator(self.normal_user))
        self.assertFalse(is_staff_user(self.normal_user))

        # Audio Reviewer
        self.assertFalse(is_normal_user(self.audio_reviewer))
        self.assertTrue(is_audio_reviewer(self.audio_reviewer))
        self.assertFalse(is_security_operator(self.audio_reviewer))
        self.assertFalse(is_maintenance_operator(self.audio_reviewer))
        self.assertFalse(is_administrator(self.audio_reviewer))
        self.assertTrue(is_staff_user(self.audio_reviewer))

        # Security Operator
        self.assertFalse(is_normal_user(self.security_operator))
        self.assertFalse(is_audio_reviewer(self.security_operator))
        self.assertTrue(is_security_operator(self.security_operator))
        self.assertFalse(is_maintenance_operator(self.security_operator))
        self.assertFalse(is_administrator(self.security_operator))
        self.assertTrue(is_staff_user(self.security_operator))

        # Maintenance Operator
        self.assertFalse(is_normal_user(self.maintenance_operator))
        self.assertFalse(is_audio_reviewer(self.maintenance_operator))
        self.assertFalse(is_security_operator(self.maintenance_operator))
        self.assertTrue(is_maintenance_operator(self.maintenance_operator))
        self.assertFalse(is_administrator(self.maintenance_operator))
        self.assertTrue(is_staff_user(self.maintenance_operator))

        # Administrator
        self.assertFalse(is_normal_user(self.administrator))
        self.assertFalse(is_audio_reviewer(self.administrator))
        self.assertFalse(is_security_operator(self.administrator))
        self.assertFalse(is_maintenance_operator(self.administrator))
        self.assertTrue(is_administrator(self.administrator))
        self.assertTrue(is_staff_user(self.administrator))

        # Superuser
        self.assertTrue(is_administrator(self.superuser))
        self.assertTrue(is_staff_user(self.superuser))

    # -------------------------------------------------------------------------
    # 1. NormalUserRequiredMixin
    # -------------------------------------------------------------------------
    def test_normal_user_mixin_allows_normal_user(self):
        """NormalUserRequiredMixin allows normal_user access."""
        self.client.force_login(self.normal_user)
        response = self.client.get("/test/cbv/normal/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "normal_user_granted")

    def test_normal_user_mixin_denies_other_roles(self):
        """NormalUserRequiredMixin denies all other roles with HTTP 403."""
        denied_users = [
            self.audio_reviewer,
            self.security_operator,
            self.maintenance_operator,
            self.administrator,
        ]
        for user in denied_users:
            self.client.force_login(user)
            response = self.client.get("/test/cbv/normal/")
            self.assertEqual(
                response.status_code,
                403,
                f"Role {user.role} should be denied access to NormalUserRequiredMixin",
            )

    def test_normal_user_mixin_redirects_unauthenticated(self):
        """NormalUserRequiredMixin redirects anonymous users to login with HTTP 302."""
        response = self.client.get("/test/cbv/normal/")
        self.assertEqual(response.status_code, 302)

    # -------------------------------------------------------------------------
    # 2. AudioReviewerRequiredMixin
    # -------------------------------------------------------------------------
    def test_audio_reviewer_mixin_allows_audio_reviewer(self):
        """AudioReviewerRequiredMixin allows audio_reviewer access."""
        self.client.force_login(self.audio_reviewer)
        response = self.client.get("/test/cbv/reviewer/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "audio_reviewer_granted")

    def test_audio_reviewer_mixin_denies_other_roles(self):
        """AudioReviewerRequiredMixin denies normal_user, security_operator, maintenance_operator, administrator."""
        denied_users = [
            self.normal_user,
            self.security_operator,
            self.maintenance_operator,
            self.administrator,
        ]
        for user in denied_users:
            self.client.force_login(user)
            response = self.client.get("/test/cbv/reviewer/")
            self.assertEqual(
                response.status_code,
                403,
                f"Role {user.role} should be denied access to AudioReviewerRequiredMixin",
            )

    def test_audio_reviewer_mixin_redirects_unauthenticated(self):
        """AudioReviewerRequiredMixin redirects anonymous users to login with HTTP 302."""
        response = self.client.get("/test/cbv/reviewer/")
        self.assertEqual(response.status_code, 302)

    # -------------------------------------------------------------------------
    # 3. SecurityOperatorRequiredMixin
    # -------------------------------------------------------------------------
    def test_security_operator_mixin_allows_security_operator(self):
        """SecurityOperatorRequiredMixin allows security_operator access."""
        self.client.force_login(self.security_operator)
        response = self.client.get("/test/cbv/security/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "security_operator_granted")

    def test_security_operator_mixin_denies_other_roles(self):
        """SecurityOperatorRequiredMixin denies normal_user, audio_reviewer, maintenance_operator, administrator."""
        denied_users = [
            self.normal_user,
            self.audio_reviewer,
            self.maintenance_operator,
            self.administrator,
        ]
        for user in denied_users:
            self.client.force_login(user)
            response = self.client.get("/test/cbv/security/")
            self.assertEqual(
                response.status_code,
                403,
                f"Role {user.role} should be denied access to SecurityOperatorRequiredMixin",
            )

    def test_security_operator_mixin_redirects_unauthenticated(self):
        """SecurityOperatorRequiredMixin redirects anonymous users to login with HTTP 302."""
        response = self.client.get("/test/cbv/security/")
        self.assertEqual(response.status_code, 302)

    # -------------------------------------------------------------------------
    # 4. MaintenanceOperatorRequiredMixin
    # -------------------------------------------------------------------------
    def test_maintenance_operator_mixin_allows_maintenance_operator(self):
        """MaintenanceOperatorRequiredMixin allows maintenance_operator access."""
        self.client.force_login(self.maintenance_operator)
        response = self.client.get("/test/cbv/maintenance/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "maintenance_operator_granted")

    def test_maintenance_operator_mixin_denies_other_roles(self):
        """MaintenanceOperatorRequiredMixin denies normal_user, audio_reviewer, security_operator, administrator."""
        denied_users = [
            self.normal_user,
            self.audio_reviewer,
            self.security_operator,
            self.administrator,
        ]
        for user in denied_users:
            self.client.force_login(user)
            response = self.client.get("/test/cbv/maintenance/")
            self.assertEqual(
                response.status_code,
                403,
                f"Role {user.role} should be denied access to MaintenanceOperatorRequiredMixin",
            )

    def test_maintenance_operator_mixin_redirects_unauthenticated(self):
        """MaintenanceOperatorRequiredMixin redirects anonymous users to login with HTTP 302."""
        response = self.client.get("/test/cbv/maintenance/")
        self.assertEqual(response.status_code, 302)

    # -------------------------------------------------------------------------
    # 5. AdministratorRequiredMixin
    # -------------------------------------------------------------------------
    def test_administrator_mixin_allows_administrator(self):
        """AdministratorRequiredMixin allows administrator access."""
        self.client.force_login(self.administrator)
        response = self.client.get("/test/cbv/admin/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "administrator_granted")

    def test_administrator_mixin_allows_superuser(self):
        """AdministratorRequiredMixin allows superuser access even with non-admin role."""
        self.client.force_login(self.superuser)
        response = self.client.get("/test/cbv/admin/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "administrator_granted")

    def test_administrator_mixin_denies_non_admin_roles(self):
        """AdministratorRequiredMixin denies normal_user, audio_reviewer, security_operator, maintenance_operator."""
        denied_users = [
            self.normal_user,
            self.audio_reviewer,
            self.security_operator,
            self.maintenance_operator,
        ]
        for user in denied_users:
            self.client.force_login(user)
            response = self.client.get("/test/cbv/admin/")
            self.assertEqual(
                response.status_code,
                403,
                f"Role {user.role} should be denied access to AdministratorRequiredMixin",
            )

    def test_administrator_mixin_redirects_unauthenticated(self):
        """AdministratorRequiredMixin redirects anonymous users to login with HTTP 302."""
        response = self.client.get("/test/cbv/admin/")
        self.assertEqual(response.status_code, 302)

    # -------------------------------------------------------------------------
    # 6. StaffRequiredMixin
    # -------------------------------------------------------------------------
    def test_staff_mixin_allows_all_staff_roles(self):
        """StaffRequiredMixin allows audio_reviewer, security_operator, maintenance_operator, administrator, superuser."""
        staff_users = [
            self.audio_reviewer,
            self.security_operator,
            self.maintenance_operator,
            self.administrator,
            self.superuser,
        ]
        for user in staff_users:
            self.client.force_login(user)
            response = self.client.get("/test/cbv/staff/")
            self.assertEqual(
                response.status_code,
                200,
                f"Staff user {user.username} ({user.role}) should be allowed access",
            )
            self.assertEqual(response.content.decode(), "staff_granted")

    def test_staff_mixin_denies_normal_user(self):
        """StaffRequiredMixin denies normal_user with HTTP 403."""
        self.client.force_login(self.normal_user)
        response = self.client.get("/test/cbv/staff/")
        self.assertEqual(response.status_code, 403)

    def test_staff_mixin_redirects_unauthenticated(self):
        """StaffRequiredMixin redirects anonymous users to login with HTTP 302."""
        response = self.client.get("/test/cbv/staff/")
        self.assertEqual(response.status_code, 302)

    # -------------------------------------------------------------------------
    # Direct View Dispatch with RequestFactory (PermissionDenied assertion)
    # -------------------------------------------------------------------------
    def test_direct_dispatch_raises_permission_denied_for_unauthorized_user(self):
        """Calling setup/dispatch directly with RequestFactory raises PermissionDenied."""
        request = self.factory.get("/test/cbv/admin/")
        request.user = self.normal_user

        view = AdministratorProtectedView.as_view()
        with self.assertRaises(PermissionDenied):
            view(request)

    def test_direct_dispatch_returns_200_for_authorized_user(self):
        """Calling setup/dispatch directly with RequestFactory returns HttpResponse for authorized user."""
        request = self.factory.get("/test/cbv/admin/")
        request.user = self.administrator

        view = AdministratorProtectedView.as_view()
        response = view(request)
        self.assertEqual(response.status_code, 200)

    # -------------------------------------------------------------------------
    # Function-Based View Decorators
    # -------------------------------------------------------------------------
    def test_decorator_require_normal_user(self):
        """require_normal_user decorator permits normal_user and denies other roles."""
        self.client.force_login(self.normal_user)
        response = self.client.get("/test/fbv/normal/")
        self.assertEqual(response.status_code, 200)

        self.client.force_login(self.audio_reviewer)
        response = self.client.get("/test/fbv/normal/")
        self.assertEqual(response.status_code, 403)

    def test_decorator_require_audio_reviewer(self):
        """require_audio_reviewer decorator permits audio_reviewer and denies normal_user."""
        self.client.force_login(self.audio_reviewer)
        response = self.client.get("/test/fbv/reviewer/")
        self.assertEqual(response.status_code, 200)

        self.client.force_login(self.normal_user)
        response = self.client.get("/test/fbv/reviewer/")
        self.assertEqual(response.status_code, 403)

    def test_decorator_require_security_operator(self):
        """require_security_operator decorator permits security_operator and denies normal_user."""
        self.client.force_login(self.security_operator)
        response = self.client.get("/test/fbv/security/")
        self.assertEqual(response.status_code, 200)

        self.client.force_login(self.normal_user)
        response = self.client.get("/test/fbv/security/")
        self.assertEqual(response.status_code, 403)

    def test_decorator_require_maintenance_operator(self):
        """require_maintenance_operator decorator permits maintenance_operator and denies normal_user."""
        self.client.force_login(self.maintenance_operator)
        response = self.client.get("/test/fbv/maintenance/")
        self.assertEqual(response.status_code, 200)

        self.client.force_login(self.normal_user)
        response = self.client.get("/test/fbv/maintenance/")
        self.assertEqual(response.status_code, 403)

    def test_decorator_require_administrator(self):
        """require_administrator decorator permits administrator and denies non-admin roles."""
        self.client.force_login(self.administrator)
        response = self.client.get("/test/fbv/admin/")
        self.assertEqual(response.status_code, 200)

        self.client.force_login(self.superuser)
        response = self.client.get("/test/fbv/admin/")
        self.assertEqual(response.status_code, 200)

        self.client.force_login(self.normal_user)
        response = self.client.get("/test/fbv/admin/")
        self.assertEqual(response.status_code, 403)

    def test_decorator_require_staff(self):
        """require_staff decorator permits any staff role and denies normal_user."""
        self.client.force_login(self.audio_reviewer)
        response = self.client.get("/test/fbv/staff/")
        self.assertEqual(response.status_code, 200)

        self.client.force_login(self.normal_user)
        response = self.client.get("/test/fbv/staff/")
        self.assertEqual(response.status_code, 403)

    def test_decorators_redirect_unauthenticated(self):
        """All decorators redirect unauthenticated users to login URL with HTTP 302."""
        endpoints = [
            "/test/fbv/normal/",
            "/test/fbv/reviewer/",
            "/test/fbv/security/",
            "/test/fbv/maintenance/",
            "/test/fbv/admin/",
            "/test/fbv/staff/",
        ]
        for url in endpoints:
            response = self.client.get(url)
            self.assertEqual(
                response.status_code,
                302,
                f"Endpoint {url} should redirect unauthenticated request to login",
            )
