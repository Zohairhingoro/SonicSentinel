from functools import wraps
from typing import Callable, Any, Union

from django.contrib.auth.mixins import UserPassesTestMixin
from django.contrib.auth.views import redirect_to_login
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect


def is_normal_user(user: Any) -> bool:
    """Check if authenticated user has normal_user role."""
    if not user or not user.is_authenticated:
        return False
    return getattr(user, "is_normal_user", False) or getattr(user, "role", None) == "normal_user"


def is_audio_reviewer(user: Any) -> bool:
    """Check if authenticated user has audio_reviewer role."""
    if not user or not user.is_authenticated:
        return False
    return getattr(user, "is_audio_reviewer", False) or getattr(user, "role", None) == "audio_reviewer"


def is_security_operator(user: Any) -> bool:
    """Check if authenticated user has security_operator role."""
    if not user or not user.is_authenticated:
        return False
    return getattr(user, "is_security_operator", False) or getattr(user, "role", None) == "security_operator"


def is_maintenance_operator(user: Any) -> bool:
    """Check if authenticated user has maintenance_operator role."""
    if not user or not user.is_authenticated:
        return False
    return getattr(user, "is_maintenance_operator", False) or getattr(user, "role", None) == "maintenance_operator"


def is_administrator(user: Any) -> bool:
    """Check if authenticated user has administrator role or is a superuser."""
    if not user or not user.is_authenticated:
        return False
    return (
        getattr(user, "is_administrator", False)
        or getattr(user, "role", None) == "administrator"
        or getattr(user, "is_superuser", False)
    )


def is_staff_user(user: Any) -> bool:
    """Check if user has any staff role (reviewer, operator, or admin)."""
    if not user or not user.is_authenticated:
        return False
    return (
        is_audio_reviewer(user)
        or is_security_operator(user)
        or is_maintenance_operator(user)
        or is_administrator(user)
    )


class BaseRoleRequiredMixin(UserPassesTestMixin):
    """Base mixin for class-based views enforcing role permissions."""

    request: HttpRequest

    def check_user_role(self, user: Any) -> bool:
        """Override in subclasses to perform specific role check."""
        raise NotImplementedError

    def test_func(self) -> bool:
        user = getattr(self.request, "user", None)
        if user is None:
            return False
        return self.check_user_role(user)

    def handle_no_permission(self) -> HttpResponseRedirect:
        user = getattr(self.request, "user", None)
        if user and user.is_authenticated:
            raise PermissionDenied("You do not have permission to access this resource.")
        return super().handle_no_permission()


class NormalUserRequiredMixin(BaseRoleRequiredMixin):
    """CBV Mixin requiring normal_user role."""

    def check_user_role(self, user: Any) -> bool:
        return is_normal_user(user)


class AudioReviewerRequiredMixin(BaseRoleRequiredMixin):
    """CBV Mixin requiring audio_reviewer role."""

    def check_user_role(self, user: Any) -> bool:
        return is_audio_reviewer(user)


class SecurityOperatorRequiredMixin(BaseRoleRequiredMixin):
    """CBV Mixin requiring security_operator role."""

    def check_user_role(self, user: Any) -> bool:
        return is_security_operator(user)


class MaintenanceOperatorRequiredMixin(BaseRoleRequiredMixin):
    """CBV Mixin requiring maintenance_operator role."""

    def check_user_role(self, user: Any) -> bool:
        return is_maintenance_operator(user)


class AdministratorRequiredMixin(BaseRoleRequiredMixin):
    """CBV Mixin requiring administrator role."""

    def check_user_role(self, user: Any) -> bool:
        return is_administrator(user)


class StaffRequiredMixin(BaseRoleRequiredMixin):
    """CBV Mixin requiring any staff role (reviewer, security/maintenance operator, admin)."""

    def check_user_role(self, user: Any) -> bool:
        return is_staff_user(user)


def role_required(check_func: Callable[[Any], bool]) -> Callable:
    """Decorator factory for function-based views enforcing role permission."""

    def decorator(view_func: Callable) -> Callable:
        @wraps(view_func)
        def _wrapped_view(request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
            user = getattr(request, "user", None)
            if not user or not user.is_authenticated:
                return redirect_to_login(request.get_full_path())
            if not check_func(user):
                raise PermissionDenied("You do not have permission to access this resource.")
            return view_func(request, *args, **kwargs)

        return _wrapped_view

    return decorator


require_normal_user = role_required(is_normal_user)
require_audio_reviewer = role_required(is_audio_reviewer)
require_security_operator = role_required(is_security_operator)
require_maintenance_operator = role_required(is_maintenance_operator)
require_administrator = role_required(is_administrator)
require_staff = role_required(is_staff_user)
