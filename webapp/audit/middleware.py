from django.contrib.auth.signals import user_logged_in, user_login_failed
from django.dispatch import receiver

from audit.models import AuditLog
from audit.services import log_action


@receiver(user_logged_in)
def _on_user_logged_in(sender, request, user, **kwargs):
    """Emit a LOGIN audit entry whenever a user successfully authenticates."""
    log_action(
        user=user,
        action=AuditLog.ACTION_LOGIN,
        detail={
            "ip": _get_client_ip(request),
            "user_agent": request.META.get("HTTP_USER_AGENT", "")[:256],
        },
    )


@receiver(user_login_failed)
def _on_user_login_failed(sender, credentials, request, **kwargs):
    """Emit a LOGIN audit entry with failure flag on failed authentication attempts."""
    log_action(
        user=None,
        action=AuditLog.ACTION_LOGIN,
        detail={
            "failed": True,
            "username": credentials.get("username", ""),
            "ip": _get_client_ip(request),
        },
    )


def _get_client_ip(request) -> str:
    """Extract the real client IP address from the request, respecting proxy headers."""
    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "")
