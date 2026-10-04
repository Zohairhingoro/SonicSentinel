from accounts.permissions import (
    is_administrator,
    is_audio_reviewer,
    is_maintenance_operator,
    is_normal_user,
    is_security_operator,
    is_staff_user,
)


def role_permissions(request):
    """Context processor providing role-permission booleans to templates directly evaluated via accounts.permissions."""
    user = getattr(request, "user", None)
    return {
        "user_is_normal": is_normal_user(user),
        "user_is_reviewer": is_audio_reviewer(user),
        "user_is_security": is_security_operator(user),
        "user_is_maintenance": is_maintenance_operator(user),
        "user_is_admin": is_administrator(user),
        "user_is_staff": is_staff_user(user),
    }
