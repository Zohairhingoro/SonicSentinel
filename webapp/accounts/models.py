import uuid
from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """Custom user model supporting role-based access control and unique user IDs."""

    ROLE_NORMAL_USER = "normal_user"
    ROLE_AUDIO_REVIEWER = "audio_reviewer"
    ROLE_SECURITY_OPERATOR = "security_operator"
    ROLE_MAINTENANCE_OPERATOR = "maintenance_operator"
    ROLE_ADMINISTRATOR = "administrator"

    ROLE_CHOICES = [
        (ROLE_NORMAL_USER, "Normal User"),
        (ROLE_AUDIO_REVIEWER, "Audio Reviewer"),
        (ROLE_SECURITY_OPERATOR, "Security Operator"),
        (ROLE_MAINTENANCE_OPERATOR, "Maintenance Operator"),
        (ROLE_ADMINISTRATOR, "Administrator"),
    ]

    user_id = models.CharField(
        max_length=64,
        unique=True,
        editable=False,
        db_index=True,
        help_text="Unique public user identifier",
    )
    role = models.CharField(
        max_length=32,
        choices=ROLE_CHOICES,
        default=ROLE_NORMAL_USER,
        db_index=True,
        help_text="User role defining access permissions",
    )

    def save(self, *args, **kwargs) -> None:
        """Auto-generate unique user_id if not present before saving."""
        if not self.user_id:
            # pyrefly: ignore [bad-assignment]
            self.user_id = f"USR-{uuid.uuid4().hex[:12].upper()}"
        super().save(*args, **kwargs)

    @property
    def is_normal_user(self) -> bool:
        return self.role == self.ROLE_NORMAL_USER

    @property
    def is_audio_reviewer(self) -> bool:
        return self.role == self.ROLE_AUDIO_REVIEWER

    @property
    def is_security_operator(self) -> bool:
        return self.role == self.ROLE_SECURITY_OPERATOR

    @property
    def is_maintenance_operator(self) -> bool:
        return self.role == self.ROLE_MAINTENANCE_OPERATOR

    @property
    def is_administrator(self) -> bool:
        return bool(self.role == self.ROLE_ADMINISTRATOR or self.is_superuser)


    def __str__(self) -> str:
        return f"{self.username} ({self.role})"
