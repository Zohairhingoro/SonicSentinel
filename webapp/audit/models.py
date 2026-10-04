from django.conf import settings
from django.db import models


class AuditLog(models.Model):
    """Immutable audit trail record for security-relevant user and system actions."""

    objects = models.Manager()

    ACTION_LOGIN = "login"
    ACTION_UPLOAD = "upload"
    ACTION_MIC_SESSION = "mic_session"
    ACTION_PREDICTION = "prediction"
    ACTION_ALERT = "alert"
    ACTION_REVIEW = "review"
    ACTION_OVERRIDE = "override"
    ACTION_EXPORT = "export"
    ACTION_MODEL_UPDATE = "model_update"

    ACTION_CHOICES = [
        (ACTION_LOGIN, "Login"),
        (ACTION_UPLOAD, "Upload"),
        (ACTION_MIC_SESSION, "Mic Session"),
        (ACTION_PREDICTION, "Prediction"),
        (ACTION_ALERT, "Alert"),
        (ACTION_REVIEW, "Review"),
        (ACTION_OVERRIDE, "Override"),
        (ACTION_EXPORT, "Export"),
        (ACTION_MODEL_UPDATE, "Model Update"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_logs",
    )
    action = models.CharField(
        max_length=32,
        choices=ACTION_CHOICES,
        db_index=True,
    )
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    detail = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self) -> str:
        username = self.user.username if self.user is not None else "anonymous"
        return f"[{self.timestamp}] {username} → {self.action}"

