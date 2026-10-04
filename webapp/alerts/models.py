from django.conf import settings
from django.db import models
from audio_events.models import AudioEvent


class AlertRule(models.Model):
    """Configurable alert rules per sound event category."""

    objects = models.Manager()

    SEVERITY_INFORMATIONAL = "Informational"
    SEVERITY_LOW = "Low"
    SEVERITY_MEDIUM = "Medium"
    SEVERITY_HIGH = "High"
    SEVERITY_CRITICAL = "Critical"

    SEVERITY_CHOICES = [
        (SEVERITY_INFORMATIONAL, "Informational"),
        (SEVERITY_LOW, "Low"),
        (SEVERITY_MEDIUM, "Medium"),
        (SEVERITY_HIGH, "High"),
        (SEVERITY_CRITICAL, "Critical"),
    ]

    QUALITY_GOOD = "Good"
    QUALITY_ACCEPTABLE = "Acceptable"
    QUALITY_POOR = "Poor"
    QUALITY_UNUSABLE = "Unusable"

    QUALITY_CHOICES = [
        (QUALITY_GOOD, "Good"),
        (QUALITY_ACCEPTABLE, "Acceptable"),
        (QUALITY_POOR, "Poor"),
        (QUALITY_UNUSABLE, "Unusable"),
    ]

    category = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        help_text="Mandatory sound category name",
    )
    min_confidence = models.FloatField(
        default=0.70,
        help_text="Minimum confidence threshold required to trigger alert",
    )
    top_two_margin = models.FloatField(
        default=0.15,
        help_text="Minimum top-two margin threshold",
    )
    required_consecutive_detections = models.IntegerField(
        default=1,
        help_text="Required repeated detections across consecutive windows",
    )
    requires_model_agreement = models.BooleanField(
        default=False,
        help_text="Whether agreement between Python and GTM models is required",
    )
    requires_min_quality = models.CharField(
        max_length=32,
        choices=QUALITY_CHOICES,
        default=QUALITY_ACCEPTABLE,
        help_text="Minimum required audio quality",
    )
    severity = models.CharField(
        max_length=32,
        choices=SEVERITY_CHOICES,
        default=SEVERITY_MEDIUM,
        help_text="Alert severity level",
    )
    recommended_action = models.TextField(
        help_text="Recommended action for operator when alert triggers"
    )
    escalation_condition = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="Condition under which alert automatically escalates",
    )

    class Meta:
        ordering = ["category"]

    def __str__(self) -> str:
        return f"AlertRule({self.category}: {self.severity})"


class Alert(models.Model):
    """Represents a generated severity-based security or maintenance alert."""

    objects = models.Manager()

    STATUS_ACTIVE = "active"
    STATUS_ACKNOWLEDGED = "acknowledged"
    STATUS_DISMISSED = "dismissed"
    STATUS_ESCALATED = "escalated"

    STATUS_CHOICES = [
        (STATUS_ACTIVE, "Active"),
        (STATUS_ACKNOWLEDGED, "Acknowledged"),
        (STATUS_DISMISSED, "Dismissed"),
        (STATUS_ESCALATED, "Escalated"),
    ]

    audio_event = models.ForeignKey(
        AudioEvent,
        on_delete=models.CASCADE,
        related_name="alerts",
    )
    alert_rule = models.ForeignKey(
        AlertRule,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="alerts",
    )
    severity = models.CharField(
        max_length=32,
        choices=AlertRule.SEVERITY_CHOICES,
        default=AlertRule.SEVERITY_MEDIUM,
    )
    status = models.CharField(
        max_length=32,
        choices=STATUS_CHOICES,
        default=STATUS_ACTIVE,
        db_index=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="resolved_alerts",
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Alert({self.audio_event.audio_id}: {self.severity} - {self.status})"
