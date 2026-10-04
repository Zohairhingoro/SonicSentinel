from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from django.conf import settings
from django.db import models

if TYPE_CHECKING:
    from django.db.models import QuerySet
    from predictions.models import PythonPrediction, GTMPrediction, ComparisonResult
    from alerts.models import Alert
    from reviews.models import ReviewRecord


def audio_event_upload_to(instance: "AudioEvent", filename: str) -> str:
    audio_id = instance.audio_id or "pending"
    return f"audio/{audio_id}/{filename}"


class AudioEvent(models.Model):
    """Represents an ingested audio recording file and its processing state."""

    STATUS_UPLOADED = "Uploaded"
    STATUS_CLASSIFIED = "Classified"
    STATUS_UNCERTAIN = "Uncertain"
    STATUS_ALERT_GENERATED = "Alert Generated"
    STATUS_MANUAL_REVIEW = "Manual Review"
    STATUS_REVIEWED = "Reviewed"
    STATUS_CLOSED = "Closed"

    STATUS_CHOICES = [
        (STATUS_UPLOADED, "Uploaded"),
        (STATUS_CLASSIFIED, "Classified"),
        (STATUS_UNCERTAIN, "Uncertain"),
        (STATUS_ALERT_GENERATED, "Alert Generated"),
        (STATUS_MANUAL_REVIEW, "Manual Review"),
        (STATUS_REVIEWED, "Reviewed"),
        (STATUS_CLOSED, "Closed"),
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

    objects = models.Manager()

    if TYPE_CHECKING:
        python_predictions: QuerySet[PythonPrediction]
        gtm_predictions: QuerySet[GTMPrediction]
        comparison_results: QuerySet[ComparisonResult]
        alerts: QuerySet[Alert]
        review_records: QuerySet[ReviewRecord]

    audio_id = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        editable=False,
        help_text="Unique indexed identifier for the audio event",
    )
    filename = models.CharField(max_length=255)
    format = models.CharField(max_length=32)
    duration = models.FloatField(help_text="Duration in seconds")
    sampling_rate = models.IntegerField(help_text="Sampling rate in Hz")
    channels = models.IntegerField(default=1)
    bit_depth = models.IntegerField(null=True, blank=True)
    file_size = models.PositiveBigIntegerField(help_text="File size in bytes")
    audio_file = models.FileField(
        upload_to=audio_event_upload_to,
        max_length=512,
        null=True,
        blank=True,
        help_text="Uploaded raw audio file stored under MEDIA_ROOT",
    )
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audio_events",
    )
    uploaded_at = models.DateTimeField(auto_now_add=True, db_index=True)
    status = models.CharField(
        max_length=32,
        choices=STATUS_CHOICES,
        default=STATUS_UPLOADED,
        db_index=True,
    )
    quality = models.CharField(
        max_length=32,
        choices=QUALITY_CHOICES,
        default=QUALITY_GOOD,
        db_index=True,
    )
    category = models.CharField(
        max_length=64,
        default="Background Noise",
        db_index=True,
        help_text="Primary detected sound category",
    )
    severity = models.CharField(
        max_length=32,
        default="Informational",
        db_index=True,
        help_text="Severity level derived from category per alert rules",
    )
    storage_path = models.CharField(max_length=512)
    hash = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        help_text="Unique SHA-256 hash of audio file contents for duplicate detection",
    )
    audio_fingerprint = models.CharField(
        max_length=128,
        blank=True,
        default="",
        db_index=True,
        help_text="Normalized amplitude envelope fingerprint for near-duplicate detection",
    )
    suspected_duplicate_of_id = models.CharField(
        max_length=64,
        blank=True,
        default="",
        db_index=True,
        help_text="Audio ID of suspected near-duplicate audio event flagged for review",
    )
    suspected_duplicate = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="suspected_duplicates",
        help_text="Reference to suspected near-duplicate AudioEvent",
    )


    class Meta:
        ordering = ["-uploaded_at"]
        indexes = [
            models.Index(fields=["category"], name="audio_evt_category_idx"),
            models.Index(fields=["severity"], name="audio_evt_severity_idx"),
            models.Index(fields=["quality"], name="audio_evt_quality_idx"),
            models.Index(fields=["uploaded_at"], name="audio_evt_uploaded_at_idx"),
        ]

    def save(self, *args, **kwargs) -> None:
        if not self.audio_id:
            self.audio_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"  # type: ignore[assignment]
        from core.constants import get_severity_for_category
        computed_sev = get_severity_for_category(str(self.category))
        if computed_sev:
            self.severity = computed_sev  # type: ignore[assignment]
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.audio_id} - {self.filename} ({self.status})"


class Segment(models.Model):
    """Represents a fixed-duration temporal window segment sliced from an AudioEvent."""

    objects = models.Manager()

    audio_event = models.ForeignKey(
        AudioEvent,
        on_delete=models.CASCADE,
        related_name="segments",
    )
    segment_index = models.IntegerField(default=0)
    start_timestamp = models.FloatField(help_text="Start offset in seconds")
    end_timestamp = models.FloatField(help_text="End offset in seconds")

    class Meta:
        ordering = ["segment_index"]

    def __str__(self) -> str:
        return f"{self.audio_event.audio_id} Segment #{self.segment_index} ({self.start_timestamp}s-{self.end_timestamp}s)"
