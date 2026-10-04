from django.conf import settings
from django.db import models
from audio_events.models import AudioEvent


class ReviewRecord(models.Model):
    """Stores manual audio event review decisions and corrections without mutating original model predictions."""

    objects = models.Manager()

    audio_event = models.ForeignKey(
        AudioEvent,
        on_delete=models.CASCADE,
        related_name="review_records",
    )
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="review_records",
    )
    decision = models.CharField(
        max_length=64,
        db_index=True,
        help_text="Confirmed or corrected sound event category",
    )
    comment = models.TextField(
        blank=True,
        default="",
        help_text="Reviewer comments explaining review decision",
    )
    recommended_action = models.TextField(
        blank=True,
        default="",
        help_text="Recommended action following manual review",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        reviewer_name = self.reviewer.username if self.reviewer else "Anonymous"
        return f"ReviewRecord({self.audio_event.audio_id}: {self.decision} by {reviewer_name})"
