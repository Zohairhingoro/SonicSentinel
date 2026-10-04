"""Django signals for the predictions app to automatically trigger comparative analysis."""

from django.db.models.signals import post_save
from django.dispatch import receiver

from predictions.comparison import create_comparison_if_ready
from predictions.models import GTMPrediction, PythonPrediction


@receiver(post_save, sender=PythonPrediction)
def trigger_comparison_on_python_prediction(sender, instance, created, **kwargs):
    """Automatically run comparison check when a PythonPrediction is created or updated."""
    if instance.audio_event:
        create_comparison_if_ready(instance.audio_event)


@receiver(post_save, sender=GTMPrediction)
def trigger_comparison_on_gtm_prediction(sender, instance, created, **kwargs):
    """Automatically run comparison check when a GTMPrediction is created or updated."""
    if instance.audio_event:
        create_comparison_if_ready(instance.audio_event)
