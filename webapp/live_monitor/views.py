from typing import Any, Dict, List

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils import timezone

from alerts.models import Alert
from audio_events.models import AudioEvent
from core.constants import get_audio_config
from predictions.models import ComparisonResult


def index(request: HttpRequest) -> HttpResponse:
    """Render live monitoring dashboard for security operators.

    Embeds FE-8a mic status widget, FE-8b capture controls, and live panels
    for current sound detection, both model results, agreement status, severity,
    and active alerts for the session.
    """
    audio_config = get_audio_config()
    window_duration_seconds = float(audio_config.get("window_duration_seconds", 3.0))

    return render(
        request,
        "live_monitor/index.html",
        {
            "window_duration_seconds": window_duration_seconds,
            "audio_config": audio_config,
        },
    )


def session_status_api(request: HttpRequest) -> JsonResponse:
    """Return JSON snapshot of the most recent audio event processed in this session.

    Polled every 5 s by the live monitor dashboard to update:
    - Current detected sound and quality
    - Python model prediction + top-3 confidence scores
    - GTM model prediction + top-3 confidence scores
    - Model agreement / match status
    - Event severity
    - Active unacknowledged/unescalated alerts (latest 5)
    """
    latest_event: AudioEvent | None = (
        AudioEvent.objects.select_related("uploaded_by")
        .prefetch_related("python_predictions", "gtm_predictions", "comparison_results", "alerts")
        .order_by("-uploaded_at")
        .first()
    )

    event_data: Dict[str, Any] = {}
    python_data: Dict[str, Any] = {}
    gtm_data: Dict[str, Any] = {}
    comparison_data: Dict[str, Any] = {}
    alert_data: List[Dict[str, Any]] = []

    if latest_event:
        event_data = {
            "audio_id": latest_event.audio_id,
            "category": latest_event.category,
            "quality": latest_event.quality,
            "severity": latest_event.severity,
            "status": latest_event.status,
            "uploaded_at": latest_event.uploaded_at.isoformat(),
        }

        py_pred = latest_event.python_predictions.order_by("-created_at").first()
        if py_pred:
            top3 = py_pred.top3
            python_data = {
                "predicted_class": py_pred.predicted_class,
                "confidence": py_pred.confidence,
                "confidence_scores": py_pred.confidence_scores,
                "model_version": py_pred.model_version,
                "top3": top3,
            }

        gtm_pred = latest_event.gtm_predictions.order_by("-created_at").first()
        if gtm_pred:
            gtm_top3 = gtm_pred.top3
            gtm_data = {
                "predicted_class": gtm_pred.predicted_class,
                "confidence_scores": gtm_pred.confidence_scores,
                "model_version": gtm_pred.model_version,
                "top3": gtm_top3,
            }

        comparison = latest_event.comparison_results.order_by("-created_at").first()
        if comparison:
            comparison_data = {
                "match_status": comparison.match_status,
                "confidence_difference": comparison.confidence_difference,
                "top_two_margin": comparison.top_two_margin,
                "is_uncertain": comparison.is_uncertain,
                "is_overlapping": comparison.is_overlapping,
            }

    active_alerts_qs = (
        Alert.objects.select_related("audio_event", "alert_rule")
        .exclude(status__in=[Alert.STATUS_ACKNOWLEDGED, Alert.STATUS_DISMISSED])
        .order_by("-created_at")[:5]
    )

    for alert in active_alerts_qs:
        diff = timezone.now() - alert.created_at
        if diff.total_seconds() < 60:
            time_ago = "Just now"
        elif diff.total_seconds() < 3600:
            time_ago = f"{int(diff.total_seconds() // 60)}m ago"
        elif diff.total_seconds() < 86400:
            time_ago = f"{int(diff.total_seconds() // 3600)}h ago"
        else:
            time_ago = f"{int(diff.total_seconds() // 86400)}d ago"

        alert_data.append(
            {
                "id": alert.id,
                "audio_id": alert.audio_event.audio_id if alert.audio_event else "",
                "category": alert.audio_event.category if alert.audio_event else "Unknown",
                "severity": alert.severity,
                "status": alert.status,
                "time_ago": time_ago,
                "recommended_action": alert.alert_rule.recommended_action if alert.alert_rule else "",
            }
        )

    return JsonResponse(
        {
            "event": event_data,
            "python_prediction": python_data,
            "gtm_prediction": gtm_data,
            "comparison": comparison_data,
            "active_alerts": alert_data,
            "active_alert_count": len(alert_data),
        }
    )
