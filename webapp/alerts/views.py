"""Alert views module for SonicSentinel AI."""

from typing import Any, Dict, List
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from accounts.permissions import (
    is_administrator,
    is_audio_reviewer,
    is_maintenance_operator,
    is_security_operator,
)
from alerts.models import Alert, AlertRule


def can_user_action_alert(user: Any, alert: Alert, action_name: str) -> bool:
    """Check if user has role-based permission to perform an action on an alert."""
    if not user or not user.is_authenticated:
        return False

    # Administrators and Audio Reviewers have full access to all alert actions
    if is_administrator(user) or is_audio_reviewer(user):
        return True

    category = alert.audio_event.category if alert.audio_event else ""

    # Security Operator handles security categories
    if is_security_operator(user):
        return category != "Machinery Fault"

    # Maintenance Operator handles machinery/equipment categories
    if is_maintenance_operator(user):
        return category in ["Machinery Fault", "Vehicle Horn", "Background Noise", "Animal Sound"]

    return False


def index(request: HttpRequest) -> HttpResponse:
    """Alerts management index endpoint rendering alerts dashboard for staff and API status for JSON."""
    if request.headers.get("Accept") == "application/json":
        return JsonResponse({"status": "ok", "app": "alerts"})

    alerts_qs = Alert.objects.select_related("audio_event", "alert_rule", "resolved_by").order_by("-created_at")[:50]
    context = {
        "alerts": alerts_qs,
        "active_count": Alert.objects.exclude(status__in=[Alert.STATUS_ACKNOWLEDGED, Alert.STATUS_DISMISSED]).count(),
    }
    return render(request, "alerts/index.html", context)


def active_alerts_api(request: HttpRequest) -> JsonResponse:
    """Return JSON payload of active unacknowledged/undismissed sound event alerts for polling."""
    active_alerts = (
        Alert.objects.select_related("audio_event", "alert_rule")
        .exclude(status__in=[Alert.STATUS_ACKNOWLEDGED, Alert.STATUS_DISMISSED])
        .order_by("-created_at")[:20]
    )

    alert_list: List[Dict[str, Any]] = []
    for alert in active_alerts:
        cat = alert.audio_event.category if alert.audio_event else "Unknown"
        audio_id = alert.audio_event.audio_id if alert.audio_event else ""

        diff = timezone.now() - alert.created_at
        if diff.total_seconds() < 60:
            time_ago = "Just now"
        elif diff.total_seconds() < 3600:
            time_ago = f"{int(diff.total_seconds() // 60)}m ago"
        elif diff.total_seconds() < 86400:
            time_ago = f"{int(diff.total_seconds() // 3600)}h ago"
        else:
            time_ago = f"{int(diff.total_seconds() // 86400)}d ago"

        alert_list.append(
            {
                "id": alert.id,
                "audio_id": audio_id,
                "category": cat,
                "severity": alert.severity,
                "status": alert.status,
                "created_at": alert.created_at.isoformat(),
                "created_at_formatted": alert.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "time_ago": time_ago,
                "event_detail_url": f"/audio/events/?search={audio_id}",
                "recommended_action": alert.alert_rule.recommended_action if alert.alert_rule else "",
                "can_acknowledge": can_user_action_alert(request.user, alert, "acknowledge"),
                "can_dismiss": can_user_action_alert(request.user, alert, "dismiss"),
                "can_escalate": can_user_action_alert(request.user, alert, "escalate") and alert.status != Alert.STATUS_ESCALATED,
            }
        )

    return JsonResponse(
        {
            "count": len(alert_list),
            "alerts": alert_list,
        }
    )


@require_POST
def acknowledge_alert(request: HttpRequest, alert_id: int) -> JsonResponse:
    """POST view to acknowledge an alert row."""
    alert = get_object_or_404(Alert, pk=alert_id)
    if not can_user_action_alert(request.user, alert, "acknowledge"):
        return JsonResponse({"error": "Permission denied for your role."}, status=403)

    alert.status = Alert.STATUS_ACKNOWLEDGED
    alert.resolved_at = timezone.now()
    if request.user.is_authenticated:
        alert.resolved_by = request.user
    alert.save(update_fields=["status", "resolved_at", "resolved_by"])

    try:
        from audit.services import log_action
        from audit.models import AuditLog

        log_action(
            user=request.user if request.user.is_authenticated else None,
            action=AuditLog.ACTION_ALERT,
            detail={
                "alert_id": alert.id,
                "action": "acknowledge",
                "audio_id": alert.audio_event.audio_id if alert.audio_event else "",
            },
        )
    except Exception:
        pass

    return JsonResponse(
        {
            "success": True,
            "alert_id": alert.id,
            "status": alert.status,
            "message": f"Alert {alert.id} acknowledged successfully.",
        }
    )


@require_POST
def dismiss_alert(request: HttpRequest, alert_id: int) -> JsonResponse:
    """POST view to dismiss an alert row."""
    alert = get_object_or_404(Alert, pk=alert_id)
    if not can_user_action_alert(request.user, alert, "dismiss"):
        return JsonResponse({"error": "Permission denied for your role."}, status=403)

    alert.status = Alert.STATUS_DISMISSED
    alert.resolved_at = timezone.now()
    if request.user.is_authenticated:
        alert.resolved_by = request.user
    alert.save(update_fields=["status", "resolved_at", "resolved_by"])

    try:
        from audit.services import log_action
        from audit.models import AuditLog

        log_action(
            user=request.user if request.user.is_authenticated else None,
            action=AuditLog.ACTION_ALERT,
            detail={
                "alert_id": alert.id,
                "action": "dismiss",
                "audio_id": alert.audio_event.audio_id if alert.audio_event else "",
            },
        )
    except Exception:
        pass

    return JsonResponse(
        {
            "success": True,
            "alert_id": alert.id,
            "status": alert.status,
            "message": f"Alert {alert.id} dismissed successfully.",
        }
    )


@require_POST
def escalate_alert(request: HttpRequest, alert_id: int) -> JsonResponse:
    """POST view to escalate an alert row to Critical severity."""
    alert = get_object_or_404(Alert, pk=alert_id)
    if not can_user_action_alert(request.user, alert, "escalate"):
        return JsonResponse({"error": "Permission denied for your role."}, status=403)

    alert.status = Alert.STATUS_ESCALATED
    alert.severity = AlertRule.SEVERITY_CRITICAL
    alert.save(update_fields=["status", "severity"])

    try:
        from audit.services import log_action
        from audit.models import AuditLog

        log_action(
            user=request.user if request.user.is_authenticated else None,
            action=AuditLog.ACTION_ALERT,
            detail={
                "alert_id": alert.id,
                "action": "escalate",
                "audio_id": alert.audio_event.audio_id if alert.audio_event else "",
            },
        )
    except Exception:
        pass

    return JsonResponse(
        {
            "success": True,
            "alert_id": alert.id,
            "status": alert.status,
            "severity": alert.severity,
            "message": f"Alert {alert.id} escalated to Critical severity.",
        }
    )
