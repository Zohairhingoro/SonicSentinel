from django.contrib.auth import get_user_model
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from accounts.permissions import require_administrator
from audit.models import AuditLog
from audit.services import get_paginated_audit_logs

User = get_user_model()


@require_administrator
def audit_log_list(request: HttpRequest) -> HttpResponse:
    """Admin-only view listing paginated and filterable AuditLog entries."""
    filters = {}
    action = request.GET.get("action", "").strip()
    user_id = request.GET.get("user_id", "").strip()
    date_from = request.GET.get("date_from", "").strip()
    date_to = request.GET.get("date_to", "").strip()

    if action:
        filters["action"] = action
    if user_id:
        filters["user_id"] = user_id
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    logs_page = get_paginated_audit_logs(request, per_page=20, filters=filters)
    users = User.objects.all().order_by("username")
    action_choices = AuditLog.ACTION_CHOICES

    context = {
        "logs_page": logs_page,
        "action_choices": action_choices,
        "users": users,
        "selected_action": action,
        "selected_user_id": user_id,
        "selected_date_from": date_from,
        "selected_date_to": date_to,
    }
    return render(request, "audit/log_list.html", context)


@require_administrator
def index(request: HttpRequest) -> HttpResponse:
    """Admin-only alias for the audit log list."""
    return audit_log_list(request)
