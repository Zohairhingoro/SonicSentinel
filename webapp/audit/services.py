import logging
from datetime import date, datetime
from typing import Any, Dict, Optional
from django.contrib.auth import get_user_model
from django.core.paginator import Page, Paginator
from django.db.models import Q, QuerySet
from django.http import HttpRequest
from django.utils.dateparse import parse_date, parse_datetime
from django.utils.timezone import is_aware, make_aware

from audit.models import AuditLog

logger = logging.getLogger(__name__)
User = get_user_model()


def log_action(
    user: Optional[Any],
    action: str,
    detail: Optional[Dict[str, Any]] = None,
) -> AuditLog:
    """Create and persist an AuditLog row for a security-relevant event.

    This is the single entry point for all audit logging across every app.
    No other app should create AuditLog rows directly.
    """
    if detail is None:
        detail = {}

    try:
        entry = AuditLog.objects.create(
            user=user if user and user.is_authenticated else None,
            action=action,
            detail=detail,
        )
        return entry
    except Exception:
        logger.exception(
            "audit.log_action failed for action=%s user=%s",
            action,
            getattr(user, "pk", None),
        )
        raise


def _parse_date_to_datetime(val: Any, is_end: bool = False) -> Optional[datetime]:
    if isinstance(val, datetime):
        return val if is_aware(val) else make_aware(val)
    if isinstance(val, date):
        dt = datetime.combine(val, datetime.max.time() if is_end else datetime.min.time())
        return make_aware(dt)
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        dt = parse_datetime(val)
        if dt is None:
            parsed_d = parse_date(val)
            if parsed_d:
                dt = datetime.combine(parsed_d, datetime.max.time() if is_end else datetime.min.time())
        if dt:
            return dt if is_aware(dt) else make_aware(dt)
    return None


def list_audit_logs(filters: Optional[Dict[str, Any]] = None) -> QuerySet[AuditLog]:
    """Compose and return a filtered AuditLog QuerySet ordered most recent first (chronological reverse)."""
    queryset = AuditLog.objects.select_related("user").order_by("-timestamp")

    if not filters:
        return queryset

    # Action filter
    action_val = filters.get("action") or filters.get("actions")
    if action_val:
        if isinstance(action_val, (list, tuple, set)):
            queryset = queryset.filter(action__in=action_val)
        elif isinstance(action_val, str) and action_val.strip():
            queryset = queryset.filter(action=action_val.strip())

    # User filter
    user_val = filters.get("user") or filters.get("user_id") or filters.get("username")
    if user_val is not None:
        if hasattr(user_val, "id"):
            queryset = queryset.filter(user=user_val)
        elif isinstance(user_val, int):
            queryset = queryset.filter(user_id=user_val)
        elif isinstance(user_val, str) and user_val.strip():
            val_str = user_val.strip()
            if val_str.isdigit():
                queryset = queryset.filter(Q(user_id=int(val_str)) | Q(user__user_id=val_str))
            else:
                queryset = queryset.filter(Q(user__username__iexact=val_str) | Q(user__user_id=val_str))

    # Date range filters on timestamp
    start_date = (
        filters.get("start_date")
        or filters.get("date_from")
        or filters.get("from_date")
        or filters.get("timestamp_gte")
    )
    end_date = (
        filters.get("end_date")
        or filters.get("date_to")
        or filters.get("to_date")
        or filters.get("timestamp_lte")
    )

    if start_date is not None:
        parsed_start = _parse_date_to_datetime(start_date, is_end=False)
        if parsed_start:
            queryset = queryset.filter(timestamp__gte=parsed_start)

    if end_date is not None:
        parsed_end = _parse_date_to_datetime(end_date, is_end=True)
        if parsed_end:
            queryset = queryset.filter(timestamp__lte=parsed_end)

    return queryset


def get_paginated_audit_logs(
    request: HttpRequest,
    per_page: int = 20,
    filters: Optional[Dict[str, Any]] = None,
    page_param: str = "page",
) -> Page:
    """Return a paginated Page of AuditLog entries."""
    if filters is None:
        filters = {}
        if request.GET.get("action"):
            filters["action"] = request.GET.get("action")
        if request.GET.get("user_id") or request.GET.get("user"):
            filters["user_id"] = request.GET.get("user_id") or request.GET.get("user")
        if request.GET.get("date_from"):
            filters["date_from"] = request.GET.get("date_from")
        if request.GET.get("date_to"):
            filters["date_to"] = request.GET.get("date_to")

    queryset = list_audit_logs(filters)
    paginator = Paginator(queryset, per_page)
    page_number = request.GET.get(page_param, 1)
    return paginator.get_page(page_number)
