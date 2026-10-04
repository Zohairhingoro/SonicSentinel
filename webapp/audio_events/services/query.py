from datetime import date, datetime
from typing import Any, Dict, Optional, Union
from django.core.paginator import Page, Paginator
from django.db.models import Q, QuerySet
from django.http import HttpRequest
from django.utils.dateparse import parse_date, parse_datetime
from django.utils.timezone import get_current_timezone, is_aware, make_aware

from audio_events.models import AudioEvent


def _parse_date_to_datetime(val: Any, is_end: bool = False) -> Optional[datetime]:
    if isinstance(val, datetime):
        return val if is_aware(val) else make_aware(val)
    if isinstance(val, date):
        dt: Optional[datetime] = datetime.combine(val, datetime.max.time() if is_end else datetime.min.time())
        return make_aware(dt)  # type: ignore[arg-type]
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


def list_audio_events(filters: Optional[Dict[str, Any]] = None) -> QuerySet[AudioEvent]:
    """Compose and return a filtered AudioEvent QuerySet based on criteria dictionary.

    Supports filtering by category, severity, quality, review_status, user, date range
    (uploaded_at), and confidence range (joining PythonPrediction).
    """
    queryset = AudioEvent.objects.select_related("uploaded_by").order_by("-uploaded_at")

    if not filters:
        return queryset

    # Category filter
    category_val = filters.get("category") or filters.get("categories")
    if category_val:
        if isinstance(category_val, (list, tuple, set)):
            queryset = queryset.filter(category__in=category_val)
        elif isinstance(category_val, str) and category_val.strip():
            queryset = queryset.filter(category=category_val.strip())

    # Severity filter
    severity_val = filters.get("severity") or filters.get("severities")
    if severity_val:
        if isinstance(severity_val, (list, tuple, set)):
            severities = [s.strip().capitalize() for s in severity_val if isinstance(s, str)]
            queryset = queryset.filter(severity__in=severities)
        elif isinstance(severity_val, str) and severity_val.strip():
            target_sev = severity_val.strip().capitalize()
            queryset = queryset.filter(severity__iexact=target_sev)

    # Quality filter
    quality_val = filters.get("quality") or filters.get("qualities")
    if quality_val:
        if isinstance(quality_val, (list, tuple, set)):
            queryset = queryset.filter(quality__in=quality_val)
        elif isinstance(quality_val, str) and quality_val.strip():
            queryset = queryset.filter(quality=quality_val.strip())

    # Review status filter
    review_status = filters.get("review_status") or filters.get("review_state")
    if review_status is None and "status" in filters:
        review_status = filters.get("status")

    if review_status is not None:
        if isinstance(review_status, bool):
            if review_status:
                queryset = queryset.filter(
                    Q(status=AudioEvent.STATUS_REVIEWED) | Q(review_records__isnull=False)
                ).distinct()
            else:
                queryset = queryset.filter(review_records__isnull=True).exclude(
                    status=AudioEvent.STATUS_REVIEWED
                )
        elif isinstance(review_status, str) and review_status.strip():
            val_str = review_status.strip()
            val_lower = val_str.lower()
            if val_lower in ["reviewed", "true"]:
                queryset = queryset.filter(
                    Q(status=AudioEvent.STATUS_REVIEWED) | Q(review_records__isnull=False)
                ).distinct()
            elif val_lower in ["unreviewed", "pending", "false"]:
                queryset = queryset.filter(review_records__isnull=True).exclude(
                    status=AudioEvent.STATUS_REVIEWED
                )
            else:
                queryset = queryset.filter(status__iexact=val_str)

    # User filter
    user_val = (
        filters.get("user")
        or filters.get("user_id")
        or filters.get("uploaded_by")
        or filters.get("uploaded_by_id")
    )
    if user_val is not None:
        if hasattr(user_val, "id"):
            queryset = queryset.filter(uploaded_by=user_val)
        elif isinstance(user_val, int):
            queryset = queryset.filter(uploaded_by_id=user_val)
        elif isinstance(user_val, str) and user_val.strip():
            val_str = user_val.strip()
            if val_str.isdigit():
                queryset = queryset.filter(
                    Q(uploaded_by_id=int(val_str)) | Q(uploaded_by__user_id=val_str)
                )
            else:
                queryset = queryset.filter(
                    Q(uploaded_by__username__iexact=val_str) | Q(uploaded_by__user_id=val_str)
                )

    # Date range filter on uploaded_at
    start_date = (
        filters.get("start_date")
        or filters.get("date_from")
        or filters.get("from_date")
        or filters.get("uploaded_at_gte")
        or filters.get("uploaded_at_min")
    )
    end_date = (
        filters.get("end_date")
        or filters.get("date_to")
        or filters.get("to_date")
        or filters.get("uploaded_at_lte")
        or filters.get("uploaded_at_max")
    )
    if "date_range" in filters and isinstance(filters["date_range"], (list, tuple)) and len(filters["date_range"]) == 2:
        start_date = filters["date_range"][0]
        end_date = filters["date_range"][1]

    if start_date is not None:
        parsed_start = _parse_date_to_datetime(start_date, is_end=False)
        if parsed_start:
            queryset = queryset.filter(uploaded_at__gte=parsed_start)

    if end_date is not None:
        parsed_end = _parse_date_to_datetime(end_date, is_end=True)
        if parsed_end:
            queryset = queryset.filter(uploaded_at__lte=parsed_end)

    # Confidence range filter joining PythonPrediction
    min_conf = (
        filters.get("min_confidence")
        or filters.get("confidence_min")
        or filters.get("confidence_gte")
        or filters.get("min_conf")
    )
    max_conf = (
        filters.get("max_confidence")
        or filters.get("confidence_max")
        or filters.get("confidence_lte")
        or filters.get("max_conf")
    )
    if "confidence_range" in filters and isinstance(filters["confidence_range"], (list, tuple)) and len(filters["confidence_range"]) == 2:
        min_conf = filters["confidence_range"][0]
        max_conf = filters["confidence_range"][1]

    joined_confidence = False
    if min_conf is not None:
        try:
            queryset = queryset.filter(python_predictions__confidence__gte=float(min_conf))
            joined_confidence = True
        except (ValueError, TypeError):
            pass

    if max_conf is not None:
        try:
            queryset = queryset.filter(python_predictions__confidence__lte=float(max_conf))
            joined_confidence = True
        except (ValueError, TypeError):
            pass

    if joined_confidence:
        queryset = queryset.distinct()

    # General search query
    if filters.get("search"):
        search_query = str(filters["search"]).strip()
        if search_query:
            queryset = queryset.filter(
                Q(filename__icontains=search_query) | Q(audio_id__icontains=search_query)
            )

    return queryset


def get_audio_events_queryset(filters: Optional[Dict[str, Any]] = None) -> QuerySet[AudioEvent]:
    """Retrieve AudioEvent queryset ordered chronologically with optional filtering."""
    return list_audio_events(filters)


def _build_filters_from_get(get: Any) -> Dict[str, Any]:
    """Extract and normalise every supported filter key from a GET QueryDict.

    Called internally by ``get_paginated_audio_events`` when no explicit
    filters dict is supplied.  All values are passed through as-is; type
    coercion and validation are the caller's responsibility (the view uses
    EventFilterForm.to_query_filters() which already does this).
    """
    mapping = {
        "search": "search",
        "category": "category",
        "severity": "severity",
        "quality": "quality",
        "review_status": "review_status",
        "status": "review_status",          # legacy alias still supported
        "start_date": "start_date",
        "end_date": "end_date",
        "min_confidence": "min_confidence",
        "max_confidence": "max_confidence",
        "user": "user",
    }
    filters: Dict[str, Any] = {}
    for param, key in mapping.items():
        val = get.get(param)
        if val is not None and str(val).strip():
            filters[key] = str(val).strip()
    return filters


def get_paginated_audio_events(
    request: HttpRequest,
    per_page: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    page_param: str = "page",
) -> Page:
    """Return a paginated Page of AudioEvents for reusable timeline rendering.

    When ``filters`` is not supplied the function reads all supported filter
    parameters directly from ``request.GET`` so callers do not need to repeat
    the mapping.  Pass an explicit ``filters`` dict (e.g. from
    ``EventFilterForm.to_query_filters()``) to take priority over GET params.
    """
    if filters is None:
        filters = _build_filters_from_get(request.GET)

    queryset = list_audio_events(filters)
    paginator = Paginator(queryset, per_page)
    page_number = request.GET.get(page_param, 1)
    return paginator.get_page(page_number)
