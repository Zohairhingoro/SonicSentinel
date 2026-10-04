import json
from datetime import datetime
from typing import Union

from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.views import View

from accounts.permissions import is_administrator, require_administrator
from audio_events.forms import EventFilterForm
from audio_events.services import get_paginated_audio_events, list_audio_events
from dashboard.analytics import (
    get_admin_summary_stats,
    get_all_analytics_summary,
    get_user_dashboard_summary,
)
from dashboard.export import generate_audio_events_export_response


def index(request: HttpRequest) -> HttpResponse:
    """Render main dashboard overview with chronological sound event timeline."""
    events_page = get_paginated_audio_events(request)
    return render(request, "home.html", {"events_page": events_page})


def user_dashboard(request: HttpRequest) -> HttpResponse:
    """Render normal-user dashboard showing user-specific uploads, detections, critical events, quality warnings, and review records."""
    summary = get_user_dashboard_summary(request.user)
    context = {
        **summary,
        "events_page": summary["timeline_page"],
        "dashboard_type": "user",
    }
    return render(request, "dashboard/user.html", context)


@require_administrator
def admin_dashboard(request: HttpRequest) -> HttpResponse:
    """Render administrator dashboard overview with summary stat cards and detection trends chart."""
    events_page = get_paginated_audio_events(request)
    summary_stats = get_admin_summary_stats()
    analytics_data = get_all_analytics_summary()
    detection_trends = analytics_data.get("detection_trends", {})
    return render(
        request,
        "dashboard/admin.html",
        {
            "events_page": events_page,
            "summary_stats": summary_stats,
            "analytics": analytics_data,
            "detection_trends_json": json.dumps(detection_trends),
            "dashboard_type": "admin",
            "is_admin": True,
        },
    )


def analytics_dashboard(request: HttpRequest) -> HttpResponse:
    """Render comprehensive analytics dashboard with summary metrics and export controls."""
    is_admin = is_administrator(request.user)
    form = EventFilterForm(request.GET or None)

    filters: dict = {}
    if form.is_bound and form.is_valid():
        filters = form.to_query_filters()
        if not is_admin:
            filters.pop("user", None)
    elif not form.is_bound:
        filters = {}

    events_page = get_paginated_audio_events(request, per_page=15, filters=filters)
    analytics_data = get_all_analytics_summary()

    return render(
        request,
        "dashboard/analytics.html",
        {
            "form": form,
            "events_page": events_page,
            "analytics": analytics_data,
            "is_admin": is_admin,
            "active_filter_count": sum(1 for v in filters.values() if v not in ("", None)),
        },
    )


class AdminAnalyticsView(LoginRequiredMixin, UserPassesTestMixin, View):
    """Admin-only dedicated analytics page serving six Chart.js charts.

    Passes serialised chart data (JSON) into the template so the client-side
    Chart.js instances can be initialised without additional XHR calls.
    """

    def test_func(self) -> bool:
        return is_administrator(self.request.user)

    def handle_no_permission(self) -> HttpResponseRedirect:
        user = getattr(self.request, "user", None)
        if user and user.is_authenticated:
            raise PermissionDenied("Analytics charts are restricted to administrators.")
        return super().handle_no_permission()

    def get(self, request: HttpRequest) -> HttpResponse:
        data = get_all_analytics_summary()

        category_freq = data["category_frequency"]
        confidence_dist = data["confidence_distribution"]
        fp_fn = data["false_positive_negative"]
        model_dis = data["model_disagreement"]
        quality = data["audio_quality_breakdown"]
        alert_rt = data["alert_response_time"]
        summary_stats = data["admin_summary_stats"]

        chart_data = {
            "category_freq": {
                "labels": list(category_freq.keys()),
                "values": list(category_freq.values()),
            },
            "confidence_dist": {
                "labels": list(confidence_dist.keys()),
                "values": list(confidence_dist.values()),
            },
            "fp_fn": {
                "labels": ["False Positives", "False Negatives"],
                "values": [fp_fn["false_positives"], fp_fn["false_negatives"]],
                "total_reviews": fp_fn["total_reviews"],
                "total_overrides": fp_fn["total_overrides"],
                "override_rate": fp_fn["override_rate"],
            },
            "model_disagreement": {
                "labels": list(model_dis["by_status"].keys()),
                "values": list(model_dis["by_status"].values()),
                "disagreement_rate": model_dis["disagreement_rate"],
                "total_comparisons": model_dis["total_comparisons"],
            },
            "audio_quality": {
                "labels": list(quality.keys()),
                "values": list(quality.values()),
            },
            "alert_response": {
                "avg_minutes": alert_rt["avg_response_minutes"],
                "avg_seconds": alert_rt["avg_response_seconds"],
                "resolved": alert_rt["resolved_count"],
                "active": alert_rt["active_count"],
            },
        }

        return render(
            request,
            "dashboard/analytics_charts.html",
            {
                "chart_data_json": json.dumps(chart_data),
                "summary_stats": summary_stats,
                "fp_fn": fp_fn,
                "model_dis": model_dis,
                "alert_rt": alert_rt,
            },
        )


class ExportEventsView(LoginRequiredMixin, UserPassesTestMixin, View):
    """View allowing administrators to export filtered AudioEvent records as CSV or Excel.

    Reuses EventFilterForm / list_audio_events (FE-20) to filter the queryset
    based on any active query parameters (search, category, severity, quality,
    review_status, start_date, end_date, user).
    """

    def test_func(self) -> bool:
        return is_administrator(self.request.user)

    def handle_no_permission(self) -> HttpResponseRedirect:
        user = getattr(self.request, "user", None)
        if user and user.is_authenticated:
            raise PermissionDenied("Exporting event analysis report is restricted to administrators.")
        return super().handle_no_permission()

    def get(self, request: HttpRequest) -> HttpResponse:

        form = EventFilterForm(request.GET or None)
        filters: dict = {}
        if form.is_bound and form.is_valid():
            filters = form.to_query_filters()

        queryset = list_audio_events(filters)

        export_format = request.GET.get("format", "csv").lower().strip()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "xlsx" if export_format in ["excel", "xlsx"] else "csv"
        filename = f"sonic_sentinel_analysis_report_{timestamp_str}.{ext}"

        return generate_audio_events_export_response(queryset, export_format=export_format, filename=filename)
