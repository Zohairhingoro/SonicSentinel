"""Analytics and aggregation services for SonicSentinel dashboard and reporting.

This module provides aggregation functions for category frequency, critical event frequency,
confidence distribution, false positive/negative counts, model disagreement rate,
audio quality breakdown, and average alert response time.

Scale Performance Note (≥20,000 records target per AGENTS.md §4):
- get_confidence_distribution: Performs a full-table scan on PythonPrediction.confidence to bin rows into Case/When buckets. Slow at ≥20,000 records. Recommend short-TTL caching or pre-calculated summary tables.
- get_false_positive_negative_counts: Evaluates ReviewRecord joins with AudioEvent in memory. Slow at scale. Recommend DB-level CASE/WHEN aggregation or pre-computed review stats.
- get_average_alert_response_time: Calculates timestamp differences across resolved Alert records using F() expressions. Slow over large Alert tables. Recommend storing response duration upon alert resolution.
"""

from typing import Any, Dict
from datetime import timedelta
from django.utils import timezone
from django.db.models import Avg, Case, CharField, Count, F, Value, When
from django.db.models.functions import TruncDate

from alerts.models import Alert, AlertRule
from audio_events.models import AudioEvent
from predictions.models import ComparisonResult, PythonPrediction
from reviews.models import ReviewRecord


def get_category_frequency() -> Dict[str, int]:
    """Aggregate sound event counts by category.

    Scale Note (≥20,000 records): GROUP BY on indexed category column is fast,
    but caching with a short TTL (e.g. 5 mins) is recommended under high concurrency.

    Returns:
        Dict mapping category name to event count.
    """
    counts = (
        AudioEvent.objects.values("category")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    return {item["category"]: item["count"] for item in counts}


def get_critical_event_frequency() -> Dict[str, Any]:
    """Aggregate total count and category breakdown of critical sound events and alerts.

    Scale Note (≥20,000 records): Uses indexed lookup on Alert.severity and audio_event__category.

    Returns:
        Dict containing total_critical count and by_category dictionary.
    """
    critical_alerts = (
        Alert.objects.filter(severity=AlertRule.SEVERITY_CRITICAL)
        .values("audio_event__category")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    by_category = {
        item["audio_event__category"]: item["count"]
        for item in critical_alerts
        if item["audio_event__category"]
    }
    total_critical = sum(by_category.values())

    return {
        "total_critical": total_critical,
        "by_category": by_category,
    }


def get_confidence_distribution() -> Dict[str, int]:
    """Aggregate Python prediction confidence scores into discrete distribution buckets.

    Scale Note (≥20,000 records): Performs full-table scan on PythonPrediction.confidence.
    At ≥20,000 records, caching or scheduled pre-aggregation is recommended.

    Returns:
        Dict mapping bucket labels ('< 0.50', '0.50 - 0.69', etc.) to prediction count.
    """
    buckets = (
        PythonPrediction.objects.annotate(
            bucket=Case(
                When(confidence__lt=0.50, then=Value("< 0.50")),
                When(confidence__lt=0.70, then=Value("0.50 - 0.69")),
                When(confidence__lt=0.85, then=Value("0.70 - 0.84")),
                default=Value("0.85 - 1.00"),
                output_field=CharField(),
            )
        )
        .values("bucket")
        .annotate(count=Count("id"))
    )

    ordered_keys = ["< 0.50", "0.50 - 0.69", "0.70 - 0.84", "0.85 - 1.00"]
    raw_dict = {item["bucket"]: item["count"] for item in buckets}
    return {k: raw_dict.get(k, 0) for k in ordered_keys}


def get_false_positive_negative_counts() -> Dict[str, Any]:
    """Calculate false positive and false negative counts based on reviewer overrides vs original predictions.

    Scale Note (≥20,000 records): Uses select_related('audio_event') to prevent N+1 queries.
    At ≥20,000 records, in-memory iteration should be replaced with DB-level CASE/WHEN aggregation or pre-computed review stats.

    Returns:
        Dict containing total_reviews, total_overrides, false_positives, false_negatives, and override_rate.
    """
    reviews = ReviewRecord.objects.select_related("audio_event").all()

    total_reviews = len(reviews)
    total_overrides = 0
    false_positives = 0
    false_negatives = 0

    critical_categories = {
        "Gunshot",
        "Glass Breaking",
        "Panic Scream",
        "Aggression or Violent Conflict",
        "Person Asking for Help",
        "Alarm or Siren",
        "Machinery Fault",
    }

    for review in reviews:
        orig_cat = review.audio_event.category
        rev_cat = review.decision
        if orig_cat != rev_cat:
            total_overrides += 1
            if orig_cat in critical_categories and rev_cat not in critical_categories:
                false_positives += 1
            elif orig_cat not in critical_categories and rev_cat in critical_categories:
                false_negatives += 1

    override_rate = (total_overrides / total_reviews) if total_reviews > 0 else 0.0

    return {
        "total_reviews": total_reviews,
        "total_overrides": total_overrides,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "override_rate": round(override_rate, 4),
    }


def get_model_disagreement_rate() -> Dict[str, Any]:
    """Calculate model prediction disagreement rate between Python model and GTM model.

    Scale Note (≥20,000 records): Uses Django ORM aggregation on indexed match_status column.

    Returns:
        Dict containing total_comparisons, disagreements, disagreement_rate, and by_status dictionary.
    """
    status_counts = (
        ComparisonResult.objects.values("match_status")
        .annotate(count=Count("id"))
    )

    by_status = {item["match_status"]: item["count"] for item in status_counts}
    total_comparisons = sum(by_status.values())
    disagreements = by_status.get(ComparisonResult.MATCH_DISAGREEMENT, 0)

    disagreement_rate = (disagreements / total_comparisons) if total_comparisons > 0 else 0.0

    return {
        "total_comparisons": total_comparisons,
        "disagreements": disagreements,
        "disagreement_rate": round(disagreement_rate, 4),
        "by_status": by_status,
    }


def get_audio_quality_breakdown() -> Dict[str, int]:
    """Aggregate audio recordings count broken down by quality classification.

    Scale Note (≥20,000 records): Uses Django ORM aggregation on indexed quality column.

    Returns:
        Dict mapping quality levels ('Good', 'Acceptable', 'Poor', 'Unusable') to event count.
    """
    counts = (
        AudioEvent.objects.values("quality")
        .annotate(count=Count("id"))
    )

    ordered_qualities = ["Good", "Acceptable", "Poor", "Unusable"]
    raw_dict = {item["quality"]: item["count"] for item in counts}
    return {q: raw_dict.get(q, 0) for q in ordered_qualities}


def get_average_alert_response_time() -> Dict[str, Any]:
    """Calculate average resolution response time for generated alerts.

    Scale Note (≥20,000 records): Evaluates F('resolved_at') - F('created_at') over resolved alerts.
    At scale, pre-storing duration or caching response time stats is recommended.

    Returns:
        Dict containing avg_response_seconds, avg_response_minutes, resolved_count, and active_count.
    """
    resolved_alerts = Alert.objects.filter(resolved_at__isnull=False)
    resolved_count = resolved_alerts.count()
    active_count = Alert.objects.filter(resolved_at__isnull=True).count()

    if resolved_count == 0:
        return {
            "avg_response_seconds": 0.0,
            "avg_response_minutes": 0.0,
            "resolved_count": 0,
            "active_count": active_count,
        }

    res = resolved_alerts.aggregate(
        avg_time=Avg(F("resolved_at") - F("created_at"))
    )
    avg_timedelta = res.get("avg_time")

    if avg_timedelta is not None:
        if isinstance(avg_timedelta, (int, float)):
            avg_seconds = float(avg_timedelta)
        else:
            avg_seconds = float(avg_timedelta.total_seconds())
    else:
        avg_seconds = 0.0

    return {
        "avg_response_seconds": round(avg_seconds, 2),
        "avg_response_minutes": round(avg_seconds / 60.0, 2),
        "resolved_count": resolved_count,
        "active_count": active_count,
    }


def get_total_audio_events() -> int:
    """Count total ingested audio events.

    Returns:
        Total AudioEvent record count.
    """
    return AudioEvent.objects.count()


def get_critical_alerts_count() -> int:
    """Count total critical severity alerts raised.

    Returns:
        Count of Alert records with severity SEVERITY_CRITICAL.
    """
    return Alert.objects.filter(severity=AlertRule.SEVERITY_CRITICAL).count()


def get_average_confidence() -> float:
    """Calculate overall average Python prediction confidence score.

    Returns:
        Average confidence score rounded to 4 decimal places, or 0.0 if no predictions exist.
    """
    result = PythonPrediction.objects.aggregate(avg_conf=Avg("confidence"))
    avg_val = result.get("avg_conf")
    return round(float(avg_val), 4) if avg_val is not None else 0.0


def get_model_disagreement_count() -> int:
    """Count total prediction disagreements between Python model and GTM model.

    Returns:
        Count of ComparisonResult records with match_status MATCH_DISAGREEMENT.
    """
    return ComparisonResult.objects.filter(
        match_status=ComparisonResult.MATCH_DISAGREEMENT
    ).count()


def get_poor_quality_count() -> int:
    """Count total audio recordings classified with Poor or Unusable quality.

    Returns:
        Count of AudioEvent records with quality in ('Poor', 'Unusable').
    """
    return AudioEvent.objects.filter(
        quality__in=[AudioEvent.QUALITY_POOR, AudioEvent.QUALITY_UNUSABLE]
    ).count()


def get_admin_summary_stats() -> Dict[str, Any]:
    """Return consolidated summary stat card metrics for the admin dashboard.

    Pulls each metric from dedicated analytics aggregation functions in this module.

    Returns:
        Dict containing total_audio_events, critical_alerts_count, average_confidence,
        model_disagreement_count, and poor_quality_count.
    """
    return {
        "total_audio_events": get_total_audio_events(),
        "critical_alerts_count": get_critical_alerts_count(),
        "average_confidence": get_average_confidence(),
        "model_disagreement_count": get_model_disagreement_count(),
        "poor_quality_count": get_poor_quality_count(),
    }


def get_detection_trends(days: int = 30) -> Dict[str, Any]:
    """Aggregate sound event counts by category over the last N days in daily buckets.

    Scale Note (≥20,000 records): GROUP BY on uploaded_at date and category.
    Uses indexed uploaded_at and category columns.

    Args:
        days: Number of days to look back (default 30).

    Returns:
        Dict containing dates list, categories list, and datasets dict mapping each
        category to a list of daily event counts.
    """
    now = timezone.now()
    start_date = (now - timedelta(days=days - 1)).date()

    date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    date_map = {d: i for i, d in enumerate(date_list)}

    trend_records = (
        AudioEvent.objects.filter(uploaded_at__date__gte=start_date)
        .annotate(day=TruncDate("uploaded_at"))
        .values("day", "category")
        .annotate(count=Count("id"))
        .order_by("day")
    )

    all_categories = [
        "Machinery Fault",
        "Glass Breaking",
        "Alarm or Siren",
        "Vehicle Horn",
        "Animal Sound",
        "Gunshot",
        "Panic Scream",
        "Aggression or Violent Conflict",
        "Person Asking for Help",
        "Background Noise",
    ]

    datasets: Dict[str, list] = {cat: [0] * days for cat in all_categories}

    for rec in trend_records:
        day_obj = rec["day"]
        if day_obj:
            day_str = day_obj.strftime("%Y-%m-%d") if hasattr(day_obj, "strftime") else str(day_obj)
            cat = rec["category"]
            if day_str in date_map and cat:
                idx = date_map[day_str]
                if cat not in datasets:
                    datasets[cat] = [0] * days
                datasets[cat][idx] = rec["count"]

    return {
        "dates": date_list,
        "categories": list(datasets.keys()),
        "datasets": datasets,
    }


def get_all_analytics_summary() -> Dict[str, Any]:
    """Return consolidated analytics dashboard metrics payload in a single call."""
    return {
        "category_frequency": get_category_frequency(),
        "critical_event_frequency": get_critical_event_frequency(),
        "confidence_distribution": get_confidence_distribution(),
        "false_positive_negative": get_false_positive_negative_counts(),
        "model_disagreement": get_model_disagreement_rate(),
        "audio_quality_breakdown": get_audio_quality_breakdown(),
        "alert_response_time": get_average_alert_response_time(),
        "admin_summary_stats": get_admin_summary_stats(),
        "detection_trends": get_detection_trends(),
    }


def get_user_dashboard_summary(user: Any) -> Dict[str, Any]:
    """Compile normal user dashboard metrics and collections.

    Returns:
        Dict containing recent_uploads (last 10), current_detections,
        critical_events, quality_warnings, review_records, latest_comparison,
        and timeline_page object for FE-7 reuse.
    """
    from django.db.models import Q
    from django.core.paginator import Paginator

    if getattr(user, "is_authenticated", False):
        user_events_qs = AudioEvent.objects.filter(uploaded_by=user)
        if not user_events_qs.exists():
            user_events_qs = AudioEvent.objects.all()
    else:
        user_events_qs = AudioEvent.objects.all()

    user_events_qs = user_events_qs.select_related("uploaded_by").prefetch_related(
        "comparison_results", "python_predictions", "gtm_predictions", "review_records"
    ).order_by("-uploaded_at")

    recent_uploads = list(user_events_qs[:10])

    current_detections = list(
        user_events_qs.filter(
            status__in=[
                AudioEvent.STATUS_UPLOADED,
                AudioEvent.STATUS_CLASSIFIED,
                AudioEvent.STATUS_ALERT_GENERATED,
                AudioEvent.STATUS_MANUAL_REVIEW,
                AudioEvent.STATUS_UNCERTAIN,
            ]
        )[:8]
    )

    critical_events = list(
        user_events_qs.filter(severity__in=["Critical", "High"])[:10]
    )

    quality_warnings = list(
        user_events_qs.filter(quality__in=[AudioEvent.QUALITY_POOR, AudioEvent.QUALITY_UNUSABLE])[:10]
    )

    if getattr(user, "is_authenticated", False):
        review_records = list(
            ReviewRecord.objects.filter(
                Q(reviewer=user) | Q(audio_event__uploaded_by=user)
            )
            .select_related("audio_event", "reviewer")
            .distinct()
            .order_by("-created_at")[:10]
        )
    else:
        review_records = list(
            ReviewRecord.objects.select_related("audio_event", "reviewer").order_by("-created_at")[:10]
        )

    first_event = recent_uploads[0] if recent_uploads else None
    latest_comparison = first_event.comparison_results.first() if first_event else None

    paginator = Paginator(user_events_qs, 10)
    timeline_page = paginator.get_page(1)

    return {
        "recent_uploads": recent_uploads,
        "current_detections": current_detections,
        "critical_events": critical_events,
        "quality_warnings": quality_warnings,
        "review_records": review_records,
        "latest_comparison": latest_comparison,
        "timeline_page": timeline_page,
    }

