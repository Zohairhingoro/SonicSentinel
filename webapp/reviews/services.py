"""Domain services for manual audio event reviews and review queue processing."""

from typing import Any, List
from audio_events.models import AudioEvent
from predictions.models import ComparisonResult
from .models import ReviewRecord


def get_routing_reason(audio_event: AudioEvent) -> str:
    """Derives the reason routed for manual review from stored AudioEvent and Comparison metrics.

    Evaluates stored quality metrics, comparison match statuses, prediction confidence scores,
    and overlapping flags to return reasons: 'low confidence', 'poor quality',
    'model disagreement', or 'overlapping sounds'.
    """
    reasons: List[str] = []

    # 1. Check quality on AudioEvent
    if audio_event.quality in [AudioEvent.QUALITY_POOR, AudioEvent.QUALITY_UNUSABLE]:
        reasons.append("poor quality")

    # Access prefetched comparison results
    comparisons = list(audio_event.comparison_results.all())
    comp = comparisons[0] if comparisons else None

    # Access prefetched predictions
    python_preds = list(audio_event.python_predictions.all())
    py_pred = python_preds[0] if python_preds else None

    gtm_preds = list(audio_event.gtm_predictions.all())
    gtm_pred = gtm_preds[0] if gtm_preds else None

    if comp:
        # 2. Check overlapping sounds flag
        if comp.is_overlapping:
            reasons.append("overlapping sounds")

        # 3. Check model disagreement
        if (
            comp.match_status == ComparisonResult.MATCH_DISAGREEMENT
            or (py_pred and gtm_pred and py_pred.predicted_class != gtm_pred.predicted_class)
        ):
            reasons.append("model disagreement")

        # 4. Check low confidence / weak match / uncertain
        if (
            comp.is_uncertain
            or comp.match_status == ComparisonResult.MATCH_WEAK
            or (comp.top_two_margin is not None and comp.top_two_margin < 0.15)
            or (py_pred and py_pred.confidence is not None and py_pred.confidence < 0.60)
        ):
            if "model disagreement" not in reasons:
                reasons.append("low confidence")
    else:
        # Fallback checks if ComparisonResult object is missing
        if py_pred and gtm_pred and py_pred.predicted_class != gtm_pred.predicted_class:
            reasons.append("model disagreement")
        if py_pred and py_pred.confidence is not None and py_pred.confidence < 0.60:
            reasons.append("low confidence")

    # Check stored ReviewRecord comments if present
    reviews = list(audio_event.review_records.all())
    if not reasons and reviews:
        latest_review = reviews[0]
        if latest_review.comment:
            comment_lower = latest_review.comment.lower()
            if "confidence" in comment_lower:
                reasons.append("low confidence")
            elif "quality" in comment_lower:
                reasons.append("poor quality")
            elif "disagreement" in comment_lower or "mismatch" in comment_lower:
                reasons.append("model disagreement")
            elif "overlap" in comment_lower:
                reasons.append("overlapping sounds")

    if reasons:
        # Deduplicate while preserving order
        unique_reasons: List[str] = []
        for r in reasons:
            if r not in unique_reasons:
                unique_reasons.append(r)
        return ", ".join(unique_reasons)

    return "low confidence"


def get_manual_review_queue() -> List[AudioEvent]:
    """Queries all AudioEvents with status 'Manual Review', sorted oldest first.

    Prefetches related predictions, comparison results, and review records to prevent N+1 queries.
    Attaches computed `routing_reason` and `confidence_diff_display` to each event instance.
    """
    events = (
        AudioEvent.objects.filter(status=AudioEvent.STATUS_MANUAL_REVIEW)
        .select_related("uploaded_by")
        .prefetch_related(
            "comparison_results",
            "python_predictions",
            "gtm_predictions",
            "review_records",
        )
        .order_by("uploaded_at")
    )

    event_list = list(events)
    for event in event_list:
        event.routing_reason = get_routing_reason(event)
        comps = list(event.comparison_results.all())
        if comps and comps[0].confidence_difference is not None:
            event.confidence_diff = comps[0].confidence_difference
            event.confidence_diff_display = f"{comps[0].confidence_difference:.4f}"
        else:
            event.confidence_diff = None
            event.confidence_diff_display = "N/A"

    return event_list


def submit_review_decision(
    audio_event: AudioEvent,
    reviewer: Any,
    decision: str,
    comment: str = "",
    recommended_action: str = "",
) -> ReviewRecord:
    """Stores manual audio event review decision as a ReviewRecord override.

    Updates AudioEvent category, severity, and status to 'Reviewed' without deleting
    or mutating original PythonPrediction or GTMPrediction rows. Logs audit action.
    """
    review = ReviewRecord.objects.create(
        audio_event=audio_event,
        reviewer=reviewer if getattr(reviewer, "is_authenticated", False) else None,
        decision=decision,
        comment=comment,
        recommended_action=recommended_action,
    )

    from core.constants import get_severity_for_category

    audio_event.category = decision
    audio_event.severity = get_severity_for_category(decision)
    audio_event.status = AudioEvent.STATUS_REVIEWED
    audio_event.save(update_fields=["category", "severity", "status"])

    try:
        from audit.services import log_action
        from audit.models import AuditLog

        log_action(
            user=reviewer if getattr(reviewer, "is_authenticated", False) else None,
            action=AuditLog.ACTION_REVIEW,
            detail={
                "audio_id": audio_event.audio_id,
                "decision": decision,
                "comment": comment,
                "recommended_action": recommended_action,
            },
        )
    except Exception:
        pass

    return review
