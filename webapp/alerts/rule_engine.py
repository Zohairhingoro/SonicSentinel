"""Alert rule engine module for evaluating sound events against configurable AlertRules."""

from typing import Any, Dict, Optional, Union
from alerts.models import Alert, AlertRule
from audio_events.models import AudioEvent
from predictions.models import ComparisonResult, PythonPrediction

QUALITY_RANKS: Dict[str, int] = {
    AlertRule.QUALITY_GOOD: 4,
    AlertRule.QUALITY_ACCEPTABLE: 3,
    AlertRule.QUALITY_POOR: 2,
    AlertRule.QUALITY_UNUSABLE: 1,
}

AGREEMENT_STATUSES = {
    ComparisonResult.MATCH_ACCEPTABLE,
    ComparisonResult.MATCH_WEAK,
}


def evaluate_alert_conditions(
    confidence: float,
    top_two_margin: float,
    match_status: str,
    quality: str,
    consecutive_count: int,
    min_confidence: float,
    required_top_two_margin: float,
    requires_model_agreement: bool,
    requires_min_quality: str,
    required_consecutive_detections: int,
) -> bool:
    """Pure evaluation function that checks event parameters against rule thresholds.

    Args:
        confidence: Top predicted class confidence score.
        top_two_margin: Margin between top predicted class and second predicted class.
        match_status: Model comparison match status string.
        quality: Audio quality string ('Good', 'Acceptable', 'Poor', 'Unusable').
        consecutive_count: Number of consecutive detections for the category.
        min_confidence: Required minimum confidence threshold.
        required_top_two_margin: Required minimum top-two margin.
        requires_model_agreement: Whether agreement between models is required.
        requires_min_quality: Minimum required audio quality level.
        required_consecutive_detections: Required number of consecutive detections.

    Returns:
        True if all rule conditions pass, False otherwise.
    """
    if confidence < min_confidence:
        return False

    if top_two_margin < required_top_two_margin:
        return False

    if requires_model_agreement and match_status not in AGREEMENT_STATUSES:
        return False

    event_quality_rank = QUALITY_RANKS.get(quality, 1)
    required_quality_rank = QUALITY_RANKS.get(requires_min_quality, 1)
    if event_quality_rank < required_quality_rank:
        return False

    if consecutive_count < required_consecutive_detections:
        return False

    return True


def evaluate_rule_against_metrics(
    rule: Union[AlertRule, Any],
    confidence: float,
    top_two_margin: float = 1.0,
    match_status: str = ComparisonResult.MATCH_ACCEPTABLE,
    quality: str = AlertRule.QUALITY_GOOD,
    consecutive_count: int = 1,
) -> bool:
    """Evaluate event metrics against an AlertRule object or duck-typed rule instance."""
    return evaluate_alert_conditions(
        confidence=confidence,
        top_two_margin=top_two_margin,
        match_status=match_status,
        quality=quality,
        consecutive_count=consecutive_count,
        min_confidence=getattr(rule, "min_confidence", 0.70),
        required_top_two_margin=getattr(rule, "top_two_margin", 0.15),
        requires_model_agreement=getattr(rule, "requires_model_agreement", False),
        requires_min_quality=getattr(rule, "requires_min_quality", AlertRule.QUALITY_ACCEPTABLE),
        required_consecutive_detections=getattr(rule, "required_consecutive_detections", 1),
    )


def get_consecutive_detections_count(audio_event: AudioEvent, category: str) -> int:
    """Query recent predictions/events for the same user or session to count consecutive category detections."""
    if audio_event.segments.exists():
        recent_preds = PythonPrediction.objects.filter(
            audio_event=audio_event
        ).order_by("-created_at")
    elif audio_event.uploaded_by_id:
        recent_preds = PythonPrediction.objects.filter(
            audio_event__uploaded_by_id=audio_event.uploaded_by_id
        ).order_by("-created_at")[:20]
    else:
        recent_preds = PythonPrediction.objects.filter(
            audio_event__filename=audio_event.filename
        ).order_by("-created_at")[:20]

    count = 0
    for pred in recent_preds:
        if pred.predicted_class == category:
            count += 1
        else:
            break

    return max(count, 1)


def evaluate_and_trigger_alert(
    audio_event: AudioEvent,
    comparison_result: Optional[ComparisonResult] = None,
    prediction: Optional[PythonPrediction] = None,
) -> Optional[Alert]:
    """Evaluate an AudioEvent + ComparisonResult against the relevant AlertRule row and trigger Alert if matched.

    Args:
        audio_event: AudioEvent instance to evaluate.
        comparison_result: Optional ComparisonResult instance.
        prediction: Optional PythonPrediction instance.

    Returns:
        Created Alert instance if rule conditions pass, None otherwise.
    """
    if prediction is None:
        prediction = audio_event.python_predictions.order_by("-created_at").first()

    if comparison_result is None:
        comparison_result = audio_event.comparison_results.order_by("-created_at").first()

    category = prediction.predicted_class if prediction else audio_event.category
    confidence = prediction.confidence if prediction else 0.0
    top_two_margin = comparison_result.top_two_margin if comparison_result else 1.0
    match_status = comparison_result.match_status if comparison_result else ComparisonResult.MATCH_ACCEPTABLE
    quality = audio_event.quality

    rule = AlertRule.objects.filter(category=category).first()
    if not rule:
        if not AlertRule.objects.exists():
            from django.core.management import call_command
            try:
                call_command("seed_alert_rules")
                rule = AlertRule.objects.filter(category=category).first()
            except Exception:
                pass

    if not rule:
        rule = AlertRule.objects.filter(category="Background Noise").first()

    if not rule:
        return None

    consecutive_count = get_consecutive_detections_count(audio_event, category)

    passes = evaluate_alert_conditions(
        confidence=confidence,
        top_two_margin=top_two_margin,
        match_status=match_status,
        quality=quality,
        consecutive_count=consecutive_count,
        min_confidence=rule.min_confidence,
        required_top_two_margin=rule.top_two_margin,
        requires_model_agreement=rule.requires_model_agreement,
        requires_min_quality=rule.requires_min_quality,
        required_consecutive_detections=rule.required_consecutive_detections,
    )

    if not passes:
        return None

    existing_alert = Alert.objects.filter(audio_event=audio_event, alert_rule=rule).first()
    if existing_alert:
        return existing_alert

    alert = Alert.objects.create(
        audio_event=audio_event,
        alert_rule=rule,
        severity=rule.severity,
        status=Alert.STATUS_ACTIVE,
    )

    audio_event.status = AudioEvent.STATUS_ALERT_GENERATED
    audio_event.severity = rule.severity
    audio_event.save(update_fields=["status", "severity"])

    return alert
