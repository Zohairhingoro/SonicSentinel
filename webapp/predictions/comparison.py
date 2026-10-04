"""Pure logic and database integration for comparing Python and GTM ML model predictions."""

import logging
from typing import Any, Dict, Optional, Tuple

from core.constants import get_alert_rules

logger = logging.getLogger(__name__)


def calculate_top_two_margin(confidence_scores: Dict[str, float]) -> float:
    """Calculate the margin between the highest and second highest confidence scores for a model.

    Args:
        confidence_scores: Mapping of category names to float confidence scores.

    Returns:
        float: Difference between top-1 confidence score and top-2 confidence score.
    """
    if not confidence_scores:
        return 0.0

    values = sorted([float(v) for v in confidence_scores.values()], reverse=True)
    if len(values) == 1:
        return values[0]

    return values[0] - values[1]


def compute_confidence_difference(py_confidence: float, gtm_confidence: float) -> float:
    """Compute absolute difference between Python top-class confidence and GTM top-class confidence.

    Args:
        py_confidence: Top class confidence score for Python model.
        gtm_confidence: Top class confidence score for GTM model.

    Returns:
        float: |py_confidence - gtm_confidence|
    """
    return abs(float(py_confidence) - float(gtm_confidence))


def evaluate_match_status(
    py_class: str,
    py_confidence: float,
    py_top_two_margin: float,
    gtm_class: str,
    gtm_confidence: float,
    gtm_top_two_margin: float,
    audio_quality: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool, bool]:
    """Evaluate match_status, is_uncertain, and is_overlapping based on rule thresholds.

    Args:
        py_class: Predicted category by Python model.
        py_confidence: Python model top-class confidence.
        py_top_two_margin: Top-two margin for Python model.
        gtm_class: Predicted category by GTM model.
        gtm_confidence: GTM model top-class confidence.
        gtm_top_two_margin: Top-two margin for GTM model.
        audio_quality: Audio quality string ('Good', 'Acceptable', 'Poor', 'Unusable').
        config: Optional pre-loaded alert_rules dictionary.

    Returns:
        Tuple[str, bool, bool]: (match_status, is_uncertain, is_overlapping)
    """
    if config is None:
        try:
            config = get_alert_rules()
        except Exception as err:
            logger.warning(f"Failed to load alert rules config: {err}")
            config = {}

    rules = config.get("rules", {})
    comp_thresholds = config.get("comparison_thresholds", {})

    py_rule = rules.get(py_class, {})
    gtm_rule = rules.get(gtm_class, {})

    py_min_conf = py_rule.get("min_confidence", comp_thresholds.get("confidence_threshold", 0.70))
    gtm_min_conf = gtm_rule.get("min_confidence", comp_thresholds.get("confidence_threshold", 0.70))

    conf_threshold = min(py_min_conf, gtm_min_conf)
    margin_threshold = comp_thresholds.get("top_two_margin_threshold", 0.15)
    max_diff_acceptable = comp_thresholds.get("max_confidence_diff_acceptable", 0.15)
    min_conf_acceptable = comp_thresholds.get("min_confidence_acceptable", 0.60)

    conf_diff = compute_confidence_difference(py_confidence, gtm_confidence)
    combined_margin = min(py_top_two_margin, gtm_top_two_margin)

    is_overlapping = (py_top_two_margin < 0.10 and py_confidence > 0.40) or (
        gtm_top_two_margin < 0.10 and gtm_confidence > 0.40
    )

    if py_class != gtm_class:
        return "Model Disagreement", True, is_overlapping

    if (
        py_confidence < min_conf_acceptable
        or gtm_confidence < min_conf_acceptable
        or combined_margin < 0.05
        or audio_quality == "Unusable"
    ):
        return "Uncertain Result", True, is_overlapping

    if (
        conf_diff > max_diff_acceptable
        or py_confidence < conf_threshold
        or gtm_confidence < conf_threshold
        or combined_margin < margin_threshold
        or audio_quality == "Poor"
        or is_overlapping
    ):
        uncertain = (
            py_confidence < conf_threshold
            or gtm_confidence < conf_threshold
            or combined_margin < margin_threshold
            or audio_quality == "Poor"
            or is_overlapping
        )
        return "Weak Match", uncertain, is_overlapping

    return "Acceptable Match", False, is_overlapping


def compare_predictions_pure(
    py_predicted_class: str,
    py_confidence_scores: Dict[str, float],
    gtm_predicted_class: str,
    gtm_confidence_scores: Dict[str, float],
    audio_quality: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pure mathematical function to calculate comparative analysis metrics without DB dependencies.

    Args:
        py_predicted_class: Category predicted by Python model.
        py_confidence_scores: Dict mapping categories to confidence scores for Python model.
        gtm_predicted_class: Category predicted by GTM model.
        gtm_confidence_scores: Dict mapping categories to confidence scores for GTM model.
        audio_quality: Quality classification string.
        config: Alert rules configuration dict.

    Returns:
        Dict[str, Any] containing:
            - confidence_difference (float)
            - top_two_margin (float)
            - py_top_two_margin (float)
            - gtm_top_two_margin (float)
            - py_top_confidence (float)
            - gtm_top_confidence (float)
            - match_status (str)
            - is_uncertain (bool)
            - is_overlapping (bool)
    """
    py_top_conf = py_confidence_scores.get(
        py_predicted_class,
        max(py_confidence_scores.values()) if py_confidence_scores else 0.0,
    )
    gtm_top_conf = gtm_confidence_scores.get(
        gtm_predicted_class,
        max(gtm_confidence_scores.values()) if gtm_confidence_scores else 0.0,
    )

    conf_diff = compute_confidence_difference(py_top_conf, gtm_top_conf)
    py_margin = calculate_top_two_margin(py_confidence_scores)
    gtm_margin = calculate_top_two_margin(gtm_confidence_scores)
    combined_margin = min(py_margin, gtm_margin)

    match_status, is_uncertain, is_overlapping = evaluate_match_status(
        py_class=py_predicted_class,
        py_confidence=py_top_conf,
        py_top_two_margin=py_margin,
        gtm_class=gtm_predicted_class,
        gtm_confidence=gtm_top_conf,
        gtm_top_two_margin=gtm_margin,
        audio_quality=audio_quality,
        config=config,
    )

    return {
        "confidence_difference": conf_diff,
        "top_two_margin": combined_margin,
        "py_top_two_margin": py_margin,
        "gtm_top_two_margin": gtm_margin,
        "py_top_confidence": float(py_top_conf),
        "gtm_top_confidence": float(gtm_top_conf),
        "match_status": match_status,
        "is_uncertain": is_uncertain,
        "is_overlapping": is_overlapping,
    }


def create_comparison_if_ready(audio_event: Any) -> Optional[Any]:
    """Check if both PythonPrediction and GTMPrediction rows exist for AudioEvent.

    If both exist, computes comparative metrics and creates or updates ComparisonResult.

    Args:
        audio_event: AudioEvent instance.

    Returns:
        Optional[ComparisonResult]: The created or updated ComparisonResult object, or None if not ready.
    """
    from predictions.models import ComparisonResult, GTMPrediction, PythonPrediction

    py_pred = PythonPrediction.objects.filter(audio_event=audio_event).order_by("-created_at").first()
    gtm_pred = GTMPrediction.objects.filter(audio_event=audio_event).order_by("-created_at").first()

    if not py_pred or not gtm_pred:
        return None

    metrics = compare_predictions_pure(
        py_predicted_class=py_pred.predicted_class,
        py_confidence_scores=py_pred.confidence_scores or {},
        gtm_predicted_class=gtm_pred.predicted_class,
        gtm_confidence_scores=gtm_pred.confidence_scores or {},
        audio_quality=getattr(audio_event, "quality", None),
    )

    comp_result, _ = ComparisonResult.objects.update_or_create(
        audio_event=audio_event,
        defaults={
            "confidence_difference": metrics["confidence_difference"],
            "top_two_margin": metrics["top_two_margin"],
            "match_status": metrics["match_status"],
            "is_uncertain": metrics["is_uncertain"],
            "is_overlapping": metrics["is_overlapping"],
        },
    )
    return comp_result
