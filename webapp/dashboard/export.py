"""Audio event analysis report export module.

Generates professional, Excel-compatible CSV reports for AudioEvent querysets,
including metadata, model predictions (Python & GTM), model comparison metrics,
alert statuses, manual review decisions, and SHA-256 hashes.
"""

import csv
import io
from datetime import datetime
from typing import Any, List, Optional
from django.db.models import QuerySet
from django.http import HttpResponse

from audio_events.models import AudioEvent


def _fmt_dt(value: Any, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return a formatted datetime string, or 'N/A' for absent/unconverted values.

    DateTimeField values are normally coerced to datetime objects by the ORM,
    but may arrive as strings when fetched via annotated or raw queries.
    """
    if value is None:
        return "N/A"
    if isinstance(value, datetime):
        return value.strftime(fmt)
    try:
        return str(value)
    except Exception:
        return "N/A"

CSV_HEADER_FIELDS = [
    "Audio ID",
    "Filename",
    "Category",
    "Quality",
    "Severity",
    "Status",
    "Duration (s)",
    "Sample Rate (Hz)",
    "Channels",
    "File Size (bytes)",
    "Format",
    "Uploaded At",
    "Uploaded By",
    "Python Predicted Class",
    "Python Confidence",
    "Python Model Version",
    "GTM Predicted Class",
    "GTM Model Version",
    "Confidence Difference",
    "Top-2 Margin",
    "Model Agreement Status",
    "Is Uncertain",
    "Is Overlapping",
    "Alert Generated",
    "Alert Severity",
    "Alert Status",
    "Alert Created At",
    "Alert Resolved At",
    "Alert Resolved By",
    "Review Decision",
    "Reviewer",
    "Review Comment",
    "Recommended Action",
    "Reviewed At",
    "SHA-256 Hash",
]


def prepare_export_queryset(queryset: QuerySet[AudioEvent]) -> QuerySet[AudioEvent]:
    """Prefetch related predictions, comparison results, alerts, and reviews to prevent N+1 queries."""
    return queryset.select_related("uploaded_by").prefetch_related(
        "python_predictions",
        "gtm_predictions",
        "comparison_results",
        "alerts",
        "alerts__resolved_by",
        "review_records",
        "review_records__reviewer",
    )


def extract_row_data(evt: AudioEvent,) -> List[Any]:
    """Extract and format analysis report row fields for an AudioEvent instance."""
    python_preds = list(evt.python_predictions.all())
    python_pred = python_preds[0] if python_preds else None

    gtm_preds = list(evt.gtm_predictions.all())
    gtm_pred = gtm_preds[0] if gtm_preds else None

    comparisons = list(evt.comparison_results.all())
    comp_res = comparisons[0] if comparisons else None

    alerts = list(evt.alerts.all())
    alert = alerts[0] if alerts else None

    reviews = list(evt.review_records.all())
    review = reviews[0] if reviews else None

    uploaded_at_str = _fmt_dt(evt.uploaded_at)
    uploaded_by_str = evt.uploaded_by.username if evt.uploaded_by else "Anonymous"

    py_class = python_pred.predicted_class if python_pred else "N/A"
    py_conf = f"{python_pred.confidence:.4f}" if python_pred and python_pred.confidence is not None else "N/A"
    py_ver = python_pred.model_version if python_pred else "N/A"

    gtm_class = gtm_pred.predicted_class if gtm_pred else "N/A"
    gtm_ver = gtm_pred.model_version if gtm_pred else "N/A"

    conf_diff = f"{comp_res.confidence_difference:.4f}" if comp_res and comp_res.confidence_difference is not None else "N/A"
    top2_margin = f"{comp_res.top_two_margin:.4f}" if comp_res and comp_res.top_two_margin is not None else "N/A"
    match_status = comp_res.match_status if comp_res else "N/A"
    is_uncertain = str(comp_res.is_uncertain) if comp_res else "N/A"
    is_overlapping = str(comp_res.is_overlapping) if comp_res else "N/A"

    has_alert = "Yes" if alert else "No"
    alert_sev = alert.severity if alert else "N/A"
    alert_status = alert.status if alert else "N/A"
    alert_created = _fmt_dt(alert.created_at) if alert else "N/A"
    alert_resolved = _fmt_dt(alert.resolved_at) if alert else "N/A"
    alert_by = alert.resolved_by.username if alert and alert.resolved_by else "N/A"

    rev_decision = review.decision if review else "N/A"
    reviewer_str = review.reviewer.username if review and review.reviewer else "N/A"
    rev_comment = review.comment if review else ""
    rev_action = review.recommended_action if review else ""
    rev_at = _fmt_dt(review.created_at) if review else "N/A"

    dur_str = f"{evt.duration:.2f}" if evt.duration is not None else "N/A"

    return [
        evt.audio_id,
        evt.filename,
        evt.category,
        evt.quality,
        evt.severity,
        evt.status,
        dur_str,
        evt.sampling_rate,
        evt.channels,
        evt.file_size,
        evt.format,
        uploaded_at_str,
        uploaded_by_str,
        py_class,
        py_conf,
        py_ver,
        gtm_class,
        gtm_ver,
        conf_diff,
        top2_margin,
        match_status,
        is_uncertain,
        is_overlapping,
        has_alert,
        alert_sev,
        alert_status,
        alert_created,
        alert_resolved,
        alert_by,
        rev_decision,
        reviewer_str,
        rev_comment,
        rev_action,
        rev_at,
        evt.hash,
    ]


def export_audio_events_to_csv_data(queryset: QuerySet[AudioEvent]) -> str:
    """Generate Excel-compatible UTF-8 CSV string payload for given AudioEvent queryset."""
    qs = prepare_export_queryset(queryset)
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    writer.writerow(CSV_HEADER_FIELDS)
    for evt in qs:
        writer.writerow(extract_row_data(evt))

    return output.getvalue()


def generate_audio_events_csv_response(
    queryset: QuerySet[AudioEvent],
    filename: str = "sonic_sentinel_analysis_report.csv",
) -> HttpResponse:
    """Return a downloadable Django HttpResponse containing Excel-compatible UTF-8 CSV analysis report."""
    csv_text = export_audio_events_to_csv_data(queryset)
    bom_utf8_bytes = b"\xef\xbb\xbf" + csv_text.encode("utf-8")

    response = HttpResponse(bom_utf8_bytes, content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def generate_audio_events_export_response(
    queryset: QuerySet[AudioEvent],
    export_format: str = "csv",
    filename: Optional[str] = None,
) -> HttpResponse:
    """Return a downloadable HttpResponse containing analysis report in CSV or Excel format."""
    fmt = (export_format or "csv").lower().strip()
    is_excel = fmt in ["excel", "xlsx"]

    if filename:
        out_filename = filename
        if is_excel and out_filename.endswith(".csv"):
            out_filename = out_filename[:-4] + ".xlsx"
    else:
        ext = "xlsx" if is_excel else "csv"
        out_filename = f"sonic_sentinel_analysis_report.{ext}"

    csv_text = export_audio_events_to_csv_data(queryset)
    bom_utf8_bytes = b"\xef\xbb\xbf" + csv_text.encode("utf-8")

    content_type = "application/vnd.ms-excel; charset=utf-8" if is_excel else "text/csv; charset=utf-8"
    response = HttpResponse(bom_utf8_bytes, content_type=content_type)
    response["Content-Disposition"] = f'attachment; filename="{out_filename}"'
    return response

