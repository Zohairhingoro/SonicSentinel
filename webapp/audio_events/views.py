import uuid
from pathlib import Path
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from django.contrib import messages
from django.shortcuts import render
from django.views import View

from .forms import BatchAudioUploadForm, EventFilterForm, SingleAudioUploadForm
from core.storage import check_duplicate_audio, check_near_duplicate_audio

from predictions.models import PythonPrediction
from .models import AudioEvent
from .services.ml_client import MLServiceError, MLServiceUnavailable, predict_audio_file
from .services.query import get_paginated_audio_events
from .services.validation import validate_audio_file
from accounts.permissions import is_administrator, is_security_operator


def _can_batch_upload(user) -> bool:
    """Return True when the user is allowed to perform batch uploads."""
    return is_administrator(user) or is_security_operator(user)


class SingleUploadView(View):
    """View rendering single-file audio upload page and performing validation."""

    def get(self, request: HttpRequest):
        form = SingleAudioUploadForm()
        return render(
            request,
            "audio_events/upload.html",
            {
                "form": form,
                "validation_result": None,
            },
        )

    def post(self, request: HttpRequest):
        form = SingleAudioUploadForm(request.POST, request.FILES)
        validation_result = None
        audio_event = None

        if form.is_valid():
            uploaded_file = form.cleaned_data["audio_file"]
            validation_result = validate_audio_file(uploaded_file)
            if validation_result.get("accepted"):
                messages.success(
                    request,
                    f"Audio file '{uploaded_file.name}' passed all validation checks successfully!",
                )
                try:
                    existing_event, file_hash = check_duplicate_audio(uploaded_file)
                    if existing_event:
                        audio_event = existing_event
                    else:
                        meta = validation_result.get("metadata", {})
                        audio_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
                        uploaded_by = request.user if getattr(request, "user", None) and request.user.is_authenticated else None
                        suspected_dup, fingerprint = check_near_duplicate_audio(uploaded_file, limit=50)
                        initial_status = AudioEvent.STATUS_UPLOADED
                        suspected_dup_id = suspected_dup.audio_id if suspected_dup else ""

                        audio_event = AudioEvent(
                            audio_id=audio_id,
                            filename=meta.get("filename", uploaded_file.name),
                            format=meta.get("format", "wav"),
                            duration=meta.get("duration", 0.0),
                            sampling_rate=meta.get("sampling_rate", 16000),
                            channels=meta.get("channels", 1),
                            bit_depth=meta.get("bit_depth"),
                            file_size=meta.get("file_size", uploaded_file.size),
                            uploaded_by=uploaded_by,
                            status=initial_status,
                            quality=meta.get("quality", AudioEvent.QUALITY_GOOD),
                            category="Background Noise",
                            hash=file_hash or meta.get("hash", ""),
                            audio_fingerprint=fingerprint,
                            suspected_duplicate_of_id=suspected_dup_id,
                            suspected_duplicate=suspected_dup,
                            audio_file=uploaded_file,
                            storage_path=f"audio/{audio_id}/{meta.get('filename', uploaded_file.name)}",
                        )
                        audio_event.save()

                        # Trigger ML inference
                        prediction_res = None
                        try:
                            if audio_event.audio_file and hasattr(audio_event.audio_file, "path") and Path(audio_event.audio_file.path).is_file():
                                prediction_res = predict_audio_file(audio_event.audio_file.path, filename=audio_event.filename)
                            else:
                                uploaded_file.seek(0)
                                prediction_res = predict_audio_file(uploaded_file, filename=audio_event.filename)
                        except (MLServiceUnavailable, MLServiceError):
                            if not audio_event.suspected_duplicate_of_id:
                                audio_event.status = AudioEvent.STATUS_UNCERTAIN
                            audio_event.save()

                        if prediction_res:
                            pred_class = str(prediction_res.get("predicted_class", "Background Noise"))
                            confidence_dict = prediction_res.get("confidence", {})
                            confidence_score = float(confidence_dict.get(pred_class, 0.0))
                            model_ver = str(prediction_res.get("model_version", "1.0.0"))

                            prediction_obj = PythonPrediction.objects.create(
                                audio_event=audio_event,
                                predicted_class=pred_class,
                                confidence=confidence_score,
                                confidence_scores=confidence_dict,
                                model_version=model_ver,
                            )

                            audio_event.category = pred_class
                            if not audio_event.suspected_duplicate_of_id:
                                audio_event.status = AudioEvent.STATUS_CLASSIFIED
                            if "quality" in prediction_res:
                                audio_event.quality = str(prediction_res["quality"])
                            audio_event.save()

                            if not audio_event.suspected_duplicate_of_id:
                                try:
                                    from alerts.rule_engine import evaluate_and_trigger_alert
                                    evaluate_and_trigger_alert(audio_event, prediction=prediction_obj)
                                    audio_event.refresh_from_db()
                                except Exception:
                                    pass
                except Exception:
                    pass
            else:
                rejection_reason = validation_result.get("reason", "Validation failed")
                messages.error(
                    request,
                    f"Audio Validation Rejected: {rejection_reason}",
                )

        return render(
            request,
            "audio_events/upload.html",
            {
                "form": form,
                "validation_result": validation_result,
                "audio_event": audio_event,
            },
        )


@csrf_exempt
def upload_audio(request: HttpRequest) -> JsonResponse:

    """Endpoint to validate, store, and trigger ML classification on uploaded audio file."""
    if request.method != "POST":
        return JsonResponse(
            {"accepted": False, "reason": "Method not allowed. Use POST to upload audio files.", "metadata": {}},
            status=405,
        )

    uploaded_file = request.FILES.get("audio_file") or request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse(
            {
                "accepted": False,
                "reason": "No audio file provided in request. Expected form field 'audio_file' or 'file'.",
                "metadata": {},
            },
            status=400,
        )

    # Early SHA-256 duplicate content hash check before validation and media storage run
    existing_event, file_hash = check_duplicate_audio(uploaded_file)
    if existing_event:
        return JsonResponse(
            {
                "accepted": True,
                "duplicate": True,
                "reason": "Duplicate audio file content detected. Returned existing audio event record.",
                "audio_id": existing_event.audio_id,
                "storage_path": existing_event.storage_path,
                "metadata": {
                    "filename": existing_event.filename,
                    "format": existing_event.format,
                    "duration": existing_event.duration,
                    "sampling_rate": existing_event.sampling_rate,
                    "channels": existing_event.channels,
                    "file_size": existing_event.file_size,
                    "hash": existing_event.hash,
                    "quality": existing_event.quality,
                },
                "prediction": {
                    "predicted_class": existing_event.category,
                    "status": existing_event.status,
                    "severity": existing_event.severity,
                    "quality": existing_event.quality,
                },
            },
            status=200,
        )

    validation_result = validate_audio_file(uploaded_file)
    if not validation_result.get("accepted", False):
        return JsonResponse(validation_result, status=400)

    meta = validation_result.get("metadata", {})
    audio_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
    uploaded_by = request.user if getattr(request, "user", None) and request.user.is_authenticated else None

    # Check for near-duplicate audio fingerprint in recent N audio events
    suspected_dup, fingerprint = check_near_duplicate_audio(uploaded_file, limit=50)

    initial_status = AudioEvent.STATUS_UPLOADED
    suspected_dup_id = ""
    if suspected_dup:
        initial_status = AudioEvent.STATUS_MANUAL_REVIEW
        suspected_dup_id = suspected_dup.audio_id

    audio_event = AudioEvent(
        audio_id=audio_id,
        filename=meta.get("filename", uploaded_file.name),
        format=meta.get("format", "wav"),
        duration=meta.get("duration", 0.0),
        sampling_rate=meta.get("sampling_rate", 16000),
        channels=meta.get("channels", 1),
        bit_depth=meta.get("bit_depth"),
        file_size=meta.get("file_size", uploaded_file.size),
        uploaded_by=uploaded_by,
        status=initial_status,
        quality=meta.get("quality", AudioEvent.QUALITY_GOOD),
        category="Background Noise",
        hash=file_hash or meta.get("hash", ""),
        audio_fingerprint=fingerprint,
        suspected_duplicate_of_id=suspected_dup_id,
        suspected_duplicate=suspected_dup,
        audio_file=uploaded_file,
        storage_path=f"audio/{audio_id}/{meta.get('filename', uploaded_file.name)}",
    )
    audio_event.save()

    prediction_res = None
    ml_error = None

    try:
        if audio_event.audio_file and hasattr(audio_event.audio_file, "path") and Path(audio_event.audio_file.path).is_file():
            prediction_res = predict_audio_file(audio_event.audio_file.path, filename=audio_event.filename)
        else:
            uploaded_file.seek(0)
            prediction_res = predict_audio_file(uploaded_file, filename=audio_event.filename)
    except (MLServiceUnavailable, MLServiceError) as exc:
        ml_error = str(exc)
        if not audio_event.suspected_duplicate_of_id:
            audio_event.status = AudioEvent.STATUS_UNCERTAIN
        audio_event.save()

    if prediction_res:
        pred_class = str(prediction_res.get("predicted_class", "Background Noise"))
        confidence_dict = prediction_res.get("confidence", {})
        confidence_score = float(confidence_dict.get(pred_class, 0.0))
        model_ver = str(prediction_res.get("model_version", "1.0.0"))

        prediction_obj = PythonPrediction.objects.create(
            audio_event=audio_event,
            predicted_class=pred_class,
            confidence=confidence_score,
            confidence_scores=confidence_dict,
            model_version=model_ver,
        )

        audio_event.category = pred_class
        if not audio_event.suspected_duplicate_of_id:
            audio_event.status = AudioEvent.STATUS_CLASSIFIED
        if "quality" in prediction_res:
            audio_event.quality = str(prediction_res["quality"])
        audio_event.save()

        # Evaluate alert rules if not routed to manual review
        if not audio_event.suspected_duplicate_of_id:
            try:
                from alerts.rule_engine import evaluate_and_trigger_alert

                alert_obj = evaluate_and_trigger_alert(audio_event, prediction=prediction_obj)
                if alert_obj:
                    audio_event.refresh_from_db()
            except Exception:
                pass


    prediction_dict = {
        "predicted_class": audio_event.category,
        "status": audio_event.status,
        "severity": audio_event.severity,
        "quality": audio_event.quality,
        "ml_error": ml_error,
    }
    if 'prediction_obj' in locals() and prediction_obj:
        prediction_dict["confidence"] = prediction_obj.confidence
        prediction_dict["confidence_scores"] = prediction_obj.confidence_scores
        prediction_dict["top3"] = prediction_obj.top3
        prediction_dict["model_version"] = prediction_obj.model_version

    return JsonResponse(
        {
            "accepted": True,
            "reason": "Audio file validated, saved, and processed",
            "audio_id": audio_event.audio_id,
            "storage_path": audio_event.storage_path,
            "audio_url": audio_event.audio_file.url if audio_event.audio_file else "",
            "metadata": meta,
            "prediction": prediction_dict,
        },
        status=201,
    )


class BatchUploadView(View):
    """Batch audio upload restricted to administrator and security_operator roles.

    Accepts multiple files in a single POST, runs each through the identical
    validate_audio_file call used by SingleUploadView (FE-4), and renders a
    per-file results table — filename, accepted/rejected, rejection reason.
    Validation logic is NOT duplicated here; this view only orchestrates the
    existing service call.
    """

    def _check_permission(self, request: HttpRequest):
        """Return None when access is granted, or an HttpResponse to abort."""
        if not request.user.is_authenticated:
            from django.contrib.auth.views import redirect_to_login
            return redirect_to_login(request.get_full_path())
        if not _can_batch_upload(request.user):
            from django.core.exceptions import PermissionDenied
            raise PermissionDenied("Batch upload is restricted to administrators and security operators.")
        return None

    def get(self, request: HttpRequest):
        denied = self._check_permission(request)
        if denied:
            return denied
        form = BatchAudioUploadForm()
        return render(request, "audio_events/batch_upload.html", {"form": form, "results": None})

    def post(self, request: HttpRequest):
        denied = self._check_permission(request)
        if denied:
            return denied

        form = BatchAudioUploadForm(request.POST, request.FILES)
        results = []

        if form.is_valid():
            uploaded_files = form.cleaned_data["audio_files"]
            for f in uploaded_files:
                outcome = validate_audio_file(f)
                results.append(
                    {
                        "filename": getattr(f, "name", "unknown"),
                        "accepted": outcome.get("accepted", False),
                        "reason": outcome.get("reason", ""),
                        "metadata": outcome.get("metadata", {}),
                    }
                )

            accepted_count = sum(1 for r in results if r["accepted"])
            rejected_count = len(results) - accepted_count
            messages.success(
                request,
                f"Batch validation complete: {accepted_count} accepted, {rejected_count} rejected out of {len(results)} file(s).",
            )

        return render(
            request,
            "audio_events/batch_upload.html",
            {
                "form": form,
                "results": results if results else None,
                "accepted_count": sum(1 for r in results if r["accepted"]),
                "rejected_count": sum(1 for r in results if not r["accepted"]),
            },
        )


def index(request: HttpRequest) -> JsonResponse:
    return JsonResponse({"status": "ok", "app": "audio_events"})


class EventListView(View):
    """Paginated, filterable audio event list.

    GET params are bound into EventFilterForm; validation errors are shown
    inline and the unfiltered list is still rendered.  The form's
    ``to_query_filters()`` method produces the filter dict passed to
    ``get_paginated_audio_events``, which delegates to ``list_audio_events``
    — no duplicate filtering logic lives in this view.

    The ``user`` filter field is rendered only when the authenticated user has
    the administrator role (or is a Django superuser).
    """

    _PER_PAGE = 15

    def get(self, request: HttpRequest):
        is_admin = (
            request.user.is_authenticated
            and (getattr(request.user, "is_administrator", False) or request.user.is_superuser)
        )

        form = EventFilterForm(request.GET or None)

        filters: dict = {}
        if form.is_bound and form.is_valid():
            filters = form.to_query_filters()
            if not is_admin:
                filters.pop("user", None)

        page_obj = get_paginated_audio_events(request, per_page=self._PER_PAGE, filters=filters)

        return render(
            request,
            "audio_events/event_list.html",
            {
                "form": form,
                "page_obj": page_obj,
                "is_admin": is_admin,
                "active_filter_count": sum(1 for v in filters.values() if v not in ("", None)),
            },
        )

