from typing import Any
from django.contrib import messages
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import redirect, render

from accounts.permissions import is_administrator, is_audio_reviewer, role_required
from audio_events.models import AudioEvent
from core.constants import get_categories
from .services import get_manual_review_queue, submit_review_decision


def is_reviewer_or_admin(user: Any) -> bool:
    """Check if user is an audio reviewer or administrator."""
    return is_audio_reviewer(user) or is_administrator(user)


@role_required(is_reviewer_or_admin)
def review_queue(request: HttpRequest) -> HttpResponse:
    """Render manual review queue page listing all AudioEvents with status 'Manual Review', sorted oldest first.

    Visible only to audio reviewer and administrator roles.
    """
    queue_events = get_manual_review_queue()
    context = {
        "queue_events": queue_events,
        "total_count": len(queue_events),
    }
    return render(request, "reviews/queue.html", context)


@role_required(is_reviewer_or_admin)
def review_detail(request: HttpRequest, audio_id: str) -> HttpResponse:
    """Render reviewer detail page for an AudioEvent.

    Displays audio player (FE-6), comparison display (FE-10), and decision form.
    On submit, stores the decision as an override (ReviewRecord), updates event status to 'Reviewed',
    and leaves original predictions intact.
    """
    event = (
        AudioEvent.objects.filter(audio_id=audio_id)
        .select_related("uploaded_by")
        .prefetch_related(
            "comparison_results",
            "python_predictions",
            "gtm_predictions",
            "review_records",
            "review_records__reviewer",
        )
        .first()
    )
    if not event:
        raise Http404(f"Audio event with ID '{audio_id}' was not found.")

    categories = get_categories()

    if request.method == "POST":
        decision = request.POST.get("decision", "").strip()
        comment = request.POST.get("comment", "").strip()
        recommended_action = request.POST.get("recommended_action", "").strip()

        if decision not in categories:
            messages.error(request, "Invalid sound category selected.")
        else:
            submit_review_decision(
                audio_event=event,
                reviewer=request.user,
                decision=decision,
                comment=comment,
                recommended_action=recommended_action,
            )
            messages.success(
                request, f"Review decision for {event.audio_id} saved successfully."
            )
            return redirect("reviews:detail", audio_id=event.audio_id)

    comparison = event.comparison_results.first()
    review_history = list(event.review_records.all())

    context = {
        "event": event,
        "audio_event": event,
        "comparison": comparison,
        "categories": categories,
        "review_history": review_history,
    }
    return render(request, "reviews/detail.html", context)

