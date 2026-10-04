"""Form definitions for audio_events app."""

from django import forms

from core.constants import get_categories
from .models import AudioEvent

# ── shared Tailwind input classes ────────────────────────────────────────────
_INPUT = (
    "w-full bg-slate-950/60 border border-slate-700/70 text-slate-200 text-sm "
    "rounded-xl px-3 py-2 placeholder-slate-500 focus:outline-none "
    "focus:ring-2 focus:ring-cyan-500/60 focus:border-cyan-500/60 transition-all"
)
_SELECT = (
    "w-full bg-slate-950/60 border border-slate-700/70 text-slate-200 text-sm "
    "rounded-xl px-3 py-2 focus:outline-none "
    "focus:ring-2 focus:ring-cyan-500/60 focus:border-cyan-500/60 transition-all"
)


class SingleAudioUploadForm(forms.Form):
    """Form accepting a single audio file for validation and ingestion."""

    audio_file = forms.FileField(
        required=True,
        label="Select Audio File",
        help_text="Supported formats: WAV, MP3, FLAC, OGG, M4A (Max size: 20 MB, Duration: 0.5s - 30.0s).",
        widget=forms.ClearableFileInput(
            attrs={
                "class": "block w-full text-sm text-slate-300 file:mr-4 file:py-2.5 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-cyan-500/20 file:text-cyan-300 hover:file:bg-cyan-500/30 cursor-pointer bg-slate-900/90 border border-slate-700/80 rounded-xl focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 transition-all",
                "accept": ".wav,.mp3,.flac,.ogg,.m4a,audio/wav,audio/mpeg,audio/flac,audio/ogg,audio/mp4,audio/x-m4a",
            }
        ),
    )


class MultipleFileInput(forms.ClearableFileInput):
    """Widget that allows selecting multiple files in a single input."""

    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    """FileField that accepts and validates multiple uploaded files."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_result = super().clean
        if not isinstance(data, (list, tuple)):
            data = [data]
        return [single_result(d, initial) for d in data]


class BatchAudioUploadForm(forms.Form):
    """Multi-file audio upload form for administrator and security_operator roles."""

    audio_files = MultipleFileField(
        required=True,
        label="Select Audio Files",
        help_text="Select one or more audio files (WAV, MP3, FLAC, OGG, M4A). Each file is validated individually.",
        widget=MultipleFileInput(
            attrs={
                "class": (
                    "block w-full text-sm text-slate-300 "
                    "file:mr-4 file:py-2.5 file:px-4 file:rounded-xl file:border-0 "
                    "file:text-xs file:font-semibold file:bg-violet-500/20 file:text-violet-300 "
                    "hover:file:bg-violet-500/30 cursor-pointer bg-slate-900/90 border "
                    "border-slate-700/80 rounded-xl focus:outline-none focus:ring-2 "
                    "focus:ring-violet-500 focus:border-violet-500 transition-all"
                ),
                "accept": ".wav,.mp3,.flac,.ogg,.m4a,audio/wav,audio/mpeg,audio/flac,audio/ogg,audio/mp4,audio/x-m4a",
                "multiple": True,
            }
        ),
    )


# ── category choices loaded from config/classes.json at import time ──────────
def _category_choices():
    blank = [("" , "All Categories")]
    return blank + [(c, c) for c in get_categories()]


class EventFilterForm(forms.Form):
    """GET-only search/filter form for the audio event list page.

    Every field is optional.  The form is submitted as query-string parameters
    so result pages are shareable via URL.  Validation via clean_* guards
    numeric ranges; the actual DB filtering is delegated to
    ``audio_events.services.query.list_audio_events`` so there is no duplicate
    filtering implementation in the view.
    """

    # ── text search ──────────────────────────────────────────────────────────
    search = forms.CharField(
        required=False,
        label="Audio ID / Filename",
        widget=forms.TextInput(attrs={
            "placeholder": "Search by Audio ID or filename…",
            "class": _INPUT,
            "id": "filter-search",
            "autocomplete": "off",
        }),
    )

    # ── categorical filters ──────────────────────────────────────────────────
    category = forms.ChoiceField(
        required=False,
        label="Sound Category",
        choices=_category_choices,
        widget=forms.Select(attrs={"class": _SELECT, "id": "filter-category"}),
    )

    severity = forms.ChoiceField(
        required=False,
        label="Severity",
        choices=[
            ("", "All Severities"),
            ("Critical", "Critical"),
            ("High", "High"),
            ("Medium", "Medium"),
            ("Low", "Low"),
            ("Informational", "Informational"),
        ],
        widget=forms.Select(attrs={"class": _SELECT, "id": "filter-severity"}),
    )

    quality = forms.ChoiceField(
        required=False,
        label="Audio Quality",
        choices=[
            ("", "All Quality Levels"),
            (AudioEvent.QUALITY_GOOD, "Good"),
            (AudioEvent.QUALITY_ACCEPTABLE, "Acceptable"),
            (AudioEvent.QUALITY_POOR, "Poor"),
            (AudioEvent.QUALITY_UNUSABLE, "Unusable"),
        ],
        widget=forms.Select(attrs={"class": _SELECT, "id": "filter-quality"}),
    )

    review_status = forms.ChoiceField(
        required=False,
        label="Review Status",
        choices=[
            ("", "Any Review Status"),
            ("reviewed", "Reviewed"),
            ("unreviewed", "Unreviewed / Pending"),
            (AudioEvent.STATUS_UPLOADED, "Uploaded"),
            (AudioEvent.STATUS_CLASSIFIED, "Classified"),
            (AudioEvent.STATUS_UNCERTAIN, "Uncertain"),
            (AudioEvent.STATUS_ALERT_GENERATED, "Alert Generated"),
            (AudioEvent.STATUS_MANUAL_REVIEW, "Manual Review"),
            (AudioEvent.STATUS_REVIEWED, "Reviewed"),
            (AudioEvent.STATUS_CLOSED, "Closed"),
        ],
        widget=forms.Select(attrs={"class": _SELECT, "id": "filter-review-status"}),
    )

    # ── date range ───────────────────────────────────────────────────────────
    start_date = forms.DateField(
        required=False,
        label="Uploaded from",
        widget=forms.DateInput(
            format="%Y-%m-%d",
            attrs={"type": "date", "class": _INPUT, "id": "filter-start-date"},
        ),
        input_formats=["%Y-%m-%d"],
    )

    end_date = forms.DateField(
        required=False,
        label="Uploaded to",
        widget=forms.DateInput(
            format="%Y-%m-%d",
            attrs={"type": "date", "class": _INPUT, "id": "filter-end-date"},
        ),
        input_formats=["%Y-%m-%d"],
    )

    # ── confidence range  (0.0 – 1.0) ────────────────────────────────────────
    min_confidence = forms.FloatField(
        required=False,
        label="Min confidence",
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            "placeholder": "0.00",
            "step": "0.01",
            "class": _INPUT,
            "id": "filter-min-confidence",
        }),
    )

    max_confidence = forms.FloatField(
        required=False,
        label="Max confidence",
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            "placeholder": "1.00",
            "step": "0.01",
            "class": _INPUT,
            "id": "filter-max-confidence",
        }),
    )

    # ── admin-only: filter by uploader ────────────────────────────────────────
    user = forms.CharField(
        required=False,
        label="Uploaded by (username or User ID)",
        widget=forms.TextInput(attrs={
            "placeholder": "username or USR-…",
            "class": _INPUT,
            "id": "filter-user",
            "autocomplete": "off",
        }),
    )

    def clean(self):
        """Cross-field validation: start_date ≤ end_date; min_conf ≤ max_conf."""
        cleaned = super().clean()
        start = cleaned.get("start_date")
        end = cleaned.get("end_date")
        if start and end and start > end:
            self.add_error("end_date", "'Uploaded to' must be on or after 'Uploaded from'.")

        min_c = cleaned.get("min_confidence")
        max_c = cleaned.get("max_confidence")
        if min_c is not None and max_c is not None and min_c > max_c:
            self.add_error("max_confidence", "Max confidence must be ≥ min confidence.")

        return cleaned

    def to_query_filters(self) -> dict:
        """Return a filter dict suitable for ``list_audio_events``.

        Only includes non-empty values so the query service never sees blank
        strings that would accidentally match every row.
        """
        if not self.is_valid():
            return {}
        cd = self.cleaned_data
        filters: dict = {}
        if cd.get("search"):
            filters["search"] = cd["search"]
        if cd.get("category"):
            filters["category"] = cd["category"]
        if cd.get("severity"):
            filters["severity"] = cd["severity"]
        if cd.get("quality"):
            filters["quality"] = cd["quality"]
        if cd.get("review_status"):
            filters["review_status"] = cd["review_status"]
        if cd.get("start_date"):
            filters["start_date"] = cd["start_date"]
        if cd.get("end_date"):
            filters["end_date"] = cd["end_date"]
        if cd.get("min_confidence") is not None:
            filters["min_confidence"] = cd["min_confidence"]
        if cd.get("max_confidence") is not None:
            filters["max_confidence"] = cd["max_confidence"]
        if cd.get("user"):
            filters["user"] = cd["user"]
        return filters
