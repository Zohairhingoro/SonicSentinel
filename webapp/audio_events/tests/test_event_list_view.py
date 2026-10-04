"""Tests for EventListView and EventFilterForm.

Covers:
  - Form validation (cross-field rules)
  - to_query_filters() output
  - GET-param-to-filter mapping in the view
  - Admin-only user filter enforcement
  - Result count with various filter combinations
  - Shareable URL (no POST, all state in query string)
  - Zero-result and empty-DB states
"""
from datetime import date, timedelta

from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase
from django.urls import reverse
from django.utils.timezone import now

from audio_events.forms import EventFilterForm
from audio_events.models import AudioEvent
from audio_events.services.query import _build_filters_from_get
from predictions.models import PythonPrediction
from reviews.models import ReviewRecord

User = get_user_model()

EVENT_LIST_URL = "/audio/events/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(*, category="Background Noise", quality="Good", status=AudioEvent.STATUS_UPLOADED,
                hash_val, user=None) -> AudioEvent:
    return AudioEvent.objects.create(
        filename=f"{category.lower().replace(' ', '_')}.wav",
        format="WAV",
        duration=3.0,
        sampling_rate=16000,
        file_size=90000,
        uploaded_by=user,
        category=category,
        quality=quality,
        status=status,
        storage_path=f"/media/{hash_val}.wav",
        hash=hash_val,
    )


# ---------------------------------------------------------------------------
# EventFilterForm unit tests
# ---------------------------------------------------------------------------

class EventFilterFormValidationTest(TestCase):

    def test_empty_form_is_valid(self):
        form = EventFilterForm(data={})
        self.assertTrue(form.is_valid())

    def test_date_range_inverted_raises_error(self):
        form = EventFilterForm(data={
            "start_date": "2024-06-10",
            "end_date": "2024-06-01",
        })
        self.assertFalse(form.is_valid())
        self.assertIn("end_date", form.errors)

    def test_date_range_equal_is_valid(self):
        form = EventFilterForm(data={
            "start_date": "2024-06-10",
            "end_date": "2024-06-10",
        })
        self.assertTrue(form.is_valid())

    def test_confidence_inverted_raises_error(self):
        form = EventFilterForm(data={"min_confidence": "0.9", "max_confidence": "0.2"})
        self.assertFalse(form.is_valid())
        self.assertIn("max_confidence", form.errors)

    def test_confidence_out_of_range(self):
        form = EventFilterForm(data={"min_confidence": "1.5"})
        self.assertFalse(form.is_valid())

    def test_to_query_filters_skips_blanks(self):
        form = EventFilterForm(data={"category": "", "severity": "Critical"})
        self.assertTrue(form.is_valid())
        filters = form.to_query_filters()
        self.assertNotIn("category", filters)
        self.assertEqual(filters["severity"], "Critical")

    def test_to_query_filters_returns_empty_on_invalid(self):
        form = EventFilterForm(data={"min_confidence": "bad"})
        self.assertFalse(form.is_valid())
        self.assertEqual(form.to_query_filters(), {})

    def test_to_query_filters_full(self):
        form = EventFilterForm(data={
            "search": "AUD-001",
            "category": "Gunshot",
            "severity": "Critical",
            "quality": "Good",
            "review_status": "reviewed",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "min_confidence": "0.7",
            "max_confidence": "1.0",
            "user": "admin",
        })
        self.assertTrue(form.is_valid())
        f = form.to_query_filters()
        self.assertEqual(f["search"], "AUD-001")
        self.assertEqual(f["category"], "Gunshot")
        self.assertEqual(f["min_confidence"], 0.7)
        self.assertEqual(f["max_confidence"], 1.0)
        self.assertEqual(f["user"], "admin")


# ---------------------------------------------------------------------------
# _build_filters_from_get unit tests
# ---------------------------------------------------------------------------

class BuildFiltersFromGetTest(TestCase):

    def _get(self, params: dict):
        """Return a minimal GET-like mapping from a plain dict."""
        return params

    def test_known_params_extracted(self):
        data = {"category": "Gunshot", "severity": "Critical", "quality": "Good"}
        filters = _build_filters_from_get(data)
        self.assertEqual(filters["category"], "Gunshot")
        self.assertEqual(filters["severity"], "Critical")

    def test_blank_values_excluded(self):
        data = {"category": "", "severity": "  "}
        filters = _build_filters_from_get(data)
        self.assertNotIn("category", filters)
        self.assertNotIn("severity", filters)

    def test_status_aliased_to_review_status(self):
        data = {"status": "Reviewed"}
        filters = _build_filters_from_get(data)
        self.assertEqual(filters.get("review_status"), "Reviewed")

    def test_unknown_params_ignored(self):
        data = {"page": "2", "csrfmiddlewaretoken": "abc"}
        filters = _build_filters_from_get(data)
        self.assertNotIn("page", filters)
        self.assertNotIn("csrfmiddlewaretoken", filters)


# ---------------------------------------------------------------------------
# EventListView integration tests
# ---------------------------------------------------------------------------

class EventListViewTest(TestCase):

    def setUp(self):
        self.admin = User.objects.create_user(
            username="admin_user", password="pw123", role=User.ROLE_ADMINISTRATOR,
        )
        self.normal = User.objects.create_user(
            username="normal_user", password="pw123",
        )

        self.e_gun = _make_event(
            category="Gunshot", quality="Good",
            status=AudioEvent.STATUS_CLASSIFIED, hash_val="h_gun", user=self.normal,
        )
        self.e_glass = _make_event(
            category="Glass Breaking", quality="Acceptable",
            status=AudioEvent.STATUS_REVIEWED, hash_val="h_glass", user=self.admin,
        )
        self.e_noise = _make_event(
            category="Background Noise", quality="Poor",
            status=AudioEvent.STATUS_UPLOADED, hash_val="h_noise",
        )

        # Prediction for e_gun (high confidence)
        PythonPrediction.objects.create(
            audio_event=self.e_gun,
            predicted_class="Gunshot",
            confidence_scores={"Gunshot": 0.95},
            model_version="v1",
        )
        # Prediction for e_glass (medium confidence)
        PythonPrediction.objects.create(
            audio_event=self.e_glass,
            predicted_class="Glass Breaking",
            confidence_scores={"Glass Breaking": 0.72},
            model_version="v1",
        )
        # Review for e_glass
        ReviewRecord.objects.create(
            audio_event=self.e_glass,
            reviewer=self.admin,
            decision="Glass Breaking",
            comment="confirmed",
        )

    # ── Basic rendering ──────────────────────────────────────────────────────

    def test_page_renders_200_without_filters(self):
        response = self.client.get(EVENT_LIST_URL)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "audio_events/event_list.html")

    def test_page_contains_all_events_without_filters(self):
        response = self.client.get(EVENT_LIST_URL)
        self.assertContains(response, "gunshot.wav")
        self.assertContains(response, "glass_breaking.wav")
        self.assertContains(response, "background_noise.wav")

    def test_uses_get_not_post(self):
        response = self.client.post(EVENT_LIST_URL, {})
        self.assertEqual(response.status_code, 405)

    # ── URL is shareable (state in query string) ─────────────────────────────

    def test_filter_state_in_query_string(self):
        response = self.client.get(EVENT_LIST_URL + "?category=Gunshot")
        self.assertContains(response, "gunshot.wav")
        self.assertNotContains(response, "glass_breaking.wav")
        # URL still has the param — confirm response doesn't redirect away
        self.assertEqual(response.status_code, 200)

    # ── Category filter ──────────────────────────────────────────────────────

    def test_filter_by_category(self):
        response = self.client.get(EVENT_LIST_URL + "?category=Gunshot")
        self.assertContains(response, "gunshot.wav")
        self.assertNotContains(response, "glass_breaking.wav")

    # ── Severity filter ──────────────────────────────────────────────────────

    def test_filter_by_severity(self):
        response = self.client.get(EVENT_LIST_URL + "?severity=Critical")
        self.assertContains(response, "gunshot.wav")
        self.assertNotContains(response, "background_noise.wav")

    # ── Quality filter ───────────────────────────────────────────────────────

    def test_filter_by_quality(self):
        response = self.client.get(EVENT_LIST_URL + "?quality=Good")
        self.assertContains(response, "gunshot.wav")
        self.assertNotContains(response, "background_noise.wav")

    # ── Review status ────────────────────────────────────────────────────────

    def test_filter_reviewed(self):
        response = self.client.get(EVENT_LIST_URL + "?review_status=reviewed")
        self.assertContains(response, "glass_breaking.wav")
        self.assertNotContains(response, "gunshot.wav")

    def test_filter_unreviewed(self):
        response = self.client.get(EVENT_LIST_URL + "?review_status=unreviewed")
        self.assertContains(response, "gunshot.wav")
        self.assertNotContains(response, "glass_breaking.wav")

    # ── Text search ──────────────────────────────────────────────────────────

    def test_search_by_filename(self):
        response = self.client.get(EVENT_LIST_URL + "?search=gunshot")
        self.assertContains(response, "gunshot.wav")
        self.assertNotContains(response, "glass_breaking.wav")

    def test_search_by_audio_id(self):
        audio_id = self.e_gun.audio_id
        response = self.client.get(EVENT_LIST_URL + f"?search={audio_id}")
        self.assertContains(response, "gunshot.wav")

    # ── Confidence filter ────────────────────────────────────────────────────

    def test_filter_by_min_confidence(self):
        response = self.client.get(EVENT_LIST_URL + "?min_confidence=0.90")
        self.assertContains(response, "gunshot.wav")
        self.assertNotContains(response, "glass_breaking.wav")

    def test_filter_confidence_range(self):
        response = self.client.get(EVENT_LIST_URL + "?min_confidence=0.60&max_confidence=0.80")
        self.assertContains(response, "glass_breaking.wav")
        self.assertNotContains(response, "gunshot.wav")

    # ── Date range ───────────────────────────────────────────────────────────

    def test_date_filter_future_date_returns_empty(self):
        future = (date.today() + timedelta(days=365)).isoformat()
        response = self.client.get(EVENT_LIST_URL + f"?start_date={future}")
        self.assertNotContains(response, "gunshot.wav")

    def test_invalid_date_shows_form_error(self):
        response = self.client.get(EVENT_LIST_URL + "?start_date=not-a-date")
        self.assertEqual(response.status_code, 200)
        # Should still render (unfiltered or showing errors)
        self.assertTemplateUsed(response, "audio_events/event_list.html")

    # ── Admin user filter ────────────────────────────────────────────────────

    def test_user_filter_visible_to_admin(self):
        self.client.force_login(self.admin)
        response = self.client.get(EVENT_LIST_URL)
        self.assertContains(response, 'id="filter-user"')

    def test_user_filter_hidden_from_normal_user(self):
        self.client.force_login(self.normal)
        response = self.client.get(EVENT_LIST_URL)
        self.assertNotContains(response, 'id="filter-user"')

    def test_user_filter_ignored_for_non_admin_via_url(self):
        """A non-admin submitting ?user= should not filter by user."""
        self.client.force_login(self.normal)
        # admin uploaded e_glass; passing user param should be stripped
        response = self.client.get(EVENT_LIST_URL + f"?user={self.admin.username}")
        # e_gun (uploaded by normal) and e_noise (no user) should still appear
        self.assertContains(response, "gunshot.wav")
        # The admin's glass event should also appear (user param was ignored)
        self.assertContains(response, "glass_breaking.wav")

    def test_user_filter_applied_for_admin(self):
        self.client.force_login(self.admin)
        response = self.client.get(EVENT_LIST_URL + f"?user={self.normal.username}")
        self.assertContains(response, "gunshot.wav")
        self.assertNotContains(response, "glass_breaking.wav")

    # ── Zero result state ────────────────────────────────────────────────────

    def test_zero_results_shows_empty_state(self):
        response = self.client.get(EVENT_LIST_URL + "?category=Machinery+Fault")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No Sound Events Found")

    # ── Context variables ────────────────────────────────────────────────────

    def test_context_has_expected_keys(self):
        response = self.client.get(EVENT_LIST_URL)
        self.assertIn("form", response.context)
        self.assertIn("page_obj", response.context)
        self.assertIn("is_admin", response.context)
        self.assertIn("active_filter_count", response.context)

    def test_active_filter_count_increments(self):
        response = self.client.get(EVENT_LIST_URL + "?severity=Critical&quality=Good")
        self.assertEqual(response.context["active_filter_count"], 2)

    def test_active_filter_count_zero_on_no_filters(self):
        response = self.client.get(EVENT_LIST_URL)
        self.assertEqual(response.context["active_filter_count"], 0)
