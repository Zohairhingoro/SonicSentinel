from datetime import timedelta
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils.timezone import now

from audio_events.models import AudioEvent
from audio_events.services import list_audio_events
from predictions.models import PythonPrediction
from reviews.models import ReviewRecord

User = get_user_model()


class ListAudioEventsServicesTestCase(TestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(username="user1", user_id="USR-001", password="password123")
        self.user2 = User.objects.create_user(username="user2", user_id="USR-002", password="password123")

        self.time_now = now()
        self.time_past_2d = self.time_now - timedelta(days=2)
        self.time_past_5d = self.time_now - timedelta(days=5)

        # Event 1: Gunshot, Critical, Good, User 1, 5 days ago
        self.event1 = AudioEvent.objects.create(
            filename="gunshot.wav",
            format="WAV",
            duration=2.5,
            sampling_rate=16000,
            file_size=80000,
            uploaded_by=self.user1,
            category="Gunshot",
            quality="Good",
            status=AudioEvent.STATUS_CLASSIFIED,
            storage_path="/media/gunshot.wav",
            hash="hash_service_1",
        )
        AudioEvent.objects.filter(id=self.event1.id).update(uploaded_at=self.time_past_5d)
        self.event1.refresh_from_db()

        # Prediction for Event 1: High confidence
        self.pred1 = PythonPrediction.objects.create(
            audio_event=self.event1,
            predicted_class="Gunshot",
            confidence_scores={"Gunshot": 0.95, "Background Noise": 0.05},
            model_version="v1.0.0",
        )

        # Event 2: Glass Breaking, High, Acceptable, User 2, 2 days ago
        self.event2 = AudioEvent.objects.create(
            filename="glass.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=16000,
            file_size=90000,
            uploaded_by=self.user2,
            category="Glass Breaking",
            quality="Acceptable",
            status=AudioEvent.STATUS_REVIEWED,
            storage_path="/media/glass.wav",
            hash="hash_service_2",
        )
        AudioEvent.objects.filter(id=self.event2.id).update(uploaded_at=self.time_past_2d)
        self.event2.refresh_from_db()

        # Prediction for Event 2: Medium confidence
        self.pred2 = PythonPrediction.objects.create(
            audio_event=self.event2,
            predicted_class="Glass Breaking",
            confidence_scores={"Glass Breaking": 0.75, "Background Noise": 0.25},
            model_version="v1.0.0",
        )

        # Review record for Event 2
        self.review2 = ReviewRecord.objects.create(
            audio_event=self.event2,
            reviewer=self.user1,
            decision="Glass Breaking",
            comment="Confirmed glass breaking sound",
        )

        # Event 3: Background Noise, Informational, Poor, User 1, recent
        self.event3 = AudioEvent.objects.create(
            filename="noise.wav",
            format="WAV",
            duration=4.0,
            sampling_rate=16000,
            file_size=100000,
            uploaded_by=self.user1,
            category="Background Noise",
            quality="Poor",
            status=AudioEvent.STATUS_UPLOADED,
            storage_path="/media/noise.wav",
            hash="hash_service_3",
        )

        # Prediction for Event 3: Low confidence
        self.pred3 = PythonPrediction.objects.create(
            audio_event=self.event3,
            predicted_class="Background Noise",
            confidence_scores={"Background Noise": 0.45, "Vehicle Horn": 0.30},
            model_version="v1.0.0",
        )

    def test_filter_by_category(self):
        qs = list_audio_events({"category": "Gunshot"})
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs.first(), self.event1)

        qs_multi = list_audio_events({"category": ["Gunshot", "Glass Breaking"]})
        self.assertEqual(qs_multi.count(), 2)

    def test_filter_by_severity(self):
        qs_critical = list_audio_events({"severity": "Critical"})
        self.assertEqual(qs_critical.count(), 1)
        self.assertEqual(qs_critical.first(), self.event1)

        qs_high = list_audio_events({"severity": "High"})
        self.assertEqual(qs_high.count(), 1)
        self.assertEqual(qs_high.first(), self.event2)

    def test_filter_by_quality(self):
        qs_good = list_audio_events({"quality": "Good"})
        self.assertEqual(qs_good.count(), 1)
        self.assertEqual(qs_good.first(), self.event1)

        qs_poor = list_audio_events({"quality": "Poor"})
        self.assertEqual(qs_poor.count(), 1)
        self.assertEqual(qs_poor.first(), self.event3)

    def test_filter_by_review_status(self):
        qs_reviewed = list_audio_events({"review_status": "reviewed"})
        self.assertEqual(qs_reviewed.count(), 1)
        self.assertEqual(qs_reviewed.first(), self.event2)

        qs_unreviewed = list_audio_events({"review_status": "unreviewed"})
        self.assertEqual(qs_unreviewed.count(), 2)
        self.assertNotIn(self.event2, qs_unreviewed)

    def test_filter_by_user(self):
        qs_user1 = list_audio_events({"user": self.user1})
        self.assertEqual(qs_user1.count(), 2)
        self.assertIn(self.event1, qs_user1)
        self.assertIn(self.event3, qs_user1)

        qs_user2_id = list_audio_events({"user_id": "USR-002"})
        self.assertEqual(qs_user2_id.count(), 1)
        self.assertEqual(qs_user2_id.first(), self.event2)

    def test_filter_by_date_range(self):
        start = self.time_now - timedelta(days=3)
        qs_recent = list_audio_events({"start_date": start})
        self.assertEqual(qs_recent.count(), 2)
        self.assertIn(self.event2, qs_recent)
        self.assertIn(self.event3, qs_recent)
        self.assertNotIn(self.event1, qs_recent)

        end = self.time_now - timedelta(days=3)
        qs_older = list_audio_events({"end_date": end})
        self.assertEqual(qs_older.count(), 1)
        self.assertEqual(qs_older.first(), self.event1)

    def test_filter_by_confidence_range(self):
        qs_high_conf = list_audio_events({"min_confidence": 0.80})
        self.assertEqual(qs_high_conf.count(), 1)
        self.assertEqual(qs_high_conf.first(), self.event1)

        qs_mid_range = list_audio_events({"min_confidence": 0.50, "max_confidence": 0.80})
        self.assertEqual(qs_mid_range.count(), 1)
        self.assertEqual(qs_mid_range.first(), self.event2)

    def test_combined_filters(self):
        filters = {
            "user": self.user1,
            "min_confidence": 0.90,
            "quality": "Good",
        }
        qs = list_audio_events(filters)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs.first(), self.event1)
