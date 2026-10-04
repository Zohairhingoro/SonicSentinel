from django.contrib.auth import get_user_model
from django.test import TestCase

from audio_events.models import AudioEvent
from predictions.models import PythonPrediction, GTMPrediction
from reviews.models import ReviewRecord

User = get_user_model()


class ReviewRecordModelTestCase(TestCase):
    def setUp(self):
        self.reviewer = User.objects.create_user(
            username="reviewer_user", password="password123", role=User.ROLE_AUDIO_REVIEWER
        )
        self.audio_event = AudioEvent.objects.create(
            filename="conflict_recording.wav",
            format="WAV",
            duration=5.0,
            sampling_rate=22050,
            file_size=150000,
            uploaded_by=self.reviewer,
            category="Background Noise",
            storage_path="/media/conflict_recording.wav",
            hash="review_test_hash_1",
        )
        # Create original predictions
        self.py_pred = PythonPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Background Noise",
            confidence_scores={"Background Noise": 0.55, "Panic Scream": 0.45},
            model_version="v1.0.0",
        )
        self.gtm_pred = GTMPrediction.objects.create(
            audio_event=self.audio_event,
            predicted_class="Background Noise",
            confidence_scores={"Background Noise": 0.60, "Panic Scream": 0.40},
            model_version="gtm-v1",
        )

    def test_review_record_creation(self):
        review = ReviewRecord.objects.create(
            audio_event=self.audio_event,
            reviewer=self.reviewer,
            decision="Panic Scream",
            comment="Re-analyzed audio: panic scream audible behind ambient noise",
            recommended_action="Escalate to security team immediately",
        )

        self.assertEqual(review.audio_event, self.audio_event)
        self.assertEqual(review.reviewer, self.reviewer)
        self.assertEqual(review.decision, "Panic Scream")
        self.assertIn("panic scream audible", review.comment)
        self.assertEqual(self.audio_event.review_records.first(), review)
        self.assertIn("Panic Scream", str(review))

    def test_review_record_does_not_mutate_predictions(self):
        # Initial prediction state
        orig_py_class = self.py_pred.predicted_class
        orig_gtm_class = self.gtm_pred.predicted_class
        orig_py_scores = dict(self.py_pred.confidence_scores)

        # Create review record overriding decision
        ReviewRecord.objects.create(
            audio_event=self.audio_event,
            reviewer=self.reviewer,
            decision="Gunshot",
            comment="Corrected classification to Gunshot",
        )

        # Refresh predictions from DB
        self.py_pred.refresh_from_db()
        self.gtm_pred.refresh_from_db()

        # Predictions MUST remain completely un-mutated
        self.assertEqual(self.py_pred.predicted_class, orig_py_class)
        self.assertEqual(self.gtm_pred.predicted_class, orig_gtm_class)
        self.assertEqual(self.py_pred.confidence_scores, orig_py_scores)
        self.assertEqual(PythonPrediction.objects.count(), 1)
        self.assertEqual(GTMPrediction.objects.count(), 1)
        self.assertEqual(ReviewRecord.objects.count(), 1)
