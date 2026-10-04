from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from audio_events.models import AudioEvent
from predictions.models import ComparisonResult, GTMPrediction, PythonPrediction
from reviews.models import ReviewRecord

User = get_user_model()


class ReviewDetailViewTestCase(TestCase):
    def setUp(self):
        # Create users for role-based access testing
        self.normal_user = User.objects.create_user(
            username="user_normal", password="password123", role=User.ROLE_NORMAL_USER
        )
        self.reviewer_user = User.objects.create_user(
            username="user_reviewer", password="password123", role=User.ROLE_AUDIO_REVIEWER
        )
        self.security_user = User.objects.create_user(
            username="user_security", password="password123", role=User.ROLE_SECURITY_OPERATOR
        )
        self.maint_user = User.objects.create_user(
            username="user_maint", password="password123", role=User.ROLE_MAINTENANCE_OPERATOR
        )
        self.admin_user = User.objects.create_user(
            username="user_admin", password="password123", role=User.ROLE_ADMINISTRATOR
        )

        # Create sample AudioEvent flagged for Manual Review
        self.event = AudioEvent.objects.create(
            filename="sample_review.wav",
            format="WAV",
            duration=4.5,
            sampling_rate=22050,
            file_size=80000,
            status=AudioEvent.STATUS_MANUAL_REVIEW,
            category="Gunshot",
            storage_path="/media/sample_review.wav",
            hash="hash_review_detail_123",
        )

        # Create original PythonPrediction
        self.py_pred = PythonPrediction.objects.create(
            audio_event=self.event,
            predicted_class="Gunshot",
            confidence=0.88,
            confidence_scores={
                "Gunshot": 0.88,
                "Background Noise": 0.07,
                "Glass Breaking": 0.05,
            },
            model_version="v1.0.0",
        )

        # Create original GTMPrediction
        self.gtm_pred = GTMPrediction.objects.create(
            audio_event=self.event,
            predicted_class="Glass Breaking",
            confidence_scores={
                "Glass Breaking": 0.65,
                "Gunshot": 0.25,
                "Background Noise": 0.10,
            },
            model_version="gtm-v1.0",
        )

        # Create ComparisonResult
        self.comparison = ComparisonResult.objects.create(
            audio_event=self.event,
            confidence_difference=0.23,
            top_two_margin=0.81,
            match_status=ComparisonResult.MATCH_DISAGREEMENT,
            is_uncertain=True,
        )

        self.url = reverse("reviews:detail", kwargs={"audio_id": self.event.audio_id})

    def test_unauthenticated_redirects_to_login(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_unauthorized_roles_receive_403(self):
        unauthorized_users = [self.normal_user, self.security_user, self.maint_user]
        for user in unauthorized_users:
            self.client.login(username=user.username, password="password123")
            response = self.client.get(self.url)
            self.assertEqual(response.status_code, 403, f"User {user.username} should get 403")
            self.client.logout()

    def test_authorized_roles_can_access_detail(self):
        authorized_users = [self.reviewer_user, self.admin_user]
        for user in authorized_users:
            self.client.login(username=user.username, password="password123")
            response = self.client.get(self.url)
            self.assertEqual(response.status_code, 200, f"User {user.username} should get 200")
            self.assertTemplateUsed(response, "reviews/detail.html")
            self.assertEqual(response.context["event"].audio_id, self.event.audio_id)
            self.assertEqual(response.context["comparison"], self.comparison)
            self.assertIn("categories", response.context)
            self.client.logout()

    def test_non_existent_audio_id_returns_404(self):
        self.client.login(username=self.reviewer_user.username, password="password123")
        invalid_url = reverse("reviews:detail", kwargs={"audio_id": "NON_EXISTENT_ID"})
        response = self.client.get(invalid_url)
        self.assertEqual(response.status_code, 404)

    def test_submit_review_decision_creates_record_and_updates_status(self):
        self.client.login(username=self.reviewer_user.username, password="password123")

        post_data = {
            "decision": "Glass Breaking",
            "comment": "Confirmed glass breaking sound upon acoustic spectral review.",
            "recommended_action": "Dispatch security operator for perimeter check.",
        }

        response = self.client.post(self.url, data=post_data)
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.url)

        # Verify AudioEvent status was updated to Reviewed
        self.event.refresh_from_db()
        self.assertEqual(self.event.status, AudioEvent.STATUS_REVIEWED)
        self.assertEqual(self.event.category, "Glass Breaking")
        self.assertEqual(self.event.severity, "High")

        # Verify ReviewRecord was created
        reviews = ReviewRecord.objects.filter(audio_event=self.event)
        self.assertEqual(reviews.count(), 1)
        record = reviews.first()
        self.assertEqual(record.decision, "Glass Breaking")
        self.assertEqual(record.reviewer, self.reviewer_user)
        self.assertEqual(record.comment, post_data["comment"])
        self.assertEqual(record.recommended_action, post_data["recommended_action"])

        # CRITICAL CONSTRAINT: Verify original predictions are preserved intact
        self.assertTrue(PythonPrediction.objects.filter(pk=self.py_pred.pk).exists())
        self.assertTrue(GTMPrediction.objects.filter(pk=self.gtm_pred.pk).exists())
        py_db = PythonPrediction.objects.get(pk=self.py_pred.pk)
        self.assertEqual(py_db.predicted_class, "Gunshot")
        self.assertEqual(py_db.confidence, 0.88)
        gtm_db = GTMPrediction.objects.get(pk=self.gtm_pred.pk)
        self.assertEqual(gtm_db.predicted_class, "Glass Breaking")

    def test_invalid_decision_shows_error(self):
        self.client.login(username=self.reviewer_user.username, password="password123")

        post_data = {
            "decision": "Invalid Category Name",
            "comment": "Testing invalid submission",
        }

        response = self.client.post(self.url, data=post_data)
        self.assertEqual(response.status_code, 200)

        # Event status should NOT change to Reviewed
        self.event.refresh_from_db()
        self.assertEqual(self.event.status, AudioEvent.STATUS_MANUAL_REVIEW)
        self.assertEqual(ReviewRecord.objects.filter(audio_event=self.event).count(), 0)
