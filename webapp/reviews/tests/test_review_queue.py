from datetime import timedelta
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from audio_events.models import AudioEvent
from predictions.models import ComparisonResult, GTMPrediction, PythonPrediction
from reviews.models import ReviewRecord
from reviews.services import get_manual_review_queue, get_routing_reason

User = get_user_model()


class ReviewQueueViewTestCase(TestCase):
    def setUp(self):
        # Create users for all roles
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
        self.superuser = User.objects.create_superuser(
            username="user_super", password="password123", email="admin@example.com"
        )

        self.url = reverse("reviews:queue")

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

    def test_authorized_roles_can_access_queue(self):
        authorized_users = [self.reviewer_user, self.admin_user, self.superuser]
        for user in authorized_users:
            self.client.login(username=user.username, password="password123")
            response = self.client.get(self.url)
            self.assertEqual(response.status_code, 200, f"User {user.username} should get 200")
            self.assertTemplateUsed(response, "reviews/queue.html")
            self.client.logout()

    def test_queue_filters_status_and_sorts_oldest_first(self):
        now = timezone.now()

        # Create older manual review event
        event_old = AudioEvent.objects.create(
            filename="old_event.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=22050,
            file_size=50000,
            status=AudioEvent.STATUS_MANUAL_REVIEW,
            category="Gunshot",
            storage_path="/media/old_event.wav",
            hash="hash_old_123",
        )
        event_old.uploaded_at = now - timedelta(hours=5)
        event_old.save()

        # Create newer manual review event
        event_new = AudioEvent.objects.create(
            filename="new_event.wav",
            format="WAV",
            duration=4.0,
            sampling_rate=22050,
            file_size=60000,
            status=AudioEvent.STATUS_MANUAL_REVIEW,
            category="Glass Breaking",
            storage_path="/media/new_event.wav",
            hash="hash_new_456",
        )
        event_new.uploaded_at = now - timedelta(hours=1)
        event_new.save()

        # Create event with Classified status (should NOT appear in queue)
        AudioEvent.objects.create(
            filename="classified_event.wav",
            format="WAV",
            duration=5.0,
            sampling_rate=22050,
            file_size=70000,
            status=AudioEvent.STATUS_CLASSIFIED,
            category="Vehicle Horn",
            storage_path="/media/classified.wav",
            hash="hash_classified_789",
        )

        self.client.login(username=self.reviewer_user.username, password="password123")
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)

        queue_events = response.context["queue_events"]
        self.assertEqual(len(queue_events), 2)
        # Verify oldest first ordering
        self.assertEqual(queue_events[0].audio_id, event_old.audio_id)
        self.assertEqual(queue_events[1].audio_id, event_new.audio_id)

    def test_routing_reasons_derivation(self):
        # 1. Poor quality event
        event_poor = AudioEvent.objects.create(
            filename="noisy.wav",
            format="WAV",
            duration=2.0,
            sampling_rate=22050,
            file_size=30000,
            status=AudioEvent.STATUS_MANUAL_REVIEW,
            quality=AudioEvent.QUALITY_POOR,
            category="Machinery Fault",
            storage_path="/media/noisy.wav",
            hash="hash_poor_1",
        )
        self.assertIn("poor quality", get_routing_reason(event_poor))

        # 2. Model disagreement event
        event_disagree = AudioEvent.objects.create(
            filename="conflict.wav",
            format="WAV",
            duration=2.0,
            sampling_rate=22050,
            file_size=30000,
            status=AudioEvent.STATUS_MANUAL_REVIEW,
            category="Panic Scream",
            storage_path="/media/conflict.wav",
            hash="hash_disagree_2",
        )
        PythonPrediction.objects.create(
            audio_event=event_disagree,
            predicted_class="Panic Scream",
            confidence_scores={"Panic Scream": 0.85},
            model_version="v1.0",
        )
        GTMPrediction.objects.create(
            audio_event=event_disagree,
            predicted_class="Background Noise",
            confidence_scores={"Background Noise": 0.80},
            model_version="gtm-1",
        )
        ComparisonResult.objects.create(
            audio_event=event_disagree,
            confidence_difference=0.05,
            top_two_margin=0.10,
            match_status=ComparisonResult.MATCH_DISAGREEMENT,
            is_uncertain=True,
        )
        self.assertIn("model disagreement", get_routing_reason(event_disagree))

        # 3. Overlapping sounds event
        event_overlap = AudioEvent.objects.create(
            filename="overlap.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=22050,
            file_size=40000,
            status=AudioEvent.STATUS_MANUAL_REVIEW,
            category="Animal Sound",
            storage_path="/media/overlap.wav",
            hash="hash_overlap_3",
        )
        ComparisonResult.objects.create(
            audio_event=event_overlap,
            confidence_difference=0.10,
            top_two_margin=0.05,
            match_status=ComparisonResult.MATCH_ACCEPTABLE,
            is_uncertain=False,
            is_overlapping=True,
        )
        self.assertIn("overlapping sounds", get_routing_reason(event_overlap))

        # 4. Low confidence event
        event_low_conf = AudioEvent.objects.create(
            filename="low_conf.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=22050,
            file_size=40000,
            status=AudioEvent.STATUS_MANUAL_REVIEW,
            category="Aggression or Violent Conflict",
            storage_path="/media/low_conf.wav",
            hash="hash_lowconf_4",
        )
        PythonPrediction.objects.create(
            audio_event=event_low_conf,
            predicted_class="Aggression or Violent Conflict",
            confidence_scores={"Aggression or Violent Conflict": 0.45},
            model_version="v1.0",
        )
        ComparisonResult.objects.create(
            audio_event=event_low_conf,
            confidence_difference=0.02,
            top_two_margin=0.03,
            match_status=ComparisonResult.MATCH_WEAK,
            is_uncertain=True,
        )
        self.assertIn("low confidence", get_routing_reason(event_low_conf))
