from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase
from django.urls import reverse

from audio_events.models import AudioEvent
from audio_events.services import get_paginated_audio_events

User = get_user_model()


class TimelineComponentTestCase(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user(username="timeline_user", password="password123")

        # Create audio events with distinct timestamps and categories
        self.event1 = AudioEvent.objects.create(
            filename="gunshot.wav",
            format="WAV",
            duration=2.5,
            sampling_rate=16000,
            file_size=80000,
            uploaded_by=self.user,
            category="Gunshot",
            storage_path="/media/gunshot.wav",
            hash="hash_1",
        )
        self.event2 = AudioEvent.objects.create(
            filename="glass.wav",
            format="WAV",
            duration=3.0,
            sampling_rate=16000,
            file_size=90000,
            uploaded_by=self.user,
            category="Glass Breaking",
            storage_path="/media/glass.wav",
            hash="hash_2",
        )
        self.event3 = AudioEvent.objects.create(
            filename="machinery.wav",
            format="WAV",
            duration=5.0,
            sampling_rate=16000,
            file_size=150000,
            uploaded_by=self.user,
            category="Machinery Fault",
            storage_path="/media/machinery.wav",
            hash="hash_3",
        )

    def test_severity_properties(self):
        self.assertEqual(self.event1.severity, "Critical")
        self.assertEqual(self.event2.severity, "High")
        self.assertEqual(self.event3.severity, "Medium")

    def test_paginated_audio_events_ordering(self):
        request = self.factory.get("/")
        page = get_paginated_audio_events(request, per_page=10)
        events = list(page.object_list)

        # Should be ordered most recent first
        self.assertEqual(events[0], self.event3)
        self.assertEqual(events[1], self.event2)
        self.assertEqual(events[2], self.event1)

    def test_paginated_audio_events_pagination(self):
        request = self.factory.get("/?page=1")
        page = get_paginated_audio_events(request, per_page=2)
        self.assertEqual(len(page.object_list), 2)
        self.assertTrue(page.has_next())

        request_p2 = self.factory.get("/?page=2")
        page2 = get_paginated_audio_events(request_p2, per_page=2)
        self.assertEqual(len(page2.object_list), 1)
        self.assertFalse(page2.has_next())

    def test_dashboard_renders_timeline(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "System Sound Detections Timeline")
        self.assertContains(response, "gunshot.wav")
        self.assertContains(response, "Critical Severity")
