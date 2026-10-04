from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase
from audio_events.models import AudioEvent, Segment

User = get_user_model()


class AudioEventModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="password123")

    def test_create_audio_event(self):
        event = AudioEvent.objects.create(
            filename="sample.wav",
            format="wav",
            duration=5.0,
            sampling_rate=22050,
            channels=1,
            bit_depth=16,
            file_size=102400,
            uploaded_by=self.user,
            storage_path="media/audio/sample.wav",
            hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )

        self.assertTrue(event.audio_id.startswith("AUD-"))
        self.assertEqual(event.status, AudioEvent.STATUS_UPLOADED)
        self.assertEqual(event.quality, AudioEvent.QUALITY_GOOD)
        self.assertEqual(event.uploaded_by, self.user)
        self.assertIn("sample.wav", str(event))

    def test_unique_hash_constraint(self):
        hash_val = "1111111111111111111111111111111111111111111111111111111111111111"
        AudioEvent.objects.create(
            filename="clip1.wav",
            format="wav",
            duration=2.0,
            sampling_rate=22050,
            channels=1,
            file_size=5000,
            storage_path="media/clip1.wav",
            hash=hash_val,
        )

        with self.assertRaises(IntegrityError):
            AudioEvent.objects.create(
                filename="clip2.wav",
                format="wav",
                duration=2.0,
                sampling_rate=22050,
                channels=1,
                file_size=5000,
                storage_path="media/clip2.wav",
                hash=hash_val,
            )

    def test_segment_creation(self):
        event = AudioEvent.objects.create(
            filename="long_audio.wav",
            format="wav",
            duration=10.0,
            sampling_rate=22050,
            channels=1,
            file_size=200000,
            storage_path="media/long_audio.wav",
            hash="2222222222222222222222222222222222222222222222222222222222222222",
        )

        seg1 = Segment.objects.create(
            audio_event=event,
            segment_index=0,
            start_timestamp=0.0,
            end_timestamp=3.0,
        )
        seg2 = Segment.objects.create(
            audio_event=event,
            segment_index=1,
            start_timestamp=2.0,
            end_timestamp=5.0,
        )

        self.assertEqual(event.segments.count(), 2)
        self.assertEqual(list(event.segments.all()), [seg1, seg2])
