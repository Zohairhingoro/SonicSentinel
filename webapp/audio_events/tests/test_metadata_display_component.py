import io
import math
import struct
import wave
from django.core.files.uploadedfile import SimpleUploadedFile
from django.template.loader import render_to_string
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from audio_events.models import AudioEvent


def create_test_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate minimal valid WAV audio file bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        num_samples = int(duration_seconds * sample_rate)
        samples = [struct.pack("<h", int(0.5 * 32767.0 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))) for i in range(num_samples)]
        wav_file.writeframes(b"".join(samples))
    buffer.seek(0)
    return buffer.read()


class MetadataDisplayComponentTestCase(TestCase):
    def setUp(self):
        self.now = timezone.now()
        self.event = AudioEvent.objects.create(
            audio_id="AUD-META0001",
            filename="machinery_fault.wav",
            format="WAV",
            duration=5.4,
            sampling_rate=44100,
            channels=1,
            bit_depth=24,
            file_size=476280,
            category="Machinery Fault",
            quality=AudioEvent.QUALITY_ACCEPTABLE,
            storage_path="audio/AUD-META0001/machinery_fault.wav",
            hash="hash_metadata_test_12345",
        )

    def test_render_metadata_display_with_event_model(self):
        html = render_to_string("components/metadata_display.html", {"event": self.event})

        # Verify all 9 mandatory extracted fields exist in output
        self.assertIn("machinery_fault.wav", html)
        self.assertIn("WAV", html)
        self.assertIn("5.4 s", html)
        self.assertIn("44100 Hz", html)
        self.assertIn("1 (Mono)", html)
        self.assertIn("24 bit", html)
        self.assertIn("476280 bytes", html)
        self.assertIn("Acceptable", html)
        self.assertIn(str(self.event.uploaded_at.year), html)
        self.assertIn("hash_metadata_test_12345", html)
        self.assertIn("AUD-META0001", html)

    def test_render_metadata_display_with_metadata_dict(self):
        meta = {
            "filename": "glass_breaking_test.wav",
            "format": "wav",
            "duration": 2.1,
            "sampling_rate": 22050,
            "channels": 2,
            "bit_depth": 16,
            "file_size": 185600,
            "quality": "Good",
            "upload_timestamp": "2026-09-25 19:55:00",
            "hash": "hash_dict_test_67890",
            "audio_id": "AUD-DICT0002",
        }
        html = render_to_string("components/metadata_display.html", {"metadata": meta, "title": "Validation Metadata"})

        self.assertIn("glass_breaking_test.wav", html)
        self.assertIn("WAV", html)
        self.assertIn("2.1 s", html)
        self.assertIn("22050 Hz", html)
        self.assertIn("2 (Stereo)", html)
        self.assertIn("16 bit", html)
        self.assertIn("185600 bytes", html)
        self.assertIn("Good", html)
        self.assertIn("2026-09-25 19:55:00", html)
        self.assertIn("hash_dict_test_67890", html)

    def test_upload_page_uses_metadata_display_component(self):
        wav_bytes = create_test_wav(duration_seconds=1.2, sample_rate=16000)
        uploaded = SimpleUploadedFile("sample_ingest.wav", wav_bytes, content_type="audio/wav")

        response = self.client.post(reverse("audio_events:single_upload"), {"audio_file": uploaded})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "audio_events/upload.html")
        self.assertTemplateUsed(response, "components/metadata_display.html")
        self.assertContains(response, "Validation Passed — Extracted Metadata")
        self.assertContains(response, "sample_ingest.wav")
        self.assertContains(response, "16000 Hz")
