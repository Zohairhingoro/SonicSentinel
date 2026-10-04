import io
import math
import struct
import wave
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse


def create_synthetic_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
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


class SingleUploadPageTest(TestCase):
    def test_single_upload_get(self):
        response = self.client.get(reverse("audio_events:single_upload"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "audio_events/upload.html")
        self.assertContains(response, "Single Audio File Ingestion")
        self.assertContains(response, "Upload Audio Recording")

    def test_single_upload_post_valid_file_shows_metadata_summary(self):
        wav_bytes = create_synthetic_wav(duration_seconds=1.5, sample_rate=16000)
        uploaded = SimpleUploadedFile("sample_clip.wav", wav_bytes, content_type="audio/wav")

        response = self.client.post(reverse("audio_events:single_upload"), {"audio_file": uploaded})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "audio_events/upload.html")

        # Verify summary panel content
        self.assertContains(response, "Validation Passed")
        self.assertContains(response, "sample_clip.wav")
        self.assertContains(response, "16000 Hz")
        self.assertContains(response, "1.5 s")
        self.assertContains(response, "1 (Mono)")

    def test_single_upload_post_unsupported_format_shows_specific_rejection_reason(self):
        uploaded = SimpleUploadedFile("invalid_file.txt", b"Plain text content", content_type="text/plain")

        response = self.client.post(reverse("audio_events:single_upload"), {"audio_file": uploaded})
        self.assertEqual(response.status_code, 200)

        # Verify specific error reason is rendered
        self.assertContains(response, "Validation Rejected")
        self.assertContains(response, "Unsupported audio format")

    def test_single_upload_post_silent_audio_shows_specific_rejection_reason(self):
        # Create silent WAV
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00" * 32000)
        buffer.seek(0)
        silent_bytes = buffer.read()

        uploaded = SimpleUploadedFile("silent_clip.wav", silent_bytes, content_type="audio/wav")

        response = self.client.post(reverse("audio_events:single_upload"), {"audio_file": uploaded})
        self.assertEqual(response.status_code, 200)

        self.assertContains(response, "Validation Rejected")
        self.assertContains(response, "silence")
