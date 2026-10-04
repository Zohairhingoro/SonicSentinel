import io
import math
import struct
import wave
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse

from audio_events.models import AudioEvent
from audio_events.services.validation import validate_audio_file


def create_in_memory_wav(
    duration_sec: float = 1.5,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.5,
    freq: float = 440.0,
) -> bytes:
    """Helper utility to generate a valid WAV audio byte stream in memory."""
    buf = io.BytesIO()
    n_samples = int(duration_sec * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        frames = bytearray()
        for i in range(n_samples):
            val = int(amplitude * 32767.0 * math.sin(2.0 * math.pi * freq * i / sample_rate)) if amplitude > 0 else 0
            frames.extend(struct.pack("<h", val))
            if channels > 1:
                frames.extend(struct.pack("<h", val))

        wf.writeframes(bytes(frames))

    return buf.getvalue()


class AudioValidationServiceTestCase(TestCase):
    def test_valid_wav_file_accepted(self):
        wav_bytes = create_in_memory_wav(duration_sec=2.0, sample_rate=16000, amplitude=0.5)
        uploaded = SimpleUploadedFile("test_valid.wav", wav_bytes, content_type="audio/wav")

        res = validate_audio_file(uploaded)
        self.assertTrue(res["accepted"])
        self.assertEqual(res["reason"], "Audio file passed all validation checks")
        meta = res["metadata"]
        self.assertEqual(meta["filename"], "test_valid.wav")
        self.assertEqual(meta["format"], "wav")
        self.assertAlmostEqual(meta["duration"], 2.0, places=1)
        self.assertEqual(meta["sampling_rate"], 16000)
        self.assertEqual(meta["channels"], 1)
        self.assertEqual(meta["bit_depth"], 16)
        self.assertTrue(len(meta["hash"]) > 0)

    def test_unsupported_format_rejected(self):
        uploaded = SimpleUploadedFile("document.pdf", b"PDF header...", content_type="application/pdf")
        res = validate_audio_file(uploaded)
        self.assertFalse(res["accepted"])
        self.assertIn("Unsupported audio format", res["reason"])

    def test_empty_file_rejected(self):
        uploaded = SimpleUploadedFile("empty.wav", b"", content_type="audio/wav")
        res = validate_audio_file(uploaded)
        self.assertFalse(res["accepted"])
        self.assertIn("empty", res["reason"].lower())

    def test_silent_audio_rejected(self):
        wav_bytes = create_in_memory_wav(duration_sec=2.0, sample_rate=16000, amplitude=0.0)
        uploaded = SimpleUploadedFile("silent.wav", wav_bytes, content_type="audio/wav")

        res = validate_audio_file(uploaded)
        self.assertFalse(res["accepted"])
        self.assertIn("silence", res["reason"].lower())

    def test_too_short_duration_rejected(self):
        wav_bytes = create_in_memory_wav(duration_sec=0.2, sample_rate=16000, amplitude=0.5)
        uploaded = SimpleUploadedFile("short.wav", wav_bytes, content_type="audio/wav")

        res = validate_audio_file(uploaded)
        self.assertFalse(res["accepted"])
        self.assertIn("shorter than minimum", res["reason"].lower())

    def test_duplicate_file_rejected(self):
        wav_bytes = create_in_memory_wav(duration_sec=1.5, sample_rate=16000, amplitude=0.4)
        uploaded1 = SimpleUploadedFile("dup.wav", wav_bytes, content_type="audio/wav")
        res1 = validate_audio_file(uploaded1)
        self.assertTrue(res1["accepted"])

        # Save AudioEvent row
        meta = res1["metadata"]
        AudioEvent.objects.create(
            filename=meta["filename"],
            format=meta["format"],
            duration=meta["duration"],
            sampling_rate=meta["sampling_rate"],
            channels=meta["channels"],
            file_size=meta["file_size"],
            storage_path="/media/dup.wav",
            hash=meta["hash"],
        )

        # Upload identical file again
        uploaded2 = SimpleUploadedFile("dup.wav", wav_bytes, content_type="audio/wav")
        res2 = validate_audio_file(uploaded2)
        self.assertFalse(res2["accepted"])
        self.assertIn("Duplicate file", res2["reason"])


class AudioUploadViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_upload_endpoint_valid_file(self):
        wav_bytes = create_in_memory_wav(duration_sec=2.0, sample_rate=16000, amplitude=0.5)
        uploaded = SimpleUploadedFile("upload_sample.wav", wav_bytes, content_type="audio/wav")

        response = self.client.post("/audio/upload/api/", {"audio_file": uploaded})
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertTrue(data["accepted"])
        self.assertIn("AUD-", data["audio_id"])
        self.assertTrue(AudioEvent.objects.filter(audio_id=data["audio_id"]).exists())

    def test_upload_endpoint_invalid_file(self):
        uploaded = SimpleUploadedFile("bad_file.txt", b"Hello World", content_type="text/plain")

        response = self.client.post("/audio/upload/api/", {"audio_file": uploaded})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["accepted"])
        self.assertIn("Unsupported audio format", data["reason"])
