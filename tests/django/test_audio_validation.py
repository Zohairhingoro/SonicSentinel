"""
tests/django/test_audio_validation.py — Unit tests for audio_events/services/validation.py.

Tests:
  - Valid audio file acceptance (metadata extraction, quality rating, passing reason)
  - Unsupported audio format rejection (.txt, .pdf, .xyz)
  - Oversized audio file rejection (> max_file_size_mb threshold)
  - Too-short duration audio file rejection (< min_duration_seconds threshold)
  - Silent audio file rejection (RMS energy < 1e-4)
  - Empty file (0 bytes) rejection
  - Corrupt / malformed audio data rejection
  - Duplicate audio file rejection (matching existing AudioEvent hash)

Uses fixture files stored in sample_audio/.
"""

import hashlib
import io
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = REPO_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from audio_events.models import AudioEvent
from audio_events.services.validation import validate_audio_file

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_AUDIO_DIR = REPO_ROOT / "sample_audio"


class AudioValidationServiceTestCase(TestCase):
    """Verifies audio validation service against configured system constraints."""

    @classmethod
    def setUpTestData(cls):
        cls.valid_wav_path = SAMPLE_AUDIO_DIR / "valid.wav"
        cls.too_short_wav_path = SAMPLE_AUDIO_DIR / "too_short.wav"
        cls.silent_wav_path = SAMPLE_AUDIO_DIR / "silent.wav"
        cls.unsupported_txt_path = SAMPLE_AUDIO_DIR / "unsupported.txt"
        cls.corrupt_wav_path = SAMPLE_AUDIO_DIR / "corrupt.wav"

    def test_fixture_files_exist(self):
        """Ensure sample_audio fixture files exist on disk before testing."""
        self.assertTrue(self.valid_wav_path.exists(), "sample_audio/valid.wav is missing")
        self.assertTrue(self.too_short_wav_path.exists(), "sample_audio/too_short.wav is missing")
        self.assertTrue(self.silent_wav_path.exists(), "sample_audio/silent.wav is missing")
        self.assertTrue(self.unsupported_txt_path.exists(), "sample_audio/unsupported.txt is missing")

    # -------------------------------------------------------------------------
    # 1. Valid File Acceptance
    # -------------------------------------------------------------------------
    def test_valid_file_acceptance_from_file_stream(self):
        """A valid 1.0s 22050Hz sine wave WAV file passes all validation checks."""
        with self.valid_wav_path.open("rb") as fh:
            result = validate_audio_file(fh)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "Audio file passed all validation checks")
        meta = result["metadata"]
        self.assertEqual(meta["format"], "wav")
        self.assertEqual(meta["duration"], 1.0)
        self.assertEqual(meta["sampling_rate"], 22050)
        self.assertEqual(meta["channels"], 1)
        self.assertEqual(meta["bit_depth"], 16)
        self.assertEqual(meta["quality"], AudioEvent.QUALITY_GOOD)
        self.assertGreater(meta["rms_energy"], 0.01)
        self.assertTrue(len(meta["hash"]) == 64)

    def test_valid_file_acceptance_from_uploaded_file(self):
        """A SimpleUploadedFile instance wrapping valid WAV data passes validation."""
        content = self.valid_wav_path.read_bytes()
        uploaded = SimpleUploadedFile("valid.wav", content, content_type="audio/wav")

        result = validate_audio_file(uploaded)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["metadata"]["filename"], "valid.wav")
        self.assertEqual(result["metadata"]["format"], "wav")

    # -------------------------------------------------------------------------
    # 2. Unsupported Format Rejection
    # -------------------------------------------------------------------------
    def test_unsupported_format_txt_rejection(self):
        """An unsupported text file (.txt) is rejected with clear message."""
        with self.unsupported_txt_path.open("rb") as fh:
            result = validate_audio_file(fh)

        self.assertFalse(result["accepted"])
        self.assertIn("Unsupported audio format 'txt'", result["reason"])
        self.assertEqual(result["metadata"], {})

    def test_unsupported_format_extensions_rejection(self):
        """Arbitrary unsupported extensions (.xyz, .pdf, .exe) are immediately rejected."""
        dummy_content = b"Some binary or text content"
        for ext in ["xyz", "pdf", "exe", "doc"]:
            uploaded = SimpleUploadedFile(f"sample.{ext}", dummy_content)
            result = validate_audio_file(uploaded)
            self.assertFalse(result["accepted"])
            self.assertIn(f"Unsupported audio format '{ext}'", result["reason"])

    # -------------------------------------------------------------------------
    # 3. Oversized File Rejection
    # -------------------------------------------------------------------------
    def test_oversized_file_rejection_with_config_threshold(self):
        """File exceeding max_file_size_mb is rejected with detailed error."""
        # valid.wav is ~44 KB. Configure max size to 0.01 MB (~10.2 KB)
        restricted_config = {
            "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"],
            "max_file_size_mb": 0.01,
            "min_duration_seconds": 0.5,
            "max_duration_seconds": 30.0,
        }
        with patch("audio_events.services.validation.get_audio_config", return_value=restricted_config):
            with self.valid_wav_path.open("rb") as fh:
                result = validate_audio_file(fh)

            self.assertFalse(result["accepted"])
            self.assertIn("exceeds maximum allowed size", result["reason"])
            self.assertIn("0.01 MB", result["reason"])

    def test_oversized_file_rejection_large_stream(self):
        """A stream exceeding the default 20 MB size limit is rejected."""
        # Create an in-memory stream of 21 MB
        oversized_bytes = b"\x00" * (21 * 1024 * 1024)
        uploaded = SimpleUploadedFile("huge_file.wav", oversized_bytes, content_type="audio/wav")

        result = validate_audio_file(uploaded)
        self.assertFalse(result["accepted"])
        self.assertIn("exceeds maximum allowed size (20 MB)", result["reason"])

    # -------------------------------------------------------------------------
    # 4. Too-Short Duration Rejection
    # -------------------------------------------------------------------------
    def test_too_short_duration_rejection(self):
        """A 0.2s WAV file is rejected because it is shorter than min_duration_seconds (0.5s)."""
        with self.too_short_wav_path.open("rb") as fh:
            result = validate_audio_file(fh)

        self.assertFalse(result["accepted"])
        self.assertIn("shorter than minimum required (0.5s)", result["reason"])
        self.assertEqual(result["metadata"]["duration"], 0.2)

    def test_too_short_duration_rejection_with_uploaded_file(self):
        """SimpleUploadedFile wrapping too_short.wav fails duration check."""
        content = self.too_short_wav_path.read_bytes()
        uploaded = SimpleUploadedFile("too_short.wav", content, content_type="audio/wav")

        result = validate_audio_file(uploaded)
        self.assertFalse(result["accepted"])
        self.assertIn("shorter than minimum required", result["reason"])

    # -------------------------------------------------------------------------
    # 5. Silent Audio Rejection
    # -------------------------------------------------------------------------
    def test_silent_audio_rejection(self):
        """An audio file containing only silence (RMS energy < 1e-4) is rejected."""
        with self.silent_wav_path.open("rb") as fh:
            result = validate_audio_file(fh)

        self.assertFalse(result["accepted"])
        self.assertIn("contains only silence or unusable low-amplitude sound", result["reason"])
        self.assertEqual(result["metadata"]["rms_energy"], 0.0)

    def test_silent_audio_rejection_with_uploaded_file(self):
        """SimpleUploadedFile wrapping silent.wav fails usable signal check."""
        content = self.silent_wav_path.read_bytes()
        uploaded = SimpleUploadedFile("silent.wav", content, content_type="audio/wav")

        result = validate_audio_file(uploaded)
        self.assertFalse(result["accepted"])
        self.assertIn("contains only silence or unusable low-amplitude sound", result["reason"])

    # -------------------------------------------------------------------------
    # 6. Additional Integrity & Duplicate Checks
    # -------------------------------------------------------------------------
    def test_empty_file_rejection(self):
        """An empty file (0 bytes) is rejected with empty file message."""
        empty_upload = SimpleUploadedFile("empty.wav", b"", content_type="audio/wav")
        result = validate_audio_file(empty_upload)
        self.assertFalse(result["accepted"])
        self.assertIn("Uploaded file is empty (0 bytes)", result["reason"])

    def test_corrupt_wav_file_rejection(self):
        """A corrupt or truncated WAV file fails integrity parsing."""
        if self.corrupt_wav_path.exists():
            with self.corrupt_wav_path.open("rb") as fh:
                result = validate_audio_file(fh)
            self.assertFalse(result["accepted"])
            self.assertIn("File integrity check failed", result["reason"])

    def test_duplicate_file_rejection(self):
        """Uploading an audio file whose SHA-256 hash already exists in DB is rejected."""
        content = self.valid_wav_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()

        # Create existing AudioEvent record with this hash
        AudioEvent.objects.create(
            audio_id="test-dup-0001",
            filename="existing_clip.wav",
            format="wav",
            hash=content_hash,
            duration=1.0,
            sampling_rate=22050,
            channels=1,
            bit_depth=16,
            file_size=len(content),
            storage_path="audio/test-dup-0001/existing_clip.wav",
        )

        uploaded = SimpleUploadedFile("valid.wav", content, content_type="audio/wav")
        result = validate_audio_file(uploaded)
        self.assertFalse(result["accepted"])
        self.assertIn("Duplicate file detected", result["reason"])
