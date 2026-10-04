import hashlib
from django.test import TestCase

from audio_events.models import AudioEvent
from core.storage import (
    are_fingerprints_near_duplicate,
    check_duplicate_audio,
    check_near_duplicate_audio,
    compute_audio_fingerprint,
    compute_file_hash,
)


class StorageHashTest(TestCase):
    def test_compute_file_hash_bytes(self):
        payload = b"test audio file bytes payload"
        expected_hash = hashlib.sha256(payload).hexdigest()

        calculated_hash = compute_file_hash(payload)
        self.assertEqual(calculated_hash, expected_hash)

    def test_check_duplicate_audio_no_match(self):
        payload = b"unique audio payload 1"
        evt, file_hash = check_duplicate_audio(payload)

        self.assertIsNone(evt)
        self.assertEqual(len(file_hash), 64)

    def test_check_duplicate_audio_with_existing_match(self):
        payload = b"identical duplicate audio bytes"
        expected_hash = hashlib.sha256(payload).hexdigest()

        existing_evt = AudioEvent.objects.create(
            filename="existing.wav",
            format="wav",
            duration=2.5,
            sampling_rate=16000,
            file_size=len(payload),
            quality=AudioEvent.QUALITY_GOOD,
            category="Gunshot",
            hash=expected_hash,
            storage_path="audio/existing.wav",
        )

        matched_evt, returned_hash = check_duplicate_audio(payload)
        self.assertIsNotNone(matched_evt)
        self.assertEqual(matched_evt.audio_id, existing_evt.audio_id)
        self.assertEqual(returned_hash, expected_hash)

    def test_compute_audio_fingerprint(self):
        payload = b"test audio fingerprint payload bytes content 1234567890"
        fingerprint = compute_audio_fingerprint(payload)

        self.assertTrue(len(fingerprint) > 0)
        self.assertIn("dur:", fingerprint)
        self.assertIn("_env:", fingerprint)

    def test_are_fingerprints_near_duplicate(self):
        fp1 = "dur:5.0_env:0123456789abcdef"
        fp2 = "dur:5.0_env:0123456789abcdeg"  # 1 char diff
        fp3 = "dur:10.0_env:0123456789abcdef" # duration diff
        fp4 = "dur:5.0_env:ffffffffffffffff" # envelope diff

        self.assertTrue(are_fingerprints_near_duplicate(fp1, fp1))
        self.assertTrue(are_fingerprints_near_duplicate(fp1, fp2))
        self.assertFalse(are_fingerprints_near_duplicate(fp1, fp3))
        self.assertFalse(are_fingerprints_near_duplicate(fp1, fp4))

    def test_check_near_duplicate_audio(self):
        payload = b"sample audio near duplicate test payload content data 999"
        fp = compute_audio_fingerprint(payload)

        existing_evt = AudioEvent.objects.create(
            filename="original_sample.wav",
            format="wav",
            duration=2.0,
            sampling_rate=16000,
            file_size=len(payload),
            quality=AudioEvent.QUALITY_GOOD,
            category="Panic Scream",
            hash="unique_hash_9999",
            audio_fingerprint=fp,
            storage_path="audio/original_sample.wav",
        )

        suspected_evt, returned_fp = check_near_duplicate_audio(payload, limit=50)

        self.assertIsNotNone(suspected_evt)
        self.assertEqual(suspected_evt.audio_id, existing_evt.audio_id)
        self.assertEqual(returned_fp, fp)
