from django.template.loader import render_to_string
from django.test import TestCase
from django.urls import reverse

from audio_events.models import AudioEvent


class AudioPreviewComponentTestCase(TestCase):
    def setUp(self):
        self.event = AudioEvent.objects.create(
            audio_id="AUD-PREVIEW0001",
            filename="siren_alert.wav",
            format="WAV",
            duration=4.2,
            sampling_rate=16000,
            file_size=134400,
            category="Alarm or Siren",
            storage_path="audio/AUD-PREVIEW0001/siren_alert.wav",
            hash="hash_preview_test",
        )

    def test_render_audio_preview_component_default(self):
        html = render_to_string("components/audio_preview.html", {})

        # Verify key HTML5 audio controls exist in component
        self.assertIn('id="audio-elem-default"', html)
        self.assertIn('id="play-btn-default"', html)
        self.assertIn('id="replay-btn-default"', html)
        self.assertIn('id="seek-bar-default"', html)
        self.assertIn('id="volume-bar-default"', html)
        self.assertIn('id="mute-btn-default"', html)
        self.assertIn('id="current-time-default"', html)
        self.assertIn('id="duration-default"', html)
        self.assertIn('id="speed-select-default"', html)
        self.assertIn("js/audio_preview.js", html)

    def test_render_audio_preview_component_with_custom_context(self):
        context = {
            "audio_url": "/media/audio/AUD-PREVIEW0001/siren_alert.wav",
            "filename": "siren_alert.wav",
            "file_input_id": "id_audio_file",
            "id_suffix": "test_clip_1",
            "title": "Custom Test Clip",
        }
        html = render_to_string("components/audio_preview.html", context)

        self.assertIn('id="audio-elem-test_clip_1"', html)
        self.assertIn('id="play-btn-test_clip_1"', html)
        self.assertIn('id="seek-bar-test_clip_1"', html)
        self.assertIn("siren_alert.wav", html)
        self.assertIn("Custom Test Clip", html)
        self.assertIn("id_audio_file", html)

    def test_upload_page_renders_audio_preview_component(self):
        response = self.client.get(reverse("audio_events:single_upload"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "audio_events/upload.html")
        self.assertTemplateUsed(response, "components/audio_preview.html")
        self.assertContains(response, "js/audio_preview.js")
        self.assertContains(response, "Selected Audio Clip Preview")
