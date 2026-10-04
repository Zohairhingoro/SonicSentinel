import json
from django.template.loader import render_to_string
from django.test import SimpleTestCase, TestCase

from audio_events.models import AudioEvent


class WaveformSpectrogramComponentTestCase(TestCase):
    def setUp(self):
        self.event = AudioEvent.objects.create(
            filename="gunshot_test.wav",
            format="WAV",
            duration=2.5,
            sampling_rate=16000,
            file_size=80000,
            category="Gunshot",
            storage_path="/media/gunshot_test.wav",
            hash="hash_viz_1",
        )

    def test_render_with_raw_arrays(self):
        waveform_data = json.dumps([0.1, -0.2, 0.8, -0.9, 0.4, -0.1])
        spectrogram_data = json.dumps([[0.1, 0.5], [0.8, 0.3]])

        context = {
            "event": self.event,
            "waveform": waveform_data,
            "spectrogram": spectrogram_data,
            "id_suffix": "test_1",
        }
        html = render_to_string("components/waveform_spectrogram.html", context)

        self.assertIn("Acoustic Signal Visualization", html)
        self.assertIn("waveform-canvas-test_1", html)
        self.assertIn("spectrogram-canvas-test_1", html)
        self.assertIn("gunshot_test.wav", html)

    def test_render_with_images(self):
        context = {
            "event": self.event,
            "waveform_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "spectrogram_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "id_suffix": "test_img",
        }
        html = render_to_string("components/waveform_spectrogram.html", context)

        self.assertIn("data:image/png;base64", html)
        self.assertIn("Waveform Image", html)
        self.assertIn("Spectrogram Image", html)
