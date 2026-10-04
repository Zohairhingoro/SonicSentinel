from django.test import TestCase
from django.urls import reverse

from core.constants import get_audio_config


class MicCaptureLiveMonitorTestCase(TestCase):
    def test_live_monitor_index_view_loads_audio_config(self):
        response = self.client.get(reverse("live_monitor:index"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "live_monitor/index.html")

        # Verify window_duration_seconds from config/audio_config.json is passed
        config = get_audio_config()
        expected_window = float(config.get("window_duration_seconds", 3.0))
        self.assertEqual(response.context["window_duration_seconds"], expected_window)

    def test_live_monitor_template_renders_capture_controls_and_script(self):
        response = self.client.get(reverse("live_monitor:index"))
        self.assertEqual(response.status_code, 200)

        # Check for container and data-window-duration attribute
        self.assertContains(response, 'id="mic-capture-container"')
        self.assertContains(response, 'data-window-duration="3.0"')

        # Check for Start / Stop buttons
        self.assertContains(response, 'id="start-mic-btn"')
        self.assertContains(response, 'id="stop-mic-btn"')
        self.assertContains(response, "Start Capture")

        # Check for FE-8a status widget
        self.assertContains(response, 'id="mic-status-widget"')

        # Check for mic_capture.js and gtm_infer.js static file references
        self.assertContains(response, "live_monitor/js/mic_capture.js")
        self.assertContains(response, "live_monitor/js/gtm_infer.js")
        self.assertContains(response, 'id="gtm-live-prediction-pill"')
