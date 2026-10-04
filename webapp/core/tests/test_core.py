from django.test import TestCase
from core.constants import (
    get_alert_rules,
    get_audio_config,
    get_categories,
    get_classes,
    get_supported_formats,
    reload_configs,
)


class AppEndpointsTest(TestCase):
    def test_all_app_routes(self):
        routes = [
            "/",
            "/accounts/login/",
            "/audio/",
            "/predictions/",
            "/alerts/",
            "/reviews/",
            "/dashboard/",
            "/live/",
            "/audit/",
            "/core/",
        ]
        for route in routes:
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertIn(response.status_code, [200, 302])


class ConstantsConfigTest(TestCase):
    def test_get_classes(self):
        classes_data = get_classes()
        self.assertIn("categories", classes_data)
        self.assertEqual(len(classes_data["categories"]), 10)

    def test_get_alert_rules(self):
        rules_data = get_alert_rules()
        self.assertIn("rules", rules_data)
        self.assertIn("Gunshot", rules_data["rules"])

    def test_get_audio_config(self):
        audio_config = get_audio_config()
        self.assertIn("supported_formats", audio_config)
        self.assertIn("wav", audio_config["supported_formats"])

    def test_get_categories_and_formats(self):
        categories = get_categories()
        self.assertEqual(len(categories), 10)
        formats = get_supported_formats()
        self.assertIn("mp3", formats)

    def test_reload_configs(self):
        _ = get_classes()
        reload_configs()
        classes_data = get_classes()
        self.assertEqual(len(classes_data["categories"]), 10)
