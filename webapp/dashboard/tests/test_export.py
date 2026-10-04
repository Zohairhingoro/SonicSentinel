import csv
import io
from django.test import TestCase

from alerts.models import Alert, AlertRule
from audio_events.models import AudioEvent
from dashboard.export import (
    CSV_HEADER_FIELDS,
    export_audio_events_to_csv_data,
    generate_audio_events_csv_response,
    prepare_export_queryset,
)
from predictions.models import ComparisonResult, GTMPrediction, PythonPrediction
from reviews.models import ReviewRecord


class ExportReportTest(TestCase):
    def setUp(self):
        self.evt = AudioEvent.objects.create(
            filename="gunshot.wav",
            format="wav",
            duration=3.5,
            sampling_rate=22050,
            channels=1,
            file_size=1500,
            quality=AudioEvent.QUALITY_GOOD,
            category="Gunshot",
            severity="Critical",
            storage_path="audio/gunshot.wav",
            hash="hash_export_123",
        )

        PythonPrediction.objects.create(
            audio_event=self.evt,
            predicted_class="Gunshot",
            confidence=0.96,
            confidence_scores={"Gunshot": 0.96},
            model_version="1.0.0",
        )

        GTMPrediction.objects.create(
            audio_event=self.evt,
            predicted_class="Gunshot",
            confidence_scores={"Gunshot": 0.94},
            model_version="gtm-1.0",
        )

        ComparisonResult.objects.update_or_create(
            audio_event=self.evt,
            defaults={
                "confidence_difference": 0.02,
                "top_two_margin": 0.5,
                "match_status": ComparisonResult.MATCH_ACCEPTABLE,
                "is_uncertain": False,
                "is_overlapping": False,
            },
        )

        rule = AlertRule.objects.create(
            category="Gunshot",
            severity=AlertRule.SEVERITY_CRITICAL,
            recommended_action="Dispatch emergency response",
        )
        Alert.objects.create(
            audio_event=self.evt,
            alert_rule=rule,
            severity=AlertRule.SEVERITY_CRITICAL,
            status=Alert.STATUS_ACTIVE,
        )

        ReviewRecord.objects.create(
            audio_event=self.evt,
            decision="Gunshot",
            comment="Confirmed gunshot event",
            recommended_action="Keep active alert",
        )

    def test_prepare_export_queryset(self):
        qs = AudioEvent.objects.filter(pk=self.evt.pk)
        prefetched_qs = prepare_export_queryset(qs)
        evt_item = prefetched_qs.first()

        self.assertIsNotNone(evt_item)
        self.assertEqual(len(list(evt_item.python_predictions.all())), 1)
        self.assertEqual(len(list(evt_item.gtm_predictions.all())), 1)
        self.assertEqual(len(list(evt_item.comparison_results.all())), 1)
        self.assertEqual(len(list(evt_item.alerts.all())), 1)
        self.assertEqual(len(list(evt_item.review_records.all())), 1)

    def test_export_audio_events_to_csv_data(self):
        qs = AudioEvent.objects.all()
        csv_str = export_audio_events_to_csv_data(qs)
        self.assertIsInstance(csv_str, str)

        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        self.assertEqual(len(rows), 2)  # Header + 1 data row
        headers = rows[0]
        self.assertEqual(headers, CSV_HEADER_FIELDS)

        data_row = rows[1]
        self.assertEqual(data_row[0], self.evt.audio_id)
        self.assertEqual(data_row[1], "gunshot.wav")
        self.assertEqual(data_row[2], "Gunshot")
        self.assertEqual(data_row[13], "Gunshot")  # Python class
        self.assertEqual(data_row[14], "0.9600")  # Python confidence
        self.assertEqual(data_row[16], "Gunshot")  # GTM class
        self.assertEqual(data_row[20], ComparisonResult.MATCH_ACCEPTABLE)
        self.assertEqual(data_row[23], "Yes")  # Alert generated
        self.assertEqual(data_row[29], "Gunshot")  # Review decision
        self.assertEqual(data_row[34], "hash_export_123")  # Hash

    def test_generate_audio_events_csv_response(self):
        qs = AudioEvent.objects.all()
        response = generate_audio_events_csv_response(qs, filename="custom_report.csv")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("custom_report.csv", response["Content-Disposition"])
        self.assertTrue(response.content.startswith(b"\xef\xbb\xbf"))  # UTF-8 BOM check
