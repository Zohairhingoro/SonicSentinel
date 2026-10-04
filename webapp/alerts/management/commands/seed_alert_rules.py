from django.core.management.base import BaseCommand
from alerts.models import AlertRule
from core.constants import get_alert_rules


class Command(BaseCommand):
    help = "Seed AlertRule database table from config/alert_rules.json."

    def handle(self, *args, **options):
        alert_rules_config = get_alert_rules()
        rules_dict = alert_rules_config.get("rules", {})

        count = 0
        for category, rule in rules_dict.items():
            severity = rule.get("severity", AlertRule.SEVERITY_MEDIUM)
            action = rule.get("action", f"Investigate {category} event")
            min_confidence = float(rule.get("min_confidence", 0.70))
            require_repeat = bool(rule.get("require_repeat", False))
            consecutive = 2 if require_repeat else 1

            AlertRule.objects.update_or_create(
                category=category,
                defaults={
                    "severity": severity,
                    "recommended_action": action,
                    "min_confidence": min_confidence,
                    "top_two_margin": float(rule.get("top_two_margin", 0.15)),
                    "required_consecutive_detections": consecutive,
                    "requires_model_agreement": bool(rule.get("requires_model_agreement", False)),
                    "requires_min_quality": rule.get("requires_min_quality", AlertRule.QUALITY_ACCEPTABLE),
                    "escalation_condition": rule.get("escalation_condition", ""),
                },
            )
            count += 1

        self.stdout.write(
            self.style.SUCCESS(f"Successfully seeded {count} alert rules into database.")
        )
