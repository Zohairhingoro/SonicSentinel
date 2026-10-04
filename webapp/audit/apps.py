from django.apps import AppConfig


class AuditConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "audit"

    def ready(self) -> None:
        import audit.middleware  # noqa: F401 — registers signal handlers
