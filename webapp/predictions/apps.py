from django.apps import AppConfig


class PredictionsConfig(AppConfig):
    """Django AppConfig for predictions application."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "predictions"

    def ready(self) -> None:
        import predictions.signals  # noqa: F401


