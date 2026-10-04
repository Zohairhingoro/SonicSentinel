from django.urls import path
from . import views

app_name = "alerts"

urlpatterns = [
    path("", views.index, name="index"),
    path("active/", views.active_alerts_api, name="active_api"),
    path("<int:alert_id>/acknowledge/", views.acknowledge_alert, name="acknowledge"),
    path("<int:alert_id>/dismiss/", views.dismiss_alert, name="dismiss"),
    path("<int:alert_id>/escalate/", views.escalate_alert, name="escalate"),
]
