from django.urls import path
from . import views

app_name = "live_monitor"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/session-status/", views.session_status_api, name="session_status_api"),
]
