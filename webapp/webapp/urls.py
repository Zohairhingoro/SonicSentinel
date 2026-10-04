from django.contrib import admin
from django.urls import include, path
from dashboard.views import index as home_view

urlpatterns = [
    path("", home_view, name="home"),
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),
    path("audio/", include("audio_events.urls")),
    path("predictions/", include("predictions.urls")),
    path("alerts/", include("alerts.urls")),
    path("reviews/", include("reviews.urls")),
    path("dashboard/", include("dashboard.urls")),
    path("live/", include("live_monitor.urls")),
    path("audit/", include("audit.urls")),
    path("core/", include("core.urls")),
]

