from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.index, name="index"),
    path("user/", views.user_dashboard, name="user"),
    path("admin/", views.admin_dashboard, name="admin"),
    path("analytics/", views.analytics_dashboard, name="analytics"),
    path("analytics/charts/", views.AdminAnalyticsView.as_view(), name="analytics_charts"),
    path("export/", views.ExportEventsView.as_view(), name="export"),
]
