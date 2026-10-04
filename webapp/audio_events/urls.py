from django.urls import path
from django.views.generic import RedirectView
from . import views

app_name = "audio_events"

urlpatterns = [
    path("", RedirectView.as_view(url="events/", permanent=False), name="index"),
    path("upload/api/", views.upload_audio, name="upload_audio"),
    path("upload/", views.SingleUploadView.as_view(), name="single_upload"),
    path("upload/single/", views.SingleUploadView.as_view(), name="single_upload_alt"),
    path("upload/batch/", views.BatchUploadView.as_view(), name="batch_upload"),
    path("events/", views.EventListView.as_view(), name="event_list"),
]
