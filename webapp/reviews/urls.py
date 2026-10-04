from django.urls import path
from . import views

app_name = "reviews"

urlpatterns = [
    path("", views.review_queue, name="queue"),
    path("queue/", views.review_queue, name="queue_alt"),
    path("index/", views.review_queue, name="index"),
    path("<str:audio_id>/", views.review_detail, name="detail"),
    path("detail/<str:audio_id>/", views.review_detail, name="detail_alt"),
]

