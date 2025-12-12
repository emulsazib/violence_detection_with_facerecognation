from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('simple_video_feed/', views.simple_video_feed, name='simple_video_feed'),
    path('capture_image/', views.capture_image, name='capture_image'),
    path('start_detection/', views.start_detection, name='start_detection'),
    path('stop_detection/', views.stop_detection, name='stop_detection'),
    path('add_camera/', views.add_camera, name='add_camera'),
    path('run_face_recognition/', views.run_face_recognition, name='run_face_recognition'),
    path('get_violence_alert/', views.get_violence_alert, name='get_violence_alert'),
    path('test_violence_alert/', views.test_violence_alert, name='test_violence_alert'),
    path('health_check/', views.health_check, name='health_check'),
]
