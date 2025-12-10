from django.urls import path
from . import views

app_name = 'training_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train, name='train'),
    path('job/<str:job_id>/', views.job_detail, name='job_detail'),
    path('api/train/', views.api_train, name='api_train'),
]
