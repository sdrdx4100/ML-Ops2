from django.urls import path
from . import views

app_name = 'inference_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('log/<int:log_id>/', views.log_detail, name='log_detail'),
    path('api/predict/', views.api_predict, name='api_predict'),
]
