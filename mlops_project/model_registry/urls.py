from django.urls import path
from . import views

app_name = 'model_registry'

urlpatterns = [
    path('', views.index, name='index'),
    path('model/<str:version>/', views.model_detail, name='model_detail'),
    path('model/<str:version>/delete/', views.delete_model, name='delete_model'),
    path('api/list/', views.api_list, name='api_list'),
    path('api/get/<str:version>/', views.api_get, name='api_get'),
]
