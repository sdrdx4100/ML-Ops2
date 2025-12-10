from django.urls import path
from . import views

app_name = 'datasets'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('detail/<str:name>/', views.detail, name='detail'),
    path('delete/<str:name>/', views.delete, name='delete'),
    path('api/list/', views.api_list, name='api_list'),
    path('api/get/<str:name>/', views.api_get, name='api_get'),
]
