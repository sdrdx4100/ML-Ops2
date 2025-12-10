from django.contrib import admin
from .models import InferenceLog


@admin.register(InferenceLog)
class InferenceLogAdmin(admin.ModelAdmin):
    list_display = ['id', 'model_version', 'timestamp']
    list_filter = ['model_version']
    search_fields = ['model_version']
    readonly_fields = ['timestamp']
