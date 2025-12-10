from django.contrib import admin
from .models import ModelArtifact


@admin.register(ModelArtifact)
class ModelArtifactAdmin(admin.ModelAdmin):
    list_display = ['version', 'model_type', 'training_mode', 'parent', 'created_at']
    list_filter = ['model_type', 'training_mode']
    search_fields = ['version', 'model_type']
    readonly_fields = ['created_at', 'file_hash']
