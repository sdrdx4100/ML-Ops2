from django.contrib import admin
from .models import TrainingJob


@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ['job_id', 'model_type', 'training_mode', 'status', 'result_version', 'started_at']
    list_filter = ['model_type', 'training_mode', 'status']
    search_fields = ['job_id', 'dataset_name']
    readonly_fields = ['started_at', 'completed_at']
