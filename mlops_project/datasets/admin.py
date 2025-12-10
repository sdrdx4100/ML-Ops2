from django.contrib import admin
from .models import Dataset


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'file_type', 'row_count', 'created_at']
    list_filter = ['file_type']
    search_fields = ['name']
    readonly_fields = ['created_at', 'updated_at', 'row_count', 'column_names', 'statistics']
