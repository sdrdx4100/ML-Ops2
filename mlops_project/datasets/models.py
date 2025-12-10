from django.db import models
import json


class Dataset(models.Model):
    """
    Model representing a dataset for ML training/inference.
    
    Attributes:
        name: Unique name identifier
        file_path: Path to the data file (CSV or Parquet)
        file_type: Type of file (csv or parquet)
        row_count: Number of rows
        column_names: JSON list of column names
        statistics: JSON statistics per column
        created_at: Timestamp of creation
        updated_at: Timestamp of last update
    """
    
    FILE_TYPE_CHOICES = [
        ('csv', 'CSV'),
        ('parquet', 'Parquet'),
    ]
    
    name = models.CharField(max_length=200, unique=True)
    file_path = models.CharField(max_length=500)
    file_type = models.CharField(
        max_length=20,
        choices=FILE_TYPE_CHOICES,
        default='csv'
    )
    row_count = models.IntegerField(default=0)
    column_names = models.JSONField(default=list)
    statistics = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Dataset'
        verbose_name_plural = 'Datasets'
    
    def __str__(self):
        return f"{self.name} ({self.row_count} rows)"
