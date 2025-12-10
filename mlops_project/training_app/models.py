from django.db import models


class TrainingJob(models.Model):
    """
    Model representing a training job.
    
    Attributes:
        job_id: Unique job identifier
        dataset_names: JSON list of dataset names used (supports multiple datasets)
        model_type: Type of model being trained
        training_mode: 'new' or 'fine_tune'
        hyperparameters: JSON hyperparameters
        status: Current job status
        error_message: Error message if failed
        result_version: Version of resulting model
        started_at: Job start time
        completed_at: Job completion time
    """
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    TRAINING_MODE_CHOICES = [
        ('new', 'New Training'),
        ('fine_tune', 'Fine Tuning'),
    ]
    
    job_id = models.CharField(max_length=100, unique=True)
    dataset_names = models.JSONField(default=list)  # Changed to support multiple datasets
    model_type = models.CharField(max_length=100)
    training_mode = models.CharField(
        max_length=20,
        choices=TRAINING_MODE_CHOICES,
        default='new'
    )
    hyperparameters = models.JSONField(default=dict)
    feature_columns = models.JSONField(default=list)
    target_column = models.CharField(max_length=100)
    base_model_version = models.CharField(max_length=50, blank=True, null=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    error_message = models.TextField(blank=True, null=True)
    result_version = models.CharField(max_length=50, blank=True, null=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        ordering = ['-started_at']
        verbose_name = 'Training Job'
        verbose_name_plural = 'Training Jobs'
    
    def __str__(self):
        dataset_count = len(self.dataset_names) if self.dataset_names else 0
        return f"Job {self.job_id}: {self.model_type} ({dataset_count} datasets, {self.status})"
    
    @property
    def dataset_names_display(self) -> str:
        """Display dataset names as a comma-separated string."""
        if not self.dataset_names:
            return ""
        return ", ".join(self.dataset_names)
