from django.db import models


class InferenceLog(models.Model):
    """
    Model representing an inference log entry.
    
    Attributes:
        model_version: Version of the model used
        input_data: JSON input data
        prediction: JSON prediction result
        analysis: JSON feature importance/contribution analysis
        timestamp: Time of inference
    """
    
    model_version = models.CharField(max_length=50)
    input_data = models.JSONField(default=dict)
    prediction = models.JSONField(default=dict)
    analysis = models.JSONField(default=dict, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Inference Log'
        verbose_name_plural = 'Inference Logs'
    
    def __str__(self):
        return f"Inference with model v{self.model_version} at {self.timestamp}"
