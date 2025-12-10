from django.db import models
import json


class ModelArtifact(models.Model):
    """
    Model representing a trained ML model artifact.
    
    Attributes:
        version: Unique version identifier (major.minor.patch)
        parent: Reference to parent model (for fine-tuning lineage)
        model_file: Path to the model file
        preprocessor_file: Path to the preprocessor file
        metadata: JSON metadata about the model
        metrics: JSON training/validation metrics
        training_mode: 'new' or 'fine_tune'
        model_type: Type of model (linear_regression, random_forest, etc.)
        file_hash: SHA256 hash of model file for verification
        created_at: Timestamp of creation
    """
    
    TRAINING_MODE_CHOICES = [
        ('new', 'New Training'),
        ('fine_tune', 'Fine Tuning'),
    ]
    
    version = models.CharField(max_length=50, unique=True)
    parent = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='children'
    )
    model_file = models.CharField(max_length=500)
    preprocessor_file = models.CharField(max_length=500, blank=True, null=True)
    metadata = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    training_mode = models.CharField(
        max_length=20,
        choices=TRAINING_MODE_CHOICES,
        default='new'
    )
    model_type = models.CharField(max_length=100)
    file_hash = models.CharField(max_length=64, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Model Artifact'
        verbose_name_plural = 'Model Artifacts'
    
    def __str__(self):
        return f"{self.model_type} v{self.version}"
    
    def get_lineage(self):
        """Get full lineage of this model (ancestors)."""
        lineage = [self]
        current = self
        while current.parent:
            lineage.append(current.parent)
            current = current.parent
        return lineage
    
    @classmethod
    def get_next_version(cls, base_version=None, increment='patch'):
        """
        Generate the next version number.
        
        Args:
            base_version: Base version to increment from (optional)
            increment: 'major', 'minor', or 'patch'
        
        Returns:
            Next version string
        """
        if base_version:
            parts = base_version.split('.')
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            
            if increment == 'major':
                return f"{major + 1}.0.0"
            elif increment == 'minor':
                return f"{major}.{minor + 1}.0"
            else:
                return f"{major}.{minor}.{patch + 1}"
        
        # Get latest version
        latest = cls.objects.order_by('-created_at').first()
        if latest:
            return cls.get_next_version(latest.version, increment)
        
        return "1.0.0"
