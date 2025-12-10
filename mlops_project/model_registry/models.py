from django.db import models
import json


class TrainingDataHistory(models.Model):
    """
    Model to track which datasets have been used to train each model version.
    This prevents overfitting by detecting duplicate data usage.
    
    Attributes:
        model_version: Reference to the model artifact
        dataset_name: Name of the dataset used
        dataset_hash: SHA256 hash of the dataset content for duplicate detection
        row_count: Number of rows used from this dataset
        trained_at: Timestamp when this dataset was used for training
    """
    
    model_version = models.ForeignKey(
        'ModelArtifact',
        on_delete=models.CASCADE,
        related_name='training_history'
    )
    dataset_name = models.CharField(max_length=200)
    dataset_hash = models.CharField(max_length=64)
    row_count = models.IntegerField(default=0)
    trained_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-trained_at']
        verbose_name = 'Training Data History'
        verbose_name_plural = 'Training Data Histories'
        # Prevent exact same dataset being used twice for the same model lineage
        indexes = [
            models.Index(fields=['model_version', 'dataset_hash']),
        ]
    
    def __str__(self):
        return f"{self.model_version.version} <- {self.dataset_name}"
    
    @classmethod
    def get_used_dataset_hashes(cls, model_version: 'ModelArtifact') -> set:
        """
        Get all dataset hashes used in the training lineage of a model.
        
        Args:
            model_version: The model artifact to check lineage for
            
        Returns:
            Set of dataset hashes used in this model's lineage
        """
        used_hashes = set()
        
        # Get lineage (current model and all ancestors)
        lineage = model_version.get_lineage()
        
        for model in lineage:
            history = cls.objects.filter(model_version=model)
            for entry in history:
                used_hashes.add(entry.dataset_hash)
        
        return used_hashes
    
    @classmethod
    def check_duplicate_datasets(
        cls, 
        model_version: 'ModelArtifact', 
        dataset_hashes: list
    ) -> list:
        """
        Check if any of the given dataset hashes have already been used.
        
        Args:
            model_version: The base model to check against
            dataset_hashes: List of dataset hashes to check
            
        Returns:
            List of duplicate dataset hashes
        """
        used_hashes = cls.get_used_dataset_hashes(model_version)
        return [h for h in dataset_hashes if h in used_hashes]


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
