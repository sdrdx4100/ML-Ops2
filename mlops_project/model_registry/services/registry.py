"""
Model Registry service for managing trained models.
"""
import os
import hashlib
import pickle
import json
from typing import Optional, Dict, Any, Tuple
from django.conf import settings

from model_registry.models import ModelArtifact
from shared.utils import get_logger
from shared.utils.exceptions import ModelRegistryError

logger = get_logger(__name__)


class ModelRegistry:
    """
    Service for managing model artifacts.
    
    Provides methods to save, load, and query trained models.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the registry.
        
        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = models_dir or os.path.join(
            settings.MEDIA_ROOT, 'models'
        )
        os.makedirs(self.models_dir, exist_ok=True)
    
    def _compute_hash(self, filepath: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def save(
        self,
        model: Any,
        model_type: str,
        metrics: Dict[str, float],
        preprocessor: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        training_mode: str = 'new',
        parent_version: Optional[str] = None,
        version: Optional[str] = None,
    ) -> ModelArtifact:
        """
        Save a trained model to the registry.
        
        Args:
            model: Trained model object
            model_type: Type of model (e.g., 'linear_regression')
            metrics: Training/validation metrics
            preprocessor: Optional preprocessor pipeline
            metadata: Optional metadata dict
            training_mode: 'new' or 'fine_tune'
            parent_version: Version of parent model (for fine-tuning)
            version: Optional explicit version (auto-generated if not provided)
        
        Returns:
            Created ModelArtifact instance
        """
        # Get parent if fine-tuning
        parent = None
        if parent_version:
            try:
                parent = ModelArtifact.objects.get(version=parent_version)
            except ModelArtifact.DoesNotExist:
                raise ModelRegistryError(f"Parent model version {parent_version} not found")
        
        # Generate version
        if version is None:
            if training_mode == 'fine_tune' and parent:
                version = ModelArtifact.get_next_version(parent.version, 'minor')
            else:
                version = ModelArtifact.get_next_version()
        
        # Save model file
        model_filename = f"model_{version.replace('.', '_')}.pkl"
        model_filepath = os.path.join(self.models_dir, model_filename)
        
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)
        
        file_hash = self._compute_hash(model_filepath)
        
        # Save preprocessor if provided
        preprocessor_filepath = None
        if preprocessor is not None:
            preprocessor_filename = f"preprocessor_{version.replace('.', '_')}.pkl"
            preprocessor_filepath = os.path.join(self.models_dir, preprocessor_filename)
            
            with open(preprocessor_filepath, 'wb') as f:
                pickle.dump(preprocessor, f)
        
        # Create artifact record
        artifact = ModelArtifact.objects.create(
            version=version,
            parent=parent,
            model_file=model_filepath,
            preprocessor_file=preprocessor_filepath,
            metadata=metadata or {},
            metrics=metrics,
            training_mode=training_mode,
            model_type=model_type,
            file_hash=file_hash,
        )
        
        logger.info(f"Saved model {model_type} version {version}")
        return artifact
    
    def load(self, version: str, verify_hash: bool = True) -> Tuple[Any, Optional[Any]]:
        """
        Load a model from the registry.
        
        Args:
            version: Model version to load
            verify_hash: Whether to verify file hash
        
        Returns:
            Tuple of (model, preprocessor)
        """
        try:
            artifact = ModelArtifact.objects.get(version=version)
        except ModelArtifact.DoesNotExist:
            raise ModelRegistryError(f"Model version {version} not found")
        
        # Verify hash
        if verify_hash and artifact.file_hash:
            current_hash = self._compute_hash(artifact.model_file)
            if current_hash != artifact.file_hash:
                raise ModelRegistryError(
                    f"Model file hash mismatch for version {version}. "
                    "File may have been modified."
                )
        
        # Load model
        with open(artifact.model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessor if exists
        preprocessor = None
        if artifact.preprocessor_file and os.path.exists(artifact.preprocessor_file):
            with open(artifact.preprocessor_file, 'rb') as f:
                preprocessor = pickle.load(f)
        
        logger.info(f"Loaded model version {version}")
        return model, preprocessor
    
    def get_artifact(self, version: str) -> ModelArtifact:
        """
        Get model artifact metadata.
        
        Args:
            version: Model version
        
        Returns:
            ModelArtifact instance
        """
        try:
            return ModelArtifact.objects.get(version=version)
        except ModelArtifact.DoesNotExist:
            raise ModelRegistryError(f"Model version {version} not found")
    
    def list_versions(
        self,
        model_type: Optional[str] = None,
        training_mode: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        List available model versions.
        
        Args:
            model_type: Filter by model type
            training_mode: Filter by training mode
            limit: Maximum number of results
        
        Returns:
            List of ModelArtifact instances
        """
        queryset = ModelArtifact.objects.all()
        
        if model_type:
            queryset = queryset.filter(model_type=model_type)
        if training_mode:
            queryset = queryset.filter(training_mode=training_mode)
        
        return list(queryset[:limit])
    
    def delete(self, version: str, delete_files: bool = True) -> None:
        """
        Delete a model from the registry.
        
        Args:
            version: Model version to delete
            delete_files: Whether to delete model files
        """
        try:
            artifact = ModelArtifact.objects.get(version=version)
        except ModelArtifact.DoesNotExist:
            raise ModelRegistryError(f"Model version {version} not found")
        
        if delete_files:
            if os.path.exists(artifact.model_file):
                os.remove(artifact.model_file)
            if artifact.preprocessor_file and os.path.exists(artifact.preprocessor_file):
                os.remove(artifact.preprocessor_file)
        
        artifact.delete()
        logger.info(f"Deleted model version {version}")
