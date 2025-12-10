"""
Tests for the training app - e2e training and lineage verification.
"""
import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mlops_project'))

import django
django.setup()

from django.test import TestCase
from training_app.models import TrainingJob
from training_app.services import Trainer
from model_registry.models import ModelArtifact
from model_registry.services import ModelRegistry
from datasets.models import Dataset
from datasets.services import DataLoader


@pytest.fixture
def temp_media_dir():
    """Create a temporary media directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_dataset(temp_media_dir):
    """Create a sample dataset for testing."""
    from django.conf import settings
    
    # Override media root for testing
    original_media_root = settings.MEDIA_ROOT
    settings.MEDIA_ROOT = temp_media_dir
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    y = 2 * X1 + 3 * X2 + np.random.randn(n_samples) * 0.1
    
    df = pd.DataFrame({'feature1': X1, 'feature2': X2, 'target': y})
    
    # Save as CSV
    os.makedirs(os.path.join(temp_media_dir, 'datasets'), exist_ok=True)
    csv_path = os.path.join(temp_media_dir, 'datasets', 'test_data.csv')
    df.to_csv(csv_path, index=False)
    
    # Create dataset record
    dataset = Dataset.objects.create(
        name='test_dataset',
        file_path=csv_path,
        file_type='csv',
        row_count=n_samples,
        column_names=['feature1', 'feature2', 'target'],
        statistics={},
    )
    
    yield dataset
    
    # Cleanup
    settings.MEDIA_ROOT = original_media_root
    dataset.delete()


@pytest.mark.django_db
class TestTrainingE2E:
    """End-to-end training tests."""
    
    def test_new_training_linear_regression(self, sample_dataset, temp_media_dir):
        """Test new training with linear regression."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        config = {
            'dataset_name': 'test_dataset',
            'model_type': 'linear_regression',
            'training_mode': 'new',
            'target_column': 'target',
            'feature_columns': ['feature1', 'feature2'],
            'hyperparameters': {},
        }
        
        trainer = Trainer()
        version, metrics = trainer.train(config)
        
        # Verify model was created
        assert version is not None
        assert 'val_mae' in metrics or 'val_mse' in metrics
        
        # Verify artifact exists
        artifact = ModelArtifact.objects.get(version=version)
        assert artifact.model_type == 'linear_regression'
        assert artifact.training_mode == 'new'
        assert artifact.parent is None
        
        # Cleanup
        artifact.delete()
    
    def test_new_training_random_forest(self, sample_dataset, temp_media_dir):
        """Test new training with random forest."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        config = {
            'dataset_name': 'test_dataset',
            'model_type': 'random_forest',
            'training_mode': 'new',
            'target_column': 'target',
            'hyperparameters': {'n_estimators': 10, 'max_depth': 5},
        }
        
        trainer = Trainer()
        version, metrics = trainer.train(config)
        
        assert version is not None
        assert 'val_r2' in metrics
        
        # Cleanup
        ModelArtifact.objects.filter(version=version).delete()


@pytest.mark.django_db
class TestLineage:
    """Tests for model lineage tracking."""
    
    def test_fine_tune_lineage(self, sample_dataset, temp_media_dir):
        """Test that fine-tuning creates correct lineage."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        # First, train a base model
        base_config = {
            'dataset_name': 'test_dataset',
            'model_type': 'sgd_regressor',
            'training_mode': 'new',
            'target_column': 'target',
            'hyperparameters': {'max_iter': 100},
        }
        
        trainer = Trainer()
        base_version, _ = trainer.train(base_config)
        
        # Now fine-tune
        fine_tune_config = {
            'dataset_name': 'test_dataset',
            'model_type': 'sgd_regressor',
            'training_mode': 'fine_tune',
            'target_column': 'target',
            'base_model_version': base_version,
            'hyperparameters': {'max_iter': 50},
        }
        
        fine_tuned_version, _ = trainer.train(fine_tune_config)
        
        # Verify lineage
        fine_tuned_artifact = ModelArtifact.objects.get(version=fine_tuned_version)
        assert fine_tuned_artifact.parent is not None
        assert fine_tuned_artifact.parent.version == base_version
        assert fine_tuned_artifact.training_mode == 'fine_tune'
        
        # Verify lineage chain
        lineage = fine_tuned_artifact.get_lineage()
        assert len(lineage) == 2
        assert lineage[0].version == fine_tuned_version
        assert lineage[1].version == base_version
        
        # Cleanup
        ModelArtifact.objects.filter(version__in=[base_version, fine_tuned_version]).delete()
    
    def test_version_increment(self, sample_dataset, temp_media_dir):
        """Test that versions are incremented correctly."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        # Clear existing models
        ModelArtifact.objects.all().delete()
        
        config = {
            'dataset_name': 'test_dataset',
            'model_type': 'linear_regression',
            'training_mode': 'new',
            'target_column': 'target',
        }
        
        trainer = Trainer()
        
        # Train first model
        v1, _ = trainer.train(config)
        assert v1 == '1.0.0'
        
        # Train second model
        v2, _ = trainer.train(config)
        assert v2 == '1.0.1'
        
        # Cleanup
        ModelArtifact.objects.all().delete()
