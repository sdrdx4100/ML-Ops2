"""
Tests for the inference app.
"""
import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil
import pickle

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mlops_project'))

import django
django.setup()

from sklearn.linear_model import LinearRegression
from inference_app.models import InferenceLog
from inference_app.services import Predictor
from model_registry.models import ModelArtifact
from model_registry.services import ModelRegistry


@pytest.fixture
def temp_media_dir():
    """Create a temporary media directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def trained_model(temp_media_dir):
    """Create a trained model for testing."""
    from django.conf import settings
    settings.MEDIA_ROOT = temp_media_dir
    
    # Train a simple model
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([3, 7, 11, 15])  # y = x1 + x2
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Save model
    os.makedirs(os.path.join(temp_media_dir, 'models'), exist_ok=True)
    model_path = os.path.join(temp_media_dir, 'models', 'model_1_0_0.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Create artifact record
    artifact = ModelArtifact.objects.create(
        version='1.0.0',
        model_file=model_path,
        model_type='linear_regression',
        training_mode='new',
        metrics={'train_mae': 0.0, 'val_mae': 0.0},
        metadata={
            'feature_columns': ['x1', 'x2'],
            'target_column': 'y',
        },
    )
    
    yield artifact
    
    # Cleanup
    artifact.delete()


@pytest.mark.django_db
class TestInference:
    """Tests for inference functionality."""
    
    def test_predict_single_input(self, trained_model, temp_media_dir):
        """Test prediction with single input."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        predictor = Predictor()
        
        # Predict
        input_data = {'x1': 2, 'x2': 3}
        result = predictor.predict('1.0.0', input_data)
        
        assert 'predictions' in result
        assert len(result['predictions']) == 1
        # Should be approximately 5 (2 + 3)
        assert abs(result['predictions'][0] - 5) < 0.1
    
    def test_predict_dataframe(self, trained_model, temp_media_dir):
        """Test prediction with DataFrame."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        predictor = Predictor()
        
        # Predict
        df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [1, 2, 3]})
        result = predictor.predict('1.0.0', df)
        
        assert 'predictions' in result
        assert len(result['predictions']) == 3
    
    def test_inference_logging(self, trained_model, temp_media_dir):
        """Test that inference is logged."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        # Clear existing logs
        InferenceLog.objects.all().delete()
        
        predictor = Predictor()
        input_data = {'x1': 1, 'x2': 1}
        predictor.predict('1.0.0', input_data)
        
        # Check that log was created
        logs = InferenceLog.objects.filter(model_version='1.0.0')
        assert logs.count() == 1
        
        log = logs.first()
        assert log.input_data is not None
        assert log.prediction is not None


@pytest.mark.django_db
class TestModelRegistry:
    """Tests for model registry."""
    
    def test_save_and_load(self, temp_media_dir):
        """Test saving and loading models."""
        from django.conf import settings
        settings.MEDIA_ROOT = temp_media_dir
        
        os.makedirs(os.path.join(temp_media_dir, 'models'), exist_ok=True)
        
        # Train model
        model = LinearRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        model.fit(X, y)
        
        # Save
        registry = ModelRegistry()
        artifact = registry.save(
            model=model,
            model_type='linear_regression',
            metrics={'train_mae': 0.0},
            metadata={'test': True},
        )
        
        assert artifact.version is not None
        assert artifact.file_hash is not None
        
        # Load
        loaded_model, _ = registry.load(artifact.version)
        
        # Verify predictions match
        pred1 = model.predict([[4]])
        pred2 = loaded_model.predict([[4]])
        
        assert np.allclose(pred1, pred2)
        
        # Cleanup
        artifact.delete()
    
    def test_hash_verification(self, temp_media_dir):
        """Test that file hash verification works."""
        from django.conf import settings
        from shared.utils.exceptions import ModelRegistryError
        settings.MEDIA_ROOT = temp_media_dir
        
        os.makedirs(os.path.join(temp_media_dir, 'models'), exist_ok=True)
        
        # Save a model
        model = LinearRegression()
        model.fit([[1], [2]], [1, 2])
        
        registry = ModelRegistry()
        artifact = registry.save(
            model=model,
            model_type='linear_regression',
            metrics={},
        )
        
        # Modify the file
        with open(artifact.model_file, 'ab') as f:
            f.write(b'corrupted')
        
        # Loading should fail with hash verification
        with pytest.raises(ModelRegistryError):
            registry.load(artifact.version, verify_hash=True)
        
        # But should work without verification
        loaded, _ = registry.load(artifact.version, verify_hash=False)
        assert loaded is not None
        
        # Cleanup
        artifact.delete()
