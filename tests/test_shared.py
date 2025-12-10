"""
Tests for the shared module.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mlops_project'))

from shared.preprocess.scaler import StandardScaler, MinMaxScaler, RobustScaler
from shared.preprocess.encoder import LabelEncoder, OneHotEncoder
from shared.preprocess.feature_engineering import (
    handle_missing_values,
    create_polynomial_features,
    PreprocessingPipeline,
)
from shared.metrics.regression import mae, mse, rmse, r2_score
from shared.metrics.classification import accuracy, precision, recall, f1_score
from shared.utils.timer import Timer
from shared.utils.exceptions import TrainingError, InferenceError


class TestScalers:
    """Tests for scaler classes."""
    
    def test_standard_scaler_fit_transform(self):
        """Test StandardScaler fit and transform."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check that mean is approximately 0
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        # Check that std is approximately 1
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_standard_scaler_inverse_transform(self):
        """Test StandardScaler inverse transform."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)
        
        assert np.allclose(X, X_recovered)
    
    def test_standard_scaler_serialization(self):
        """Test StandardScaler serialization."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Serialize and deserialize
        data = scaler.to_dict()
        scaler2 = StandardScaler.from_dict(data)
        
        # Both should produce same result
        X_scaled1 = scaler.transform(X)
        X_scaled2 = scaler2.transform(X)
        
        assert np.allclose(X_scaled1, X_scaled2)
    
    def test_minmax_scaler(self):
        """Test MinMaxScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check that min is 0 and max is 1
        assert np.allclose(X_scaled.min(axis=0), 0)
        assert np.allclose(X_scaled.max(axis=0), 1)
    
    def test_robust_scaler(self):
        """Test RobustScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6], [100, 200]])  # With outlier
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Robust scaler should handle outliers better
        assert scaler.fitted
        assert X_scaled is not None


class TestEncoders:
    """Tests for encoder classes."""
    
    def test_label_encoder(self):
        """Test LabelEncoder."""
        y = np.array(['a', 'b', 'c', 'a', 'b'])
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        assert encoder.fitted
        assert len(encoder.classes_) == 3
        assert y_encoded.shape == y.shape
    
    def test_label_encoder_inverse(self):
        """Test LabelEncoder inverse transform."""
        y = np.array(['a', 'b', 'c', 'a', 'b'])
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        y_recovered = encoder.inverse_transform(y_encoded)
        
        assert np.array_equal(y, y_recovered)
    
    def test_onehot_encoder(self):
        """Test OneHotEncoder."""
        X = np.array([['a'], ['b'], ['c'], ['a']])
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)
        
        assert encoder.fitted
        assert X_encoded.shape == (4, 3)  # 4 samples, 3 unique values


class TestFeatureEngineering:
    """Tests for feature engineering functions."""
    
    def test_handle_missing_mean(self):
        """Test handle_missing_values with mean strategy."""
        df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [1, np.nan, 3, 4]})
        result = handle_missing_values(df, strategy='mean')
        
        assert result['a'].isnull().sum() == 0
        assert result['b'].isnull().sum() == 0
    
    def test_handle_missing_median(self):
        """Test handle_missing_values with median strategy."""
        df = pd.DataFrame({'a': [1, 2, np.nan, 4]})
        result = handle_missing_values(df, strategy='median')
        
        assert result['a'].isnull().sum() == 0
    
    def test_handle_missing_constant(self):
        """Test handle_missing_values with constant strategy."""
        df = pd.DataFrame({'a': [1, 2, np.nan, 4]})
        result = handle_missing_values(df, strategy='constant', fill_value=0)
        
        assert result['a'].isnull().sum() == 0
        assert result['a'].tolist() == [1, 2, 0, 4]
    
    def test_polynomial_features(self):
        """Test create_polynomial_features."""
        X = np.array([[1, 2], [3, 4]])
        X_poly = create_polynomial_features(X, degree=2)
        
        # Should have original features plus squared features
        assert X_poly.shape[1] == 4  # 2 original + 2 squared
    
    def test_preprocessing_pipeline(self):
        """Test PreprocessingPipeline."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0],
            'b': [10.0, 20.0, 30.0, 40.0],
        })
        
        pipeline = PreprocessingPipeline()
        pipeline.add_scaler('standard')
        
        result = pipeline.fit_transform(df)
        
        assert pipeline.fitted
        assert result.shape == df.shape


class TestMetrics:
    """Tests for metrics functions."""
    
    def test_mae(self):
        """Test Mean Absolute Error."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        
        assert mae(y_true, y_pred) == 0
        
        y_pred = np.array([2, 3, 4, 5])
        assert mae(y_true, y_pred) == 1
    
    def test_mse(self):
        """Test Mean Squared Error."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        
        assert mse(y_true, y_pred) == 0
        
        y_pred = np.array([2, 3, 4, 5])
        assert mse(y_true, y_pred) == 1
    
    def test_rmse(self):
        """Test Root Mean Squared Error."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 3, 4, 5])
        
        assert rmse(y_true, y_pred) == 1
    
    def test_r2_score(self):
        """Test R-squared score."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        
        assert r2_score(y_true, y_pred) == 1.0
    
    def test_accuracy(self):
        """Test accuracy."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        
        assert accuracy(y_true, y_pred) == 1.0
        
        y_pred = np.array([1, 0, 1, 0])
        assert accuracy(y_true, y_pred) == 0.5
    
    def test_precision(self):
        """Test precision."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        
        assert precision(y_true, y_pred) == 1.0
    
    def test_recall(self):
        """Test recall."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        
        assert recall(y_true, y_pred) == 1.0
    
    def test_f1_score(self):
        """Test F1 score."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        
        assert f1_score(y_true, y_pred) == 1.0


class TestUtils:
    """Tests for utility classes."""
    
    def test_timer(self):
        """Test Timer context manager."""
        import time
        
        with Timer("test", log=False) as t:
            time.sleep(0.1)
        
        assert t.elapsed is not None
        assert t.elapsed >= 0.1
    
    def test_exceptions(self):
        """Test custom exceptions."""
        with pytest.raises(TrainingError):
            raise TrainingError("Test error")
        
        with pytest.raises(InferenceError):
            raise InferenceError("Test error")
