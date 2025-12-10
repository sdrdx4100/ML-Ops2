"""
Scalers for data preprocessing.
"""
import numpy as np
from typing import Optional, Dict, Any
import pickle


class BaseScaler:
    """Base class for scalers."""
    
    def __init__(self):
        self.fitted = False
        self.feature_names: Optional[list] = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None) -> 'BaseScaler':
        """Fit the scaler to data."""
        raise NotImplementedError
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        raise NotImplementedError
    
    def fit_transform(self, X: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        """Fit and transform data."""
        self.fit(X, feature_names)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler to dictionary."""
        raise NotImplementedError
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseScaler':
        """Deserialize scaler from dictionary."""
        raise NotImplementedError


class StandardScaler(BaseScaler):
    """
    Standard scaler that centers data to zero mean and unit variance.
    """
    
    def __init__(self):
        super().__init__()
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None) -> 'StandardScaler':
        """Fit the scaler to data."""
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        self.feature_names = feature_names
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to zero mean and unit variance."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return X * self.std_ + self.mean_
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler to dictionary."""
        return {
            'type': 'StandardScaler',
            'mean': self.mean_.tolist() if self.mean_ is not None else None,
            'std': self.std_.tolist() if self.std_ is not None else None,
            'feature_names': self.feature_names,
            'fitted': self.fitted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardScaler':
        """Deserialize scaler from dictionary."""
        scaler = cls()
        scaler.mean_ = np.array(data['mean']) if data['mean'] is not None else None
        scaler.std_ = np.array(data['std']) if data['std'] is not None else None
        scaler.feature_names = data.get('feature_names')
        scaler.fitted = data.get('fitted', False)
        return scaler


class MinMaxScaler(BaseScaler):
    """
    Min-max scaler that scales data to a specified range.
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        super().__init__()
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None) -> 'MinMaxScaler':
        """Fit the scaler to data."""
        X = np.asarray(X, dtype=np.float64)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        data_range = self.max_ - self.min_
        data_range[data_range == 0] = 1.0
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.feature_names = feature_names
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to specified range."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        X_std = (X - self.min_) * self.scale_
        return X_std + self.feature_range[0]
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        X_std = (X - self.feature_range[0]) / self.scale_
        return X_std + self.min_
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler to dictionary."""
        return {
            'type': 'MinMaxScaler',
            'min': self.min_.tolist() if self.min_ is not None else None,
            'max': self.max_.tolist() if self.max_ is not None else None,
            'scale': self.scale_.tolist() if self.scale_ is not None else None,
            'feature_range': self.feature_range,
            'feature_names': self.feature_names,
            'fitted': self.fitted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MinMaxScaler':
        """Deserialize scaler from dictionary."""
        scaler = cls(feature_range=tuple(data.get('feature_range', (0, 1))))
        scaler.min_ = np.array(data['min']) if data['min'] is not None else None
        scaler.max_ = np.array(data['max']) if data['max'] is not None else None
        scaler.scale_ = np.array(data['scale']) if data['scale'] is not None else None
        scaler.feature_names = data.get('feature_names')
        scaler.fitted = data.get('fitted', False)
        return scaler


class RobustScaler(BaseScaler):
    """
    Robust scaler using median and IQR, resistant to outliers.
    """
    
    def __init__(self):
        super().__init__()
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None) -> 'RobustScaler':
        """Fit the scaler to data."""
        X = np.asarray(X, dtype=np.float64)
        self.median_ = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.iqr_ = q3 - q1
        # Avoid division by zero
        self.iqr_[self.iqr_ == 0] = 1.0
        self.feature_names = feature_names
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using median and IQR."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.median_) / self.iqr_
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return X * self.iqr_ + self.median_
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler to dictionary."""
        return {
            'type': 'RobustScaler',
            'median': self.median_.tolist() if self.median_ is not None else None,
            'iqr': self.iqr_.tolist() if self.iqr_ is not None else None,
            'feature_names': self.feature_names,
            'fitted': self.fitted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RobustScaler':
        """Deserialize scaler from dictionary."""
        scaler = cls()
        scaler.median_ = np.array(data['median']) if data['median'] is not None else None
        scaler.iqr_ = np.array(data['iqr']) if data['iqr'] is not None else None
        scaler.feature_names = data.get('feature_names')
        scaler.fitted = data.get('fitted', False)
        return scaler
