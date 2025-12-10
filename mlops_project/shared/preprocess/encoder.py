"""
Encoders for categorical data preprocessing.
"""
import numpy as np
from typing import Optional, Dict, Any, List


class LabelEncoder:
    """
    Label encoder that transforms categorical values to integers.
    """
    
    def __init__(self):
        self.classes_: Optional[np.ndarray] = None
        self.class_to_int_: Optional[Dict[Any, int]] = None
        self.fitted = False
    
    def fit(self, y: np.ndarray) -> 'LabelEncoder':
        """Fit the encoder to data."""
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.class_to_int_ = {c: i for i, c in enumerate(self.classes_)}
        self.fitted = True
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform data to integer labels."""
        if not self.fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        y = np.asarray(y).ravel()
        return np.array([self.class_to_int_[val] for val in y])
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform integer labels to original values."""
        if not self.fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        y = np.asarray(y).ravel()
        return self.classes_[y]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize encoder to dictionary."""
        return {
            'type': 'LabelEncoder',
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'fitted': self.fitted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LabelEncoder':
        """Deserialize encoder from dictionary."""
        encoder = cls()
        if data.get('classes') is not None:
            encoder.classes_ = np.array(data['classes'])
            encoder.class_to_int_ = {c: i for i, c in enumerate(encoder.classes_)}
        encoder.fitted = data.get('fitted', False)
        return encoder


class OneHotEncoder:
    """
    One-hot encoder that transforms categorical values to binary arrays.
    """
    
    def __init__(self, handle_unknown: str = 'error'):
        """
        Initialize encoder.
        
        Args:
            handle_unknown: How to handle unknown categories during transform.
                           'error' raises an error, 'ignore' returns all zeros.
        """
        self.handle_unknown = handle_unknown
        self.categories_: Optional[List[np.ndarray]] = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """Fit the encoder to data."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.categories_ = []
        for col in range(X.shape[1]):
            self.categories_.append(np.unique(X[:, col]))
        
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to one-hot encoding."""
        if not self.fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Calculate total number of output columns
        n_features = sum(len(cats) for cats in self.categories_)
        result = np.zeros((X.shape[0], n_features), dtype=np.float64)
        
        col_offset = 0
        for col_idx, categories in enumerate(self.categories_):
            for row_idx, val in enumerate(X[:, col_idx]):
                indices = np.where(categories == val)[0]
                if len(indices) > 0:
                    result[row_idx, col_offset + indices[0]] = 1.0
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{val}' in column {col_idx}")
            col_offset += len(categories)
        
        return result
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        self.fit(X)
        return self.transform(X)
    
    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get feature names for output columns."""
        if not self.fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        if input_features is None:
            input_features = [f"x{i}" for i in range(len(self.categories_))]
        
        names = []
        for i, categories in enumerate(self.categories_):
            for cat in categories:
                names.append(f"{input_features[i]}_{cat}")
        
        return names
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize encoder to dictionary."""
        return {
            'type': 'OneHotEncoder',
            'categories': [c.tolist() for c in self.categories_] if self.categories_ else None,
            'handle_unknown': self.handle_unknown,
            'fitted': self.fitted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OneHotEncoder':
        """Deserialize encoder from dictionary."""
        encoder = cls(handle_unknown=data.get('handle_unknown', 'error'))
        if data.get('categories') is not None:
            encoder.categories_ = [np.array(c) for c in data['categories']]
        encoder.fitted = data.get('fitted', False)
        return encoder
