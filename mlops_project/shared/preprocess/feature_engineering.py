"""
Feature engineering utilities for preprocessing.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union
from .scaler import StandardScaler, MinMaxScaler, RobustScaler
from .encoder import LabelEncoder, OneHotEncoder


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    fill_value: Optional[Any] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values. Options:
                 'mean', 'median', 'mode', 'constant', 'drop'
        fill_value: Value to fill when strategy='constant'
        columns: Columns to apply the strategy to (None for all)
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif strategy == 'constant':
            df[col] = df[col].fillna(fill_value)
        elif strategy == 'drop':
            df = df.dropna(subset=[col])
    
    return df


def create_polynomial_features(
    X: np.ndarray,
    degree: int = 2,
    include_bias: bool = False
) -> np.ndarray:
    """
    Create polynomial features up to specified degree.
    
    Args:
        X: Input array of shape (n_samples, n_features)
        degree: Maximum degree of polynomial features
        include_bias: Whether to include a bias column of ones
    
    Returns:
        Array with polynomial features
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    features = [X]
    
    # Add polynomial features
    for d in range(2, degree + 1):
        for i in range(n_features):
            features.append(X[:, i:i+1] ** d)
    
    result = np.hstack(features)
    
    if include_bias:
        result = np.hstack([np.ones((n_samples, 1)), result])
    
    return result


class PreprocessingPipeline:
    """
    Pipeline for chaining multiple preprocessing steps.
    """
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.fitted = False
    
    def add_scaler(
        self,
        scaler_type: str = 'standard',
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'PreprocessingPipeline':
        """Add a scaling step."""
        scaler_map = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
        }
        
        scaler_class = scaler_map.get(scaler_type)
        if scaler_class is None:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.steps.append({
            'type': 'scaler',
            'scaler_type': scaler_type,
            'columns': columns,
            'scaler': scaler_class(**kwargs) if kwargs else scaler_class(),
        })
        return self
    
    def add_encoder(
        self,
        encoder_type: str = 'label',
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'PreprocessingPipeline':
        """Add an encoding step."""
        encoder_map = {
            'label': LabelEncoder,
            'onehot': OneHotEncoder,
        }
        
        encoder_class = encoder_map.get(encoder_type)
        if encoder_class is None:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.steps.append({
            'type': 'encoder',
            'encoder_type': encoder_type,
            'columns': columns,
            'encoder': encoder_class(**kwargs) if kwargs else encoder_class(),
        })
        return self
    
    def fit(self, df: pd.DataFrame) -> 'PreprocessingPipeline':
        """Fit all preprocessing steps."""
        df = df.copy()
        
        for step in self.steps:
            if step['type'] == 'scaler':
                cols = step['columns'] or df.select_dtypes(include=[np.number]).columns.tolist()
                if cols:
                    step['scaler'].fit(df[cols].values, feature_names=cols)
            elif step['type'] == 'encoder':
                cols = step['columns']
                if cols:
                    for col in cols:
                        if col in df.columns:
                            encoder = type(step['encoder'])()
                            encoder.fit(df[col].values)
                            if 'encoders' not in step:
                                step['encoders'] = {}
                            step['encoders'][col] = encoder
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps."""
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        df = df.copy()
        
        for step in self.steps:
            if step['type'] == 'scaler':
                cols = step['columns'] or df.select_dtypes(include=[np.number]).columns.tolist()
                if cols:
                    df[cols] = step['scaler'].transform(df[cols].values)
            elif step['type'] == 'encoder':
                encoders = step.get('encoders', {})
                for col, encoder in encoders.items():
                    if col in df.columns:
                        if isinstance(encoder, LabelEncoder):
                            df[col] = encoder.transform(df[col].values)
                        elif isinstance(encoder, OneHotEncoder):
                            encoded = encoder.transform(df[col].values)
                            feature_names = encoder.get_feature_names([col])
                            for i, name in enumerate(feature_names):
                                df[name] = encoded[:, i]
                            df = df.drop(columns=[col])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data."""
        self.fit(df)
        return self.transform(df)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline to dictionary."""
        serialized_steps = []
        for step in self.steps:
            step_data = {
                'type': step['type'],
                'columns': step['columns'],
            }
            if step['type'] == 'scaler':
                step_data['scaler_type'] = step['scaler_type']
                step_data['scaler'] = step['scaler'].to_dict()
            elif step['type'] == 'encoder':
                step_data['encoder_type'] = step['encoder_type']
                if 'encoders' in step:
                    step_data['encoders'] = {
                        col: enc.to_dict() for col, enc in step['encoders'].items()
                    }
            serialized_steps.append(step_data)
        
        return {
            'steps': serialized_steps,
            'fitted': self.fitted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingPipeline':
        """Deserialize pipeline from dictionary."""
        pipeline = cls()
        
        scaler_classes = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
        }
        
        encoder_classes = {
            'label': LabelEncoder,
            'onehot': OneHotEncoder,
        }
        
        for step_data in data.get('steps', []):
            step = {
                'type': step_data['type'],
                'columns': step_data['columns'],
            }
            
            if step_data['type'] == 'scaler':
                step['scaler_type'] = step_data['scaler_type']
                scaler_class = scaler_classes[step_data['scaler_type']]
                step['scaler'] = scaler_class.from_dict(step_data['scaler'])
            elif step_data['type'] == 'encoder':
                step['encoder_type'] = step_data['encoder_type']
                encoder_class = encoder_classes[step_data['encoder_type']]
                step['encoder'] = encoder_class()
                if 'encoders' in step_data:
                    step['encoders'] = {
                        col: encoder_class.from_dict(enc_data)
                        for col, enc_data in step_data['encoders'].items()
                    }
            
            pipeline.steps.append(step)
        
        pipeline.fitted = data.get('fitted', False)
        return pipeline


def apply_preprocessing_pipeline(
    df: pd.DataFrame,
    config: Dict[str, Any],
    pipeline: Optional[PreprocessingPipeline] = None
) -> tuple:
    """
    Apply preprocessing based on configuration.
    
    Args:
        df: Input DataFrame
        config: Preprocessing configuration
        pipeline: Optional existing pipeline (for inference)
    
    Returns:
        Tuple of (processed_df, pipeline)
    """
    if pipeline is not None:
        # Use existing pipeline (inference mode)
        processed_df = pipeline.transform(df)
        return processed_df, pipeline
    
    # Create new pipeline (training mode)
    pipeline = PreprocessingPipeline()
    
    # Handle missing values first
    missing_strategy = config.get('missing_strategy', 'mean')
    df = handle_missing_values(df, strategy=missing_strategy)
    
    # Add scaling
    if config.get('scale', True):
        scaler_type = config.get('scaler_type', 'standard')
        numeric_cols = config.get('numeric_columns')
        pipeline.add_scaler(scaler_type=scaler_type, columns=numeric_cols)
    
    # Add encoding
    categorical_cols = config.get('categorical_columns', [])
    if categorical_cols:
        encoder_type = config.get('encoder_type', 'label')
        pipeline.add_encoder(encoder_type=encoder_type, columns=categorical_cols)
    
    processed_df = pipeline.fit_transform(df)
    return processed_df, pipeline
