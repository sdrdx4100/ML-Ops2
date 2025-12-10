"""
Trainer service for model training.
"""
import uuid
import numpy as np
import pandas as pd
from django.utils import timezone
from typing import Dict, Any, Optional, List, Tuple
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from training_app.models import TrainingJob
from model_registry.services import ModelRegistry
from datasets.services import DataLoader
from shared.utils import get_logger, Timer
from shared.utils.exceptions import TrainingError
from shared.preprocess import apply_preprocessing_pipeline
from shared.metrics import mae, mse, rmse, r2_score, accuracy, f1_score

logger = get_logger(__name__)


class TorchMLP(nn.Module):
    """Simple MLP for regression/classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class Trainer:
    """
    Service for training ML models.
    
    Supports scikit-learn and PyTorch models.
    """
    
    SUPPORTED_MODELS = [
        'linear_regression',
        'random_forest',
        'sgd_regressor',
        'random_forest_classifier',
        'torch_mlp',
    ]
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.registry = ModelRegistry()
    
    def _create_sklearn_model(
        self,
        model_type: str,
        hyperparameters: Dict[str, Any]
    ):
        """Create a scikit-learn model instance."""
        if model_type == 'linear_regression':
            return LinearRegression(**hyperparameters)
        elif model_type == 'random_forest':
            params = {
                'n_estimators': hyperparameters.get('n_estimators', 100),
                'max_depth': hyperparameters.get('max_depth', None),
                'min_samples_split': hyperparameters.get('min_samples_split', 2),
                'random_state': hyperparameters.get('random_state', 42),
            }
            return RandomForestRegressor(**params)
        elif model_type == 'sgd_regressor':
            params = {
                'max_iter': hyperparameters.get('max_iter', 1000),
                'tol': hyperparameters.get('tol', 1e-3),
                'random_state': hyperparameters.get('random_state', 42),
            }
            return SGDRegressor(**params)
        elif model_type == 'random_forest_classifier':
            params = {
                'n_estimators': hyperparameters.get('n_estimators', 100),
                'max_depth': hyperparameters.get('max_depth', None),
                'min_samples_split': hyperparameters.get('min_samples_split', 2),
                'random_state': hyperparameters.get('random_state', 42),
            }
            return RandomForestClassifier(**params)
        else:
            raise TrainingError(f"Unknown model type: {model_type}")
    
    def _create_torch_model(
        self,
        input_dim: int,
        hyperparameters: Dict[str, Any]
    ) -> TorchMLP:
        """Create a PyTorch MLP model."""
        hidden_dims = hyperparameters.get('hidden_dims', [64, 32])
        output_dim = hyperparameters.get('output_dim', 1)
        dropout = hyperparameters.get('dropout', 0.2)
        
        return TorchMLP(input_dim, hidden_dims, output_dim, dropout)
    
    def _train_sklearn_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        is_classifier: bool = False
    ) -> Dict[str, float]:
        """Train a scikit-learn model and compute metrics."""
        with Timer("sklearn training"):
            model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        if is_classifier:
            metrics = {
                'train_accuracy': accuracy(y_train, y_pred_train),
                'val_accuracy': accuracy(y_val, y_pred_val),
                'train_f1': f1_score(y_train, y_pred_train, average='macro'),
                'val_f1': f1_score(y_val, y_pred_val, average='macro'),
            }
        else:
            metrics = {
                'train_mae': mae(y_train, y_pred_train),
                'val_mae': mae(y_val, y_pred_val),
                'train_mse': mse(y_train, y_pred_train),
                'val_mse': mse(y_val, y_pred_val),
                'train_r2': r2_score(y_train, y_pred_train),
                'val_r2': r2_score(y_val, y_pred_val),
            }
        
        return metrics
    
    def _train_torch_model(
        self,
        model: TorchMLP,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Train a PyTorch model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Prepare data
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        batch_size = hyperparameters.get('batch_size', 32)
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        lr = hyperparameters.get('learning_rate', 0.001)
        epochs = hyperparameters.get('epochs', 100)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        with Timer("torch training"):
            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
        
        # Compute metrics
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train_t).cpu().numpy().ravel()
            y_pred_val = model(X_val_t).cpu().numpy().ravel()
        
        metrics = {
            'train_mae': mae(y_train, y_pred_train),
            'val_mae': mae(y_val, y_pred_val),
            'train_mse': mse(y_train, y_pred_train),
            'val_mse': mse(y_val, y_pred_val),
            'train_r2': r2_score(y_train, y_pred_train),
            'val_r2': r2_score(y_val, y_pred_val),
        }
        
        return metrics
    
    def _fine_tune_sklearn(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        is_classifier: bool = False
    ) -> Dict[str, float]:
        """Fine-tune a scikit-learn model using partial_fit if available."""
        if hasattr(model, 'partial_fit'):
            with Timer("sklearn fine-tuning"):
                # For classifiers, may need to pass classes
                if is_classifier and hasattr(model, 'classes_'):
                    model.partial_fit(X_train, y_train)
                else:
                    model.partial_fit(X_train, y_train)
        else:
            # Fall back to full retraining
            with Timer("sklearn retraining"):
                model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        if is_classifier:
            metrics = {
                'train_accuracy': accuracy(y_train, y_pred_train),
                'val_accuracy': accuracy(y_val, y_pred_val),
                'train_f1': f1_score(y_train, y_pred_train, average='macro'),
                'val_f1': f1_score(y_val, y_pred_val, average='macro'),
            }
        else:
            metrics = {
                'train_mae': mae(y_train, y_pred_train),
                'val_mae': mae(y_val, y_pred_val),
                'train_mse': mse(y_train, y_pred_train),
                'val_mse': mse(y_val, y_pred_val),
                'train_r2': r2_score(y_train, y_pred_train),
                'val_r2': r2_score(y_val, y_pred_val),
            }
        
        return metrics
    
    def train(self, config: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """
        Train a model based on configuration.
        
        Args:
            config: Training configuration dict containing:
                - dataset_name: Name of the dataset
                - model_type: Type of model to train
                - training_mode: 'new' or 'fine_tune'
                - feature_columns: List of feature column names
                - target_column: Target column name
                - hyperparameters: Model hyperparameters
                - preprocess_config: Preprocessing configuration
                - base_model_version: (for fine_tune) Version of base model
        
        Returns:
            Tuple of (model_version, metrics)
        """
        # Create job record
        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob.objects.create(
            job_id=job_id,
            dataset_name=config['dataset_name'],
            model_type=config['model_type'],
            training_mode=config.get('training_mode', 'new'),
            hyperparameters=config.get('hyperparameters', {}),
            feature_columns=config.get('feature_columns', []),
            target_column=config['target_column'],
            base_model_version=config.get('base_model_version'),
            status='running'
        )
        
        try:
            # Load data
            df = self.data_loader.load(config['dataset_name'])
            
            # Extract features and target
            feature_cols = config.get('feature_columns', [])
            if not feature_cols:
                feature_cols = [c for c in df.columns if c != config['target_column']]
            
            X = df[feature_cols].values
            y = df[config['target_column']].values
            
            # Apply preprocessing
            preprocess_config = config.get('preprocess_config', {'scale': True})
            X_df = pd.DataFrame(X, columns=feature_cols)
            X_processed, preprocessor = apply_preprocessing_pipeline(X_df, preprocess_config)
            X = X_processed.values
            
            # Split data
            test_size = config.get('test_size', 0.2)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            model_type = config['model_type']
            training_mode = config.get('training_mode', 'new')
            hyperparameters = config.get('hyperparameters', {})
            
            is_classifier = 'classifier' in model_type.lower()
            
            if training_mode == 'fine_tune' and config.get('base_model_version'):
                # Load base model
                model, _ = self.registry.load(config['base_model_version'])
                
                if model_type == 'torch_mlp':
                    metrics = self._train_torch_model(
                        model, X_train, y_train, X_val, y_val, hyperparameters
                    )
                else:
                    metrics = self._fine_tune_sklearn(
                        model, X_train, y_train, X_val, y_val, is_classifier
                    )
            else:
                # New training
                if model_type == 'torch_mlp':
                    model = self._create_torch_model(X_train.shape[1], hyperparameters)
                    metrics = self._train_torch_model(
                        model, X_train, y_train, X_val, y_val, hyperparameters
                    )
                else:
                    model = self._create_sklearn_model(model_type, hyperparameters)
                    metrics = self._train_sklearn_model(
                        model, X_train, y_train, X_val, y_val, is_classifier
                    )
            
            # Save to registry
            metadata = {
                'dataset_name': config['dataset_name'],
                'feature_columns': feature_cols,
                'target_column': config['target_column'],
                'preprocess_config': preprocess_config,
                'hyperparameters': hyperparameters,
            }
            
            artifact = self.registry.save(
                model=model,
                model_type=model_type,
                metrics=metrics,
                preprocessor=preprocessor,
                metadata=metadata,
                training_mode=training_mode,
                parent_version=config.get('base_model_version'),
            )
            
            # Update job
            job.status = 'completed'
            job.result_version = artifact.version
            job.completed_at = timezone.now()
            job.save()
            
            logger.info(f"Training completed. Model version: {artifact.version}")
            return artifact.version, metrics
            
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save()
            
            logger.error(f"Training failed: {str(e)}")
            raise TrainingError(str(e)) from e


def train_model(config: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    """
    Convenience function to train a model.
    
    Args:
        config: Training configuration
    
    Returns:
        Tuple of (model_version, metrics)
    """
    trainer = Trainer()
    return trainer.train(config)
