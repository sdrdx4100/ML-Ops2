"""
Predictor service for model inference.
"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, List, Union

from inference_app.models import InferenceLog
from model_registry.services import ModelRegistry
from model_registry.models import ModelArtifact
from shared.utils import get_logger
from shared.utils.exceptions import InferenceError

logger = get_logger(__name__)


class Predictor:
    """
    Service for running model inference.
    
    Supports both scikit-learn and PyTorch models with
    optional SHAP/LIME feature analysis.
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
        self._model_cache: Dict[str, tuple] = {}
    
    def _load_model(self, version: str) -> tuple:
        """Load model from cache or registry."""
        if version not in self._model_cache:
            model, preprocessor = self.registry.load(version)
            self._model_cache[version] = (model, preprocessor)
        return self._model_cache[version]
    
    def _preprocess_input(
        self,
        data: Union[Dict, pd.DataFrame],
        preprocessor: Optional[Any],
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Preprocess input data."""
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Get feature columns from metadata
        feature_columns = metadata.get('feature_columns', [])
        if feature_columns:
            # Ensure columns are in correct order
            for col in feature_columns:
                if col not in df.columns:
                    raise InferenceError(f"Missing feature column: {col}")
            df = df[feature_columns]
        
        # Apply preprocessor if available
        if preprocessor is not None:
            if hasattr(preprocessor, 'transform'):
                df = preprocessor.transform(df)
        
        if isinstance(df, pd.DataFrame):
            return df.values
        return df
    
    def _predict_sklearn(self, model, X: np.ndarray) -> np.ndarray:
        """Run prediction with scikit-learn model."""
        return model.predict(X)
    
    def _predict_torch(self, model, X: np.ndarray) -> np.ndarray:
        """Run prediction with PyTorch model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        X_t = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            predictions = model(X_t).cpu().numpy()
        
        return predictions.ravel()
    
    def _compute_shap_values(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        is_torch: bool = False
    ) -> Dict[str, Any]:
        """Compute SHAP values for feature importance."""
        try:
            import shap
            
            if is_torch:
                # Use DeepExplainer for PyTorch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                model.eval()
                
                # Use a subset for background
                background = torch.FloatTensor(X[:min(100, len(X))]).to(device)
                explainer = shap.DeepExplainer(model, background)
                X_t = torch.FloatTensor(X).to(device)
                shap_values = explainer.shap_values(X_t)
            else:
                # Use TreeExplainer for tree models, or KernelExplainer otherwise
                if hasattr(model, 'estimators_') or hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    # Use KernelExplainer as fallback
                    background = shap.sample(X, min(100, len(X)))
                    explainer = shap.KernelExplainer(model.predict, background)
                
                shap_values = explainer.shap_values(X)
            
            # Convert to feature importance
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
            
            shap_values = np.asarray(shap_values)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Ensure feature names match shap values dimensions
            if len(feature_names) != len(mean_abs_shap):
                feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]
            
            importance = dict(zip(feature_names, mean_abs_shap.tolist()))
            
            return {
                'method': 'shap',
                'feature_importance': importance,
            }
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {str(e)}")
            return {'method': 'shap', 'error': str(e)}
    
    def _compute_lime_explanation(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        is_torch: bool = False
    ) -> Dict[str, Any]:
        """Compute LIME explanation for a single prediction."""
        try:
            import lime
            import lime.lime_tabular
            
            # Create predictor function
            if is_torch:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                model.eval()
                
                def predict_fn(x):
                    x_t = torch.FloatTensor(x).to(device)
                    with torch.no_grad():
                        return model(x_t).cpu().numpy().ravel()
            else:
                predict_fn = model.predict
            
            # Create explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=feature_names,
                mode='regression'
            )
            
            # Explain first instance
            exp = explainer.explain_instance(X[0], predict_fn, num_features=len(feature_names))
            
            # Extract feature importance
            importance = dict(exp.as_list())
            
            return {
                'method': 'lime',
                'feature_importance': importance,
            }
            
        except Exception as e:
            logger.warning(f"LIME analysis failed: {str(e)}")
            return {'method': 'lime', 'error': str(e)}
    
    def predict(
        self,
        version: str,
        data: Union[Dict, pd.DataFrame, List[Dict]],
        analyze: bool = False,
        analysis_method: str = 'shap'
    ) -> Dict[str, Any]:
        """
        Run inference with a trained model.
        
        Args:
            version: Model version to use
            data: Input data (single dict, DataFrame, or list of dicts)
            analyze: Whether to include feature analysis
            analysis_method: 'shap' or 'lime'
        
        Returns:
            Dictionary with predictions and optional analysis
        """
        # Load model
        model, preprocessor = self._load_model(version)
        artifact = self.registry.get_artifact(version)
        metadata = artifact.metadata
        
        # Handle different input formats
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Preprocess
        X = self._preprocess_input(df, preprocessor, metadata)
        
        # Predict
        is_torch = artifact.model_type == 'torch_mlp'
        
        if is_torch:
            predictions = self._predict_torch(model, X)
        else:
            predictions = self._predict_sklearn(model, X)
        
        result = {
            'version': version,
            'predictions': predictions.tolist(),
        }
        
        # Feature analysis
        if analyze:
            feature_names = metadata.get('feature_columns', [])
            if not feature_names:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            if analysis_method == 'lime':
                analysis = self._compute_lime_explanation(model, X, feature_names, is_torch)
            else:
                analysis = self._compute_shap_values(model, X, feature_names, is_torch)
            
            result['analysis'] = analysis
        
        # Log inference
        input_json = df.to_dict(orient='records')
        InferenceLog.objects.create(
            model_version=version,
            input_data=input_json,
            prediction={'predictions': result['predictions']},
            analysis=result.get('analysis', {}),
        )
        
        logger.info(f"Inference completed with model v{version}")
        return result
    
    def predict_file(
        self,
        version: str,
        file_path: str,
        analyze: bool = False,
        analysis_method: str = 'shap'
    ) -> Dict[str, Any]:
        """
        Run inference on a file.
        
        Args:
            version: Model version to use
            file_path: Path to CSV or Parquet file
            analyze: Whether to include feature analysis
            analysis_method: 'shap' or 'lime'
        
        Returns:
            Dictionary with predictions and optional analysis
        """
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
        
        return self.predict(version, df, analyze, analysis_method)
