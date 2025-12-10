"""
Custom exceptions for ML-Ops platform.
"""


class MLOpsException(Exception):
    """Base exception for ML-Ops platform."""
    pass


class TrainingError(MLOpsException):
    """Exception raised during model training."""
    pass


class InferenceError(MLOpsException):
    """Exception raised during inference."""
    pass


class DataError(MLOpsException):
    """Exception raised during data operations."""
    pass


class ModelRegistryError(MLOpsException):
    """Exception raised during model registry operations."""
    pass


class PreprocessingError(MLOpsException):
    """Exception raised during preprocessing."""
    pass
