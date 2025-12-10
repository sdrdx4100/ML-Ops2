"""
Plotting functions for ML-Ops platform.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
import io
import base64


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def plot_learning_curve(
    train_scores: List[float],
    val_scores: List[float],
    title: str = "Learning Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Score"
) -> str:
    """
    Plot learning curve.
    
    Args:
        train_scores: Training scores per epoch
        val_scores: Validation scores per epoch
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    
    Returns:
        Base64-encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_scores) + 1)
    ax.plot(epochs, train_scores, 'b-', label='Training')
    ax.plot(epochs, val_scores, 'r-', label='Validation')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return _fig_to_base64(fig)


def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    title: str = "Feature Importance",
    top_n: int = 20
) -> str:
    """
    Plot feature importance.
    
    Args:
        feature_names: Names of features
        importances: Importance scores
        title: Plot title
        top_n: Number of top features to display
    
    Returns:
        Base64-encoded PNG image
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = [importances[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importances[::-1], align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    return _fig_to_base64(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False
) -> str:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Plot title
        normalize: Whether to normalize values
    
    Returns:
        Base64-encoded PNG image
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    if class_names is None:
        class_names = [str(c) for c in classes]
    
    # Build confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')
    
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual"
) -> str:
    """
    Plot predicted vs actual values for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
    
    Returns:
        Base64-encoded PNG image
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, color='steelblue')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return _fig_to_base64(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot"
) -> str:
    """
    Plot residuals for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
    
    Returns:
        Base64-encoded PNG image
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, color='steelblue')
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title)
    fig.tight_layout()
    
    return _fig_to_base64(fig)
