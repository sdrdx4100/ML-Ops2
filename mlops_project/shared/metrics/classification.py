"""
Classification metrics.
"""
import numpy as np
from typing import Optional


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
    
    Returns:
        Accuracy score
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary',
    pos_label: int = 1
) -> float:
    """
    Calculate precision score.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        average: 'binary', 'micro', 'macro', or 'weighted'
        pos_label: Positive class label for binary classification
    
    Returns:
        Precision score
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if average == 'binary':
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    weights = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        precisions.append(prec)
        weights.append(np.sum(y_true == cls))
    
    if average == 'macro':
        return float(np.mean(precisions))
    elif average == 'weighted':
        return float(np.average(precisions, weights=weights))
    elif average == 'micro':
        tp_total = sum(np.sum((y_pred == cls) & (y_true == cls)) for cls in classes)
        fp_total = sum(np.sum((y_pred == cls) & (y_true != cls)) for cls in classes)
        return float(tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
    
    return float(np.mean(precisions))


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary',
    pos_label: int = 1
) -> float:
    """
    Calculate recall score.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        average: 'binary', 'micro', 'macro', or 'weighted'
        pos_label: Positive class label for binary classification
    
    Returns:
        Recall score
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if average == 'binary':
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    weights = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        recalls.append(rec)
        weights.append(np.sum(y_true == cls))
    
    if average == 'macro':
        return float(np.mean(recalls))
    elif average == 'weighted':
        return float(np.average(recalls, weights=weights))
    elif average == 'micro':
        tp_total = sum(np.sum((y_pred == cls) & (y_true == cls)) for cls in classes)
        fn_total = sum(np.sum((y_pred != cls) & (y_true == cls)) for cls in classes)
        return float(tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0
    
    return float(np.mean(recalls))


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary',
    pos_label: int = 1
) -> float:
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        average: 'binary', 'micro', 'macro', or 'weighted'
        pos_label: Positive class label for binary classification
    
    Returns:
        F1 score
    """
    prec = precision(y_true, y_pred, average=average, pos_label=pos_label)
    rec = recall(y_true, y_pred, average=average, pos_label=pos_label)
    
    if prec + rec == 0:
        return 0.0
    
    return float(2 * (prec * rec) / (prec + rec))


def roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: int = 1
) -> float:
    """
    Calculate ROC AUC (Area Under the Receiver Operating Characteristic Curve).
    
    Uses the trapezoidal rule to compute the area under the ROC curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates or decision function)
        pos_label: Positive class label
    
    Returns:
        ROC AUC score
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    # Convert to binary
    y_binary = (y_true == pos_label).astype(int)
    
    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_binary = y_binary[desc_score_indices]
    
    # Get distinct threshold values
    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.concatenate(([0], distinct_indices + 1))
    
    # Calculate TPR and FPR
    tps = np.cumsum(y_binary)
    fps = np.cumsum(1 - y_binary)
    
    n_pos = np.sum(y_binary)
    n_neg = len(y_binary) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Add endpoints
    tpr = np.concatenate([[0], tpr, [1]])
    fpr = np.concatenate([[0], fpr, [1]])
    
    # Calculate AUC using trapezoidal rule
    auc = float(np.trapz(tpr, fpr))
    
    return auc
