"""
Metrics Utility for KernelVision.
Computes classification performance metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from typing import Dict, Tuple

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard classification evaluation metrics.
    
    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    
    Returns:
    - Dict with accuracy, precision, recall, and f1 score.
    """
    # Using zero_division=0 to prevent errors if a class is never predicted
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }

def get_confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes the confusion matrix.
    
    Returns:
    - 2x2 confusion matrix array
    """
    return confusion_matrix(y_true, y_pred)
