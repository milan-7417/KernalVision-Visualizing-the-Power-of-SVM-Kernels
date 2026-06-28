"""
SVM Model Utility for KernelVision.
Handles SVM training, prediction, and retrieval of model parameters.
"""

from sklearn.svm import SVC
import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Any

def train_svm(
    X: Union[np.ndarray, pd.DataFrame], 
    y: Union[np.ndarray, pd.Series], 
    kernel: str = 'rbf', 
    C: float = 1.0, 
    gamma: Union[str, float] = 'scale', 
    degree: int = 3, 
    coef0: float = 0.0
) -> SVC:
    """
    Trains a Support Vector Classifier model on the provided data.
    
    Parameters:
    - X: Training vectors (features)
    - y: Target values (labels)
    - kernel: Kernel function type ('linear', 'poly', 'rbf', 'sigmoid')
    - C: Regularization parameter
    - gamma: Kernel coefficient (for 'rbf', 'poly', 'sigmoid')
    - degree: Degree of polynomial kernel function (ignored by other kernels)
    - coef0: Independent term in kernel function (for 'poly' and 'sigmoid')
    
    Returns:
    - Fitted SVC model
    """
    # Clean input check
    if isinstance(gamma, str) and gamma not in ['scale', 'auto']:
        try:
            gamma = float(gamma)
        except ValueError:
            gamma = 'scale' # default fallback
            
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        probability=True,  # helpful for decision contours and soft prediction
        random_state=42
    )
    model.fit(X, y)
    return model

def predict_svm(model: SVC, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Predicts classes for the given feature set.
    """
    return model.predict(X)

def get_support_vectors_info(model: SVC) -> Dict[str, Any]:
    """
    Retrieves support vector matrices and metrics.
    
    Returns a dictionary with:
    - 'support_vectors': array of support vectors
    - 'n_support': total number of support vectors
    - 'n_support_per_class': list showing count of support vectors per class
    - 'support_indices': indices of support vectors in training set
    """
    return {
        "support_vectors": model.support_vectors_,
        "n_support": len(model.support_),
        "n_support_per_class": model.n_support_.tolist(),
        "support_indices": model.support_
    }
