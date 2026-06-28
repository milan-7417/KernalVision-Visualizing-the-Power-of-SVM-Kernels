"""
Dataset Generator Utility for KernelVision.
Provides functions to generate standard classification datasets:
- Two Moons
- Concentric Circles
- Blobs
- XOR Dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_blobs

def generate_moons(n_samples: int = 300, noise: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a 2D dataset of two interleaving half circles (Moons).
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    # Scale points slightly for visual layout
    X = X * 5.0
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y.astype(int)
    return df

def generate_circles(n_samples: int = 300, noise: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a 2D dataset of concentric circles.
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    X = X * 5.0
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y.astype(int)
    return df

def generate_blobs(n_samples: int = 300, noise: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a 2D dataset of two isotropic Gaussian clusters.
    """
    # Map noise [0.0, 1.0] to cluster standard deviation [0.2, 4.0]
    cluster_std = 0.2 + noise * 3.8
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=cluster_std, random_state=random_state)
    # Shift to center around 0
    X = X - X.mean(axis=0)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y.astype(int)
    return df

def generate_xor(n_samples: int = 300, noise: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a 2D dataset representing the XOR logic boundary.
    Composed of 4 clusters, opposite corners sharing the same class.
    """
    np.random.seed(random_state)
    n_per_cluster = n_samples // 4
    # Map noise to standard deviation
    std = 0.2 + noise * 1.8
    
    # Define centers
    c0_1 = np.random.normal(loc=[-2.5, -2.5], scale=std, size=(n_per_cluster, 2))
    c0_2 = np.random.normal(loc=[2.5, 2.5], scale=std, size=(n_per_cluster, 2))
    c1_1 = np.random.normal(loc=[-2.5, 2.5], scale=std, size=(n_per_cluster, 2))
    c1_2 = np.random.normal(loc=[2.5, -2.5], scale=std, size=(n_per_cluster, 2))
    
    X = np.vstack([c0_1, c0_2, c1_1, c1_2])
    y = np.array([0]*len(c0_1) + [0]*len(c0_2) + [1]*len(c1_1) + [1]*len(c1_2))
    
    # Handle remainder samples
    rem = n_samples % 4
    if rem > 0:
        for _ in range(rem):
            cluster = np.random.choice([0, 1, 2, 3])
            if cluster == 0:
                pt = np.random.normal(loc=[-2.5, -2.5], scale=std, size=(1, 2))
                lbl = 0
            elif cluster == 1:
                pt = np.random.normal(loc=[2.5, 2.5], scale=std, size=(1, 2))
                lbl = 0
            elif cluster == 2:
                pt = np.random.normal(loc=[-2.5, 2.5], scale=std, size=(1, 2))
                lbl = 1
            else:
                pt = np.random.normal(loc=[2.5, -2.5], scale=std, size=(1, 2))
                lbl = 1
            X = np.vstack([X, pt])
            y = np.append(y, lbl)
            
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y.astype(int)
    return df
