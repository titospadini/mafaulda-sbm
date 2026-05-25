"""
Multiclass SBM Model-Matrix Construction and Error Vector Generation

This script handles Step 3 of the Rotating-Machine Fault Diagnosis pipeline:
1. Loads the 46-dimensional feature matrices (X_train_features, X_test_features)
   and labels (y_train, y_test) from the data/ directory.
2. Implements the Wegerich Similarity Function (WSF) using L1 norm and gamma =
   0.01.
3. Computes the geometric median for each unique fault class in the training set
   using Weiszfeld's algorithm to seed the class dictionaries (D_c).
4. Constructs the representative dictionary matrices D_c using the threshold
   method (tau = 0.9) strictly on the training set to prevent data leakage.
5. Performs SBM estimation (Model B configuration) for both training and testing
   samples using the learned dictionaries.
6. Computes the estimation error vector using the best-matching estimate
   (maximizing WSF similarity between original sample and its estimate).
7. Concatenates the original 46 features with the 46-dimensional error vectors
   to create 92-dimensional extended feature matrices.
8. Exports the extended matrices to the data/ directory.
"""

import os
import sys
import time
from typing import Dict
import numpy as np

from mafaulda.gpu_utils import is_gpu_available, to_tensor, to_numpy

# Constants from the paper
GAMMA = 0.0010
TAU = 0.85
EXPECTED_ORIGINAL_FEATURES = 46
EXPECTED_EXTENDED_FEATURES = 92


def wegerich_similarity_torch(
    x1: "torch.Tensor",
    x2: "torch.Tensor",
    gamma: float = GAMMA
) -> "torch.Tensor":
    """GPU-accelerated Wegerich Similarity Function using PyTorch Tensors."""
    import torch
    l1_dist = torch.sum(torch.abs(x1 - x2), dim=-1)
    return 1.0 / (1.0 + gamma * l1_dist)


def compute_sbm_estimates_torch(
    X_tensor: "torch.Tensor",
    D_c_dict_tensors: Dict[str, "torch.Tensor"],
    gamma: float = GAMMA
) -> Dict[str, "torch.Tensor"]:
    """GPU-accelerated SBM estimations using PyTorch batched tensor logic."""
    import torch
    estimates = {}

    for class_name, D_c in D_c_dict_tensors.items():
        # Pairwise memory similarities
        G_c = wegerich_similarity_torch(D_c.unsqueeze(1), D_c.unsqueeze(0), gamma=gamma)

        # Pseudo-inverse of G_c (extremely fast on GPU via SVD)
        G_c_pinv = torch.linalg.pinv(G_c)

        # Input-to-memory similarity mapping
        A_c = wegerich_similarity_torch(X_tensor.unsqueeze(1), D_c.unsqueeze(0), gamma=gamma)

        # Interpolation weights
        W_c = torch.matmul(A_c, G_c_pinv)

        # Normalize weights
        w_norms = torch.sum(torch.abs(W_c), dim=1)
        w_norms = torch.clamp(w_norms, min=1e-12)
        W_c_normalized = W_c / w_norms.unsqueeze(1)

        # Compute final estimate
        X_hat_c = torch.matmul(W_c_normalized, D_c)
        estimates[class_name] = X_hat_c

    return estimates


def wegerich_similarity(
    x1: np.ndarray,
    x2: np.ndarray,
    gamma: float = GAMMA
) -> np.ndarray:
    """
    Computes the Wegerich Similarity Function (WSF) between two vectors, or
    between a matrix and a vector, using the L1 norm.
    """
    l1_dist = np.sum(np.abs(x1 - x2), axis=-1)
    return 1.0 / (1.0 + gamma * l1_dist)


def compute_geometric_median(
    X: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Finds the geometric median of a multi-dimensional dataset X using the robust
    iterative Weiszfeld's algorithm.
    """
    if len(X) == 0:
        raise ValueError("Cannot compute geometric median of an empty set.")
    if len(X) == 1:
        return X[0].copy()

    # Initialize the estimate with the mean of the points
    y = np.mean(X, axis=0)

    for _ in range(max_iter):
        distances = np.linalg.norm(X - y, axis=1)
        weights = 1.0 / np.maximum(distances, 1e-12)
        sum_weights = np.sum(weights)
        new_y = np.sum(X * weights[:, np.newaxis], axis=0) / sum_weights
        if np.linalg.norm(new_y - y) < tol:
            return new_y
        y = new_y

    return y


def construct_class_dictionary(
    X_c: np.ndarray,
    tau: float = TAU,
    gamma: float = GAMMA
) -> np.ndarray:
    """
    Constructs the compact SBM representative state dictionary (memory matrix)
    D_c for class c using the threshold method seeded with the class geometric median.
    """
    g_median = compute_geometric_median(X_c, max_iter=2000, tol=1e-6)
    D_c_list = [g_median]

    for x in X_c:
        D_c_arr = np.array(D_c_list)
        similarities = wegerich_similarity(D_c_arr, x, gamma=gamma)
        if all(sim < tau for sim in similarities):
            D_c_list.append(x)

    return np.array(D_c_list)


def compute_sbm_estimates(
    X: np.ndarray,
    D_c_dict: Dict[str, np.ndarray],
    gamma: float = GAMMA
) -> Dict[str, np.ndarray]:
    """
    Computes SBM state estimations on CPU for all input samples X across all fault classes.
    """
    estimates = {}

    for class_name, D_c in D_c_dict.items():
        G_c = wegerich_similarity(D_c[:, np.newaxis, :], D_c[np.newaxis, :, :], gamma=gamma)
        G_c_pinv = np.linalg.pinv(G_c)
        A_c = wegerich_similarity(X[:, np.newaxis, :], D_c[np.newaxis, :, :], gamma=gamma)
        W_c = A_c @ G_c_pinv
        w_norms = np.sum(np.abs(W_c), axis=1)
        w_norms = np.maximum(w_norms, 1e-12)
        W_c_normalized = W_c / w_norms[:, np.newaxis]
        X_hat_c = W_c_normalized @ D_c
        estimates[class_name] = X_hat_c

    return estimates


def generate_extended_features(
    X: np.ndarray,
    D_c_dict: Dict[str, np.ndarray],
    gamma: float = GAMMA,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Constructs the 92-dimensional extended feature matrix by appending the
    best-matching SBM estimation error vector to the original 46 features.
    """
    if use_gpu and is_gpu_available():
        import torch
        X_tensor = to_tensor(X)
        D_c_dict_tensors = {k: to_tensor(v) for k, v in D_c_dict.items()}

        estimates = compute_sbm_estimates_torch(X_tensor, D_c_dict_tensors, gamma=gamma)
        class_names = list(D_c_dict.keys())
        N = len(X)

        similarities_matrix = torch.zeros((N, len(class_names)), device=X_tensor.device)
        for c_idx, class_name in enumerate(class_names):
            X_hat_c = estimates[class_name]
            similarities_matrix[:, c_idx] = wegerich_similarity_torch(X_tensor, X_hat_c, gamma=gamma)

        best_class_indices = torch.argmax(similarities_matrix, dim=1)

        X_hat_best = torch.zeros_like(X_tensor)
        for c_idx, class_name in enumerate(class_names):
            mask = (best_class_indices == c_idx)
            if torch.any(mask):
                X_hat_best[mask] = estimates[class_name][mask]

        error_vectors = X_tensor - X_hat_best
        X_extended_tensor = torch.hstack([X_tensor, error_vectors])
        return to_numpy(X_extended_tensor)

    # CPU Fallback
    estimates = compute_sbm_estimates(X, D_c_dict, gamma=gamma)
    class_names = list(D_c_dict.keys())
    N = len(X)

    similarities_matrix = np.zeros((N, len(class_names)))
    for c_idx, class_name in enumerate(class_names):
        X_hat_c = estimates[class_name]
        similarities_matrix[:, c_idx] = wegerich_similarity(X, X_hat_c, gamma=gamma)

    best_class_indices = np.argmax(similarities_matrix, axis=1)

    X_hat_best = np.zeros_like(X)
    for c_idx, class_name in enumerate(class_names):
        mask = (best_class_indices == c_idx)
        if np.any(mask):
            X_hat_best[mask] = estimates[class_name][mask]

    error_vectors = X - X_hat_best
    X_extended = np.hstack([X, error_vectors])
    return X_extended


def generate_similarity_extended_features(
    X: np.ndarray,
    D_c_dict: Dict[str, np.ndarray],
    gamma: float = GAMMA,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Constructs the 52-dimensional extended feature matrix by appending the
    6-class SBM similarity scores directly to the original 46 features.
    """
    if use_gpu and is_gpu_available():
        import torch
        X_tensor = to_tensor(X)
        D_c_dict_tensors = {k: to_tensor(v) for k, v in D_c_dict.items()}

        estimates = compute_sbm_estimates_torch(X_tensor, D_c_dict_tensors, gamma=gamma)
        class_names = sorted(list(D_c_dict.keys()))
        N = len(X)

        similarities_matrix = torch.zeros((N, len(class_names)), device=X_tensor.device)
        for c_idx, class_name in enumerate(class_names):
            X_hat_c = estimates[class_name]
            similarities_matrix[:, c_idx] = wegerich_similarity_torch(X_tensor, X_hat_c, gamma=gamma)

        X_extended_tensor = torch.hstack([X_tensor, similarities_matrix])
        return to_numpy(X_extended_tensor)

    # CPU Fallback
    estimates = compute_sbm_estimates(X, D_c_dict, gamma=gamma)
    class_names = sorted(list(D_c_dict.keys()))
    N = len(X)

    similarities_matrix = np.zeros((N, len(class_names)))
    for c_idx, class_name in enumerate(class_names):
        X_hat_c = estimates[class_name]
        similarities_matrix[:, c_idx] = wegerich_similarity(X, X_hat_c, gamma=gamma)

    X_extended = np.hstack([X, similarities_matrix])
    return X_extended
