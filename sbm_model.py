"""
Multiclass SBM Model-Matrix Construction and Error Vector Generation

This script handles Step 3 of the Rotating-Machine Fault Diagnosis pipeline:
1. Loads the 46-dimensional feature matrices (X_train_features, X_test_features)
   and labels (y_train, y_test) from the data/ directory.
2. Implements the Wegerich Similarity Function (WSF) using L1 norm and gamma = 0.01.
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
import numpy as np

# Constants from the paper
GAMMA = 0.01
TAU = 0.9
EXPECTED_ORIGINAL_FEATURES = 46
EXPECTED_EXTENDED_FEATURES = 92


def wegerich_similarity(x1: np.ndarray, x2: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """
    Computes the Wegerich Similarity Function (WSF) between two vectors,
    or a matrix and a vector, using the L1 norm.

    Formula: s(x1, x2) = 1.0 / (1.0 + gamma * sum(abs(x1 - x2)))

    Parameters:
        x1 (np.ndarray): First vector or array of vectors.
        x2 (np.ndarray): Second vector or array of vectors.
        gamma (float): Sensitivity parameter (default 0.01).

    Returns:
        np.ndarray or float: The computed WSF similarity score(s).
    """
    l1_dist = np.sum(np.abs(x1 - x2), axis=-1)
    return 1.0 / (1.0 + gamma * l1_dist)


def compute_geometric_median(X: np.ndarray, max_iter: int = 2000, tol: float = 1e-6) -> np.ndarray:
    """
    Computes the geometric median of a set of multidimensional points X
    using Weiszfeld's algorithm.

    Parameters:
        X (np.ndarray): Input points of shape (K, d).
        max_iter (int): Maximum number of iterations (default 2000 for strict convergence).
        tol (float): Convergence tolerance (default 1e-6).

    Returns:
        np.ndarray: The geometric median vector of shape (d,).
    """
    if len(X) == 0:
        raise ValueError("Cannot compute geometric median of an empty set.")
    if len(X) == 1:
        return X[0].copy()

    # Initialize the estimate with the mean of the points
    y = np.mean(X, axis=0)

    for _ in range(max_iter):
        # Compute Euclidean (L2) distances from current estimate y to all points in X
        distances = np.linalg.norm(X - y, axis=1)

        # Avoid division by zero using a robust lower bound
        weights = 1.0 / np.maximum(distances, 1e-12)
        sum_weights = np.sum(weights)

        new_y = np.sum(X * weights[:, np.newaxis], axis=0) / sum_weights

        # Check for convergence
        if np.linalg.norm(new_y - y) < tol:
            return new_y
        y = new_y

    return y


def construct_class_dictionary(X_c: np.ndarray, tau: float = TAU, gamma: float = GAMMA) -> np.ndarray:
    """
    Constructs the representative dictionary matrix D_c for a class c
    using the threshold method.

    Parameters:
        X_c (np.ndarray): Training samples for class c of shape (K, 46).
        tau (float): Similarity threshold (default 0.9).
        gamma (float): WSF sensitivity parameter (default 0.01).

    Returns:
        np.ndarray: Representative dictionary matrix D_c of shape (M_c, 46).
    """
    # 1. Calculate the geometric median of class training samples (max_iter=2000, tol=1e-6)
    g_median = compute_geometric_median(X_c, max_iter=2000, tol=1e-6)

    # 2. Set this geometric median as the first representative state (first row) of D_c
    D_c_list = [g_median]

    # 3. Iterate through all training samples in X_c
    for x in X_c:
        D_c_arr = np.array(D_c_list)
        # Compute WSF similarity between x and all currently selected rows in D_c
        similarities = wegerich_similarity(D_c_arr, x, gamma=gamma)

        # Append sample x as a new row ONLY if its similarity to EVERY SINGLE already existing element in D_c
        # is strictly less than the threshold tau = 0.9.
        if all(sim < tau for sim in similarities):
            D_c_list.append(x)

    return np.array(D_c_list)


def compute_sbm_estimates(X: np.ndarray, D_c_dict: dict, gamma: float = GAMMA) -> dict:
    """
    Computes the SBM estimate matrix x_hat_n_c for all samples X and all classes.

    Parameters:
        X (np.ndarray): Input feature matrix of shape (N, 46).
        D_c_dict (dict): Dictionary mapping class names to D_c matrices.
        gamma (float): WSF sensitivity parameter.

    Returns:
        dict: Mapping from class names to estimate matrices of shape (N, 46).
    """
    estimates = {}

    for class_name, D_c in D_c_dict.items():
        # D_c shape: (M_c, 46)
        # 1. Calculate matrix G_c (pairwise similarities between rows in D_c)
        # Mathematical Equation: (G_c)_ij = s((D_c)_i, (D_c)_j)
        # By utilizing broadcasting, we construct the symmetric similarity matrix
        # using the unified Wegerich Similarity Function (WSF) and L1 norm.
        G_c = wegerich_similarity(D_c[:, np.newaxis, :], D_c[np.newaxis, :, :], gamma=gamma)  # shape: (M_c, M_c)

        # Robust pseudo-inverse computation for the similarity matrix G_c
        G_c_pinv = np.linalg.pinv(G_c)  # shape: (M_c, M_c)

        # 2. Calculate matrix A_c (similarities between all samples in X and rows of D_c)
        # Mathematical Equation: (A_c)_nj = s(x_n, (D_c)_j)
        # We compute this across all samples using the unified WSF.
        A_c = wegerich_similarity(X[:, np.newaxis, :], D_c[np.newaxis, :, :], gamma=gamma)  # shape: (N, M_c)

        # 3. Calculate weight vectors: w_n_c = G_c_pinv @ a_n_c
        # In matrix notation for N samples: W_c = A_c @ G_c_pinv (since G_c_pinv is symmetric)
        # Each row n of W_c corresponds to the transposed weight vector w_n_c.T of shape (M_c,)
        W_c = A_c @ G_c_pinv  # shape: (N, M_c)

        # 4. Normalize weights using L1 norm strictly BEFORE the matrix product
        # Mathematical Equation: w_prime_n_c = w_n_c / sum(abs(w_n_c))
        # We compute the L1 norm of each weight vector (each row of W_c)
        w_norms = np.sum(np.abs(W_c), axis=1)  # shape: (N,)
        w_norms = np.maximum(w_norms, 1e-12)  # Robust lower bound to avoid division by zero
        W_c_normalized = W_c / w_norms[:, np.newaxis]  # shape: (N, M_c)

        # 5. Compute final estimate: x_hat_n_c = D_c.T @ w_prime_n_c
        # In matrix notation for N samples: X_hat_c = W_c_normalized @ D_c
        # This is mathematically equivalent to D_c.T @ w_prime_n_c for each sample vector
        # because (D_c.T @ w_prime_n_c).T = w_prime_n_c.T @ D_c = W_c_normalized[n] @ D_c
        X_hat_c = W_c_normalized @ D_c  # shape: (N, 46)

        estimates[class_name] = X_hat_c

    return estimates


def generate_extended_features(X: np.ndarray, D_c_dict: dict, gamma: float = GAMMA) -> np.ndarray:
    """
    Computes SBM estimates, finds the best estimate for each sample, generates
    the estimation error vector, and concatenates it with the original features.

    Parameters:
        X (np.ndarray): Original feature matrix of shape (N, 46).
        D_c_dict (dict): Dictionary mapping class names to D_c matrices.
        gamma (float): WSF sensitivity parameter.

    Returns:
        np.ndarray: The 92-dimensional extended feature matrix of shape (N, 92).
    """
    # 1. Compute estimates for all classes
    estimates = compute_sbm_estimates(X, D_c_dict, gamma=gamma)
    class_names = list(D_c_dict.keys())
    N = len(X)

    # 2. Compute similarity between each sample and its estimate for each class
    similarities_matrix = np.zeros((N, len(class_names)))
    for c_idx, class_name in enumerate(class_names):
        X_hat_c = estimates[class_name]
        similarities_matrix[:, c_idx] = wegerich_similarity(X, X_hat_c, gamma=gamma)

    # 3. Find the best matching class c* that maximizes similarity
    best_class_indices = np.argmax(similarities_matrix, axis=1)

    # 4. Reconstruct the best estimate matrix X_hat_best
    X_hat_best = np.zeros_like(X)
    for c_idx, class_name in enumerate(class_names):
        mask = (best_class_indices == c_idx)
        if np.any(mask):
            X_hat_best[mask] = estimates[class_name][mask]

    # 5. Calculate the estimation error vectors: error = X - X_hat_best
    error_vectors = X - X_hat_best

    # 6. Concatenate original 46 features with the 46-dimensional error vectors
    X_extended = np.hstack([X, error_vectors])
    return X_extended

