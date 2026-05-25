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

# Constants from the paper
GAMMA = 0.0010
TAU = 0.85
EXPECTED_ORIGINAL_FEATURES = 46
EXPECTED_EXTENDED_FEATURES = 92


def wegerich_similarity(
    x1: np.ndarray,
    x2: np.ndarray,
    gamma: float = GAMMA
) -> np.ndarray:
    """
    Computes the Wegerich Similarity Function (WSF) between two vectors, or
    between a matrix
    and a vector, using the L1 norm.

    Pedagogical Context:
        In Similarity-Based Modeling (SBM), similarity replaces distance as the
        core coordinate
        space metric. The WSF coupled with the L1 norm (Manhattan distance) is
        mathematically defined as:

        $$s(x_1, x_2) = \\frac{1}{1 + \\gamma \\cdot \\|x_1 - x_2\\|_1}$$

        Where:
          - $\\|x_1 - x_2\\|_1$ is the L1 norm, calculated as the sum of
            absolute coordinate differences.
          - $\\gamma$ (gamma) is a sensitivity scaling parameter. A lower
            $\\gamma$ value (e.g. 0.0010)
            makes the similarity score decay more slowly as distance increases,
            allowing SBM weight
            interpolation to remain highly stable even for slightly noisy
            features.

    Parameters:
        x1 (np.ndarray): First input vector or multi-dimensional array of
        vectors.
        x2 (np.ndarray): Second input vector or multi-dimensional array of
        vectors.
        gamma (float): Sensitivity parameter (defaults to the optimized GAMMA =
        0.0010).

    Returns:
        np.ndarray: The computed WSF similarity score(s) bounded between (0.0,
        1.0].
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
    iterative
    Weiszfeld's algorithm.

    Pedagogical Context:
        Unlike the coordinate-wise arithmetic mean or standard median, the
        geometric median is the
        point $y$ that strictly minimizes the sum of Euclidean (L2) distances to
        all points in $X$:

        $$y = \\arg\\min_{y} \\sum_{i=1}^{K} \\|x_i - y\\|_2$$

        This property makes the geometric median highly robust against signal
        outliers and impulse
        shocks. In our pipeline, it serves as the stable anchor seed (the very
        first state) of each
        SBM class dictionary $D_c$.

        Weiszfeld's algorithm iteratively updates the estimate $y^{(k)}$ using
        weighted averages:

        $$y^{(k+1)} = \\frac{\\sum_{i} x_i / \\|x_i - y^{(k)}\\|_2}{\\sum_{i} 1
        / \\|x_i - y^{(k)}\\|_2}$$

        Epsilon bounds are integrated to robustly handle zero-distance division.

    Parameters:
        X (np.ndarray): Input training data points of shape (K, d).
        max_iter (int): Maximum convergence iterations (defaults to 2000 for
        strict mathematical convergence).
        tol (float): Convergence criteria tolerance (defaults to 1e-6).

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
        # Compute Euclidean (L2) distances from current estimate y to all points
        # in X
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


def construct_class_dictionary(
    X_c: np.ndarray,
    tau: float = TAU,
    gamma: float = GAMMA
) -> np.ndarray:
    """
    Constructs the compact SBM representative state dictionary (memory matrix)
    D_c for class c
    using the threshold method seeded with the class geometric median.

    Pedagogical Context:
        An SBM class dictionary $D_c$ acts as a clean normal manifold
        representation of class $c$.
        Instead of memorizing all $K$ samples (which is noisy and
        computationally expensive), we build
        a compact subset of $M_c \\ll K$ states using the Threshold Method:

        1. Initialize $D_c$ with the robust geometric median of $X_c$.
        2. Recursively iterate through each candidate training sample $x \\in
           X_c$.
         3. Compute the WSF similarity between $x$ and all existing states inside
           $D_c$.
        4. Append $x$ as a new state in $D_c$ ONLY if its similarity to all
           existing states is strictly
           less than the threshold $\\tau$ (tau).

        This ensures that newly added states represent structurally unique
        operating behaviors while
        preventing redundant or noisy features from polluting the dictionary.

    Parameters:
        X_c (np.ndarray): Original training samples for class c of shape (K,
        46).
        tau (float): Similarity threshold (defaults to the optimized TAU =
        0.85).
        gamma (float): WSF sensitivity parameter (defaults to the optimized
        GAMMA = 0.0010).

    Returns:
        np.ndarray: The representative state dictionary matrix D_c of shape
        (M_c, 46).
    """
    # 1. Calculate the geometric median of class training samples
    # (max_iter=2000, tol=1e-6)
    g_median = compute_geometric_median(X_c, max_iter=2000, tol=1e-6)

    # 2. Set this geometric median as the first representative state (first row)
    # of D_c
    D_c_list = [g_median]

    # 3. Iterate through all training samples in X_c
    for x in X_c:
        D_c_arr = np.array(D_c_list)
        # Compute WSF similarity between x and all currently selected rows in
        # D_c
        similarities = wegerich_similarity(D_c_arr, x, gamma=gamma)

        # Append sample x as a new row ONLY if its similarity to EVERY SINGLE
        # already existing element in D_c
        # is strictly less than the threshold tau = 0.9.
        if all(sim < tau for sim in similarities):
            D_c_list.append(x)

    return np.array(D_c_list)


def compute_sbm_estimates(
    X: np.ndarray,
    D_c_dict: Dict[str, np.ndarray],
    gamma: float = GAMMA
) -> Dict[str, np.ndarray]:
    """
    Computes SBM state estimations for all input samples X across all fault
    classes.

    Pedagogical Context:
        SBM computes an estimate $\\hat{x}_{n,c}$ of an input vector $x_n$ by
        interpolating the
        representative states stored in the memory matrix $D_c$. For each class
        $c$, SBM executes:

        1. Pairwise Memory Similarity ($G_c$):
           Calculates similarity between all representative states in $D_c$:
           $$(G_c)_{ij} = s((D_c)_i, (D_c)_j)$$

        2. Input-to-Memory Similarity ($A_c$):
           Calculates similarity between each sample $x_n$ and all
           representative states in $D_c$:
           $$(A_c)_{nj} = s(x_n, (D_c)_j)$$

        3. Raw Interpolation Weights ($W_c$):
           Solves for the interpolation weights using the pseudo-inverse of
           $G_c$:
           $$W_c = A_c \\cdot G_c^{\\dagger}$$

        4. Normalized Weights ($W_{c,\\text{normalized}}$):
           Normalizes weights by their L1 norm to preserve amplitude bounds:
           $$w'_{n,c} = \\frac{w_{n,c}}{\\sum |w_{n,c}|}$$

        5. Final Estimate ($X_{\\hat{c}}$):
           Reconstructs the signal using normalized weights:
           $$\\hat{x}_{n,c} = D_c^T \\cdot w'_{n,c}$$

    Parameters:
        X (np.ndarray): Original feature matrix of shape (N, 46).
        D_c_dict (Dict[str, np.ndarray]): Dictionary mapping class name strings
        to D_c arrays.
        gamma (float): WSF sensitivity parameter (defaults to the optimized
        GAMMA = 0.0010).

    Returns:
        Dict[str, np.ndarray]: Mapping from class names to estimate matrices of
        shape (N, 46).
    """
    estimates = {}

    for class_name, D_c in D_c_dict.items():
        # D_c shape: (M_c, 46)
        # 1. Calculate matrix G_c (pairwise similarities between rows in D_c)
        # Mathematical Equation: (G_c)_ij = s((D_c)_i, (D_c)_j)
        # By utilizing broadcasting, we construct the symmetric similarity
        # matrix
        # using the unified Wegerich Similarity Function (WSF) and L1 norm.
        G_c = wegerich_similarity(D_c[:, np.newaxis, :], D_c[np.newaxis, :, :], gamma=gamma)  # shape: (M_c, M_c)

        # Robust pseudo-inverse computation for the similarity matrix G_c
        G_c_pinv = np.linalg.pinv(G_c)  # shape: (M_c, M_c)

        # 2. Calculate matrix A_c (similarities between all samples in X and
        # rows of D_c)
        # Mathematical Equation: (A_c)_nj = s(x_n, (D_c)_j)
        # We compute this across all samples using the unified WSF.
        A_c = wegerich_similarity(X[:, np.newaxis, :], D_c[np.newaxis, :, :], gamma=gamma)  # shape: (N, M_c)

        # 3. Calculate weight vectors: w_n_c = G_c_pinv @ a_n_c
        # In matrix notation for N samples: W_c = A_c @ G_c_pinv (since G_c_pinv
        # is symmetric)
        # Each row n of W_c corresponds to the transposed weight vector w_n_c.T
        # of shape (M_c,)
        W_c = A_c @ G_c_pinv  # shape: (N, M_c)

        # 4. Normalize weights using L1 norm strictly BEFORE the matrix product
        # Mathematical Equation: w_prime_n_c = w_n_c / sum(abs(w_n_c))
        # We compute the L1 norm of each weight vector (each row of W_c)
        w_norms = np.sum(np.abs(W_c), axis=1)  # shape: (N,)
        w_norms = np.maximum(w_norms, 1e-12)  # Robust lower bound to avoid division by zero
        W_c_normalized = W_c / w_norms[:, np.newaxis]  # shape: (N, M_c)

        # 5. Compute final estimate: x_hat_n_c = D_c.T @ w_prime_n_c
        # In matrix notation for N samples: X_hat_c = W_c_normalized @ D_c
        # This is mathematically equivalent to D_c.T @ w_prime_n_c for each
        # sample vector
        # because (D_c.T @ w_prime_n_c).T = w_prime_n_c.T @ D_c =
        # W_c_normalized[n] @ D_c
        X_hat_c = W_c_normalized @ D_c  # shape: (N, 46)

        estimates[class_name] = X_hat_c

    return estimates


def generate_extended_features(
    X: np.ndarray,
    D_c_dict: Dict[str, np.ndarray],
    gamma: float = GAMMA
) -> np.ndarray:
    """
    Constructs the 92-dimensional extended feature matrix by appending the
    best-matching
    SBM estimation error vector to the original 46 features.

    Pedagogical Context:
        This replicates the primary SBM Model B extension pipeline described in
        the paper:

        1. Generate estimates $\\hat{x}_{n,c}$ for all 6 fault classes.
        2. Compute the WSF similarity score between the input $x_n$ and its
           estimate for each class.
        3. Identify the best-matching class $c^*$ that maximizes this
           similarity:
           $$c^* = \\arg\\max_{c} s(x_n, \\hat{x}_{n,c})$$
        4. Calculate the estimation error vector $e_n$ using this best-matching
           estimate:
           $$e_n = x_n - \\hat{x}_{n,c^*}$$
        5. Appends $e_n$ to $x_n$ to construct the 92-dimensional extended
           feature representation:
           $$x_{n,\\text{extended}} = [x_n^T, e_n^T]^T$$

        The error vector provides the classifier with high-resolution residuals
        that highlight exactly
        where the machine's behavior deviates from the learned normal fault
        manifolds.

    Parameters:
        X (np.ndarray): Original feature matrix of shape (N, 46).
        D_c_dict (Dict[str, np.ndarray]): Dictionary mapping class name strings
        to D_c arrays.
        gamma (float): WSF sensitivity parameter (defaults to the optimized
        GAMMA = 0.0010).

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


def generate_similarity_extended_features(
    X: np.ndarray,
    D_c_dict: Dict[str, np.ndarray],
    gamma: float = GAMMA
) -> np.ndarray:
    """
    Constructs the 52-dimensional extended feature matrix by appending the
    6-class SBM
    similarity scores directly to the original 46 features.

    Pedagogical Context:
        This replicates Configuration 3 of Experiment 3 from the scientific
        paper.
        Instead of using physical estimation error residuals, we feed the
        similarity coordinates
        themselves directly to the classifier.

        For each sample $x_n$, we compute the WSF similarity to its SBM estimate
        $\\hat{x}_{n,c}$
        across all 6 fault classes, yielding a 6-dimensional similarity vector:

        $$s_n = [s(x_n, \\hat{x}_{n,c_1}), \\dots, s(x_n, \\hat{x}_{n,c_6})]^T$$

        This vector is appended to the original 46 features, resulting in a
        52-dimensional representation.
        Because SBM similarities offer extremely clear class boundary
        separation, this compact 52-dimensional
        matrix achieves exceptional classification performance while using half
        the extension size of the error-based model.

    Parameters:
        X (np.ndarray): Original feature matrix of shape (N, 46).
        D_c_dict (Dict[str, np.ndarray]): Dictionary mapping class name strings
        to D_c arrays.
        gamma (float): WSF sensitivity parameter (defaults to the optimized
        GAMMA = 0.0010).

    Returns:
        np.ndarray: The 52-dimensional extended feature matrix of shape (N, 52).
    """
    # 1. Compute estimates for all classes
    estimates = compute_sbm_estimates(X, D_c_dict, gamma=gamma)

    # Sort class names alphabetically to ensure deterministic ordering of the 6
    # classes
    class_names = sorted(list(D_c_dict.keys()))
    N = len(X)

    # 2. Compute similarity between each sample and its estimate for each class
    similarities_matrix = np.zeros((N, len(class_names)))
    for c_idx, class_name in enumerate(class_names):
        X_hat_c = estimates[class_name]
        similarities_matrix[:, c_idx] = wegerich_similarity(X, X_hat_c, gamma=gamma)

    # 3. Concatenate original 46 features with the 6-dimensional similarity
    # vector
    X_extended = np.hstack([X, similarities_matrix])
    return X_extended
