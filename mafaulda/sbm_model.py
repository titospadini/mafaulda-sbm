"""
Multiclass SBM Model-Matrix Construction and Error Vector Generation (Pure-Python Version)

This module handles Step 3 of the pipeline:
1. Implements Wegerich Similarity Function (WSF) using L1 norm.
2. Seeds class dictionaries with Weiszfeld's Geometric Median.
3. Grows class representative memory matrices (D_c) using the Threshold Method.
4. Performs SBM projection / estimation using Moore-Penrose pseudo-inverse (mat_inverse_regularized).
5. Generates 92-dimensional error residual and 52-dimensional similarity extended features.
"""

from typing import List, Dict
from mafaulda.pure_math import (
    vec_sub,
    vec_scale,
    vec_add,
    vec_l1_norm,
    mat_inverse_regularized,
    compute_geometric_median
)

# Constants from the paper
GAMMA = 0.0010
TAU = 0.85
EXPECTED_ORIGINAL_FEATURES = 46
EXPECTED_EXTENDED_FEATURES = 92


def wegerich_similarity(u: List[float], v: List[float], gamma: float = GAMMA) -> float:
    """Computes the Wegerich Similarity Function (WSF) between two vectors."""
    l1_dist = sum(abs(x - y) for x, y in zip(u, v))
    return 1.0 / (1.0 + gamma * l1_dist)


def construct_class_dictionary(
    X_c: List[List[float]],
    tau: float = TAU,
    gamma: float = GAMMA,
    g_median: List[float] = None
) -> List[List[float]]:
    """
    Constructs the compact SBM representative state dictionary (memory matrix) D_c
    using the threshold method seeded with the class geometric median.
    """
    if not X_c:
        return []

    # 1. Calculate geometric median seed if not pre-computed
    if g_median is None:
        g_median = compute_geometric_median(X_c)

    D_c = [g_median]

    # 2. Iterate through all training samples in X_c
    for x in X_c:
        # Append sample x as a new row only if its similarity to EVERY existing element in D_c is strictly < tau
        if all(wegerich_similarity(x, state, gamma) < tau for state in D_c):
            D_c.append(x)

    return D_c


def compute_sbm_estimates(
    X: List[List[float]],
    D_c_dict: Dict[str, List[List[float]]],
    gamma: float = GAMMA
) -> Dict[str, List[List[float]]]:
    """Computes SBM clean signal estimations for all input samples X across all fault classes."""
    estimates = {}

    for class_name, D_c in D_c_dict.items():
        m_c = len(D_c)
        if m_c == 0:
            estimates[class_name] = [list(row) for row in X]
            continue

        # 1. Pairwise similarity matrix G_c for the dictionary
        G_c = [[0.0] * m_c for _ in range(m_c)]
        for i in range(m_c):
            for j in range(i, m_c):
                sim = wegerich_similarity(D_c[i], D_c[j], gamma)
                G_c[i][j] = sim
                G_c[j][i] = sim

        # Robust matrix inverse (regularized)
        G_c_inv = mat_inverse_regularized(G_c, reg=1e-9)

        estimates_c = []
        for x in X:
            # 2. Calculate input-to-memory similarity vector A_c
            A_c = [wegerich_similarity(x, state, gamma) for state in D_c]

            # 3. Calculate raw interpolation weights w_c = G_c_inv @ A_c
            w_c = [0.0] * m_c
            for r in range(m_c):
                w_c[r] = sum(G_c_inv[r][c] * A_c[c] for c in range(m_c))

            # 4. Normalize weights using L1 norm
            w_norm = sum(abs(w) for w in w_c)
            w_norm = max(w_norm, 1e-12)
            w_c_normalized = [w / w_norm for w in w_c]

            # 5. Reconstruct final estimate: x_hat = D_c^T @ w_c_normalized
            x_hat = [0.0] * len(x)
            for state_idx, weight in enumerate(w_c_normalized):
                state = D_c[state_idx]
                for d in range(len(x)):
                    x_hat[d] += weight * state[d]

            estimates_c.append(x_hat)

        estimates[class_name] = estimates_c

    return estimates


def generate_extended_features(
    X: List[List[float]],
    D_c_dict: Dict[str, List[List[float]]],
    gamma: float = GAMMA
) -> List[List[float]]:
    """Constructs the 92-dimensional extended feature matrix using SBM error residuals (Model B)."""
    # 1. Compute estimates for all classes
    estimates = compute_sbm_estimates(X, D_c_dict, gamma=gamma)
    class_names = list(D_c_dict.keys())
    num_samples = len(X)
    num_features = len(X[0])

    X_extended = []
    for i in range(num_samples):
        x = X[i]

        # 2. Find best-matching class c* maximizing reconstruction similarity
        best_sim = -1.0
        best_class = class_names[0]

        for class_name in class_names:
            x_hat_c = estimates[class_name][i]
            sim = wegerich_similarity(x, x_hat_c, gamma)
            if sim > best_sim:
                best_sim = sim
                best_class = class_name

        # 3. Reconstruct estimation error vector: error = x - x_hat_best
        x_hat_best = estimates[best_class][i]
        error_vector = vec_sub(x, x_hat_best)

        # 4. Concatenate original features with the error residual
        X_extended.append(x + error_vector)

    return X_extended


def generate_similarity_extended_features(
    X: List[List[float]],
    D_c_dict: Dict[str, List[List[float]]],
    gamma: float = GAMMA
) -> List[List[float]]:
    """Constructs the 52-dimensional extended feature matrix by appending direct similarity scores."""
    # 1. Compute estimates for all classes
    estimates = compute_sbm_estimates(X, D_c_dict, gamma=gamma)
    class_names = sorted(list(D_c_dict.keys()))
    num_samples = len(X)

    X_extended = []
    for i in range(num_samples):
        x = X[i]

        # 2. Compute similarity coordinates
        similarities = []
        for class_name in class_names:
            x_hat_c = estimates[class_name][i]
            similarities.append(wegerich_similarity(x, x_hat_c, gamma))

        # 3. Concatenate original features with similarities
        X_extended.append(x + similarities)

    return X_extended
