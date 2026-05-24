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
        diff_D = D_c[:, np.newaxis, :] - D_c[np.newaxis, :, :]
        l1_D = np.sum(np.abs(diff_D), axis=-1)
        G_c = 1.0 / (1.0 + gamma * l1_D)  # shape: (M_c, M_c)

        # Robust pseudo-inverse computation
        G_c_pinv = np.linalg.pinv(G_c)  # shape: (M_c, M_c)

        # 2. Calculate matrix A_c (similarities between all samples in X and rows of D_c)
        # X: (N, 46), D_c: (M_c, 46) -> diff_X_D shape: (N, M_c, 46)
        diff_X_D = X[:, np.newaxis, :] - D_c[np.newaxis, :, :]
        l1_X_D = np.sum(np.abs(diff_X_D), axis=-1)
        A_c = 1.0 / (1.0 + gamma * l1_X_D)  # shape: (N, M_c)

        # 3. Calculate weight vectors: w_n_c = G_c_pinv @ a_n_c
        # For N samples: W_c = A_c @ G_c_pinv
        W_c = A_c @ G_c_pinv  # shape: (N, M_c)

        # 4. Normalize weights using L1 norm (handling zero norm robustly)
        w_norms = np.sum(np.abs(W_c), axis=1)  # shape: (N,)
        w_norms = np.maximum(w_norms, 1e-12)
        W_c_normalized = W_c / w_norms[:, np.newaxis]  # shape: (N, M_c)

        # 5. Compute final estimate: x_hat = D_c.T @ w_norm
        # For N samples: X_hat = W_norm @ D_c
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


if __name__ == '__main__':
    print("=== MaFaulDa Step 3: SBM Model-Matrix and Error Vector Generation ===")
    start_time = time.time()

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')

    X_train_path = os.path.join(data_dir, 'X_train_features.npy')
    X_test_path = os.path.join(data_dir, 'X_test_features.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')
    y_test_path = os.path.join(data_dir, 'y_test.npy')

    # Load original feature matrices and labels
    print(f"Loading data from {data_dir}...")
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    # Basic validations
    assert X_train.shape[1] == EXPECTED_ORIGINAL_FEATURES, f"Expected {EXPECTED_ORIGINAL_FEATURES} features, got {X_train.shape[1]}"
    assert X_test.shape[1] == EXPECTED_ORIGINAL_FEATURES, f"Expected {EXPECTED_ORIGINAL_FEATURES} features, got {X_test.shape[1]}"

    unique_classes = np.unique(y_train)
    print(f"Loaded {len(X_train)} training samples across {len(unique_classes)} unique classes:")
    for cls in unique_classes:
        print(f"  - {cls}: {np.sum(y_train == cls)} training samples")

    # Construct dictionaries for each unique class in the training set
    print("\nConstructing class dictionary matrices (D_c) using Weiszfeld's and Threshold methods...")
    D_c_dict = {}
    for cls in unique_classes:
        class_start_time = time.time()
        # Filter training samples belonging to class cls
        X_c = X_train[y_train == cls]
        # Build dictionary
        D_c = construct_class_dictionary(X_c, tau=TAU, gamma=GAMMA)
        D_c_dict[cls] = D_c
        class_elapsed = time.time() - class_start_time
        print(f"  - Class '{cls}': built D_c shape {D_c.shape} from {len(X_c)} samples in {class_elapsed:.2f}s")

    # Generate 92-dimensional extended feature matrices
    print("\nGenerating extended 92-dimensional feature matrices (SBM Model B)...")
    X_train_extended = generate_extended_features(X_train, D_c_dict, gamma=GAMMA)
    X_test_extended = generate_extended_features(X_test, D_c_dict, gamma=GAMMA)

    # Validations on the extended feature matrices
    assert X_train_extended.shape == (len(X_train), EXPECTED_EXTENDED_FEATURES), f"X_train_extended shape mismatch: {X_train_extended.shape}"
    assert X_test_extended.shape == (len(X_test), EXPECTED_EXTENDED_FEATURES), f"X_test_extended shape mismatch: {X_test_extended.shape}"
    assert not np.isnan(X_train_extended).any(), "Found NaN values in X_train_extended!"
    assert not np.isnan(X_test_extended).any(), "Found NaN values in X_test_extended!"

    print("\nVerification Checks Passed:")
    print(f"  X_train_extended shape: {X_train_extended.shape} (Expected: ({len(X_train)}, {EXPECTED_EXTENDED_FEATURES}))")
    print(f"  X_test_extended shape:  {X_test_extended.shape} (Expected: ({len(X_test)}, {EXPECTED_EXTENDED_FEATURES}))")
    print("  No NaN values detected in the extended feature matrices!")

    # Save the new extended feature matrices
    X_train_ext_path = os.path.join(data_dir, 'X_train_extended.npy')
    X_test_ext_path = os.path.join(data_dir, 'X_test_extended.npy')

    np.save(X_train_ext_path, X_train_extended)
    np.save(X_test_ext_path, X_test_extended)

    total_elapsed = time.time() - start_time
    print(f"\nAll files saved successfully in {data_dir} in {total_elapsed:.2f} seconds!")
    print(f"  - X_train_extended.npy ({os.path.getsize(X_train_ext_path) / 1024:.1f} KB)")
    print(f"  - X_test_extended.npy ({os.path.getsize(X_test_ext_path) / 1024:.1f} KB)")
    print("====================================================")
