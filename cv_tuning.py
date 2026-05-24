"""
MaFaulDa SBM Hyperparameter Tuning via 10-Fold Stratified Cross-Validation

This module handles the hyperparameter tuning step (Step 5) of the rotating-machine
fault diagnosis pipeline, exposing function interfaces to run the Stratified 10-Fold
Cross-Validation grid search over gamma and tau values.
"""

import os
import time
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from sbm_model import construct_class_dictionary, generate_extended_features
from rf_classifier import train_classifier


def run_tuning(data_dir: str) -> None:
    """
    Executes a 10-fold Stratified Cross-Validation grid search to optimize the SBM hyperparameters
    (WSF sensitivity gamma and threshold tau) strictly on the training set.

    Pedagogical Context:
        Cross-Validation (CV) is the standard method for parameter tuning.
        - 10-Fold Stratified Split: The training set is split into 10 subsets (folds). The stratification
          preserves the relative proportions of each of the 6 fault classes in every fold, preventing
          small classes like Normal from being excluded.
        - Strict Manifold Modeling per Fold: In each fold iteration, the SBM dictionaries are built
          from scratch using the fold's training portion, and extended features are calculated for both
          fold train and fold validation portions. This ensures zero data leakage.
        - Parameter Grid: We evaluate gamma $\\gamma \\in \\{0.0005, 0.0010, 0.0100, 0.1000\\}$ and
          tau $\\tau \\in \\{0.75, 0.80, 0.85, 0.90\\}$.
        - Outcome: Evaluates the optimal SBM parameter curve to find the maximum cross-validated accuracy.

    Parameters:
        data_dir (str): Absolute path to the directory containing pre-extracted feature files.

    Raises:
        FileNotFoundError: If the pre-extracted `X_train_features.npy` files do not exist under `data_dir`.
    """
    print("\n" + "="*60)
    print("=== STEP 5: SBM Hyperparameter Tuning (10-Fold Stratified CV) ===")
    print("="*60)

    X_train_path = os.path.join(data_dir, 'X_train_features.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')

    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError("Training features not found! Please run the pipeline first to generate them.")

    print(f"Loading training features from {data_dir}...")
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path, allow_pickle=True)

    print(f"  Training features shape: {X_train.shape}")
    print(f"  Training labels shape:   {y_train.shape}")

    print("\nSetting up 10-fold Stratified Cross-Validation (random_state=42)...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Grid search parameters
    gammas = [0.0005, 0.001, 0.01, 0.1]
    taus = [0.75, 0.80, 0.85, 0.90]

    results = []
    grid_start_time = time.time()

    for gamma in gammas:
        for tau in taus:
            comb_start = time.time()
            print(f"\nEvaluating combination: gamma={gamma}, tau={tau}...")
            fold_accuracies = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # Dictionary matrix construction
                D_c_dict = {}
                unique_classes = np.unique(y_tr)
                for cls in unique_classes:
                    X_c = X_tr[y_tr == cls]
                    D_c = construct_class_dictionary(X_c, tau=tau, gamma=gamma)
                    D_c_dict[cls] = D_c

                # SBM estimation and error vector generation
                X_tr_ext = generate_extended_features(X_tr, D_c_dict, gamma=gamma)
                X_val_ext = generate_extended_features(X_val, D_c_dict, gamma=gamma)

                # Train Random Forest Classifier concurrently via our rf_classifier module
                clf = train_classifier(X_tr_ext, y_tr)

                y_pred = clf.predict(X_val_ext)
                acc = accuracy_score(y_val, y_pred)
                fold_accuracies.append(acc)

            mean_acc = np.mean(fold_accuracies)
            results.append((gamma, tau, mean_acc))
            print(f"  -> Mean 10-Fold CV Accuracy: {mean_acc * 100.0:.2f}% (computed in {time.time() - comb_start:.2f}s)")

    grid_elapsed = time.time() - grid_start_time
    print(f"\nGrid search completed in {grid_elapsed:.2f} seconds.")

    # Sort results
    results.sort(key=lambda x: x[2], reverse=True)
    best_gamma, best_tau, best_acc = results[0]

    # Print table
    print_results_table(results)

    print("\n" + "="*52)
    print("================== OPTIMAL CONFIGURATION ==================")
    print(f"Best WSF Gamma (γ):       {best_gamma:.4f}")
    print(f"Best Threshold Tau (τ):   {best_tau:.2f}")
    print(f"Maximum CV Accuracy:      {best_acc * 100.0:.2f}%")
    print("="*52)
    print("="*60)


def print_results_table(results: List[Tuple[float, float, float]]) -> None:
    """
    Outputs a beautifully aligned ASCII text table displaying all evaluated hyperparameter combinations
    and their corresponding 10-fold cross-validation accuracies.

    Pedagogical Context:
        Helps identify the sensitivity of the SBM parameter landscape. Presenting grid search results
        in a structured table allows the engineer to inspect how changes in dictionary memorization bounds
        (tau) and weight sensitivities (gamma) impact overall classifier performance.

    Parameters:
        results (List[Tuple[float, float, float]]): List of tuples containing:
          - gamma (float): WSF sensitivity parameter.
          - tau (float): SBM dictionary threshold.
          - mean_accuracy (float): Average stratified cross-validation accuracy across the 10 folds.
    """
    print("\n=================== GRID SEARCH RESULTS ===================")
    print(f"{'WSF Gamma (γ)':^15} | {'Threshold Tau (τ)':^19} | {'10-Fold CV Accuracy':^21}")
    print("-" * 63)
    for gamma, tau, mean_acc in results:
        acc_pct = mean_acc * 100.0
        print(f"{gamma:^15.4f} | {tau:^19.2f} | {acc_pct:^20.2f}%")
    print("===========================================================")
