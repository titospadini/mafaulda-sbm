"""
MaFaulDa SBM Hyperparameter Tuning (Pure-Python Version)

This module handles hyperparameter tuning:
- Uses a custom pure-python Stratified K-Fold cross-validation splitter.
- Evaluates grid search combinations over gamma and tau.
- Uses standard library pickle for saving/loading cached features.
"""

import os
import time
import random
import pickle
from typing import List, Tuple, Dict, Any

from mafaulda.sbm_model import (
    construct_class_dictionary,
    generate_extended_features,
)
from mafaulda.rf_classifier import train_classifier
from mafaulda.logging_utils import log


def stratified_k_fold(
    y: List[str],
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Computes Stratified K-Fold split indices preserving class proportions in pure Python.
    """
    random.seed(random_state)

    # Group indices by class label
    label_to_indices = {}
    for idx, lbl in enumerate(y):
        label_to_indices.setdefault(lbl, []).append(idx)

    # Shuffle indices inside each class
    if shuffle:
        for lbl in label_to_indices:
            random.shuffle(label_to_indices[lbl])

    # Partition each class's indices into n_splits folds
    folds = [[] for _ in range(n_splits)]
    for lbl, indices in label_to_indices.items():
        for i, idx in enumerate(indices):
            folds[i % n_splits].append(idx)

    # Create train and validation index sets for each fold
    splits = []
    for val_fold_idx in range(n_splits):
        val_indices = folds[val_fold_idx]
        train_indices = []
        for f_idx in range(n_splits):
            if f_idx != val_fold_idx:
                train_indices.extend(folds[f_idx])
        splits.append((train_indices, val_indices))

    return splits


def run_tuning(data_dir: str, gammas: List[float] = None, taus: List[float] = None) -> None:
    """
    Executes a 10-fold Stratified Cross-Validation grid search to optimize SBM hyperparameters
    strictly using pure Python.
    """
    log("\n" + "="*60, level=1)
    log("=== STEP 5: SBM Hyperparameter Tuning (10-Fold Stratified CV) ===", level=1)
    log("="*60, level=1)

    X_train_path = os.path.join(data_dir, 'X_train_features.pkl')
    y_train_path = os.path.join(data_dir, 'y_train.pkl')

    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError("Training features not found! Please run the pipeline first to generate them.")

    log(f"Loading training features from {data_dir}...", level=1)
    with open(X_train_path, 'rb') as f:
        X_train = pickle.load(f)
    with open(y_train_path, 'rb') as f:
        y_train = pickle.load(f)

    log(f"  Training features samples: {len(X_train)}", level=1)
    log(f"  Training labels samples:   {len(y_train)}", level=1)

    log("\nSetting up 10-fold Stratified Cross-Validation (random_state=42)...", level=1)
    splits = stratified_k_fold(y_train, n_splits=10, shuffle=True, random_state=42)

    # Grid search parameters (default if not provided)
    if gammas is None:
        gammas = [0.0005, 0.001, 0.01, 0.1]
    if taus is None:
        taus = [0.75, 0.80, 0.85, 0.90]

    # Pre-compute geometric medians for each fold and fault class
    # to completely eliminate redundant Weiszfeld optimization iterations across the grid combinations!
    log("Pre-computing geometric medians for all folds and classes...", level=1)
    fold_medians = []
    pre_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr = [X_train[i] for i in train_idx]
        y_tr = [y_train[i] for i in train_idx]
        unique_classes = sorted(list(set(y_tr)))
        medians = {}
        for cls in unique_classes:
            X_c = [X_tr[i] for i, lbl in enumerate(y_tr) if lbl == cls]
            from mafaulda.pure_math import compute_geometric_median
            medians[cls] = compute_geometric_median(X_c)
        fold_medians.append(medians)
    log(f"Pre-computation of geometric medians finished in {time.time() - pre_start:.2f}s.\n", level=1)

    results = []
    grid_start_time = time.time()

    for gamma in gammas:
        for tau in taus:
            comb_start = time.time()
            fold_accuracies = []

            for fold, (train_idx, val_idx) in enumerate(splits):
                log(f"\rEvaluating combination: gamma={gamma}, tau={tau} | Fold {fold + 1}/{len(splits)}...", level=1, end="", flush=True)
                X_tr = [X_train[i] for i in train_idx]
                X_val = [X_train[i] for i in val_idx]
                y_tr = [y_train[i] for i in train_idx]
                y_val = [y_train[i] for i in val_idx]

                # Dictionary matrix construction using pre-computed geometric median
                D_c_dict = {}
                unique_classes = sorted(list(set(y_tr)))
                for cls in unique_classes:
                    X_c = [X_tr[i] for i, lbl in enumerate(y_tr) if lbl == cls]
                    # Pass the pre-computed median
                    g_med = fold_medians[fold][cls]
                    D_c = construct_class_dictionary(X_c, tau=tau, gamma=gamma, g_median=g_med)
                    D_c_dict[cls] = D_c

                # SBM estimation and error vector generation
                X_tr_ext = generate_extended_features(X_tr, D_c_dict, gamma=gamma)
                X_val_ext = generate_extended_features(X_val, D_c_dict, gamma=gamma)

                # Train Random Forest Classifier
                clf = train_classifier(X_tr_ext, y_tr)

                y_pred = clf.predict(X_val_ext)
                acc = sum(1 for yt, yp in zip(y_val, y_pred) if yt == yp) / len(y_val)
                fold_accuracies.append(acc)

            mean_acc = sum(fold_accuracies) / len(fold_accuracies)
            results.append((gamma, tau, mean_acc))
            # Cleanly overwrite the carriage return line with final results for this combination
            log(f"\rEvaluating combination: gamma={gamma}, tau={tau} -> Mean 10-Fold CV Accuracy: {mean_acc * 100.0:.2f}% (computed in {time.time() - comb_start:.2f}s)", level=1)

    grid_elapsed = time.time() - grid_start_time
    log(f"\nGrid search completed in {grid_elapsed:.2f} seconds.", level=1)

    # Sort results
    results.sort(key=lambda x: x[2], reverse=True)
    best_gamma, best_tau, best_acc = results[0]

    # Print table
    print_results_table(results)

    log("\n" + "="*52, level=1)
    log("================== OPTIMAL CONFIGURATION ==================", level=1)
    log(f"Best WSF Gamma (γ):       {best_gamma:.4f}", level=1)
    log(f"Best Threshold Tau (τ):   {best_tau:.2f}", level=1)
    log(f"Maximum CV Accuracy:      {best_acc * 100.0:.2f}%", level=1)
    log("="*52, level=1)
    log("="*60, level=1)


def print_results_table(
    results: List[Tuple[float, float, float]]
) -> None:
    """Outputs a beautifully aligned ASCII text table of the results."""
    log("\n=================== GRID SEARCH RESULTS ===================", level=1)
    log(f"{'WSF Gamma (γ)':^15} | {'Threshold Tau (τ)':^19} | {'10-Fold CV Accuracy':^21}", level=1)
    log("-" * 63, level=1)
    for gamma, tau, mean_acc in results:
        acc_pct = mean_acc * 100.0
        log(f"{gamma:^15.4f} | {tau:^19.2f} | {acc_pct:^20.2f}%", level=1)
    log("===========================================================", level=1)
