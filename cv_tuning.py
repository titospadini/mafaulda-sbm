"""
MaFaulDa SBM Hyperparameter Tuning via 10-Fold Stratified Cross-Validation

This script implements Step 5 of the Rotating-Machine Fault Diagnosis pipeline:
1. Loads the original 46-dimensional features and labels for the TRAINING set only.
2. Sets up a 10-fold Stratified Cross-Validation using StratifiedKFold (random_state=42).
3. Defines a grid search over:
   - gamma: [0.01, 0.1, 0.5, 1.0]
   - tau: [0.80, 0.85, 0.90, 0.95]
4. Executes the complete SBM matrix construction, feature extension, and
   Random Forest training loop inside each fold.
5. Employs RandomForestClassifier(n_estimators=500, max_features='sqrt',
   class_weight='balanced', n_jobs=-1, random_state=42) for highly optimized,
   multicore training.
6. Computes average validation accuracy across all 10 splits for each parameter combination.
7. Prints a formatted summary table of results and highlights the optimal configuration.
"""

import os
import sys
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure sbm_model imports work correctly by adding the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sbm_model import construct_class_dictionary, generate_extended_features


def print_results_table(results: list) -> None:
    """
    Prints a beautifully formatted, aligned text-based table
    showing all evaluated hyperparameter combinations and their accuracies.

    Parameters:
        results (list): List of tuples containing (gamma, tau, mean_accuracy).
    """
    print("\n=================== GRID SEARCH RESULTS ===================")
    print(f"{'WSF Gamma (γ)':^15} | {'Threshold Tau (τ)':^19} | {'10-Fold CV Accuracy':^21}")
    print("-" * 63)
    for gamma, tau, mean_acc in results:
        acc_pct = mean_acc * 100.0
        print(f"{gamma:^15.2f} | {tau:^19.2f} | {acc_pct:^20.2f}%")
    print("===========================================================")


if __name__ == '__main__':
    print("=== MaFaulDa Step 5: SBM Hyperparameter Tuning (10-Fold Stratified CV) ===")
    overall_start_time = time.time()

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')

    X_train_path = os.path.join(data_dir, 'X_train_features.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')

    # Load original training features and labels
    print(f"Loading original training data from {data_dir}...")
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path, allow_pickle=True)

    print(f"  Training features shape: {X_train.shape}")
    print(f"  Training labels shape:   {y_train.shape}")

    # Set up 10-fold Stratified Cross-Validation
    # StratifiedKFold ensures that the class representation is preserved in each split
    print("\nSetting up 10-fold Stratified Cross-Validation (random_state=42)...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Grid search parameters
    gammas = [0.01, 0.1, 0.5, 1.0]
    taus = [0.80, 0.85, 0.90, 0.95]

    results = []

    # Iterating over the parameter grid
    grid_start_time = time.time()
    for gamma in gammas:
        for tau in taus:
            comb_start = time.time()
            print(f"\nEvaluating combination: gamma={gamma}, tau={tau}...")
            fold_accuracies = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                # Split train folds and validation fold
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # 1. Dictionary matrix construction (D_c) on 9 train folds
                D_c_dict = {}
                unique_classes = np.unique(y_tr)
                for cls in unique_classes:
                    X_c = X_tr[y_tr == cls]
                    # Weiszfeld's is called with tighter convergence parameters
                    D_c = construct_class_dictionary(X_c, tau=tau, gamma=gamma)
                    D_c_dict[cls] = D_c

                # 2. SBM estimation and error vector generation (extending to 92-dimensional features)
                X_tr_ext = generate_extended_features(X_tr, D_c_dict, gamma=gamma)
                X_val_ext = generate_extended_features(X_val, D_c_dict, gamma=gamma)

                # 3. Train Random Forest Classifier
                # We use n_jobs=-1 to train on all 8 CPU cores concurrently
                clf = RandomForestClassifier(
                    n_estimators=500,
                    max_features='sqrt',
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42
                )
                clf.fit(X_tr_ext, y_tr)

                # 4. Evaluate classification accuracy on the validation fold
                y_pred = clf.predict(X_val_ext)
                acc = accuracy_score(y_val, y_pred)
                fold_accuracies.append(acc)

            mean_acc = np.mean(fold_accuracies)
            results.append((gamma, tau, mean_acc))
            comb_elapsed = time.time() - comb_start
            print(f"  -> Mean 10-Fold CV Accuracy: {mean_acc * 100.0:.2f}% (computed in {comb_elapsed:.2f}s)")

    grid_elapsed = time.time() - grid_start_time
    print(f"\nGrid search completed in {grid_elapsed:.2f} seconds.")

    # Sort results to find the best configuration
    results.sort(key=lambda x: x[2], reverse=True)
    best_gamma, best_tau, best_acc = results[0]

    # Print results table
    print_results_table(results)

    # Highlight optimal hyperparameter combination
    print("\n================== OPTIMAL CONFIGURATION ==================")
    print(f"Best WSF Gamma (γ):       {best_gamma:.2f}")
    print(f"Best Threshold Tau (τ):   {best_tau:.2f}")
    print(f"Maximum CV Accuracy:      {best_acc * 100.0:.2f}%")
    print("===========================================================")

    total_elapsed = time.time() - overall_start_time
    print(f"\nHyperparameter tuning completed in {total_elapsed:.2f} seconds!")
    print("===========================================================")
