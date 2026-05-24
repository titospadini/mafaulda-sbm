"""
Experiment 3 Configuration 3 Replication Script

This script feeds the Random Forest with SBM similarities instead of estimation
errors,
following the third configuration of Experiment 3 from the paper:
1. Loads the 46-dimensional features (X_train_features, X_test_features) and
   labels (y_train, y_test).
2. Builds SBM dictionaries with optimal SBM parameters: threshold tau = 0.85 and
   gamma = 0.0010.
3. Generates the 52-dimensional extended feature matrices using WSF similarity
   scores for each of the 6 classes.
4. Saves the new extended matrices as X_train_extended_sim.npy and
   X_test_extended_sim.npy.
5. Trains and evaluates the Random Forest classifier on the 90% training set and
   10% test set.
6. Prints the final Test Accuracy and Confusion Matrix.
"""

import os
import sys
import time
import numpy as np

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sbm_model import (
    construct_class_dictionary,
    generate_similarity_extended_features,
)

from rf_classifier import (
    train_classifier,
    evaluate_classifier,
)


def run_replication() -> None:
    """
    Executes the replication pipeline for Experiment 3 Configuration 3 from the
    scientific paper.

    Pedagogical Context:
        This script implements the class-similarity feature extension
        methodology:
          1. Loads the 46-dimensional hand-crafted training and testing signal
             features.
          2. Reconstructs SBM dictionaries for all 6 fault classes using the
             optimized hyperparameters:
             similarity threshold $\\tau = 0.85$ and L1-based WSF sensitivity
             $\\gamma = 0.0010$.
          3. Generates 52-dimensional extended feature matrices by concatenating
             the 46 original signal
             features with the 6 SBM class similarity scores.
          4. Trains the Random Forest ensemble on the similarity-extended
             training set.
          5. Evaluates model fidelity on the test set, outputting overall
             classification accuracy,
             a detailed labeled confusion matrix, and full precision/recall
             metrics.
    """
    print("="*60)
    print("   Experiment 3 Configuration 3: SBM Similarity Features    ")
    print("="*60)

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')

    X_train_path = os.path.join(data_dir, 'X_train_features.npy')
    X_test_path = os.path.join(data_dir, 'X_test_features.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')
    y_test_path = os.path.join(data_dir, 'y_test.npy')

    # 1. Load original feature matrices and labels
    print(f"Loading features from {data_dir}...")
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    print(f"  Training set size: {len(X_train)} samples")
    print(f"  Testing set size:  {len(X_test)} samples")

    # 2. Build SBM dictionaries using Weiszfeld's and Threshold methods
    # Optimal parameters: tau = 0.85, gamma = 0.0010
    tau = 0.85
    gamma = 0.0010
    unique_classes = np.unique(y_train)

    print(f"\nBuilding class dictionaries with optimal SBM parameters (tau={tau}, gamma={gamma})...")
    D_c_dict = {}
    for cls in unique_classes:
        class_start_time = time.time()
        X_c = X_train[y_train == cls]
        # Construct dictionary
        D_c = construct_class_dictionary(X_c, tau=tau, gamma=gamma)
        D_c_dict[cls] = D_c
        print(f"  - Class '{cls}': built D_c shape {D_c.shape} from {len(X_c)} samples in {time.time() - class_start_time:.2f}s")

    # 3. Generate 52-dimensional similarity extended feature matrices
    print("\nGenerating extended 52-dimensional feature matrices (SBM similarity scores)...")
    X_train_extended_sim = generate_similarity_extended_features(X_train, D_c_dict, gamma=gamma)
    X_test_extended_sim = generate_similarity_extended_features(X_test, D_c_dict, gamma=gamma)

    print(f"  X_train_extended_sim shape: {X_train_extended_sim.shape} (Expected: ({len(X_train)}, 52))")
    print(f"  X_test_extended_sim shape:  {X_test_extended_sim.shape} (Expected: ({len(X_test)}, 52))")

    # 4. Save new extended feature matrices
    X_train_ext_path = os.path.join(data_dir, 'X_train_extended_sim.npy')
    X_test_ext_path = os.path.join(data_dir, 'X_test_extended_sim.npy')
    np.save(X_train_ext_path, X_train_extended_sim)
    np.save(X_test_ext_path, X_test_extended_sim)
    print(f"\nNew extended matrices saved successfully to data/:")
    print(f"  - X_train_extended_sim.npy ({os.path.getsize(X_train_ext_path) / 1024:.1f} KB)")
    print(f"  - X_test_extended_sim.npy ({os.path.getsize(X_test_ext_path) / 1024:.1f} KB)")

    # 5. Train and evaluate the Random Forest Classifier per se
    print("\nTraining Random Forest Classifier on SBM similarity extended features...")
    train_start = time.time()
    clf = train_classifier(X_train_extended_sim, y_train)
    print(f"Training completed in {time.time() - train_start:.2f} seconds.")

    print("Evaluating classifier on testing set...")
    evaluate_classifier(clf, X_test_extended_sim, y_test, y_train)
    print("="*60)


if __name__ == '__main__':
    run_replication()
