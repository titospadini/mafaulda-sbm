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

# Ensure root is in sys.path so 'mafaulda' package can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mafaulda.sbm_model import (
    construct_class_dictionary,
    generate_similarity_extended_features,
)

from mafaulda.rf_classifier import (
    train_classifier,
    evaluate_classifier,
)


import argparse
from mafaulda.logging_utils import log, set_verbosity


def run_replication(use_gpu: bool = False) -> None:
    """
    Executes the replication pipeline for Experiment 3 Configuration 3 from the
    scientific paper.

    Pedagogical Context:
        This script implements the class-similarity feature extension
        methodology:
          - Loads the 46-dimensional hand-crafted training and testing signal
             features.
          - Reconstructs SBM dictionaries for all 6 fault classes using the
             optimized hyperparameters:
             similarity threshold $\\tau = 0.85$ and L1-based WSF sensitivity
             $\\gamma = 0.0010$.
          - Generates 52-dimensional extended feature matrices by concatenating
             the 46 original signal
             features with the 6 SBM class similarity scores.
          - Trains the Random Forest ensemble on the similarity-extended
             training set.
          - Evaluates model fidelity on the test set, outputting overall
             classification accuracy,
             a detailed labeled confusion matrix, and full precision/recall
             metrics.
    """
    log("="*60, level=1)
    log("   Experiment 3 Configuration 3: SBM Similarity Features    ", level=1)
    log("="*60, level=1)

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')

    X_train_path = os.path.join(data_dir, 'X_train_features.npy')
    X_test_path = os.path.join(data_dir, 'X_test_features.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')
    y_test_path = os.path.join(data_dir, 'y_test.npy')

    # 1. Load original feature matrices and labels
    log(f"Loading features from {data_dir}...", level=2)
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    log(f"  Training set size: {len(X_train)} samples", level=1)
    log(f"  Testing set size:  {len(X_test)} samples", level=1)

    # Check GPU availability and fallback
    from mafaulda.gpu_utils import is_gpu_available
    active_gpu = use_gpu and is_gpu_available()
    if active_gpu:
        log("Using GPU Acceleration for SBM similarity computations.", level=1)
    elif use_gpu:
        log("GPU requested, but PyTorch or CUDA is not available. Falling back to CPU.", level=1)

    # 2. Build SBM dictionaries using Weiszfeld's and Threshold methods
    # Optimal parameters: tau = 0.85, gamma = 0.0010
    tau = 0.85
    gamma = 0.0010
    unique_classes = np.unique(y_train)

    log(f"\nBuilding class dictionaries with optimal SBM parameters (tau={tau}, gamma={gamma})...", level=1)
    D_c_dict = {}
    for cls in unique_classes:
        class_start_time = time.time()
        X_c = X_train[y_train == cls]
        # Construct dictionary
        D_c = construct_class_dictionary(X_c, tau=tau, gamma=gamma)
        D_c_dict[cls] = D_c
        log(f"  - Class '{cls}': built D_c shape {D_c.shape} from {len(X_c)} samples in {time.time() - class_start_time:.2f}s", level=2)

    # 3. Generate 52-dimensional similarity extended feature matrices
    log("\nGenerating extended 52-dimensional feature matrices (SBM similarity scores)...", level=1)
    X_train_extended_sim = generate_similarity_extended_features(X_train, D_c_dict, gamma=gamma, use_gpu=active_gpu)
    X_test_extended_sim = generate_similarity_extended_features(X_test, D_c_dict, gamma=gamma, use_gpu=active_gpu)

    log(f"  X_train_extended_sim shape: {X_train_extended_sim.shape} (Expected: ({len(X_train)}, 52))", level=3)
    log(f"  X_test_extended_sim shape:  {X_test_extended_sim.shape} (Expected: ({len(X_test)}, 52))", level=3)

    # 4. Save new extended feature matrices
    X_train_ext_path = os.path.join(data_dir, 'X_train_extended_sim.npy')
    X_test_ext_path = os.path.join(data_dir, 'X_test_extended_sim.npy')
    np.save(X_train_ext_path, X_train_extended_sim)
    np.save(X_test_ext_path, X_test_extended_sim)
    log(f"\nNew extended matrices saved successfully to data/:", level=2)
    log(f"  - X_train_extended_sim.npy ({os.path.getsize(X_train_ext_path) / 1024:.1f} KB)", level=2)
    log(f"  - X_test_extended_sim.npy ({os.path.getsize(X_test_ext_path) / 1024:.1f} KB)", level=2)

    # 5. Train and evaluate the Random Forest Classifier per se
    log("\nTraining Random Forest Classifier on SBM similarity extended features...", level=1)
    train_start = time.time()
    clf = train_classifier(X_train_extended_sim, y_train)
    log(f"Training completed in {time.time() - train_start:.2f} seconds.", level=2)

    log("Evaluating classifier on testing set...", level=2)
    evaluate_classifier(clf, X_test_extended_sim, y_test, y_train)
    log("="*60, level=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replication Script for Experiment 3 Configuration 3 SBM Similarity Features.")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity level (-v for detailed, -vv for debug).')
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2, 3], default=None,
                        help='Directly set verbosity level (0=silent, 1=default, 2=detailed, 3=debug).')
    parser.add_argument('--gpu', action='store_true',
                        help='Enable optional GPU acceleration using PyTorch and CUDA.')

    args = parser.parse_args()

    # Determine verbosity level
    if args.verbosity is not None:
        verbosity_level = args.verbosity
    else:
        verbosity_level = 1 + args.verbose
    set_verbosity(verbosity_level)

    try:
        run_replication(use_gpu=args.gpu)
    except Exception as e:
        log(f"\n[ERROR] Replication script failed with exception: {e}", level=0, file=sys.stderr)
        import traceback
        if verbosity_level > 0:
            traceback.print_exc()
        sys.exit(1)
