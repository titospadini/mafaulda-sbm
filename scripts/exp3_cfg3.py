"""
Experiment 3 Configuration 3 Replication Script (Pure-Python Version)

This script feeds the Random Forest with SBM similarities instead of estimation errors:
1. Loads the 46-dimensional features and labels from pickle files.
2. Builds SBM dictionaries with optimal SBM parameters.
3. Generates 52-dimensional extended feature matrices.
4. Trains and evaluates the Random Forest classifier.
"""

import os
import sys
import time
import pickle
import argparse

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

from mafaulda.logging_utils import log, set_verbosity


def run_replication() -> None:
    """
    Executes the replication pipeline for Experiment 3 Configuration 3 using only pure Python.
    """
    log("="*60, level=1)
    log("   Experiment 3 Configuration 3: SBM Similarity Features    ", level=1)
    log("="*60, level=1)

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')

    X_train_path = os.path.join(data_dir, 'X_train_features.pkl')
    X_test_path = os.path.join(data_dir, 'X_test_features.pkl')
    y_train_path = os.path.join(data_dir, 'y_train.pkl')
    y_test_path = os.path.join(data_dir, 'y_test.pkl')

    # 1. Load original feature matrices and labels
    log(f"Loading features from {data_dir}...", level=2)
    with open(X_train_path, 'rb') as f:
        X_train = pickle.load(f)
    with open(X_test_path, 'rb') as f:
        X_test = pickle.load(f)
    with open(y_train_path, 'rb') as f:
        y_train = pickle.load(f)
    with open(y_test_path, 'rb') as f:
        y_test = pickle.load(f)

    log(f"  Training set size: {len(X_train)} samples", level=1)
    log(f"  Testing set size:  {len(X_test)} samples", level=1)

    # 2. Build SBM dictionaries using Weiszfeld's and Threshold methods
    tau = 0.85
    gamma = 0.0010
    unique_classes = sorted(list(set(y_train)))

    log(f"\nBuilding class dictionaries with optimal SBM parameters (tau={tau}, gamma={gamma})...", level=1)
    D_c_dict = {}
    for cls in unique_classes:
        class_start_time = time.time()
        X_c = [X_train[i] for i, lbl in enumerate(y_train) if lbl == cls]
        # Construct dictionary
        D_c = construct_class_dictionary(X_c, tau=tau, gamma=gamma)
        D_c_dict[cls] = D_c
        log(f"  - Class '{cls}': built D_c shape {len(D_c)}x{len(D_c[0]) if D_c else 0} from {len(X_c)} samples in {time.time() - class_start_time:.2f}s", level=2)

    # 3. Generate 52-dimensional similarity extended feature matrices
    log("\nGenerating extended 52-dimensional feature matrices (SBM similarity scores)...", level=1)
    X_train_extended_sim = generate_similarity_extended_features(X_train, D_c_dict, gamma=gamma)
    X_test_extended_sim = generate_similarity_extended_features(X_test, D_c_dict, gamma=gamma)

    # 4. Save new extended feature matrices
    X_train_ext_path = os.path.join(data_dir, 'X_train_extended_sim.pkl')
    X_test_ext_path = os.path.join(data_dir, 'X_test_extended_sim.pkl')
    with open(X_train_ext_path, 'wb') as f:
        pickle.dump(X_train_extended_sim, f)
    with open(X_test_ext_path, 'wb') as f:
        pickle.dump(X_test_extended_sim, f)
    log(f"\nNew extended matrices saved successfully to data/:", level=2)

    # 5. Train and evaluate the Random Forest Classifier
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
                        help='Increase verbosity level.')
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2, 3], default=None,
                        help='Directly set verbosity level.')

    args = parser.parse_args()

    if args.verbosity is not None:
        verbosity_level = args.verbosity
    else:
        verbosity_level = 1 + args.verbose
    set_verbosity(verbosity_level)

    try:
        run_replication()
    except Exception as e:
        log(f"\n[ERROR] Replication script failed with exception: {e}", level=0, file=sys.stderr)
        import traceback
        if verbosity_level > 0:
            traceback.print_exc()
        sys.exit(1)
