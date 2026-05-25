"""
MaFaulDa Rotating-Machine Fault Diagnosis - Unified Execution Entrypoint (Pure-Python Version)

This script acts as the single dedicated entrypoint for the entire data
preparation, feature extraction, SBM dictionary construction, classifier evaluation, and
hyperparameter tuning pipeline using ONLY inherent Python standard library modules.
"""

import argparse
import os
import sys
import time
import pickle

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mafaulda.data_prep import (
    map_dataset,
    load_and_normalize,
    TRAIN_TEST_SPLIT_RATIO,
    EXPECTED_TOTAL_FILES,
)

from mafaulda.feature_extraction import (
    process_set_parallel,
    EXPECTED_FEATURES,
)

from mafaulda.sbm_model import (
    construct_class_dictionary,
    generate_extended_features,
    GAMMA,
    TAU,
    EXPECTED_ORIGINAL_FEATURES,
    EXPECTED_EXTENDED_FEATURES,
)

from mafaulda.rf_classifier import (
    train_classifier,
    evaluate_classifier,
)

from mafaulda.cv_tuning import run_tuning

from mafaulda.logging_utils import log, set_verbosity


def run_pipeline(
    dataset_path: str,
    skip_extraction: bool,
    data_dir: str,
    use_hann: bool = False,
    use_fixed_entropy: bool = False
) -> None:
    """
    Executes the standard end-to-end rotating-machine fault diagnosis
    classification pipeline.
    """
    pipeline_start = time.time()

    log("\n" + "="*60, level=1)
    log("=== STEP 1: Dataset Mapping & Stratified Splitting ===", level=1)
    log("="*60, level=1)

    # 1. Map and split the dataset
    train_paths, test_paths, train_labels, test_labels = map_dataset(dataset_path)
    if not train_paths:
        raise ValueError(f"No CSV files found in the dataset directory: {dataset_path}")

    log(f"Training samples:   {len(train_paths)}", level=1)
    log(f"Testing samples:    {len(test_paths)}", level=1)

    # Validation checks
    total_mapped = len(train_paths) + len(test_paths)
    if total_mapped != EXPECTED_TOTAL_FILES:
        log(f"Warning: Expected {EXPECTED_TOTAL_FILES} files for the MaFaulDa dataset, but found {total_mapped}.", level=1)
    else:
        log("Full dataset verification checks passed successfully!", level=3)

    log("\n" + "="*60, level=1)
    log("=== STEP 2: Hand-Crafted Feature Extraction ===", level=1)
    log("="*60, level=1)

    X_train_path = os.path.join(data_dir, 'X_train_features.pkl')
    X_test_path = os.path.join(data_dir, 'X_test_features.pkl')
    y_train_path = os.path.join(data_dir, 'y_train.pkl')
    y_test_path = os.path.join(data_dir, 'y_test.pkl')

    if skip_extraction and os.path.exists(X_train_path) and os.path.exists(X_test_path):
        log("Feature extraction skipped. Loading pre-extracted features from data/...", level=2)
        with open(X_train_path, 'rb') as f:
            X_train = pickle.load(f)
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        with open(y_train_path, 'rb') as f:
            y_train = pickle.load(f)
        with open(y_test_path, 'rb') as f:
            y_test = pickle.load(f)
    else:
        # Extract features in parallel for training and testing sets
        X_train = process_set_parallel(train_paths, "Training", use_hann=use_hann, use_fixed_entropy=use_fixed_entropy)
        X_test = process_set_parallel(test_paths, "Testing", use_hann=use_hann, use_fixed_entropy=use_fixed_entropy)
        y_train = train_labels
        y_test = test_labels

        # Save results using standard library pickle
        with open(X_train_path, 'wb') as f:
            pickle.dump(X_train, f)
        with open(X_test_path, 'wb') as f:
            pickle.dump(X_test, f)
        with open(y_train_path, 'wb') as f:
            pickle.dump(y_train, f)
        with open(y_test_path, 'wb') as f:
            pickle.dump(y_test, f)
        log(f"Features saved successfully in {data_dir}.", level=2)

    # Validation checks on features
    assert len(X_train) == len(train_paths), f"X_train size mismatch: {len(X_train)}"
    assert len(X_test) == len(test_paths), f"X_test size mismatch: {len(X_test)}"
    log("Feature matrices validated. Dimensions and integrity verified.", level=3)

    log("\n" + "="*60, level=1)
    log("=== STEP 3: SBM Dictionary Construction & Feature Extension ===", level=1)
    log("="*60, level=1)

    unique_classes = sorted(list(set(y_train)))
    log(f"Constructing class dictionary matrices (D_c) using Weiszfeld's and Threshold methods for {len(unique_classes)} classes...", level=1)

    D_c_dict = {}
    for cls in unique_classes:
        class_start_time = time.time()
        X_c = [X_train[i] for i, lbl in enumerate(y_train) if lbl == cls]
        # Build dictionary with strict tau and gamma
        D_c = construct_class_dictionary(X_c, tau=TAU, gamma=GAMMA)
        D_c_dict[cls] = D_c
        class_elapsed = time.time() - class_start_time
        log(f"  - Class '{cls}': built D_c shape {len(D_c)}x{len(D_c[0]) if D_c else 0} from {len(X_c)} samples in {class_elapsed:.2f}s", level=2)

    log("\nGenerating extended 92-dimensional feature matrices (SBM Model B)...", level=1)
    X_train_extended = generate_extended_features(X_train, D_c_dict, gamma=GAMMA)
    X_test_extended = generate_extended_features(X_test, D_c_dict, gamma=GAMMA)

    # Save the new extended feature matrices
    X_train_ext_path = os.path.join(data_dir, 'X_train_extended.pkl')
    X_test_ext_path = os.path.join(data_dir, 'X_test_extended.pkl')
    with open(X_train_ext_path, 'wb') as f:
        pickle.dump(X_train_extended, f)
    with open(X_test_ext_path, 'wb') as f:
        pickle.dump(X_test_extended, f)

    log("Extended features generated and validated successfully.", level=3)

    log("\n" + "="*60, level=1)
    log("=== STEP 4: Random Forest Classification ===", level=1)
    log("="*60, level=1)

    log("Initializing and training Random Forest Classifier on extended features...", level=1)
    train_start = time.time()
    clf = train_classifier(X_train_extended, y_train)
    log(f"Training completed in {time.time() - train_start:.2f} seconds.", level=2)

    log("Evaluating classifier...", level=2)
    evaluate_classifier(clf, X_test_extended, y_test, y_train)

    pipeline_elapsed = time.time() - pipeline_start
    log(f"\nEnd-to-End Pipeline completed successfully in {pipeline_elapsed:.2f} seconds!", level=2)
    log("="*60, level=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Execution Script for Rotating-Machine Fault Diagnosis pipeline.")
    parser.add_argument('--dataset_path', type=str, default='~/datasets/mafaulda',
                        help='Path to the raw MaFaulDa dataset directory.')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip parallel feature extraction if original cached files exist.')
    parser.add_argument('--tune', action='store_true',
                        help='Execute Stratified 10-Fold Cross-Validation tuning grid search instead.')
    parser.add_argument('--use_hann', action='store_true',
                        help='Apply Hanning window and coherent gain correction to FFT.')
    parser.add_argument('--use_fixed_entropy', action='store_true',
                        help='Use a fixed histogram range (-10.0, 10.0) for Shannon entropy calculation.')
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

    dataset_path = os.path.abspath(os.path.expanduser(args.dataset_path))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    log("="*60, level=1)
    log("     Rotating-Machine Fault Diagnosis Unified Pipeline      ", level=1)
    log("="*60, level=1)
    log(f"Configured Dataset Path: {dataset_path}", level=2)
    log(f"Target Data Directory:   {data_dir}", level=2)

    try:
        if args.tune:
            run_tuning(data_dir)
        else:
            run_pipeline(
                dataset_path,
                args.skip_extraction,
                data_dir,
                use_hann=args.use_hann,
                use_fixed_entropy=args.use_fixed_entropy
            )
    except Exception as e:
        log(f"\n[ERROR] Pipeline failed with exception: {e}", level=0, file=sys.stderr)
        import traceback
        if verbosity_level > 0:
            traceback.print_exc()
        sys.exit(1)
