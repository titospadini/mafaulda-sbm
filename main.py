"""
MaFaulDa Rotating-Machine Fault Diagnosis - Unified Execution Entrypoint

This script acts as the single dedicated entrypoint for the entire data
preparation,
feature extraction, SBM dictionary construction, classifier evaluation, and
hyperparameter tuning pipeline.

Workflow:
1. Parse command-line arguments to specify dataset path, actions, and
   optimization options.
2. Step 1: Map dataset files and perform stratified train/test split.
3. Step 2: Run parallel feature extraction (46 features per file) or skip if
   cached.
4. Step 3: Construct class dictionaries (D_c) and SBM extended feature matrices
   (92 features).
5. Step 4: Train Random Forest classifier and perform evaluation (accuracy,
   confusion matrix, report).
6. Optional Step 5: Execute 10-fold Stratified Cross-Validation grid search for
   hyperparameter tuning.
"""

import argparse
import os
import sys
import time
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_prep import (
    map_dataset,
    load_and_normalize,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_STATE,
    EXPECTED_TOTAL_FILES,
)

from feature_extraction import (
    process_set_parallel,
    EXPECTED_FEATURES,
)

from sbm_model import (
    construct_class_dictionary,
    generate_extended_features,
    GAMMA,
    TAU,
    EXPECTED_ORIGINAL_FEATURES,
    EXPECTED_EXTENDED_FEATURES,
)

from rf_classifier import (
    train_classifier,
    evaluate_classifier,
)

from cv_tuning import run_tuning


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

    Pedagogical Context:
        This orchestrator coordinates the entire multi-stage pipeline:
          1. Step 1: Mapping & Splitting: Reads raw files, verifies total
             counts, and performs a 90/10
             stratified train/test split to guarantee representativeness.
          2. Step 2: Feature Extraction: Runs process-parallel feature
             extraction (46 features per file)
             or skips it to load cached matrices from `data/`.
          3. Step 3: SBM Dictionary & Projection: Builds 6 class memory matrices
             (dictionaries) strictly
             on the training features, then projects all training and testing
             features into 92-dimensional
             extended feature matrices using best-matching SBM estimation error
             vectors.
          4. Step 4: Random Forest Classification: Trains the Random Forest
             ensemble and outputs comprehensive
             classification accuracy, confusion matrix, and precision/recall
             diagnostics.

    Parameters:
        dataset_path (str): Path to the raw directory of the MaFaulDa database.
        skip_extraction (bool): If True, reuses pre-extracted signal features
        from files under `./data`
          to speed up iterations.
        data_dir (str): Directory where intermediate features and labels are
        saved.
        use_hann (bool): Whether to apply a Hanning window and coherent gain
        correction to DFT signals.
        use_fixed_entropy (bool): Whether to lock the Shannon entropy histogram
        range to (-10.0, 10.0).
    """
    pipeline_start = time.time()

    print("\n" + "="*60)
    print("=== STEP 1: Dataset Mapping & Stratified Splitting ===")
    print("="*60)

    # 1. Map the dataset
    filepaths, labels = map_dataset(dataset_path)
    if not filepaths:
        raise ValueError(f"No CSV files found in the dataset directory: {dataset_path}")

    print(f"Total mapped files: {len(filepaths)}")

    # Perform stratified split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        filepaths,
        labels,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    print(f"Training samples:   {len(train_paths)}")
    print(f"Testing samples:    {len(test_paths)}")

    # Validation checks
    if len(filepaths) != EXPECTED_TOTAL_FILES:
        print(f"Warning: Expected {EXPECTED_TOTAL_FILES} files for the full MaFaulDa dataset, but found {len(filepaths)}.")
    else:
        expected_test = int(EXPECTED_TOTAL_FILES * TRAIN_TEST_SPLIT_RATIO)
        expected_train = EXPECTED_TOTAL_FILES - expected_test
        assert abs(len(train_paths) - expected_train) <= 5, f"Expected ~{expected_train} train samples, got {len(train_paths)}"
        assert abs(len(test_paths) - expected_test) <= 5, f"Expected ~{expected_test} test samples, got {len(test_paths)}"
        print("Full dataset verification checks passed successfully!")

    # Check sample file
    if len(train_paths) > 0:
        sample_path = train_paths[0]
        norm_data = load_and_normalize(sample_path)
        assert norm_data.ndim == 2 and norm_data.shape[1] == 8, f"Expected shape (N, 8), got {norm_data.shape}"
        print("Sample signal loaded and normalized successfully.")

    print("\n" + "="*60)
    print("=== STEP 2: Hand-Crafted Feature Extraction ===")
    print("="*60)

    X_train_path = os.path.join(data_dir, 'X_train_features.npy')
    X_test_path = os.path.join(data_dir, 'X_test_features.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')
    y_test_path = os.path.join(data_dir, 'y_test.npy')

    if skip_extraction and os.path.exists(X_train_path) and os.path.exists(X_test_path):
        print("Feature extraction skipped. Loading pre-extracted features from data/...")
        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)
        y_train = np.load(y_train_path, allow_pickle=True)
        y_test = np.load(y_test_path, allow_pickle=True)
    else:
        # Extract features in parallel for training and testing sets
        X_train = process_set_parallel(train_paths, "Training", use_hann=use_hann, use_fixed_entropy=use_fixed_entropy)
        X_test = process_set_parallel(test_paths, "Testing", use_hann=use_hann, use_fixed_entropy=use_fixed_entropy)
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        # Save results
        np.save(X_train_path, X_train)
        np.save(X_test_path, X_test)
        np.save(y_train_path, y_train)
        np.save(y_test_path, y_test)
        print(f"Features saved successfully in {data_dir}.")

    # Validation checks on features
    assert X_train.shape == (len(train_paths), EXPECTED_FEATURES), f"X_train shape mismatch: {X_train.shape}"
    assert X_test.shape == (len(test_paths), EXPECTED_FEATURES), f"X_test shape mismatch: {X_test.shape}"
    assert not np.isnan(X_train).any(), "Found NaN values in X_train!"
    assert not np.isnan(X_test).any(), "Found NaN values in X_test!"
    print("Feature matrices validated. Dimensions and integrity verified.")

    print("\n" + "="*60)
    print("=== STEP 3: SBM Dictionary Construction & Feature Extension ===")
    print("="*60)

    unique_classes = np.unique(y_train)
    print(f"Constructing class dictionary matrices (D_c) using Weiszfeld's and Threshold methods for {len(unique_classes)} classes...")

    D_c_dict = {}
    for cls in unique_classes:
        class_start_time = time.time()
        X_c = X_train[y_train == cls]
        # Build dictionary with strict tau and gamma
        D_c = construct_class_dictionary(X_c, tau=TAU, gamma=GAMMA)
        D_c_dict[cls] = D_c
        class_elapsed = time.time() - class_start_time
        print(f"  - Class '{cls}': built D_c shape {D_c.shape} from {len(X_c)} samples in {class_elapsed:.2f}s")

    print("\nGenerating extended 92-dimensional feature matrices (SBM Model B)...")
    X_train_extended = generate_extended_features(X_train, D_c_dict, gamma=GAMMA)
    X_test_extended = generate_extended_features(X_test, D_c_dict, gamma=GAMMA)

    # Save the new extended feature matrices
    X_train_ext_path = os.path.join(data_dir, 'X_train_extended.npy')
    X_test_ext_path = os.path.join(data_dir, 'X_test_extended.npy')
    np.save(X_train_ext_path, X_train_extended)
    np.save(X_test_ext_path, X_test_extended)

    # Verify extended matrices
    assert X_train_extended.shape == (len(X_train), EXPECTED_EXTENDED_FEATURES), f"X_train_extended shape mismatch: {X_train_extended.shape}"
    assert X_test_extended.shape == (len(X_test), EXPECTED_EXTENDED_FEATURES), f"X_test_extended shape mismatch: {X_test_extended.shape}"
    assert not np.isnan(X_train_extended).any(), "Found NaN values in X_train_extended!"
    assert not np.isnan(X_test_extended).any(), "Found NaN values in X_test_extended!"
    print("Extended features generated and validated successfully.")

    print("\n" + "="*60)
    print("=== STEP 4: Random Forest Classification ===")
    print("="*60)

    print("Initializing and training Random Forest Classifier on extended features...")
    train_start = time.time()
    clf = train_classifier(X_train_extended, y_train)
    print(f"Training completed in {time.time() - train_start:.2f} seconds.")

    print("Evaluating classifier...")
    evaluate_classifier(clf, X_test_extended, y_test, y_train)

    pipeline_elapsed = time.time() - pipeline_start
    print(f"\nEnd-to-End Pipeline completed successfully in {pipeline_elapsed:.2f} seconds!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Execution Script for Rotating-Machine Fault Diagnosis pipeline.")
    parser.add_argument('--dataset_path', type=str, default='~/datasets/mafaulda',
                        help='Path to the raw MaFaulDa dataset directory.')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip parallel feature extraction if original .npy files exist in data/.')
    parser.add_argument('--tune', action='store_true',
                        help='Execute Stratified 10-Fold Cross-Validation tuning grid search instead.')
    parser.add_argument('--use_hann', action='store_true',
                        help='Apply Hanning window and coherent gain correction to FFT.')
    parser.add_argument('--use_fixed_entropy', action='store_true',
                        help='Use a fixed histogram range (-10.0, 10.0) for Shannon entropy calculation.')

    args = parser.parse_args()

    # Resolve absolute paths
    dataset_path = os.path.abspath(os.path.expanduser(args.dataset_path))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    print("="*60)
    print("     Rotating-Machine Fault Diagnosis Unified Pipeline      ")
    print("="*60)
    print(f"Configured Dataset Path: {dataset_path}")
    print(f"Target Data Directory:   {data_dir}")

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
        print(f"\n[ERROR] Pipeline failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
