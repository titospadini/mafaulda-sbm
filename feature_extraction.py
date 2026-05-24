"""
Feature Extraction Script for MaFaulDa Dataset

This script handles Step 2 of our Rotating-Machine Fault Diagnosis pipeline:
1. Re-constructs the identical stratified train/test split from data_prep.py.
2. Extracts exactly 46 hand-crafted features per scenario (1 rotation frequency,
   21 spectral features, and 24 statistical features) from the raw sensor signals.
3. Leverages parallel processing (ProcessPoolExecutor) for fast, concurrent extraction.
4. Exports the final NumPy matrices (X_train, X_test, y_train, y_test) into the data/ directory.
"""

import os
import sys
import time
from typing import List, Tuple
import numpy as np
from scipy.stats import kurtosis, entropy
from concurrent.futures import ProcessPoolExecutor

# Import mapping, normalization, and configuration from data_prep
# Adding current directory to sys.path to ensure correct importing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import map_dataset, load_and_normalize, TRAIN_TEST_SPLIT_RATIO, RANDOM_STATE

# Constants
SAMPLING_RATE = 50000  # 50 kHz
EXPECTED_FEATURES = 46


def extract_features_for_file(filepath: str) -> np.ndarray:
    """
    Extracts exactly 46 features from a single CSV scenario file.

    The 46 features consist of:
      - 1 Rotation Frequency (fr): Peak DFT frequency of tachometer (column 7) within [5, 120] Hz.
      - 21 Spectral Features: DFT magnitude of the first 7 signals at fr, 2fr, and 3fr (7 * 3 = 21).
      - 24 Statistical Features: Mean, Shannon entropy, and kurtosis of all 8 signals (8 * 3 = 24).

    Parameters:
        filepath (str): The absolute path to the CSV file.

    Returns:
        np.ndarray: A 46-dimensional feature vector of shape (46,).
    """
    # 1. Load and normalize signal matrix (shape: N, 8)
    # This divides each sensor channel by its standard deviation for unit variance.
    normalized_data = load_and_normalize(filepath)
    N = len(normalized_data)

    # 2. Extract rotation frequency fr from tachometer signal (column 7)
    tacho = normalized_data[:, 7]
    fft_tacho = np.fft.rfft(tacho)
    freqs = np.fft.rfftfreq(N, d=1/SAMPLING_RATE)
    mags_tacho = np.abs(fft_tacho)

    # Restrict search for rotation speed peak to physical range [5, 120] Hz to avoid high-frequency noise
    mask = (freqs >= 5.0) & (freqs <= 120.0)
    masked_freqs = freqs[mask]
    masked_mags = mags_tacho[mask]

    if len(masked_mags) > 0:
        peak_sub_idx = np.argmax(masked_mags)
        f_r = masked_freqs[peak_sub_idx]
        # Find exact peak index in full frequency array
        peak_idx = np.argmin(np.abs(freqs - f_r))
    else:
        # Fallback to global peak (excluding DC component)
        peak_idx = np.argmax(mags_tacho[1:]) + 1
        f_r = freqs[peak_idx]

    # 3. Determine spectral bin indices for fr, 2fr, and 3fr
    idx_1 = peak_idx
    idx_2 = np.argmin(np.abs(freqs - (2.0 * f_r)))
    idx_3 = np.argmin(np.abs(freqs - (3.0 * f_r)))

    # Assemble feature vector
    feature_vector = []

    # Feature 0: Rotation frequency fr
    feature_vector.append(f_r)

    # Features 1-21: Spectral magnitudes of the first 7 sensor signals at fr, 2fr, 3fr
    # We normalize DFT magnitudes by N to make them independent of sample length.
    for col_idx in range(7):
        col_signal = normalized_data[:, col_idx]
        fft_col = np.fft.rfft(col_signal)
        mags_col = np.abs(fft_col) / N
        feature_vector.append(mags_col[idx_1])
        feature_vector.append(mags_col[idx_2])
        feature_vector.append(mags_col[idx_3])

    # Features 22-45: Statistical features (mean, Shannon entropy, kurtosis) for all 8 signals
    for col_idx in range(8):
        col_signal = normalized_data[:, col_idx]

        # Mean
        mean_val = np.mean(col_signal)

        # Shannon Entropy (estimated using 100-bin histogram)
        counts, _ = np.histogram(col_signal, bins=100)
        probs = counts / np.sum(counts)
        # Use scipy.stats.entropy which computes Shannon entropy (base=2 is standard)
        entropy_val = entropy(probs, base=2)

        # Kurtosis (Fisher definition: normal distribution = 0.0)
        kurt_val = kurtosis(col_signal, fisher=True)

        feature_vector.append(mean_val)
        feature_vector.append(entropy_val)
        feature_vector.append(kurt_val)

    # Return as float64 array
    return np.array(feature_vector, dtype=np.float64)


def process_set_parallel(filepaths: List[str], set_name: str) -> np.ndarray:
    """
    Extracts features for all files in a dataset set in parallel.

    Parameters:
        filepaths (List[str]): List of absolute file paths to CSV files.
        set_name (str): Label describing the set (e.g. 'Training' or 'Testing').

    Returns:
        np.ndarray: Combined feature matrix of shape (num_samples, 46).
    """
    total_files = len(filepaths)
    print(f"\nStarting feature extraction for {set_name} set ({total_files} samples) in parallel...")

    start_time = time.time()

    # Process files concurrently using all available CPU cores
    with ProcessPoolExecutor() as executor:
        # map returns a generator, which we resolve to a list
        feature_vectors = list(executor.map(extract_features_for_file, filepaths))

    elapsed_time = time.time() - start_time
    print(f"Completed {set_name} feature extraction in {elapsed_time:.2f} seconds!")

    # Convert list of vectors to a 2D numpy array
    feature_matrix = np.vstack(feature_vectors)
    return feature_matrix


if __name__ == '__main__':
    # Raw dataset path
    dataset_path = '~/datasets/mafaulda'

    print("=== MaFaulDa Step 2: Feature Extraction Pipeline ===")

    # 1. Map the dataset and obtain splits
    filepaths, labels = map_dataset(dataset_path)

    from sklearn.model_selection import train_test_split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        filepaths,
        labels,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    # 2. Extract features in parallel for training and testing sets
    X_train = process_set_parallel(train_paths, "Training")
    X_test = process_set_parallel(test_paths, "Testing")

    # Convert labels lists to numpy arrays
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # 3. Validate feature matrix shapes and properties
    assert X_train.shape == (len(train_paths), EXPECTED_FEATURES), f"X_train shape mismatch: {X_train.shape}"
    assert X_test.shape == (len(test_paths), EXPECTED_FEATURES), f"X_test shape mismatch: {X_test.shape}"
    assert y_train.shape == (len(train_labels),), f"y_train shape mismatch: {y_train.shape}"
    assert y_test.shape == (len(test_labels),), f"y_test shape mismatch: {y_test.shape}"

    print("\nVerification Checks Passed:")
    print(f"  X_train shape: {X_train.shape} (Expected: ({len(train_paths)}, {EXPECTED_FEATURES}))")
    print(f"  X_test shape:  {X_test.shape} (Expected: ({len(test_paths)}, {EXPECTED_FEATURES}))")
    print(f"  y_train shape: {y_train.shape} (Expected: ({len(train_labels)},))")
    print(f"  y_test shape:  {y_test.shape} (Expected: ({len(test_labels)},))")

    # Check for NaN values
    assert not np.isnan(X_train).any(), "Found NaN values in X_train!"
    assert not np.isnan(X_test).any(), "Found NaN values in X_test!"
    print("  No NaN values detected in the feature matrices!")

    # 4. Create target data/ directory and save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(output_dir, exist_ok=True)

    X_train_path = os.path.join(output_dir, 'X_train_features.npy')
    X_test_path = os.path.join(output_dir, 'X_test_features.npy')
    y_train_path = os.path.join(output_dir, 'y_train.npy')
    y_test_path = os.path.join(output_dir, 'y_test.npy')

    np.save(X_train_path, X_train)
    np.save(X_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)

    print(f"\nAll files saved successfully in {output_dir}:")
    print(f"  - X_train_features.npy ({os.path.getsize(X_train_path) / 1024:.1f} KB)")
    print(f"  - X_test_features.npy ({os.path.getsize(X_test_path) / 1024:.1f} KB)")
    print(f"  - y_train.npy ({os.path.getsize(y_train_path) / 1024:.1f} KB)")
    print(f"  - y_test.npy ({os.path.getsize(y_test_path) / 1024:.1f} KB)")
    print("====================================================")
