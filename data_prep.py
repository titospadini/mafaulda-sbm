"""
Data Preparation Script for MaFaulDa Dataset

This script handles the initial data pipeline for reproducing the scientific paper
on rotating-machine fault diagnosis using the MaFaulDa database.

Key steps included:
1. Mapping: Recursively reads operational scenario files (.csv) and extracts their fault label.
2. Stratification: Performs a train/test split ensuring the original fault class proportions.
3. Normalization: Normalizes the raw sensor signals to unit variance.
"""
import argparse
import os
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

# Configuration Constants
EXPECTED_TOTAL_FILES = 1951
TRAIN_TEST_SPLIT_RATIO = 0.10
RANDOM_STATE = 42


def load_and_normalize(filepath: str) -> np.ndarray:
    """
    Reads the CSV file data (8 sensor columns) and normalizes each signal
    so that they possess unit variance by dividing by their standard deviation.

    Parameters:
        filepath (str): The absolute path to the CSV file.

    Returns:
        np.ndarray: The normalized signal matrix with unit variance columns.
    """
    # Load CSV data. It is comma-separated and has no header.
    data = np.loadtxt(filepath, delimiter=',')

    # Calculate standard deviation for each of the 8 columns (axis=0)
    std_devs = np.std(data, axis=0)

    # Avoid division by zero for any constant signals (if any exist)
    # np.where checks the condition (std_devs == 0.0). If true, it replaces the value with 1.0,
    # otherwise it keeps the original standard deviation.
    std_devs = np.where(std_devs == 0.0, 1.0, std_devs)

    # Perform unit variance normalization
    normalized_data = data / std_devs

    return normalized_data


def map_dataset(dataset_dir: str) -> Tuple[List[str], List[str]]:
    """
    Traverses the dataset directory recursively, finding all CSV files and
    mapping them to their corresponding fault label based on the top-level
    subdirectory name directly under dataset_dir.

    Parameters:
        dataset_dir (str): Path to the raw dataset directory.

    Returns:
        Tuple[List[str], List[str]]: (filepaths list, labels list)
    """
    filepaths: List[str] = []
    labels: List[str] = []
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                abs_path = os.path.abspath(os.path.join(root, file))
                # Get path relative to the dataset root
                rel_path = os.path.relpath(abs_path, dataset_dir)
                # The top-level folder name right under mafaulda/ represents the fault class (e.g., 'normal', 'imbalance').
                # We extract it by taking the first element of the split relative path.
                label = rel_path.split(os.sep)[0]
                filepaths.append(abs_path)
                labels.append(label)

    return filepaths, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare the MaFaulDa dataset.")
    parser.add_argument('--dataset_path', type=str, default='~/datasets/mafaulda',
                        help='Path to the raw MaFaulDa dataset directory.')
    args = parser.parse_args()

    # Raw dataset path
    dataset_path: str = args.dataset_path

    # Map the dataset
    filepaths, labels = map_dataset(dataset_path)

    if not filepaths:
        raise ValueError(f"No CSV files found in the dataset directory: {dataset_path}")

    # Perform a stratified train/test split to create disjoint sets for training and validation.
    # The 'stratify' parameter strictly maintains the original class proportions.
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        filepaths,
        labels,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    # Print validation information to terminal
    print(f"Total mapped files: {len(filepaths)}")
    print(f"Training samples:   {len(train_paths)}")
    print(f"Testing samples:    {len(test_paths)}")

    # Simple validation checks
    if len(filepaths) != EXPECTED_TOTAL_FILES:
        print(f"Warning: Expected {EXPECTED_TOTAL_FILES} files for the full MaFaulDa dataset, but found {len(filepaths)}.")
    else:
        expected_test = int(EXPECTED_TOTAL_FILES * TRAIN_TEST_SPLIT_RATIO)
        expected_train = EXPECTED_TOTAL_FILES - expected_test

        # When using stratify, exact expected numbers can be off by a few samples due to per-class rounding.
        assert abs(len(train_paths) - expected_train) <= 5, f"Expected roughly {expected_train} training samples, but got {len(train_paths)}"
        assert abs(len(test_paths) - expected_test) <= 5, f"Expected roughly {expected_test} testing samples, but got {len(test_paths)}"
        assert len(train_paths) + len(test_paths) == EXPECTED_TOTAL_FILES, "Total samples mismatch after split."
        print("Full dataset verification checks passed successfully!")

    # Test load_and_normalize on a sample file
    if len(train_paths) > 0:
        sample_path = train_paths[0]
        norm_data = load_and_normalize(sample_path)

        # Verify the shape is (N, 8)
        assert norm_data.ndim == 2 and norm_data.shape[1] == 8, f"Expected shape (N, 8), got {norm_data.shape}"

        # Verify standard deviation of normalized columns is 1.0 (within numerical tolerance)
        std_vals = np.std(norm_data, axis=0)
        assert np.allclose(std_vals, 1.0, atol=1e-5), f"Expected standard deviations to be close to 1.0, got {std_vals}"
        print(f"Sample normalization test passed! Standard deviations: {std_vals}")
