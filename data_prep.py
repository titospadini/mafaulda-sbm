import os
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


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

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.csv'):
                abs_path = os.path.abspath(os.path.join(root, file))
                # Get path relative to the dataset root
                rel_path = os.path.relpath(abs_path, dataset_dir)
                # The top-level folder name right under mafaulda/ is the first element
                label = rel_path.split(os.sep)[0]
                filepaths.append(abs_path)
                labels.append(label)

    return filepaths, labels


if __name__ == '__main__':
    # Raw dataset path
    dataset_path: str = '~/datasets/mafaulda'

    # Map the dataset
    filepaths, labels = map_dataset(dataset_path)

    # Perform a stratified train/test split: 90% train, 10% test
    # strictly maintaining class proportions, random_state=42
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        filepaths,
        labels,
        test_size=0.10,
        stratify=labels,
        random_state=42
    )

    # Print validation information to terminal
    print(f"Total mapped files: {len(filepaths)}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Testing samples: {len(test_paths)}")

    # Simple validation checks
    assert len(filepaths) == 1951, f"Expected 1951 files, but got {len(filepaths)}"
    assert len(train_paths) == 1755, f"Expected 1755 training samples, but got {len(train_paths)}"
    assert len(test_paths) == 196, f"Expected 196 testing samples, but got {len(test_paths)}"
    print("Verification checks passed successfully!")

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
