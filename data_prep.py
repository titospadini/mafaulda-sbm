"""
Data Preparation Script for MaFaulDa Dataset

This script handles the initial data pipeline for reproducing the scientific
paper
on rotating-machine fault diagnosis using the MaFaulDa database.

Key steps included:
1. Mapping: Recursively reads operational scenario files (.csv) and extracts
   their fault label.
2. Stratification: Performs a train/test split ensuring the original fault class
   proportions.
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
    Loads raw CSV data corresponding to a single operational scenario and
    normalizes
    each of the 8 signal channels to possess unit variance.

    Pedagogical Context:
        The MaFaulDa database CSV files contain 8 column channels sampled at 50
        kHz:
          - Columns 0-5: 6 Accelerometers (measuring vibration along different
            axes/locations)
          - Column 6: 1 Microphone (capturing acoustic emissions and ambient
            noise)
          - Column 7: 1 Tachometer (outputting a pulse train for rotational
            speed estimation)

        Dividing each channel by its standard deviation scales all physical
        sensors to
        the exact same variance baseline. This prevents channels with large
        numeric scale
        variations (like the tachometer pulses) from dominating the subsequent
        statistical
        and spectral feature extractions.

    Parameters:
        filepath (str): The absolute path to the CSV file representing the
        operational state.

    Returns:
        np.ndarray: A normalized signal matrix of shape (N, 8) with unit
        variance columns.
    """
    # Load CSV data. It is comma-separated and has no header.
    data = np.loadtxt(filepath, delimiter=',')

    # Calculate standard deviation for each of the 8 columns (axis=0)
    std_devs = np.std(data, axis=0)

    # Avoid division by zero for any constant signals (if any exist)
    # np.where checks the condition (std_devs == 0.0). If true, it replaces the
    # value with 1.0,
    # otherwise it keeps the original standard deviation.
    std_devs = np.where(std_devs == 0.0, 1.0, std_devs)

    # Perform unit variance normalization
    normalized_data = data / std_devs

    return normalized_data


def map_dataset(dataset_dir: str) -> Tuple[List[str], List[str]]:
    """
    Recursively scans the MaFaulDa dataset directory to discover all operational
    scenario CSV files
    and maps them to their respective mechanical fault classes.

    Pedagogical Context:
        The MaFaulDa database organizes fault categories by directory naming
        conventions:
          - `normal`: Normal system operations (no faults).
          - `imbalance`: Rotational mass imbalance faults.
          - `horizontal-misalignment`: Shaft horizontal alignment deviations.
          - `vertical-misalignment`: Shaft vertical alignment deviations.
          - `overhang`: Bearing fault located on the overhang side of the rotor.
          - `underhang`: Bearing fault located on the underhang side of the
            rotor.

        This function identifies the correct label by taking the directory name
        immediately
        nested under the dataset root directory (the top-level relative path
        segment).

    Parameters:
        dataset_dir (str): Absolute or relative path to the raw MaFaulDa
        database directory.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing:
          - filepaths (List[str]): List of absolute paths to all CSV files.
          - labels (List[str]): Matching string labels denoting the fault class
            for each file.

    Raises:
        FileNotFoundError: If the specified directory does not exist or is not a
        directory.
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
                # The top-level folder name right under mafaulda/ represents the
                # fault class (e.g., 'normal', 'imbalance').
                # We extract it by taking the first element of the split
                # relative path.
                label = rel_path.split(os.sep)[0]
                filepaths.append(abs_path)
                labels.append(label)

    return filepaths, labels
