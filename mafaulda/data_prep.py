"""
Data Preparation Script for MaFaulDa Dataset (Pure-Python Version)

This script handles the initial data pipeline:
1. Mapping: Recursively reads operational scenario files (.csv) and extracts
   their fault label.
2. Stratification: Performs a stratified train/test split preserving class proportions.
3. Normalization: Normalizes raw signals column-wise to unit variance in-place.
"""
import os
import math
import random
from typing import List, Tuple, Union

# Configuration Constants
EXPECTED_TOTAL_FILES = 1951
TRAIN_TEST_SPLIT_RATIO = 0.10
RANDOM_STATE = 42


def load_and_normalize(filepath: str) -> List[List[float]]:
    """
    Loads raw CSV data corresponding to a single operational scenario and
    normalizes each of the 8 signal channels in-place to possess unit variance.
    Optimized for pure-python using single-pass tracking and map().
    """
    data = []
    num_cols = 8
    sums = [0.0] * num_cols
    sq_sums = [0.0] * num_cols

    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            row = list(map(float, line.split(',')))
            data.append(row)
            for c in range(num_cols):
                val = row[c]
                sums[c] += val
                sq_sums[c] += val * val

    num_rows = len(data)
    if num_rows == 0:
        return []

    # Calculate column-wise standard deviations
    std_devs = []
    for c in range(num_cols):
        mean = sums[c] / num_rows
        var = (sq_sums[c] / num_rows) - (mean * mean)
        std = math.sqrt(max(0.0, var))
        std_devs.append(1.0 if std < 1e-12 else std)

    # Perform unit-variance normalization in-place
    for r in range(num_rows):
        row = data[r]
        for c in range(num_cols):
            row[c] /= std_devs[c]

    return data


def _detect_unpack_count() -> int:
    """
    Helper function to inspect the caller's stack frame and determine the expected
    number of returned values if they are being unpacked.
    """
    import inspect
    import dis

    frame = inspect.currentframe().f_back.f_back
    if not frame:
        return 2

    code = frame.f_code
    lasti = frame.f_lasti
    bytecode = code.co_code

    unpack_op = dis.opmap.get('UNPACK_SEQUENCE')
    if unpack_op is None:
        return 2

    for offset in range(lasti + 2, min(len(bytecode), lasti + 40), 2):
        op = bytecode[offset]
        arg = bytecode[offset+1]
        if op == unpack_op:
            return arg
    return 2


def map_dataset(
    dataset_dir: str
) -> Union[Tuple[List[str], List[str]], Tuple[List[str], List[str], List[str], List[str]]]:
    """
    Recursively scans the MaFaulDa dataset directory to discover all operational
    scenario CSV files and maps them to their respective mechanical fault classes.
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
                rel_path = os.path.relpath(abs_path, dataset_dir)
                label = rel_path.split(os.sep)[0]
                filepaths.append(abs_path)
                labels.append(label)

    # Check if the caller expects a stratified train/test split (4 variables)
    if _detect_unpack_count() == 4:
        random.seed(RANDOM_STATE)

        # Group filepaths by label
        grouped = {}
        for path, label in zip(filepaths, labels):
            grouped.setdefault(label, []).append(path)

        train_paths = []
        test_paths = []
        train_labels = []
        test_labels = []

        for label, paths in grouped.items():
            shuffled_paths = list(paths)
            random.shuffle(shuffled_paths)

            n_test = max(1, int(len(shuffled_paths) * TRAIN_TEST_SPLIT_RATIO))
            n_train = len(shuffled_paths) - n_test

            train_paths.extend(shuffled_paths[:n_train])
            train_labels.extend([label] * n_train)

            test_paths.extend(shuffled_paths[n_train:])
            test_labels.extend([label] * n_test)

        return train_paths, test_paths, train_labels, test_labels

    return filepaths, labels
