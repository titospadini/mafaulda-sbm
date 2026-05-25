"""
Feature Extraction Script for MaFaulDa Dataset (Pure-Python Version)

This script extracts 46 hand-crafted diagnostic features per operational scenario file:
- 1 rotation frequency
- 21 spectral harmonic features
- 24 statistical features (mean, Shannon entropy, Fisher kurtosis)

Completely implemented in pure Python using our optimized pure_math library.
"""

import os
import sys
import time
from typing import List
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from mafaulda.data_prep import load_and_normalize
from mafaulda.pure_math import (
    rfft_mags,
    linear_interp,
    compute_mean,
    compute_shannon_entropy,
    compute_kurtosis,
    pad_to_power_of_2
)
from mafaulda.logging_utils import log

# Constants
SAMPLING_RATE = 50000  # 50 kHz
EXPECTED_FEATURES = 46


def extract_features_for_file(
    filepath: str,
    use_hann: bool = False,
    use_fixed_entropy: bool = False
) -> List[float]:
    """
    Extracts exactly 46 diagnostic features from a single CSV scenario file.
    Optimized for pure Python using decimation for Fast Fourier Transform (FFT) analysis.
    """
    # 1. Load and normalize signal matrix (shape: N, 8)
    normalized_data = load_and_normalize(filepath)
    N = len(normalized_data)
    if N == 0:
        return [0.0] * EXPECTED_FEATURES

    # Decimate signal for FFT calculations (factor of 4 reduces N from 250,000 to 62,500)
    # Effective sampling rate is 12,500 Hz, Nyquist limit 6,250 Hz (extremely precise for 360 Hz 3rd harmonic)
    DEC_FACTOR = 4
    SAMPLING_RATE_DEC = SAMPLING_RATE / DEC_FACTOR

    # 2. Extract rotation frequency fr from tachometer signal (column 7)
    tacho_dec = [normalized_data[r][7] for r in range(0, N, DEC_FACTOR)]
    mags_tacho = rfft_mags(tacho_dec, use_hann=use_hann)

    # Determine frequency bins
    n_pad = len(pad_to_power_of_2(tacho_dec))
    df = SAMPLING_RATE_DEC / n_pad

    # Restrict search for rotation speed peak to physical range [5, 120] Hz
    k_min = int(math_ceil(5.0 / df))
    k_max = int(math_floor(120.0 / df))

    best_mag = -1.0
    best_k = 1

    k_min = max(1, min(k_min, len(mags_tacho) - 1))
    k_max = max(1, min(k_max, len(mags_tacho) - 1))

    for k in range(k_min, k_max + 1):
        if mags_tacho[k] > best_mag:
            best_mag = mags_tacho[k]
            best_k = k

    f_r = best_k * df

    # Assemble feature vector
    feature_vector = [f_r]

    # Features 1-21: Spectral magnitudes of the first 7 sensor signals at fr, 2fr, 3fr
    for col_idx in range(7):
        col_signal_dec = [normalized_data[r][col_idx] for r in range(0, N, DEC_FACTOR)]
        mags_col = rfft_mags(col_signal_dec, use_hann=use_hann)

        mag_1 = linear_interp(f_r, 0.0, df, mags_col)
        mag_2 = linear_interp(2.0 * f_r, 0.0, df, mags_col)
        mag_3 = linear_interp(3.0 * f_r, 0.0, df, mags_col)

        feature_vector.extend([mag_1, mag_2, mag_3])

    # Features 22-45: Statistical features (mean, Shannon entropy, kurtosis) for all 8 signals
    for col_idx in range(8):
        col_signal = [normalized_data[r][col_idx] for r in range(N)]

        mean_val = compute_mean(col_signal)
        entropy_val = compute_shannon_entropy(col_signal, use_fixed=use_fixed_entropy)
        kurt_val = compute_kurtosis(col_signal)

        feature_vector.extend([mean_val, entropy_val, kurt_val])

    return feature_vector


def math_ceil(x: float) -> int:
    """Helper integer ceiling."""
    i = int(x)
    return i + 1 if x > i else i


def math_floor(x: float) -> int:
    """Helper integer flooring."""
    return int(x)


def process_set_parallel(
    filepaths: List[str],
    set_name: str,
    use_hann: bool = False,
    use_fixed_entropy: bool = False
) -> List[List[float]]:
    """
    Spawns concurrent worker processes to extract features in parallel using only standard library.
    """
    total_files = len(filepaths)
    log(f"\nStarting feature extraction for {set_name} set ({total_files} samples) in parallel...", level=1)

    start_time = time.time()

    # Use functools.partial to pass the additional flags
    extract_func = partial(extract_features_for_file, use_hann=use_hann, use_fixed_entropy=use_fixed_entropy)

    with ProcessPoolExecutor() as executor:
        feature_vectors = list(executor.map(extract_func, filepaths))

    elapsed_time = time.time() - start_time
    log(f"Completed {set_name} feature extraction in {elapsed_time:.2f} seconds!", level=2)

    return feature_vectors
