"""
Feature Extraction Script for MaFaulDa Dataset

This script handles Step 2 of our Rotating-Machine Fault Diagnosis pipeline:
1. Re-constructs the identical stratified train/test split from data_prep.py.
2. Extracts exactly 46 hand-crafted features per scenario (1 rotation frequency,
   21 spectral features, and 24 statistical features) from the raw sensor
   signals.
3. Leverages parallel processing (ProcessPoolExecutor) for fast, concurrent
   extraction.
4. Exports the final NumPy matrices (X_train, X_test, y_train, y_test) into the
   data/ directory.
"""

import os
import sys
import time

from typing import (
    List,
    Tuple,
)

from functools import partial
import numpy as np

from scipy.stats import (
    kurtosis,
    entropy,
)

from concurrent.futures import ProcessPoolExecutor

# Import mapping, normalization, and configuration from data_prep
# Adding current directory to sys.path to ensure correct importing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_prep import (
    map_dataset,
    load_and_normalize,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_STATE,
)

# Constants
SAMPLING_RATE = 50000  # 50 kHz
EXPECTED_FEATURES = 46


def extract_features_for_file(filepath: str, use_hann: bool = False, use_fixed_entropy: bool = False) -> np.ndarray:
    """
    Extracts exactly 46 diagnostic features from a single CSV time-series
    scenario file.

    Pedagogical Context:
        To perform rotating-machine fault diagnosis, raw vibration, acoustic,
        and rotation
        time-series must be compressed into representative statistical and
        spectral metrics.
        This function implements three layers of feature engineering:

        1. Rotation Frequency (fr) Estimation:
           - Computes the Discrete Fourier Transform (DFT) of the tachometer
             pulse train (channel 7).
           - Identifies the dominant frequency peak within the physically
             constrained range [5, 120] Hz.
             This range filters out high-frequency sensor noise and
             low-frequency DC baseline drift.

        2. Spectral Magnitude Features (21 features):
           - Extracts exact frequency magnitudes for the first 7 physical
             sensors at the fundamental
             frequency (fr), its second harmonic (2fr), and third harmonic
             (3fr).
           - Because these harmonic frequencies may not align exactly with
             discrete FFT bin centers,
             linear interpolation (np.interp) is used to calculate high-fidelity
             continuous spectral peaks.
           - Total features = 7 sensors * 3 harmonics = 21 features.

        3. Statistical Signal Features (24 features):
           - Computes three core descriptors across all 8 sensor channels (8 * 3
             = 24 features):
             a. Mean: Represents physical baseline shifts or offsets.
             b. Shannon Entropy: Quantifies structural complexity and noise
             levels, calculated via
                probability bin counts of a 100-bin histogram.
             c. Kurtosis: Measures the "peakedness" of the signal distribution,
             identifying mechanical
                impact impulses (sudden shocks/clicks caused by gear tooth
                cracks or bearing spalls).

        4. DSP Enhancements (Optional Flags):
           - Hanning Window (`use_hann`): Dampens the start and end of the
             signal to zero to prevent
             spectral leakage. It applies coherent gain correction (scale factor
             2.0) to maintain amplitude fidelity.
           - Fixed Entropy Range (`use_fixed_entropy`): Locks the histogram
             calculation range to `(-10.0, 10.0)`.
             This resolves a critical bug where dynamic histogram bin scaling
             causes a sudden mechanical shock impulse
             to artificially alter the overall entropy scale.

    Parameters:
        filepath (str): The absolute path to the CSV file representing the
        operational state.
        use_hann (bool): Whether to apply a Hanning window and coherent gain
        correction to FFT calculations.
        use_fixed_entropy (bool): Whether to lock the Shannon entropy histogram
        range to (-10.0, 10.0).

    Returns:
        np.ndarray: A 46-dimensional float64 feature vector of shape (46,).
    """
    # 1. Load and normalize signal matrix (shape: N, 8)
    # This divides each sensor channel by its standard deviation for unit
    # variance.
    normalized_data = load_and_normalize(filepath)
    N = len(normalized_data)

    # 2. Extract rotation frequency fr from tachometer signal (column 7)
    tacho = normalized_data[:, 7]
    if use_hann:
        window = np.hanning(N)
        fft_tacho = np.fft.rfft(tacho * window)
        # Coherent gain correction: divide by window average (0.5), yielding
        # denominator (N / 4.0)
        mags_tacho = np.abs(fft_tacho) / (N / 4.0)
    else:
        fft_tacho = np.fft.rfft(tacho)
        mags_tacho = np.abs(fft_tacho) / (N / 2.0)

    freqs = np.fft.rfftfreq(N, d=1/SAMPLING_RATE)

    # Restrict search for rotation speed peak to physical range [5, 120] Hz to
    # avoid high-frequency noise
    mask = (freqs >= 5.0) & (freqs <= 120.0)
    masked_freqs = freqs[mask]
    masked_mags = mags_tacho[mask]

    if len(masked_mags) > 0:
        peak_sub_idx = np.argmax(masked_mags)
        f_r = masked_freqs[peak_sub_idx]
    else:
        # Fallback to global peak (excluding DC component)
        peak_idx = np.argmax(mags_tacho[1:]) + 1
        f_r = freqs[peak_idx]

    # Assemble feature vector
    feature_vector = []

    # Feature 0: Rotation frequency fr
    feature_vector.append(f_r)

    # Features 1-21: Spectral magnitudes of the first 7 sensor signals at fr,
    # 2fr, 3fr
    # We normalize DFT magnitudes to make them independent of sample length.
    # We use linear interpolation (np.interp) to extract magnitudes at exact
    # continuous frequencies,
    # preventing discretization/bin-quantization errors.
    for col_idx in range(7):
        col_signal = normalized_data[:, col_idx]
        if use_hann:
            window = np.hanning(N)
            fft_col = np.fft.rfft(col_signal * window)
            mags_col = np.abs(fft_col) / (N / 4.0)
        else:
            fft_col = np.fft.rfft(col_signal)
            mags_col = np.abs(fft_col) / (N / 2.0)

        mag_1 = np.interp(f_r, freqs, mags_col)
        mag_2 = np.interp(2.0 * f_r, freqs, mags_col)
        mag_3 = np.interp(3.0 * f_r, freqs, mags_col)

        feature_vector.append(mag_1)
        feature_vector.append(mag_2)
        feature_vector.append(mag_3)

    # Features 22-45: Statistical features (mean, Shannon entropy, kurtosis) for
    # all 8 signals
    for col_idx in range(8):
        col_signal = normalized_data[:, col_idx]

        # Mean
        mean_val = np.mean(col_signal)

        # Shannon Entropy (estimated using 100-bin histogram)
        if use_fixed_entropy:
            counts, _ = np.histogram(col_signal, bins=100, range=(-10.0, 10.0))
        else:
            counts, _ = np.histogram(col_signal, bins=100)

        probs = counts / np.sum(counts)
        # Use scipy.stats.entropy which computes Shannon entropy (base=2 is
        # standard)
        entropy_val = entropy(probs, base=2)

        # Kurtosis (Fisher definition: normal distribution = 0.0)
        kurt_val = kurtosis(col_signal, fisher=True)

        feature_vector.append(mean_val)
        feature_vector.append(entropy_val)
        feature_vector.append(kurt_val)

    # Return as float64 array
    return np.array(feature_vector, dtype=np.float64)


def process_set_parallel(filepaths: List[str], set_name: str, use_hann: bool = False, use_fixed_entropy: bool = False) -> np.ndarray:
    """
    Spawns concurrent worker processes to extract the 46 hand-crafted features
    in parallel across
    all CSV files in a given dataset split.

    Pedagogical Context:
        Processing 1,951 files sequentially would require significant
        computation time due to I/O overhead
        and CPU-bound Fast Fourier Transforms (FFT). By leveraging a
        ProcessPoolExecutor, this function
        distributes the workload across all available CPU cores, achieving
        optimal multi-core performance.
        It uses `functools.partial` to cleanly package the runtime DSP flags for
        parallel execution.

    Parameters:
        filepaths (List[str]): List of absolute file paths pointing to the raw
        CSV files.
        set_name (str): Human-readable name of the dataset split (e.g.,
        'Training' or 'Testing') for log outputs.
        use_hann (bool): Whether to apply a Hanning window and coherent gain
        correction to FFT.
        use_fixed_entropy (bool): Whether to use a fixed histogram range (-10.0,
        10.0) for Shannon entropy.

    Returns:
        np.ndarray: A 2D feature matrix of shape (num_samples, 46) containing
        the extracted features.
    """
    total_files = len(filepaths)
    print(f"\nStarting feature extraction for {set_name} set ({total_files} samples) in parallel...")

    start_time = time.time()

    # Use functools.partial to pass the additional flags to mapped parallel
    # process execution
    extract_func = partial(extract_features_for_file, use_hann=use_hann, use_fixed_entropy=use_fixed_entropy)

    # Process files concurrently using all available CPU cores
    with ProcessPoolExecutor() as executor:
        feature_vectors = list(executor.map(extract_func, filepaths))

    elapsed_time = time.time() - start_time
    print(f"Completed {set_name} feature extraction in {elapsed_time:.2f} seconds!")

    # Convert list of vectors to a 2D numpy array
    feature_matrix = np.vstack(feature_vectors)
    return feature_matrix
