"""
Pure-Python Mathematical Utilities for the SBM Pipeline

This module provides high-performance mathematical operations using only Python
inherent standard library features. It implements vector/matrix algebra,
a Cooley-Tukey iterative Radix-2 FFT, continuous linear interpolation, Weiszfeld's
geometric median algorithm, and signal metrics (Mean, Entropy, Kurtosis).
"""

import math
from typing import List, Tuple, Union

# Vector Algebra

def vec_add(v1: List[float], v2: List[float]) -> List[float]:
    """Adds two vectors coordinate-wise."""
    return [x1 + x2 for x1, x2 in zip(v1, v2)]

def vec_sub(v1: List[float], v2: List[float]) -> List[float]:
    """Subtracts v2 from v1 coordinate-wise."""
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def vec_scale(v: List[float], s: float) -> List[float]:
    """Scales a vector by a scalar factor."""
    return [x * s for x in v]

def vec_dot(v1: List[float], v2: List[float]) -> float:
    """Computes the dot product of two vectors."""
    return sum(x1 * x2 for x1, x2 in zip(v1, v2))

def vec_l1_norm(v: List[float]) -> float:
    """Computes the L1 norm (Manhattan distance) of a vector."""
    return sum(abs(x) for x in v)

def vec_l2_norm(v: List[float]) -> float:
    """Computes the L2 norm (Euclidean distance) of a vector."""
    return math.sqrt(sum(x * x for x in v))


# Matrix Algebra

def mat_transpose(m: List[List[float]]) -> List[List[float]]:
    """Transposes a 2D matrix (represented as list of lists)."""
    if not m or not m[0]:
        return []
    return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]

def mat_mul(m1: List[List[float]], m2: List[List[float]]) -> List[List[float]]:
    """Multiplies two 2D matrices."""
    r1, c1 = len(m1), len(m1[0])
    r2, c2 = len(m2), len(m2[0])
    if c1 != r2:
        raise ValueError(f"Matrix dimension mismatch: {c1} != {r2}")

    # Transpose m2 for cache-friendly row-by-row dot product
    m2_t = mat_transpose(m2)
    return [[sum(x1 * x2 for x1, x2 in zip(row1, row2)) for row2 in m2_t] for row1 in m1]

def mat_inverse_regularized(m: List[List[float]], reg: float = 1e-9) -> List[List[float]]:
    """
    Computes the inverse of a square symmetric matrix with diagonal regularization
    using Gauss-Jordan elimination with partial pivoting.

    This acts as a highly robust and fast pseudo-inverse approximation for SBM.
    """
    n = len(m)
    # Create augmented matrix [A + reg*I | I]
    aug = []
    for r in range(n):
        row = list(m[r])
        row[r] += reg
        identity = [0.0] * n
        identity[r] = 1.0
        aug.append(row + identity)

    for i in range(n):
        # Find pivot row
        pivot_row = i
        max_val = abs(aug[i][i])
        for r in range(i + 1, n):
            val = abs(aug[r][i])
            if val > max_val:
                max_val = val
                pivot_row = r

        if max_val < 1e-18:
            # Fallback to pure diagonal inversion if extremely singular
            for r in range(n):
                for c in range(n):
                    aug[r][n + c] = 1.0 / (m[r][r] + reg) if r == c else 0.0
            break

        # Swap rows
        if pivot_row != i:
            aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

        # Divide pivot row by diagonal element
        pivot_val = aug[i][i]
        for col in range(i, 2 * n):
            aug[i][col] /= pivot_val

        # Subtract pivot row from other rows
        for r in range(n):
            if r != i:
                factor = aug[r][i]
                for col in range(i, 2 * n):
                    aug[r][col] -= factor * aug[i][col]

    # Extract the inverted matrix from the augmented part
    return [row[n:] for row in aug]


# Digital Signal Processing (FFT, Windowing, Interpolation)

def hanning(n: int) -> List[float]:
    """Generates a standard Hanning window of size n."""
    factor = 2.0 * math.pi / (n - 1)
    return [0.5 - 0.5 * math.cos(i * factor) for i in range(n)]

def pad_to_power_of_2(x: List[float]) -> List[float]:
    """Pads the input signal with zeros to the next power of 2."""
    n = len(x)
    if n == 0:
        return []
    # Find next power of 2
    pow2 = 1
    while pow2 < n:
        pow2 <<= 1
    if pow2 == n:
        return x
    return x + [0.0] * (pow2 - n)

def cooley_tukey_fft(x: List[complex]) -> List[complex]:
    """
    Highly optimized iterative Cooley-Tukey Radix-2 FFT.
    Updates the input list of complex numbers in-place and returns it.
    """
    n = len(x)
    # Bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]

    # Butterfly computations
    length = 2
    while length <= n:
        half = length // 2
        angle = -2.0 * math.pi / length
        w_len = complex(math.cos(angle), math.sin(angle))
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = x[i + k]
                t = w * x[i + k + half]
                x[i + k] = u + t
                x[i + k + half] = u - t
                w *= w_len
        length *= 2
    return x

def rfft_mags(signal: List[float], use_hann: bool = False) -> List[float]:
    """
    Computes the real FFT magnitude spectrum of a real-valued signal.
    Applies Hanning windowing and coherent gain correction if use_hann is True.
    """
    n_orig = len(signal)
    if use_hann:
        win = hanning(n_orig)
        # Pad Hanning-windowed signal
        padded = pad_to_power_of_2([s * w for s, w in zip(signal, win)])
    else:
        padded = pad_to_power_of_2(signal)

    n_pad = len(padded)
    # Convert to complex numbers
    complex_signal = [complex(s, 0.0) for s in padded]
    cooley_tukey_fft(complex_signal)

    # Positive frequencies: only keep bins 0 to n_pad // 2
    num_bins = n_pad // 2 + 1

    # Scale magnitudes depending on Hanning window coherent gain
    if use_hann:
        # Coherent gain correction factor: divide by window average (0.5), hence (n_pad / 4.0)
        scale = n_pad / 4.0
    else:
        scale = n_pad / 2.0

    return [abs(complex_signal[k]) / scale for k in range(num_bins)]

def linear_interp(x: float, xp_start: float, xp_step: float, fp: List[float]) -> float:
    """
    Performs fast O(1) continuous linear interpolation for uniformly spaced x-coordinates.
    xp_start: the starting x value (first element of xp).
    xp_step: the uniform step size between elements of xp.
    fp: the corresponding continuous y values.
    """
    n = len(fp)
    if n == 0:
        return 0.0
    if n == 1:
        return fp[0]

    # Calculate index position
    idx = (x - xp_start) / xp_step
    if idx <= 0:
        return fp[0]
    if idx >= n - 1:
        return fp[-1]

    idx_low = int(idx)
    idx_high = idx_low + 1
    t = idx - idx_low

    return fp[idx_low] * (1.0 - t) + fp[idx_high] * t


# Statistical Operations

def compute_mean(values: List[float]) -> float:
    """Computes the arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)

def compute_shannon_entropy(values: List[float], use_fixed: bool = False) -> float:
    """
    Computes Shannon entropy (base 2) using a 100-bin histogram.
    If use_fixed is True, the bins partition the locked range [-10.0, 10.0].
    """
    n = len(values)
    if n == 0:
        return 0.0

    if use_fixed:
        v_min, v_max = -10.0, 10.0
    else:
        v_min = min(values)
        v_max = max(values)

    if v_max == v_min:
        return 0.0

    num_bins = 100
    counts = [0] * num_bins
    bin_width = (v_max - v_min) / num_bins

    for val in values:
        if val < v_min:
            idx = 0
        elif val >= v_max:
            idx = num_bins - 1
        else:
            idx = int((val - v_min) / bin_width)
            if idx >= num_bins:
                idx = num_bins - 1
        counts[idx] += 1

    entropy_val = 0.0
    for count in counts:
        if count > 0:
            p = count / n
            entropy_val -= p * math.log2(p)

    return entropy_val

def compute_kurtosis(values: List[float]) -> float:
    """Computes the Fisher excess kurtosis of a distribution (Gaussian distribution = 0.0)."""
    n = len(values)
    if n < 4:
        return 0.0

    mean_val = sum(values) / n
    var_sum = 0.0
    m4_sum = 0.0

    for val in values:
        diff = val - mean_val
        diff_sq = diff * diff
        var_sum += diff_sq
        m4_sum += diff_sq * diff_sq

    variance = var_sum / n
    if variance < 1e-12:
        return 0.0

    m4 = m4_sum / n
    return (m4 / (variance * variance)) - 3.0


# SBM Dictionary Spatial Centroid (Weiszfeld)

def compute_geometric_median(X: List[List[float]], max_iter: int = 500, tol: float = 1e-6) -> List[float]:
    """
    Computes the spatial geometric median of a dataset X using Weiszfeld's algorithm.
    """
    if not X:
        raise ValueError("Cannot compute geometric median of an empty set.")
    if len(X) == 1:
        return list(X[0])

    num_features = len(X[0])
    # Initialize with arithmetic mean
    y = [sum(X[r][c] for r in range(len(X))) / len(X) for c in range(num_features)]

    for _ in range(max_iter):
        num_sum = [0.0] * num_features
        den_sum = 0.0

        for row in X:
            # Inline L2 distance calculation to avoid list creation and function overhead
            dist_sq = 0.0
            for c in range(num_features):
                diff_c = row[c] - y[c]
                dist_sq += diff_c * diff_c
            dist = math.sqrt(dist_sq)

            weight = 1.0 / (dist if dist > 1e-12 else 1e-12)
            for c in range(num_features):
                num_sum[c] += row[c] * weight
            den_sum += weight

        new_y = [val / den_sum for val in num_sum]

        # Check convergence
        diff_norm_sq = 0.0
        for c in range(num_features):
            diff_c = new_y[c] - y[c]
            diff_norm_sq += diff_c * diff_c
        if math.sqrt(diff_norm_sq) < tol:
            return new_y
        y = new_y

    return y
