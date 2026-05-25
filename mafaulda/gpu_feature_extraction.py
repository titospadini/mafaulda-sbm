import numpy as np
import torch
from mafaulda.gpu_utils import to_tensor, to_numpy

SAMPLING_RATE = 50000  # 50 kHz


def interp_fft_mags_batch(x: torch.Tensor, freqs: torch.Tensor, mags_col: torch.Tensor) -> torch.Tensor:
    """Linearly interpolates FFT magnitudes at continuous frequency targets on GPU for a batch.
    x shape: (B,)
    freqs shape: (freqs_len,)
    mags_col shape: (B, freqs_len)
    """
    df = freqs[1] - freqs[0]
    idx = x / df
    idx_low = torch.floor(idx).long()
    idx_high = torch.clamp(idx_low + 1, max=len(freqs) - 1)
    weight = idx - idx_low.to(x.dtype)

    batch_idx = torch.arange(len(x), device=x.device)
    val_low = mags_col[batch_idx, idx_low]
    val_high = mags_col[batch_idx, idx_high]
    return (1.0 - weight) * val_low + weight * val_high


def kurtosis_torch_batch(v: torch.Tensor, fisher: bool = True) -> torch.Tensor:
    """Computes the fourth standardized moment (Kurtosis) of a batch tensor (B, N) on GPU."""
    mean = torch.mean(v, dim=1, keepdim=True)
    diffs = v - mean
    var = torch.mean(diffs**2, dim=1, keepdim=True)
    std = torch.sqrt(var)
    z = diffs / torch.clamp(std, min=1e-12)
    kurt = torch.mean(z**4, dim=1)
    if fisher:
        kurt = kurt - 3.0
    return kurt


def extract_features_batch_gpu(
    batch_signals: np.ndarray,
    use_hann: bool = False,
    use_fixed_entropy: bool = False
) -> np.ndarray:
    """
    Computes 46 diagnostic features for an entire batch of signals simultaneously on the GPU.
    batch_signals: shape (B, N, 8) np.ndarray
    """
    # 1. Push batch to CUDA
    device_data = to_tensor(batch_signals)  # shape: (B, N, 8) on CUDA
    B, N, _ = device_data.shape

    # 2. Extract rotation frequency fr from tachometer signal (column 7)
    tacho = device_data[:, :, 7]  # shape: (B, N)
    if use_hann:
        window = torch.hann_window(N, periodic=False, device=device_data.device, dtype=device_data.dtype)
        fft_tacho = torch.fft.rfft(tacho * window, dim=1)
        mags_tacho = torch.abs(fft_tacho) / (N / 4.0)
    else:
        fft_tacho = torch.fft.rfft(tacho, dim=1)
        mags_tacho = torch.abs(fft_tacho) / (N / 2.0)

    freqs = torch.fft.rfftfreq(N, d=1.0/SAMPLING_RATE, device=device_data.device)

    # Restrict search for rotation speed peak to physical range [5, 120] Hz
    mask = (freqs >= 5.0) & (freqs <= 120.0)
    masked_freqs = freqs[mask]
    masked_mags = mags_tacho[:, mask]

    peak_sub_idx = torch.argmax(masked_mags, dim=1)
    f_r = masked_freqs[peak_sub_idx]  # shape: (B,)

    feature_vector = [f_r]

    # 3. Spectral magnitudes of the first 7 sensor signals at fr, 2fr, 3fr
    for col_idx in range(7):
        col_signal = device_data[:, :, col_idx]
        if use_hann:
            window = torch.hann_window(N, periodic=False, device=device_data.device, dtype=device_data.dtype)
            fft_col = torch.fft.rfft(col_signal * window, dim=1)
            mags_col = torch.abs(fft_col) / (N / 4.0)
        else:
            fft_col = torch.fft.rfft(col_signal, dim=1)
            mags_col = torch.abs(fft_col) / (N / 2.0)

        mag_1 = interp_fft_mags_batch(f_r, freqs, mags_col)
        mag_2 = interp_fft_mags_batch(2.0 * f_r, freqs, mags_col)
        mag_3 = interp_fft_mags_batch(3.0 * f_r, freqs, mags_col)

        feature_vector.append(mag_1)
        feature_vector.append(mag_2)
        feature_vector.append(mag_3)

    # 4. Statistical features (mean, Shannon entropy, kurtosis) for all 8 signals
    for col_idx in range(8):
        col_signal = device_data[:, :, col_idx]

        # Mean
        mean_val = torch.mean(col_signal, dim=1)

        # Shannon Entropy (estimated using 100-bin histogram on GPU via bincount offset trick)
        if use_fixed_entropy:
            v_clamped = torch.clamp(col_signal, min=-10.0, max=10.0)
            idx = (v_clamped - (-10.0)) / (20.0 / 100.0)
            bin_indices = torch.clamp(torch.floor(idx).long(), min=0, max=99)
        else:
            min_val = torch.min(col_signal, dim=1, keepdim=True)[0]
            max_val = torch.max(col_signal, dim=1, keepdim=True)[0]
            span = max_val - min_val
            span = torch.where(span == 0.0, torch.ones_like(span), span)
            idx = (col_signal - min_val) / (span / 100.0)
            bin_indices = torch.clamp(torch.floor(idx).long(), min=0, max=99)

        # Offset trick for batch bincount
        offset = torch.arange(B, device=device_data.device).unsqueeze(1) * 100
        flat_indices = (bin_indices + offset).view(-1)
        flat_counts = torch.bincount(flat_indices, minlength=B * 100)
        counts = flat_counts.view(B, 100)

        probs = counts.to(device_data.dtype) / torch.sum(counts, dim=1, keepdim=True)
        entropy_val = -torch.sum(probs * torch.log2(torch.clamp(probs, min=1e-12)), dim=1)

        # Kurtosis
        kurt_val = kurtosis_torch_batch(col_signal, fisher=True)

        feature_vector.append(mean_val)
        feature_vector.append(entropy_val)
        feature_vector.append(kurt_val)

    # 5. Convert list of GPU tensors of shape (B,) back to single NumPy array of shape (B, 46)
    feature_tensor = torch.stack(feature_vector, dim=1)
    return to_numpy(feature_tensor)
