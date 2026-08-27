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
    Computes 46 diagnostic features for an entire batch of signals simultaneously on the GPU,
    vectorized across all 8 channels with zero per-channel sequential Python loops.
    batch_signals: shape (B, N, 8) np.ndarray
    """
    device_data = to_tensor(batch_signals)  # shape: (B, N, 8) on CUDA
    B, N, _ = device_data.shape

    # 1. Batched all-channel FFT
    if use_hann:
        window = torch.hann_window(N, periodic=False, device=device_data.device, dtype=device_data.dtype).view(1, N, 1)
        fft_all = torch.fft.rfft(device_data * window, dim=1)
        mags_all = torch.abs(fft_all) / (N / 4.0)
    else:
        fft_all = torch.fft.rfft(device_data, dim=1)
        mags_all = torch.abs(fft_all) / (N / 2.0)

    # 2. Extract rotation frequency fr from tachometer signal (channel 7)
    freqs = torch.fft.rfftfreq(N, d=1.0 / SAMPLING_RATE, device=device_data.device)
    mask = (freqs >= 5.0) & (freqs <= 120.0)
    masked_freqs = freqs[mask]
    masked_mags = mags_all[:, mask, 7]

    peak_sub_idx = torch.argmax(masked_mags, dim=1)
    f_r = masked_freqs[peak_sub_idx]  # shape: (B,)

    # 3. Vectorized spectral magnitudes of the first 7 sensors at fr, 2fr, 3fr (21 features)
    targets = torch.stack([f_r, 2.0 * f_r, 3.0 * f_r], dim=1)  # shape: (B, 3)
    df = freqs[1] - freqs[0]
    idx = targets / df
    idx_low = torch.floor(idx).long()
    idx_high = torch.clamp(idx_low + 1, max=len(freqs) - 1)
    weight = (idx - idx_low.to(device_data.dtype)).unsqueeze(1)  # shape: (B, 1, 3)

    mags_7 = mags_all[:, :, :7].permute(0, 2, 1)  # shape: (B, 7, freqs_len)
    batch_idx = torch.arange(B, device=device_data.device).view(B, 1, 1).expand(B, 7, 3)
    sensor_idx = torch.arange(7, device=device_data.device).view(1, 7, 1).expand(B, 7, 3)
    low_exp = idx_low.unsqueeze(1).expand(B, 7, 3)
    high_exp = idx_high.unsqueeze(1).expand(B, 7, 3)

    val_low = mags_7[batch_idx, sensor_idx, low_exp]
    val_high = mags_7[batch_idx, sensor_idx, high_exp]
    interp_mags = (1.0 - weight) * val_low + weight * val_high  # shape: (B, 7, 3)
    harmonic_features = interp_mags.reshape(B, 21)

    # 4. Statistical descriptors (mean, Shannon entropy, kurtosis) across all 8 signals simultaneously
    mean_vals = torch.mean(device_data, dim=1)  # shape: (B, 8)
    diffs = device_data - mean_vals.unsqueeze(1)
    var = torch.mean(diffs ** 2, dim=1)
    std = torch.sqrt(var)
    z = diffs / torch.clamp(std.unsqueeze(1), min=1e-12)
    kurt_vals = torch.mean(z ** 4, dim=1) - 3.0  # shape: (B, 8)

    # Vectorized 100-bin Shannon Entropy across (B, 8)
    if use_fixed_entropy:
        v_clamped = torch.clamp(device_data, min=-10.0, max=10.0)
        idx_ent = (v_clamped - (-10.0)) / (20.0 / 100.0)
        bin_indices = torch.clamp(torch.floor(idx_ent).long(), min=0, max=99)
    else:
        min_val = torch.amin(device_data, dim=1, keepdim=True)
        max_val = torch.amax(device_data, dim=1, keepdim=True)
        span = max_val - min_val
        span = torch.where(span == 0.0, torch.ones_like(span), span)
        idx_ent = (device_data - min_val) / (span / 100.0)
        bin_indices = torch.clamp(torch.floor(idx_ent).long(), min=0, max=99)

    batch_offset = torch.arange(B, device=device_data.device).view(B, 1, 1) * 800
    channel_offset = torch.arange(8, device=device_data.device).view(1, 1, 8) * 100
    total_offset = batch_offset + channel_offset
    flat_indices = (bin_indices + total_offset).view(-1)
    flat_counts = torch.bincount(flat_indices, minlength=B * 800)
    counts = flat_counts.view(B, 8, 100)
    probs = counts.to(device_data.dtype) / torch.sum(counts, dim=2, keepdim=True)
    entropy_vals = -torch.sum(probs * torch.log2(torch.clamp(probs, min=1e-12)), dim=2)  # shape: (B, 8)

    # Interleave statistics for each sensor: [mean(i), entropy(i), kurt(i)]
    stats_interleaved = torch.stack([mean_vals, entropy_vals, kurt_vals], dim=2).reshape(B, 24)

    # 5. Assemble all 46 diagnostic features
    feature_tensor = torch.cat([f_r.unsqueeze(1), harmonic_features, stats_interleaved], dim=1)
    return to_numpy(feature_tensor)

