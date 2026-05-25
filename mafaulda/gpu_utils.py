import numpy as np

_torch_available = False
try:
    import torch
    if torch.cuda.is_available():
        _torch_available = True
except ImportError:
    pass

def is_gpu_available() -> bool:
    """Returns True if PyTorch is installed and CUDA GPU is available."""
    return _torch_available

def to_tensor(arr: np.ndarray) -> "torch.Tensor":
    """Converts a NumPy array into a PyTorch CUDA tensor."""
    import torch
    return torch.from_numpy(arr).cuda()

def to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    """Converts a PyTorch CUDA tensor back to a NumPy array."""
    return tensor.cpu().numpy()
