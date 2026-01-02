"""Device detection utilities for PyTorch.

Automatically detects the best available device:
1. MPS (Apple Silicon GPU) if available
2. CUDA (NVIDIA GPU) if available
3. CPU as fallback
"""

import torch


def get_device():
    """Get the best available device for PyTorch.
    
    Returns:
        torch.device: The best available device (mps, cuda, or cpu)
    """
    if torch.backends.mps.is_available():
        # Apple Silicon GPU
        return torch.device("mps")
    elif torch.cuda.is_available():
        # NVIDIA GPU
        return torch.device("cuda")
    else:
        # CPU fallback
        return torch.device("cpu")


def get_device_name():
    """Get the name of the current device.
    
    Returns:
        str: Device name ('mps', 'cuda', or 'cpu')
    """
    device = get_device()
    return device.type


def print_device_info():
    """Print information about the current device."""
    device = get_device()
    device_name = device.type
    
    print(f"Using device: {device_name}")
    
    if device_name == "mps":
        print("  - Apple Silicon GPU (MPS) detected")
        print("  - Good for inference and small-scale training")
    elif device_name == "cuda":
        print(f"  - NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print("  - Optimal for all tasks")
    else:
        print("  - Using CPU (slower)")
        print("  - Consider using a GPU for better performance")


# Print device info when module is imported
if __name__ != "__main__":
    device = get_device()
    print(f"[Device] Using: {device.type}")
