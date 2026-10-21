"""
Reproducibility utilities.

Provides seed control and deterministic configuration helpers.
"""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and (if available) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed yet; will be added in later phases


def get_device(preferred: str = "cpu") -> str:
    """Return a valid device string. Falls back to CPU if preferred is unavailable."""
    if preferred == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    elif preferred == "mps":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
    return "cpu"
