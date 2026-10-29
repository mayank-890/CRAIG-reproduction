"""Centralized reproducibility and hardware/device utilities."""

import os
import random
import logging
from typing import Optional, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds across Python random, NumPy, PyTorch CPU, MPS, and CUDA.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    if torch.backends.mps.is_available():
        # MPS seed is set by torch.manual_seed
        pass
        
    logger.debug("Centralized seed set to %d", seed)


def get_device(override_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Detect available hardware accelerator or apply manual override.
    
    Priority order when 'auto': CUDA -> MPS -> CPU.
    
    Args:
        override_device: 'auto', 'cuda', 'mps', 'cpu', or torch.device.
        
    Returns:
        torch.device instance.
    """
    if override_device is not None and str(override_device).lower() != "auto":
        dev_str = str(override_device).lower()
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU/MPS.")
        elif dev_str == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available. Falling back to CPU.")
        else:
            return torch.device(dev_str)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
        
    return device


def get_memory_info_mb(device: torch.device) -> Optional[float]:
    """Measure peak or allocated memory in MB where available."""
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    # CPU/MPS memory tracking:
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS ru_maxrss is in bytes, on Linux in KB
        import sys
        if sys.platform == "darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        else:
            return rusage.ru_maxrss / 1024
    except Exception:
        return None
