"""
Utility Functions for BEAM-Net
=================================
Common utilities: reproducibility, device management, calibration metrics.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Get compute device with fallback."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def compute_ece(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).
    
    Measures how well predicted probabilities match actual accuracy.
    A perfectly calibrated model has ECE = 0.
    
    Critical metric for BEAM-Net: the Dirichlet attention (§3.3) claims
    to produce calibrated uncertainty, so ECE should be lower than
    deterministic baselines.
    
    Parameters
    ----------
    probs : torch.Tensor, shape (N, C)
        Predicted class probabilities (after softmax).
    targets : torch.Tensor, shape (N,)
        True class labels.
    n_bins : int
        Number of confidence bins.
    
    Returns
    -------
    ece : float
        Expected Calibration Error in [0, 1].
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(targets)
    
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].float().mean()
            ece += (avg_confidence - avg_accuracy).abs() * prop_in_bin
    
    return ece.item()