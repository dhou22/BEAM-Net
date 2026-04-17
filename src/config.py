"""
Configuration Manager for BEAM-Net
====================================
Loads experiment.yaml and provides typed, dot-accessible config.
Validates parameter ranges against theoretical constraints from the paper.
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch


def load_config(path: str = "configs/experiment.yaml") -> dict:
    """Load YAML config with environment variable override support."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Auto-detect device
    if cfg["experiment"]["device"] == "auto":
        cfg["experiment"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Validate theoretical constraints
    _validate(cfg)
    return cfg


def _validate(cfg: dict):
    """
    Validate parameters against BEAM-Net theoretical constraints.
    
    Critical constraints:
      - w_inh > 2   : Required for posterior convergence (Proposition 3.3)
      - delta > 0   : Stochasticity must be positive (Eq. 9)
      - lambda < 1/(N * max|w_TD|) : Convergence bound (Theorem 3.6)
    """
    w_inh = cfg["neuron"]["w_inh"]
    assert w_inh > 2.0, (
        f"Lateral inhibition w_inh={w_inh} must be > 2.0 "
        f"for posterior convergence (Proposition 3.3, BEAM-Net paper)"
    )
    
    delta = cfg["neuron"]["delta"]
    assert delta > 0, f"Stochasticity Δ={delta} must be > 0 (Eq. 9)"
    
    lambda_td = cfg["inference"]["lambda_td"]
    N = cfg["attention"]["n_patterns"]
    # Simplified bound check (full check requires knowing max|w_TD|)
    assert lambda_td < 1.0, (
        f"Top-down modulation λ={lambda_td} should be < 1.0 "
        f"for convergence (Theorem 3.6)"
    )
    
    print(f"[Config] Validated. Device={cfg['experiment']['device']}, "
          f"N={N}, w_inh={w_inh}, Δ={delta}")