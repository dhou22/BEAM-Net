"""
Energy Profiler for BEAM-Net (§5)
====================================
Implements the theoretical energy consumption model (Eq. 25):

  E_BEAM = d · E_spike           (Encoding)
         + (d + ρdN) · E_spike   (Coincidence)
         + (1.1N² + L·N) · E_spike  (Selection + Iteration)

where:
  E_spike = 23.6 pJ on Loihi 2 (Davies et al., 2021)
  ρ ≈ 0.05 is the sparsity factor
  L ≤ 5 is the number of bidirectional iterations

This provides the 22–30× energy reduction claim vs GPU attention (Table 1).

Note: These are theoretical projections for neuromorphic deployment.
Actual measurements require physical Loihi 2 hardware.
"""

import numpy as np
from typing import Dict


# Energy constants from literature
E_SPIKE_LOIHI2_PJ = 23.6       # picojoules per spike on Loihi 2 (Davies et al., 2021)
E_GPU_ATTENTION_NJ = 150.0      # nanojoules per attention op on A100 (estimated)
E_SPIKFORMER_NJ = 25.0          # SpikFormer estimate
E_BIOLOGICAL_NJ = 0.77          # Cortical attention (Lennie, 2003)


def compute_energy_estimate(cfg: dict) -> float:
    """
    Compute theoretical energy consumption per inference (Eq. 25).
    
    Parameters
    ----------
    cfg : dict
        Experiment configuration.
    
    Returns
    -------
    energy_nJ : float
        Energy per inference in nanojoules.
    """
    d = cfg["attention"]["n_patterns"]
    N = cfg["attention"]["n_patterns"]
    rho = 0.05
    L = cfg["inference"]["max_iterations"]
    E_spike = E_SPIKE_LOIHI2_PJ
    
    e_encoding = d * E_spike
    e_coincidence = (d + rho * d * N) * E_spike
    e_selection = (1.1 * N**2 + L * N) * E_spike
    
    total_pJ = e_encoding + e_coincidence + e_selection
    total_nJ = total_pJ / 1000.0
    
    return total_nJ


def compute_energy_comparison(cfg: dict) -> Dict[str, float]:
    """
    Full comparative energy analysis (Table 1 from paper).
    """
    beam_nJ = compute_energy_estimate(cfg)
    
    return {
        "GPU_A100_nJ": E_GPU_ATTENTION_NJ,
        "SpikFormer_nJ": E_SPIKFORMER_NJ,
        "BEAM_Net_Loihi2_nJ": beam_nJ,
        "Biological_cortex_nJ": E_BIOLOGICAL_NJ,
        "reduction_vs_GPU": E_GPU_ATTENTION_NJ / beam_nJ,
        "reduction_vs_SpikFormer": E_SPIKFORMER_NJ / beam_nJ,
    }


def compute_scaling_analysis(
    d: int = 512,
    N_range: list = None,
    rho: float = 0.05,
    L: int = 5,
) -> Dict[str, list]:
    """
    Compute energy scaling as pattern count N grows (§5.3).
    """
    if N_range is None:
        N_range = [100, 500, 1000, 2000, 5000, 10000]
    
    beam_energies = []
    gpu_energies = []
    
    for N in N_range:
        gpu_ops = N * d
        gpu_energy = gpu_ops * 0.3
        gpu_energies.append(gpu_energy / 1000)
        
        beam_ops = rho * N * d + L * N + d
        beam_energy = beam_ops * E_SPIKE_LOIHI2_PJ
        beam_energies.append(beam_energy / 1000)
    
    return {
        "N_values": N_range,
        "GPU_energy_nJ": gpu_energies,
        "BEAM_energy_nJ": beam_energies,
    }