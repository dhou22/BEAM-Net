"""
Component 2: Bayesian Spiking Neuron (BEAM-Net §3.2)
======================================================
Implements the stochastic threshold LIF neuron that performs
posterior sampling over latent causes.

Key equations:
  - Membrane dynamics (Eq. 8): τ_m dV_j/dt = -V_j + S_j(ξ) - Σ w_inh α(t-t_k)
  - Firing probability (Eq. 9): P(spike|V) = σ((V - θ) / Δ)
  - Adaptive threshold (Eq. 10): θ_j(t) = θ_0 + γ ∫ exp(-(t-s)/τ_θ) n_j(s) ds
  - Posterior convergence (Prop. 3.3): r_j → P(cause_j | ξ)

Biological basis:
  Stein model of cortical neuron variability (Stein, 1967).
  The stochastic threshold captures trial-to-trial variability observed
  in cortical neurons, where identical stimuli produce variable spike times.

Connection to W-TCRL (Fois & Girau, 2023):
  W-TCRL uses deterministic LIF with CuBa synapses (§2.1.1).
  BEAM-Net generalizes this to stochastic LIF, adding uncertainty
  quantification while preserving the temporal coding benefits.
"""

import torch
import torch.nn as nn
from typing import Tuple


class BayesianLIFNeuron(nn.Module):
    """
    Bayesian Leaky Integrate-and-Fire neuron with stochastic threshold.
    
    Unlike deterministic LIF, this neuron:
      1. Fires probabilistically (Eq. 9) — implements posterior sampling
      2. Adapts its threshold based on firing history (Eq. 10) — 
         confident neurons raise threshold, uncertain ones stay sensitive
      3. Converges to posterior P(cause_j | ξ) in population activity (Prop. 3.3)
    
    Parameters
    ----------
    N : int
        Number of neurons (= number of stored patterns).
    tau_m : float
        Membrane time constant (ms). Controls integration speed.
    tau_theta : float
        Threshold adaptation recovery time constant (ms).
    theta_0 : float
        Base firing threshold.
    delta : float
        Stochasticity parameter (Δ). Controls uncertainty width.
        Δ = 1/β maps to inverse temperature in Hopfield energy.
    gamma : float
        Threshold adaptation strength.
    w_inh : float
        Lateral inhibition weight. Must be > 2 for posterior convergence.
    tau_syn : float
        Synaptic time constant for inhibitory PSC kernel α(t).
    dt : float
        Simulation timestep (ms).
    """
    
    def __init__(
        self,
        N: int,
        tau_m: float = 10.0,
        tau_theta: float = 50.0,
        theta_0: float = 1.0,
        delta: float = 0.5,
        gamma: float = 0.1,
        w_inh: float = 2.5,
        tau_syn: float = 5.0,
        dt: float = 0.5,
    ):
        super().__init__()
        self.N = N
        self.tau_m = tau_m
        self.tau_theta = tau_theta
        self.theta_0 = theta_0
        self.delta = delta
        self.gamma = gamma
        self.w_inh = w_inh
        self.tau_syn = tau_syn
        self.dt = dt
        
        # Precomputed decay constants (Euler discretization)
        self.alpha_m = dt / tau_m
        self.alpha_theta = dt / tau_theta
        self.alpha_syn = dt / tau_syn
    
    def init_state(self, batch_size: int, device: torch.device) -> dict:
        """Initialize all neuron state variables to resting values."""
        return {
            "V": torch.zeros(batch_size, self.N, device=device),
            "theta": torch.full((batch_size, self.N), self.theta_0, device=device),
            "I_inh": torch.zeros(batch_size, self.N, device=device),
            "spike_count": torch.zeros(batch_size, self.N, device=device),
        }
    
    def forward(
        self,
        S: torch.Tensor,
        state: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single timestep update of Bayesian LIF neurons.
        
        Parameters
        ----------
        S : torch.Tensor, shape (batch, N)
            Similarity scores S_j(ξ) from coincidence detection.
        state : dict
            Current neuron state (V, theta, I_inh, spike_count).
        
        Returns
        -------
        spikes : torch.Tensor, shape (batch, N)
            Binary spike output (0 or 1).
        state : dict
            Updated neuron state.
        """
        V = state["V"]
        theta = state["theta"]
        I_inh = state["I_inh"]
        
        # ---- Eq. 8: Membrane dynamics ----
        # τ_m dV/dt = -V + S_j(ξ) - I_inh
        dV = self.alpha_m * (-V + S - I_inh)
        V_new = V + dV
        
        # ---- Eq. 9: Stochastic firing ----
        # P(spike | V) = σ((V - θ) / Δ)
        firing_prob = torch.sigmoid((V_new - theta) / self.delta)
        
        if self.training:
            spikes = torch.bernoulli(firing_prob)
        else:
            spikes = (firing_prob > 0.5).float()
        
        # ---- Lateral inhibition ----
        total_spikes = spikes.sum(dim=-1, keepdim=True)
        other_spikes = total_spikes - spikes
        I_inh_new = I_inh * (1.0 - self.alpha_syn) + self.w_inh * other_spikes
        
        # ---- Eq. 10: Adaptive threshold ----
        theta_new = theta + self.alpha_theta * (self.theta_0 - theta) + self.gamma * spikes
        
        # ---- Reset membrane potential after spike ----
        V_reset = V_new * (1.0 - spikes)
        
        new_state = {
            "V": V_reset,
            "theta": theta_new,
            "I_inh": I_inh_new,
            "spike_count": state["spike_count"] + spikes,
        }
        
        return spikes, new_state


class CoincidenceDetector(nn.Module):
    """
    Temporal coincidence detection layer (BEAM-Net §4.2).
    
    Computes similarity between query spike train and stored patterns
    using a temporal kernel κ (Eq. 22).
    
    S_j(ξ) = Σ_i w_i · κ(t_i(ξ) - t_i(x_j))    — Eq. 21
    
    κ(Δt) = exp(-|Δt|/τ_s) if |Δt| < Δt_max     — Eq. 22
    
    By Lemma 4.2, this approximates the exponential dot product
    S_j(ξ) ≈ exp(β ξ^T x_j), recovering Hopfield retrieval.
    
    Biological basis: NMDA receptor coincidence detection with
    ~5 ms temporal windows (Larkum, 2013; Markram et al., 1997).
    
    Parameters
    ----------
    d : int
        Input dimensionality (after encoding).
    N : int
        Number of stored patterns.
    tau_s : float
        Synaptic kernel time constant (ms).
    dt_max : float
        Coincidence window (ms).
    """
    
    def __init__(self, d: int, N: int, tau_s: float = 3.0, dt_max: float = 5.0):
        super().__init__()
        self.d = d
        self.N = N
        self.tau_s = tau_s
        self.dt_max = dt_max
        
        # Stored patterns as learnable spike times: X ∈ R^{N × d}
        self.patterns = nn.Parameter(torch.randn(N, d) * 0.1)
        
        # Per-dimension synaptic weights (Eq. 21)
        self.weights = nn.Parameter(torch.ones(d))
    
    def temporal_kernel(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Temporal kernel κ(Δt) — Eq. 22.
        Exponential decay within coincidence window, zero outside.
        """
        within_window = (dt.abs() < self.dt_max).float()
        return torch.exp(-dt.abs() / self.tau_s) * within_window
    
    def forward(self, query_times: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores via temporal coincidence.
        
        Parameters
        ----------
        query_times : torch.Tensor, shape (batch, d)
            Query spike times.
        
        Returns
        -------
        S : torch.Tensor, shape (batch, N)
            Similarity scores for each stored pattern.
        """
        # Pattern spike times from stored parameters
        pattern_times = 20.0 - 15.0 * torch.sigmoid(self.patterns)  # (N, d)
        
        # Compute temporal differences
        # query: (batch, d) → (batch, 1, d)
        # patterns: (N, d) → (1, N, d)
        dt = query_times.unsqueeze(1) - pattern_times.unsqueeze(0)  # (batch, N, d)
        
        # Apply temporal kernel
        kappa = self.temporal_kernel(dt)  # (batch, N, d)
        
        # Weighted sum: S_j = Σ_i w_i · κ(Δt_i)
        w = torch.abs(self.weights).unsqueeze(0).unsqueeze(0)  # (1, 1, d)
        S = (w * kappa).sum(dim=-1)  # (batch, N)
        
        return S