"""
Component 1: Native Event Encoding (BEAM-Net §3.1)
=====================================================
Converts input data (continuous values or DVS events) into spike times
using rank-order temporal coding.

Key equations:
  - Rank-order encoding (Eq. 7):  t_i(ξ) = T_ref - τ · ξ_i
  - Event stream encoding (Eq. 6): t_enc_i = min{t_k : e_k ∈ R_i, p_k = +1}

Biological basis:
  Retinal ganglion → LGN → V1 Layer IV processing, where stimulus intensity
  modulates first-spike latency with 1–5 ms precision (Gerstner et al., 2014).
  
  Population-based latency coding from W-TCRL (Fois & Girau, 2023, §2.2)
  uses Gaussian receptive fields to distribute each input dimension across
  l neurons, encoding values in relative spike latencies.

Connection to neuroscience:
  The rank-order code is consistent with Thorpe et al. (2001) temporal coding
  hypothesis: the most salient features produce earliest spikes, and downstream
  neurons can decode information from relative spike order alone.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class RankOrderEncoder(nn.Module):
    """
    Rank-order temporal encoder (BEAM-Net Eq. 7).
    
    Maps continuous values ξ ∈ [0, 1]^d to spike times t ∈ R^d.
    Larger values → earlier spikes: ξ_i > ξ_j ⟹ t_i < t_j.
    
    This is the simplest encoding — one spike per input dimension.
    
    Parameters
    ----------
    T_ref : float
        Reference time (ms). All spikes occur before T_ref.
    tau : float
        Temporal spread constant (ms). Controls separation between
        earliest and latest spikes.
    """
    
    def __init__(self, T_ref: float = 20.0, tau: float = 15.0):
        super().__init__()
        self.T_ref = T_ref
        self.tau = tau
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous input into spike times.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, d)
            Input values in [0, 1]. Should be normalized.
        
        Returns
        -------
        spike_times : torch.Tensor, shape (batch, d)
            Spike times in [T_ref - τ, T_ref]. Earlier = more salient.
        """
        # Eq. 7: t_i(ξ) = T_ref - τ · ξ_i
        x_clamped = torch.clamp(x, 0.0, 1.0)
        spike_times = self.T_ref - self.tau * x_clamped
        return spike_times


class PopulationLatencyEncoder(nn.Module):
    """
    Population-based latency encoder (inspired by W-TCRL §2.2, Fois & Girau).
    
    Each input dimension is encoded by a population of l neurons with
    Gaussian receptive fields. The neuron whose center μ_i is closest to
    the input value fires first; others fire later proportionally to distance.
    
    This provides richer temporal patterns than rank-order coding and is
    more robust to noise (W-TCRL §4, Discussion on noise robustness).
    
    Output dimensionality: d_input × l (population expansion)
    
    Parameters
    ----------
    d_input : int
        Input dimensionality.
    l : int
        Population size (neurons per dimension). Default 10 (W-TCRL).
    sigma : float
        Receptive field width. Default 0.6 (W-TCRL).
    T_ref : float
        Reference time (ms).
    tau : float
        Temporal spread (ms).
    """
    
    def __init__(
        self,
        d_input: int,
        l: int = 10,
        sigma: float = 0.6,
        T_ref: float = 20.0,
        tau: float = 15.0,
    ):
        super().__init__()
        self.d_input = d_input
        self.l = l
        self.sigma = sigma
        self.T_ref = T_ref
        self.tau = tau
        
        # Gaussian receptive field centers: uniformly spaced in [0.05, 0.95]
        # (W-TCRL §2.2: "centers μ_i are uniformly spread between 0.05 and 0.95")
        mu = torch.linspace(0.05, 0.95, l)
        self.register_buffer("mu", mu)  # (l,)
    
    @property
    def output_dim(self) -> int:
        """Expanded dimensionality after population coding."""
        return self.d_input * self.l
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input through population of Gaussian receptive fields.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, d_input)
            Normalized input in [0, 1].
        
        Returns
        -------
        spike_times : torch.Tensor, shape (batch, d_input * l)
            Spike times for entire population. Shape is flattened:
            [dim0_neuron0, dim0_neuron1, ..., dim0_neuronL, dim1_neuron0, ...]
        """
        batch = x.shape[0]
        x_clamped = torch.clamp(x, 0.0, 1.0)
        
        # x: (batch, d) → (batch, d, 1)
        # mu: (l,) → (1, 1, l)
        # Gaussian activation: A = exp(-((x - μ)^2) / (2σ^2))
        x_exp = x_clamped.unsqueeze(-1)           # (batch, d, 1)
        mu_exp = self.mu.unsqueeze(0).unsqueeze(0)  # (1, 1, l)
        
        # Activation levels (W-TCRL Eq. in §2.2)
        activation = torch.exp(-((x_exp - mu_exp) ** 2) / (2 * self.sigma ** 2))
        # activation: (batch, d, l)
        
        # Convert activations to spike times
        # Higher activation → earlier spike (same principle as Eq. 7)
        spike_times = self.T_ref - self.tau * activation
        
        # Flatten population dimension: (batch, d, l) → (batch, d * l)
        spike_times = spike_times.reshape(batch, -1)
        
        return spike_times


class DVSEventEncoder(nn.Module):
    """
    Native DVS event stream encoder (BEAM-Net §3.1.1, Eq. 6).
    
    Converts raw DVS events into spike time tensors by taking the
    first event arrival time in each spatial receptive field.
    
    For use with tonic library's event representations.
    
    Parameters
    ----------
    spatial_dims : Tuple[int, int]
        Sensor resolution (H, W).
    n_bins : int
        Number of temporal bins for discretization.
    """
    
    def __init__(self, spatial_dims: Tuple[int, int] = (34, 34), n_bins: int = 10):
        super().__init__()
        self.H, self.W = spatial_dims
        self.n_bins = n_bins
    
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Convert binned event frames to first-spike-time representation.
        
        Parameters
        ----------
        events : torch.Tensor, shape (batch, n_bins, 2, H, W)
            Temporal bins of event polarities (from tonic transforms).
            Channel 0 = ON events, Channel 1 = OFF events.
        
        Returns
        -------
        spike_times : torch.Tensor, shape (batch, H * W)
            First-spike latency per spatial pixel. 
            Pixels with no events get T_ref (latest time).
        """
        batch = events.shape[0]
        
        # Aggregate ON polarity across channels: (batch, n_bins, H, W)
        on_events = events[:, :, 0, :, :]  # ON events only
        
        # Find first bin with an event for each pixel
        # has_event: (batch, n_bins, H, W), bool
        has_event = on_events > 0
        
        # First event time: argmax over temporal bins (first True)
        # If no event, argmax returns 0 — we mask these
        any_event = has_event.any(dim=1)  # (batch, H, W)
        first_bin = has_event.float().argmax(dim=1)  # (batch, H, W)
        
        # Normalize to [0, 1] then convert to spike times
        # Eq. 6: t_enc_i = first event time (normalized)
        spike_times = first_bin.float() / self.n_bins
        # No event → latest spike time (1.0)
        spike_times[~any_event] = 1.0
        
        # Flatten spatial dims
        spike_times = spike_times.reshape(batch, -1)
        
        return spike_times