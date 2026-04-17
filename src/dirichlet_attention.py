"""
Component 3: Multi-Cause Generative Attention (BEAM-Net §3.3)
================================================================
Replaces winner-take-all selection with Dirichlet-distributed posterior
over cause attributions, enabling simultaneous tracking of multiple hypotheses.

Key equations:
  - Dirichlet posterior (Eq. 12): π ~ Dir(α), α_j = α_0 + η · S_j(ξ)
  - Retrieved output (Eq. 13):   ξ_out = Σ E[π_j] · x_j
  - Epistemic uncertainty (Eq. 14): H[E[π]] - E[H[π]]
  - Aleatoric uncertainty (Eq. 15): E[H[π]]

Biological basis:
  Population coding (Knill & Pouget, 2004) where neural populations
  encode probability distributions over stimulus features.
  
  The Dirichlet distribution is the conjugate prior for categorical
  data — biologically, this maps to divisive normalization in cortical
  circuits (Carandini & Heeger, 2012).

Connection to Hopfield energy:
  Standard attention performs single-cause selection via softmax (Eq. 3).
  Dirichlet attention generalizes this: when η → ∞, we recover WTA;
  when η → 0, uniform averaging. This is analogous to β (inverse
  temperature) in the Hopfield energy (Remark 2.3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class DirichletAttention(nn.Module):
    """
    Dirichlet-distributed multi-cause attention mechanism.
    
    Instead of hard argmax or soft softmax, maintains a full Dirichlet
    posterior over cause probabilities, providing calibrated uncertainty.
    
    Parameters
    ----------
    N : int
        Number of stored patterns (causes).
    alpha_0 : float
        Prior concentration parameter. Higher = more uniform prior.
    eta : float
        Evidence scale. Controls how much similarity scores influence posterior.
    """
    
    def __init__(self, N: int, alpha_0: float = 1.0, eta: float = 5.0):
        super().__init__()
        self.N = N
        self.alpha_0 = alpha_0
        self.eta = eta
    
    def compute_dirichlet_params(self, S: torch.Tensor) -> torch.Tensor:
        """
        Compute Dirichlet concentration parameters from similarity scores.
        
        α_j = α_0 + η · ReLU(S_j)    — Eq. 12 (with ReLU for positivity)
        
        Parameters
        ----------
        S : torch.Tensor, shape (batch, N)
            Similarity scores from coincidence detection.
        
        Returns
        -------
        alpha : torch.Tensor, shape (batch, N)
            Dirichlet concentration parameters.
        """
        return self.alpha_0 + self.eta * F.relu(S)
    
    def expected_probabilities(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        E[π_j] = α_j / Σ_k α_k    — mean of Dirichlet distribution.
        """
        return alpha / alpha.sum(dim=-1, keepdim=True)
    
    def forward(
        self,
        S: torch.Tensor,
        patterns: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform Dirichlet attention retrieval with uncertainty quantification.
        
        Parameters
        ----------
        S : torch.Tensor, shape (batch, N)
            Similarity scores.
        patterns : torch.Tensor, shape (N, d)
            Stored memory patterns.
        
        Returns
        -------
        output : torch.Tensor, shape (batch, d)
            Retrieved output: ξ_out = Σ E[π_j] · x_j  (Eq. 13)
        info : dict
            Contains uncertainty decomposition and posterior parameters:
              - alpha: Dirichlet params (batch, N)
              - posterior_mean: E[π] (batch, N)
              - epistemic_uncertainty: scalar per sample (batch,)
              - aleatoric_uncertainty: scalar per sample (batch,)
              - total_uncertainty: H[E[π]] (batch,)
        """
        # Dirichlet concentration parameters
        alpha = self.compute_dirichlet_params(S)  # (batch, N)
        
        # Posterior mean
        pi_mean = self.expected_probabilities(alpha)  # (batch, N)
        
        # Retrieved output (Eq. 13)
        output = torch.matmul(pi_mean, patterns)  # (batch, d)
        
        # ---- Uncertainty decomposition (Eqs. 14-15) ----
        total_uncertainty = self._entropy(pi_mean)  # H[E[π]]
        
        # Expected entropy E[H[π]] under Dirichlet
        alpha_0_sum = alpha.sum(dim=-1)  # (batch,)
        expected_entropy = self._dirichlet_expected_entropy(alpha, alpha_0_sum)
        
        # Epistemic = Total - Aleatoric (mutual information decomposition)
        epistemic = total_uncertainty - expected_entropy
        aleatoric = expected_entropy
        
        info = {
            "alpha": alpha,
            "posterior_mean": pi_mean,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "total_uncertainty": total_uncertainty,
        }
        
        return output, info
    
    @staticmethod
    def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Shannon entropy H[p] = -Σ p_j log(p_j)."""
        return -(p * torch.log(p + eps)).sum(dim=-1)
    
    @staticmethod
    def _dirichlet_expected_entropy(
        alpha: torch.Tensor, alpha_0: torch.Tensor
    ) -> torch.Tensor:
        """
        Expected entropy of a Dirichlet distribution.
        E[H[π]] = log B(α) + (α_0 - N)ψ(α_0) - Σ(α_j - 1)ψ(α_j)
        where ψ is the digamma function.
        """
        N = alpha.shape[-1]
        # Digamma approximation (stable for α > 0.5)
        psi_alpha = torch.digamma(alpha)
        psi_alpha0 = torch.digamma(alpha_0.unsqueeze(-1)).squeeze(-1)
        
        # Log multivariate beta
        log_B = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(alpha_0)
        
        expected_H = log_B + (alpha_0 - N) * psi_alpha0 - (
            (alpha - 1.0) * psi_alpha
        ).sum(dim=-1)
        
        return expected_H