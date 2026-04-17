"""
Component 4: Bidirectional Inference Loop (BEAM-Net §3.4)
============================================================
Implements hierarchical predictive coding through two interacting streams:
  - Bottom-Up: events → encoding → coincidence → similarity scores
  - Top-Down: current beliefs modulate encoding layer thresholds

Key equations:
  - Top-down modulation (Eq. 16): θ_i^enc(t) = θ_0_i - λ Σ P(cause_j|ξ) w_ji^TD
  - Convergence criterion (Eq. 17): ||P^(t+1) - P^(t)||_1 < ε
  - Lyapunov functional (Eq. 18): L(t) = F[q^(t)] + λ/2 Σ (V_j^(t) - V_j^(t-1))²

Theorem 3.6 guarantees convergence in ≤ L = ⌈3τ_m/τ_s⌉ iterations
(100–250 ms), matching cortical attention latencies (Desimone & Duncan, 1995).

Biological basis:
  Predictive coding (Rao & Ballard, 1999) posits that cortical circuits
  implement approximate Bayesian inference with prediction errors propagated
  across hierarchical levels. L1/L5 feedback connections implement top-down
  modulation (Larkum, 2013).
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class BidirectionalInference(nn.Module):
    """
    Bidirectional inference loop combining bottom-up evidence accumulation
    with top-down predictive modulation.
    
    Parameters
    ----------
    d : int
        Encoded input dimensionality.
    N : int
        Number of stored patterns.
    lambda_td : float
        Top-down modulation strength (must satisfy Theorem 3.6 bound).
    max_iterations : int
        Maximum number of bidirectional iterations L.
    convergence_eps : float
        Convergence threshold ε for posterior stability.
    """
    
    def __init__(
        self,
        d: int,
        N: int,
        lambda_td: float = 0.1,
        max_iterations: int = 5,
        convergence_eps: float = 0.01,
    ):
        super().__init__()
        self.d = d
        self.N = N
        self.lambda_td = lambda_td
        self.max_iterations = max_iterations
        self.convergence_eps = convergence_eps
        
        # Top-down synaptic weights: w_TD ∈ R^{N × d}
        # Maps posterior beliefs back to encoding space (Eq. 16)
        self.w_td = nn.Parameter(torch.randn(N, d) * 0.01)
    
    def top_down_modulate(
        self,
        query_times: torch.Tensor,
        posterior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply top-down predictive modulation to encoding thresholds.
        
        Eq. 16: θ_i^enc(t) = θ_0_i - λ Σ_j P(cause_j|ξ) · w_ji^TD
        
        In practice, we modulate the spike times (equivalent effect):
        earlier spikes for features predicted by current beliefs.
        
        Parameters
        ----------
        query_times : torch.Tensor, shape (batch, d)
            Original query spike times.
        posterior : torch.Tensor, shape (batch, N)
            Current posterior over causes P(cause_j | ξ).
        
        Returns
        -------
        modulated_times : torch.Tensor, shape (batch, d)
            Modulated spike times incorporating top-down predictions.
        """
        # Top-down prediction: Σ_j P(cause_j|ξ) · w_TD_j
        # posterior: (batch, N), w_td: (N, d) → prediction: (batch, d)
        prediction = torch.matmul(posterior, self.w_td)
        
        # Modulate query times: predicted features get earlier spikes
        modulated = query_times - self.lambda_td * prediction
        
        return modulated
    
    def check_convergence(
        self,
        posterior_new: torch.Tensor,
        posterior_old: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check convergence criterion (Eq. 17).
        ||P^(t+1) - P^(t)||_1 < ε
        
        Returns per-sample convergence as boolean tensor.
        """
        diff = (posterior_new - posterior_old).abs().sum(dim=-1)
        return diff < self.convergence_eps
    
    def forward(
        self,
        query_times: torch.Tensor,
        coincidence_fn,
        neuron_fn,
        attention_fn,
        patterns: torch.Tensor,
        n_sim_steps: int = 50,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run full bidirectional inference loop.
        
        Parameters
        ----------
        query_times : torch.Tensor, shape (batch, d)
            Initial query spike times.
        coincidence_fn : callable
            Coincidence detector (maps spike times → similarity scores).
        neuron_fn : BayesianLIFNeuron
            Bayesian LIF neuron layer.
        attention_fn : DirichletAttention
            Dirichlet attention mechanism.
        patterns : torch.Tensor, shape (N, d)
            Stored memory patterns.
        n_sim_steps : int
            Number of LIF simulation steps per iteration.
        
        Returns
        -------
        output : torch.Tensor, shape (batch, d)
            Final retrieved output.
        info : dict
            Iteration history, convergence info, uncertainty metrics.
        """
        batch_size = query_times.shape[0]
        device = query_times.device
        
        # Initialize posterior uniformly
        posterior = torch.ones(batch_size, self.N, device=device) / self.N
        current_times = query_times
        
        iteration_history = []
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for it in range(self.max_iterations):
            posterior_old = posterior.clone()
            
            # ---- Bottom-up pass ----
            # 1. Coincidence detection on (possibly modulated) spike times
            S = coincidence_fn(current_times)  # (batch, N)
            
            # 2. Run Bayesian LIF neurons
            state = neuron_fn.init_state(batch_size, device)
            for t in range(n_sim_steps):
                spikes, state = neuron_fn(S, state)
            
            # 3. Firing rates → posterior approximation (Prop. 3.3)
            firing_rates = state["spike_count"] / n_sim_steps
            firing_rates = firing_rates / (firing_rates.sum(dim=-1, keepdim=True) + 1e-8)
            
            # 4. Dirichlet attention
            output, attn_info = attention_fn(S, patterns)
            posterior = attn_info["posterior_mean"]
            
            # ---- Top-down pass ----
            # Modulate spike times based on current beliefs
            current_times = self.top_down_modulate(query_times, posterior)
            
            # ---- Check convergence (Eq. 17) ----
            converged = self.check_convergence(posterior, posterior_old)
            
            iteration_history.append({
                "iteration": it,
                "posterior": posterior.detach().clone(),
                "convergence": converged.float().mean().item(),
                "epistemic_unc": attn_info["epistemic_uncertainty"].mean().item(),
                "aleatoric_unc": attn_info["aleatoric_uncertainty"].mean().item(),
            })
            
            # Early exit if all samples converged
            if converged.all():
                break
        
        info = {
            "n_iterations": it + 1,
            "converged_fraction": converged.float().mean().item(),
            "iteration_history": iteration_history,
            "firing_rates": firing_rates,
            **attn_info,
        }
        
        return output, info