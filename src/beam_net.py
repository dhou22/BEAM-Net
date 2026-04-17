"""
BEAM-Net: Full Architecture (§3 Assembly)
============================================
Assembles all four components into the end-to-end architecture:
  1. Native Event Encoding (§3.1)     → temporal_encoder.py
  2. Bayesian Spiking Neuron (§3.2)   → bayesian_lif.py
  3. Dirichlet Attention (§3.3)       → dirichlet_attention.py
  4. Bidirectional Inference (§3.4)   → bidirectional_inference.py

Energy functional minimized (Eq. 19):
  E[S, q] = -ln(Σ exp(S_j)) + KL[q(z) || p(z)] + ½||ξ||²
  
This unifies Hopfield energy, variational free energy, and regularization
into a single objective with guaranteed monotonic descent (Theorem 4.1).

The model supports two modes:
  - Classification: posterior over causes → class prediction
  - Reconstruction: posterior-weighted retrieval → decoded output
    (following W-TCRL evaluation protocol, Fois & Girau, 2023, §3.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from src.temporal_encoder import RankOrderEncoder, PopulationLatencyEncoder, DVSEventEncoder
from src.bayesian_lif import BayesianLIFNeuron, CoincidenceDetector
from src.dirichlet_attention import DirichletAttention
from src.bidirectional_inference import BidirectionalInference


class BEAMNet(nn.Module):
    """
    Bayesian Event-driven Attentional Memory Network.
    
    End-to-end spiking architecture for classification or reconstruction
    on event-based or continuous data with uncertainty quantification.
    
    Parameters
    ----------
    d_input : int
        Raw input dimensionality (e.g., 34*34 = 1156 for N-MNIST).
    n_classes : int
        Number of output classes (for classification mode).
    cfg : dict
        Full configuration dictionary from experiment.yaml.
    mode : str
        "classification" or "reconstruction".
    """
    
    def __init__(
        self,
        d_input: int,
        n_classes: int,
        cfg: dict,
        mode: str = "classification",
    ):
        super().__init__()
        self.d_input = d_input
        self.n_classes = n_classes
        self.mode = mode
        self.cfg = cfg
        
        enc_cfg = cfg["encoding"]
        neu_cfg = cfg["neuron"]
        coin_cfg = cfg["coincidence"]
        att_cfg = cfg["attention"]
        inf_cfg = cfg["inference"]
        
        N = att_cfg["n_patterns"]
        
        # ---- Component 1: Encoder ----
        if enc_cfg["method"] == "population_latency":
            self.encoder = PopulationLatencyEncoder(
                d_input=d_input,
                l=enc_cfg["l_population"],
                sigma=enc_cfg["sigma_rf"],
                T_ref=enc_cfg["T_ref"],
                tau=enc_cfg["tau_spread"],
            )
            d_encoded = self.encoder.output_dim
        else:
            self.encoder = RankOrderEncoder(
                T_ref=enc_cfg["T_ref"],
                tau=enc_cfg["tau_spread"],
            )
            d_encoded = d_input
        
        # ---- Component 2: Coincidence + Bayesian LIF ----
        self.coincidence = CoincidenceDetector(
            d=d_encoded, N=N,
            tau_s=coin_cfg["tau_s"],
            dt_max=coin_cfg["dt_max"],
        )
        
        self.neurons = BayesianLIFNeuron(
            N=N,
            tau_m=neu_cfg["tau_m"],
            tau_theta=neu_cfg["tau_theta"],
            theta_0=neu_cfg["theta_0"],
            delta=neu_cfg["delta"],
            gamma=neu_cfg["gamma"],
            w_inh=neu_cfg["w_inh"],
            tau_syn=neu_cfg["tau_syn"],
            dt=neu_cfg["dt"],
        )
        
        # ---- Component 3: Dirichlet Attention ----
        self.attention = DirichletAttention(
            N=N,
            alpha_0=att_cfg["alpha_0"],
            eta=att_cfg["eta"],
        )
        
        # ---- Component 4: Bidirectional Inference ----
        self.bidir = BidirectionalInference(
            d=d_encoded, N=N,
            lambda_td=inf_cfg["lambda_td"],
            max_iterations=inf_cfg["max_iterations"],
            convergence_eps=inf_cfg["convergence_eps"],
        )
        
        # ---- Output head ----
        if mode == "classification":
            # Map posterior over N patterns → class logits
            self.classifier = nn.Linear(N, n_classes)
        else:
            # Reconstruction: decoder from pattern space to input space
            self.decoder = nn.Linear(d_encoded, d_input)
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass through BEAM-Net.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data. Shape depends on encoder:
              - Rank-order: (batch, d_input) continuous values in [0,1]
              - DVS: (batch, n_bins, 2, H, W) event frames
        return_uncertainty : bool
            Whether to compute and return uncertainty decomposition.
        
        Returns
        -------
        logits_or_recon : torch.Tensor
            Classification logits (batch, n_classes) or reconstruction (batch, d_input).
        info : dict
            Uncertainty metrics, convergence info, sparsity stats.
        """
        n_sim = self.cfg["inference"]["n_sim_steps"]
        
        # ---- Encode ----
        spike_times = self.encoder(x)  # (batch, d_encoded)
        
        # ---- Bidirectional inference ----
        output, info = self.bidir(
            query_times=spike_times,
            coincidence_fn=self.coincidence,
            neuron_fn=self.neurons,
            attention_fn=self.attention,
            patterns=self.coincidence.patterns,
            n_sim_steps=n_sim,
        )
        
        # ---- Output ----
        if self.mode == "classification":
            logits = self.classifier(info["posterior_mean"])
            result = logits
        else:
            recon = self.decoder(output)
            result = torch.sigmoid(recon)  # Normalize to [0, 1]
        
        # ---- Sparsity metric (cf. W-TCRL §3.1.2, Eq. 7) ----
        firing_rates = info.get("firing_rates", info["posterior_mean"])
        sparsity = (firing_rates > 0.01).float().mean(dim=-1)  # Fraction active
        info["sparsity"] = sparsity
        
        return result, info
    
    def compute_loss(
        self,
        result: torch.Tensor,
        target: torch.Tensor,
        info: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute BEAM-Net loss incorporating energy functional terms (Eq. 19).
        
        Loss = Task_loss + β_kl · KL_term + β_sparse · Sparsity_penalty
        
        Parameters
        ----------
        result : torch.Tensor
            Model output (logits or reconstruction).
        target : torch.Tensor
            Ground truth (class labels or original input).
        info : dict
            Forward pass info containing uncertainty metrics.
        
        Returns
        -------
        loss : torch.Tensor
            Total loss.
        loss_dict : dict
            Breakdown of loss components.
        """
        if self.mode == "classification":
            task_loss = F.cross_entropy(result, target)
        else:
            task_loss = F.mse_loss(result, target)
        
        # KL regularization: penalize deviation from uniform prior
        # This is the "complexity" term in Eq. 19
        posterior = info["posterior_mean"]  # (batch, N)
        N = posterior.shape[-1]
        uniform = torch.ones_like(posterior) / N
        kl_loss = F.kl_div(
            torch.log(posterior + 1e-8), uniform, reduction="batchmean"
        )
        
        # Sparsity penalty: encourage sparse activity
        sparsity_loss = info["sparsity"].mean()
        
        # Total loss with weighting
        beta_kl = 0.01
        beta_sparse = 0.1
        loss = task_loss + beta_kl * kl_loss + beta_sparse * sparsity_loss
        
        loss_dict = {
            "total_loss": loss.item(),
            "task_loss": task_loss.item(),
            "kl_loss": kl_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "epistemic_unc": info["epistemic_uncertainty"].mean().item(),
            "aleatoric_unc": info["aleatoric_uncertainty"].mean().item(),
        }
        
        return loss, loss_dict