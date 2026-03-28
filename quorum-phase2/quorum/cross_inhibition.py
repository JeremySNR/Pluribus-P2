"""Hopfield Energy Cross-Inhibition Engine (§3.1, §3.2).

Energy function:
  E(B) = -β⁻¹ log Σᵢ exp(β · δᵢᵀ B_slot) + ½‖B_slot‖² + λ_repel · Σᵢ<ⱼ max(0, δᵢᵀδⱼ)

Update rule (gradient descent on E):
  B_slot^{t+1} = Σᵢ wᵢ(t) · δᵢ
  where wᵢ(t) = exp(β(t) · δᵢᵀ B_slot^t) / Σⱼ exp(β(t) · δⱼᵀ B_slot^t)

Bee-inspired stop-signal dynamics (§3.2):
  dsᵢ/dt = γ·qᵢ·(1 − Σⱼ sⱼ) − α·sᵢ − Σⱼ≠ᵢ σ·sᵢ·sⱼ

[PROVEN]: Hopfield energy = transformer attention (Ramsauer et al., 2020).
[SPECULATIVE]: Repulsion penalty and bee dynamics for LLM activation competition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .utils import log_sum_exp


@dataclass
class CrossInhibitionConfig:
    """Configuration for the cross-inhibition engine."""
    beta_min: float = 0.1
    beta_max: float = 10.0
    tau_beta: float = 3.0
    lambda_repel: float = 0.1
    sigma: float = 0.3
    alpha_decay: float = 0.05
    gamma_recruit: float = 0.1
    agreement_threshold: float = 0.8


def compute_beta(
    t: int,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    tau_beta: float = 3.0,
    agreement: Optional[float] = None,
    agreement_threshold: float = 0.8,
) -> float:
    """Compute the β temperature parameter with exponential warmup (§3.3 Mechanism 3).

    β(t) = β_min + (β_max − β_min) · (1 − exp(−t/τ_β))
    If agreement exceeds threshold, jump to β_max immediately.

    Args:
        t: Current round index.
        beta_min: Minimum β (soft averaging).
        beta_max: Maximum β (near winner-take-all).
        tau_beta: Annealing time constant.
        agreement: Mean pairwise cosine similarity of deltas.
        agreement_threshold: If agreement exceeds this, jump to β_max.
    """
    if agreement is not None and agreement > agreement_threshold:
        return beta_max

    import math
    return beta_min + (beta_max - beta_min) * (1.0 - math.exp(-t / tau_beta))


def hopfield_energy(
    B_slot: torch.Tensor,
    deltas: torch.Tensor,
    beta: float,
    lambda_repel: float = 0.1,
) -> torch.Tensor:
    """Compute the Hopfield energy for a single slot (§3.1).

    E(B) = -β⁻¹ log Σᵢ exp(β · δᵢᵀ B_slot) + ½‖B_slot‖² + λ_repel · Σᵢ<ⱼ max(0, δᵢᵀδⱼ)

    Args:
        B_slot: [D] current buffer state for this slot.
        deltas: [N, D] agent deltas for this slot.
        beta: Temperature parameter.
        lambda_repel: Repulsion penalty weight.
    """
    N = deltas.shape[0]

    similarities = deltas @ B_slot  # [N]
    attractor = -(1.0 / max(beta, 1e-8)) * log_sum_exp(beta * similarities, dim=0)

    norm_reg = 0.5 * B_slot.pow(2).sum()

    repulsion = torch.tensor(0.0, device=B_slot.device)
    if lambda_repel > 0 and N > 1:
        for i in range(N):
            for j in range(i + 1, N):
                dot = (deltas[i] * deltas[j]).sum()
                repulsion = repulsion + torch.relu(dot)
        repulsion = lambda_repel * repulsion

    return attractor + norm_reg + repulsion


def hopfield_update(
    B_slot: torch.Tensor,
    deltas: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute the Hopfield-energy update for a single slot (§3.1).

    B_slot^{t+1} = Σᵢ wᵢ(t) · δᵢ
    where wᵢ(t) = softmax(β · δᵢᵀ B_slot^t)

    Args:
        B_slot: [D] current buffer state for this slot.
        deltas: [N, D] agent deltas for this slot.
        beta: Temperature parameter.

    Returns:
        Updated slot vector [D].
    """
    similarities = deltas @ B_slot  # [N]
    weights = F.softmax(beta * similarities, dim=0)  # [N]
    return (weights.unsqueeze(1) * deltas).sum(dim=0)  # [D]


def bee_inhibition_step(
    strengths: torch.Tensor,
    qualities: torch.Tensor,
    sigma: float = 0.3,
    alpha: float = 0.05,
    gamma: float = 0.1,
) -> torch.Tensor:
    """Bee-inspired stop-signal dynamics update (§3.2).

    dsᵢ/dt = γ·qᵢ·(1 − Σⱼ sⱼ) − α·sᵢ − Σⱼ≠ᵢ σ·sᵢ·sⱼ

    Args:
        strengths: [N] current influence strength per agent.
        qualities: [N] proposal quality scores.
        sigma: Cross-inhibition rate.
        alpha: Spontaneous decay rate.
        gamma: Recruitment rate.

    Returns:
        Updated strengths [N], normalized to sum to 1.
    """
    uncommitted = (1.0 - strengths.sum()).clamp(min=0.01)

    recruitment = gamma * qualities * uncommitted
    decay = alpha * strengths

    total_strength = strengths.sum()
    cross_inhib = sigma * strengths * (total_strength - strengths)

    d_strengths = recruitment - decay - cross_inhib
    new_strengths = (strengths + d_strengths).clamp(min=0.0, max=1.0)

    total = new_strengths.sum()
    if total > 1e-8:
        new_strengths = new_strengths / total

    return new_strengths


def compute_agreement(deltas: List[torch.Tensor]) -> float:
    """Compute mean pairwise cosine similarity of deltas.

    Used to determine if agents agree enough to jump β to β_max.
    """
    if len(deltas) < 2:
        return 1.0

    N = len(deltas)
    flat = torch.stack([d.flatten() for d in deltas])
    sims = []
    for i in range(N):
        for j in range(i + 1, N):
            sim = F.cosine_similarity(flat[i:i+1], flat[j:j+1]).item()
            sims.append(sim)
    return sum(sims) / len(sims) if sims else 1.0
