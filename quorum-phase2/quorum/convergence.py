"""Convergence Controller — Halting Criteria and PonderNet (§4.3).

Two modes:
  1. Quorum-based (cold start): residual < ε for 3 consecutive rounds.
  2. PonderNet-style adaptive halting (requires training with task loss).

[PROVEN]: PonderNet (Banino et al., DeepMind 2021).
Application to multi-agent convergence detection is [PLAUSIBLE].
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConvergenceConfig:
    """Configuration for convergence control."""
    tol: float = 0.01
    consecutive_required: int = 3
    max_rounds: int = 20
    use_pondernet: bool = False
    lambda_prior: float = 0.2
    beta_kl: float = 0.01


class QuorumHaltCriterion:
    """Quorum-based halting: residual < ε for K consecutive rounds (§4.3 fallback).

    Standard DEQ convergence criterion.
    """

    def __init__(self, tol: float = 0.01, consecutive_required: int = 3):
        self.tol = tol
        self.consecutive_required = consecutive_required
        self._consecutive_count = 0

    def reset(self) -> None:
        self._consecutive_count = 0

    def should_halt(self, residual: float) -> bool:
        """Check if convergence criterion is met.

        Args:
            residual: Relative residual ‖B_{t+1} − B_t‖ / ‖B_t‖.
        Returns:
            True if the criterion is satisfied.
        """
        if residual < self.tol:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0
        return self._consecutive_count >= self.consecutive_required

    @property
    def consecutive_count(self) -> int:
        return self._consecutive_count


class PonderNetHalter(nn.Module):
    """PonderNet-style adaptive halting network (§4.3).

    [SPECULATIVE] — structure only; training requires a task loss.

    Inputs: [relative_residual, max_strength, entropy_strengths, t/T_max]
    Output: λ_t = halt probability at round t.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        residual: float,
        max_strength: float,
        strength_entropy: float,
        t_normalised: float,
    ) -> torch.Tensor:
        """Compute halt probability λ_t.

        Args:
            residual: ‖B_t − B_{t−1}‖ / ‖B_t‖.
            max_strength: max(s_i) — how dominant the winner is.
            strength_entropy: entropy(s) — how decided the competition is.
            t_normalised: t / T_max — normalised round number.

        Returns:
            Scalar halt probability in [0, 1].
        """
        features = torch.tensor(
            [residual, max_strength, strength_entropy, t_normalised],
            dtype=torch.float32,
        )
        return self.net(features).squeeze()


def compute_pondernet_loss(
    halt_probs: list[torch.Tensor],
    task_losses: list[torch.Tensor],
    lambda_prior: float = 0.2,
    beta_kl: float = 0.01,
) -> torch.Tensor:
    """Compute the PonderNet training loss (§4.3).

    L_halt = Σ_t p_t · L_task(B_t) + β_KL · KL(p || Geom(λ_prior))

    where p_t = λ_t · Π_{j<t}(1−λ_j) is the unconditional halt probability.

    Args:
        halt_probs: List of λ_t values from each round.
        task_losses: List of task losses at each round.
        lambda_prior: Geometric distribution prior parameter.
        beta_kl: KL divergence weight.

    Returns:
        Total PonderNet loss.
    """
    T = len(halt_probs)
    device = halt_probs[0].device if halt_probs else torch.device("cpu")

    p_uncond = []
    cum_not_halt = torch.tensor(1.0, device=device)
    for t in range(T):
        p_t = halt_probs[t] * cum_not_halt
        p_uncond.append(p_t)
        cum_not_halt = cum_not_halt * (1.0 - halt_probs[t])

    task_term = torch.tensor(0.0, device=device)
    for t in range(T):
        task_term = task_term + p_uncond[t] * task_losses[t]

    geom_probs = []
    cum_prior = 1.0
    for t in range(T):
        geom_probs.append(lambda_prior * cum_prior)
        cum_prior *= (1.0 - lambda_prior)

    kl_term = torch.tensor(0.0, device=device)
    for t in range(T):
        p = p_uncond[t].clamp(min=1e-8)
        q = max(geom_probs[t], 1e-8)
        kl_term = kl_term + p * (p.log() - torch.tensor(q, device=device).log())

    return task_term + beta_kl * kl_term


def compute_residual(B_new: torch.Tensor, B_old: torch.Tensor) -> float:
    """Compute relative residual ‖B_new − B_old‖ / ‖B_new‖."""
    diff_norm = (B_new - B_old).norm().item()
    new_norm = B_new.norm().clamp(min=1e-8).item()
    return diff_norm / new_norm
