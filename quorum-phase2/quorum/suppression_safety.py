"""Collapse Prevention Mechanisms (§3.3).

Five mechanisms to prevent cross-inhibition from killing all proposals
(amplitude death) or locking into a single agent (winner-lock).

[PLAUSIBLE] — each mechanism has biological analogs; their composition
for multi-agent buffers is untested.
"""

from __future__ import annotations

import torch
import math
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class SuppressionSafetyConfig:
    """Configuration for suppression safety mechanisms."""
    epsilon_floor: float = 0.01
    A_min: float = 0.1
    homeostatic_k: float = 10.0
    sigma_base: float = 0.3
    refractory_threshold: float = 0.05
    winner_lock_threshold: float = 0.95
    winner_lock_rounds: int = 5


class SuppressionSafetyState:
    """Tracks mutable state for suppression safety across rounds."""

    def __init__(self, num_agents: int, config: Optional[SuppressionSafetyConfig] = None):
        self.config = config or SuppressionSafetyConfig()
        self.num_agents = num_agents
        self.refractory_flags = torch.zeros(num_agents, dtype=torch.bool)
        self.winner_lock_counter = torch.zeros(num_agents)
        self.round_idx = 0

    def reset(self) -> None:
        self.refractory_flags.zero_()
        self.winner_lock_counter.zero_()
        self.round_idx = 0


def apply_activity_floor(
    strengths: torch.Tensor,
    epsilon_floor: float = 0.01,
) -> torch.Tensor:
    """Mechanism 1 — Minimum activity floor (§3.3).

    No agent's strength can drop below ε_floor. Guarantees every agent
    retains at least ε_floor influence.

    Args:
        strengths: [N] agent strengths.
        epsilon_floor: Minimum allowed strength.
    Returns:
        Clamped and renormalised strengths.
    """
    clamped = strengths.clamp(min=epsilon_floor)
    return clamped / clamped.sum()


def homeostatic_sigma(
    strengths: torch.Tensor,
    effective_deltas: torch.Tensor,
    sigma_base: float = 0.3,
    A_min: float = 0.1,
    k: float = 10.0,
) -> float:
    """Mechanism 2 — Homeostatic regulation (§3.3).

    Monitor total buffer activity A(t) = Σᵢ ‖δᵢ^effective‖.
    σ_effective(t) = σ₀ · sigmoid(k · (A(t) − A_min))

    When total activity is healthy, σ_effective ≈ σ₀.
    When activity collapses, σ_effective → 0, releasing all inhibition.
    """
    A_t = effective_deltas.norm(dim=-1).sum().item() if effective_deltas.dim() > 1 else effective_deltas.norm().item()
    return sigma_base * torch.sigmoid(torch.tensor(k * (A_t - A_min))).item()


def apply_inhibition_cap(
    strengths: torch.Tensor,
    inhibition: torch.Tensor,
) -> torch.Tensor:
    """Mechanism 4 — Normalized inhibition cap (§3.3).

    Total inhibition on any agent cannot exceed its current activity.
    Prevents inhibition from driving strengths negative.
    """
    capped = torch.min(inhibition, strengths)
    return capped


def apply_refractory_period(
    strengths: torch.Tensor,
    state: SuppressionSafetyState,
) -> torch.Tensor:
    """Mechanism 5 — Refractory period (§3.3).

    After an agent's strength drops below the refractory threshold,
    it cannot be further inhibited for one round.
    """
    below_threshold = strengths < state.config.refractory_threshold
    new_refractory = below_threshold & ~state.refractory_flags

    protected = state.refractory_flags.clone()
    state.refractory_flags = new_refractory

    return protected


def safe_bee_step(
    strengths: torch.Tensor,
    qualities: torch.Tensor,
    effective_deltas: torch.Tensor,
    state: SuppressionSafetyState,
) -> torch.Tensor:
    """Combined suppression-safe bee dynamics update.

    Applies all five mechanisms in order:
      1. Activity floor
      2. Homeostatic sigma adjustment
      3. (β annealing handled externally)
      4. Inhibition cap
      5. Refractory period

    Args:
        strengths: [N] current agent strengths.
        qualities: [N] proposal quality scores.
        effective_deltas: [N, ...] effective deltas (for activity measurement).
        state: Mutable suppression safety state.

    Returns:
        Updated strengths [N].
    """
    cfg = state.config

    sigma_eff = homeostatic_sigma(
        strengths, effective_deltas,
        sigma_base=cfg.sigma_base, A_min=cfg.A_min, k=cfg.homeostatic_k,
    )

    refractory_protected = apply_refractory_period(strengths, state)

    uncommitted = (1.0 - strengths.sum()).clamp(min=0.01)
    recruitment = cfg.sigma_base * 0.33 * qualities * uncommitted  # γ = sigma_base/3
    decay = 0.05 * strengths

    total_strength = strengths.sum()
    cross_inhib = sigma_eff * strengths * (total_strength - strengths)

    cross_inhib = apply_inhibition_cap(strengths, cross_inhib)

    cross_inhib = cross_inhib * (~refractory_protected).float()

    d_strengths = recruitment - decay - cross_inhib
    new_strengths = (strengths + d_strengths).clamp(min=0.0, max=1.0)

    new_strengths = apply_activity_floor(new_strengths, cfg.epsilon_floor)

    state.round_idx += 1
    return new_strengths
