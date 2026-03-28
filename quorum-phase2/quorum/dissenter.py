"""Dissenter Channel — Carol's Structural Immunity (§5).

Slot 5 operates under different rules:
  1. No cross-inhibition or repulsion penalty
  2. Divergence-maximizing write access (most divergent agent writes)
  3. Quality gate (confidence-weighted magnitude)
  4. Mandatory read (all agents attend to Slot 5)

[SPECULATIVE] — no published system implements structural immunity for
minority positions in a shared latent buffer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass

from .utils import DISSENTER_SLOT


@dataclass
class DissenterConfig:
    """Configuration for the dissenter channel."""
    epsilon_floor: float = 0.01
    quality_threshold: float = 0.1
    slot_index: int = DISSENTER_SLOT


def select_dissenter(deltas: List[torch.Tensor]) -> int:
    """Select the most divergent agent as the round's dissenter (§5.1).

    The agent whose delta has the lowest cosine similarity to the consensus
    (mean of all agents' deltas) writes to the dissenter slot.

    Args:
        deltas: List of N tensors, each [S, D].

    Returns:
        Index of the most divergent agent.
    """
    if len(deltas) == 1:
        return 0

    consensus = torch.stack(deltas).mean(dim=0)  # [S, D]
    consensus_flat = consensus.flatten()
    c_norm = consensus_flat.norm()

    if c_norm < 1e-8:
        return 0

    similarities = []
    for d in deltas:
        d_flat = d.flatten()
        sim = F.cosine_similarity(d_flat.unsqueeze(0), consensus_flat.unsqueeze(0)).item()
        similarities.append(sim)

    return int(torch.tensor(similarities).argmin().item())


def compute_dissenter_delta(
    deltas: List[torch.Tensor],
    qualities: torch.Tensor,
    dissenter_idx: int,
    slot_index: int = DISSENTER_SLOT,
    epsilon_floor: float = 0.01,
) -> torch.Tensor:
    """Compute the dissenter's contribution to Slot 5 (§5.1).

    The dissenter writes with quality-gated magnitude:
      δ_dissenter^effective = max(q_i, ε_floor) · δ_i[slot_5]

    Args:
        deltas: List of N tensors, each [S, D].
        qualities: [N] quality scores.
        dissenter_idx: Index of the selected dissenter.
        slot_index: The dissenter slot index (default: 5).
        epsilon_floor: Minimum quality weight.

    Returns:
        [D] dissenter delta for the slot.
    """
    delta = deltas[dissenter_idx][slot_index]
    quality = max(qualities[dissenter_idx].item(), epsilon_floor)
    return quality * delta


def apply_dissenter_rules(
    resolved: torch.Tensor,
    deltas: List[torch.Tensor],
    qualities: torch.Tensor,
    config: Optional[DissenterConfig] = None,
) -> tuple[torch.Tensor, int]:
    """Apply dissenter channel rules to a resolved buffer update.

    Replaces the dissenter slot with the most divergent agent's
    quality-gated contribution.

    Args:
        resolved: [S, D] resolved delta from TIES-Resolve.
        deltas: List of N raw deltas [S, D].
        qualities: [N] quality scores.
        config: Dissenter configuration.

    Returns:
        (updated_resolved, dissenter_idx)
    """
    if config is None:
        config = DissenterConfig()

    dissenter_idx = select_dissenter(deltas)
    dissenter_delta = compute_dissenter_delta(
        deltas, qualities, dissenter_idx,
        slot_index=config.slot_index,
        epsilon_floor=config.epsilon_floor,
    )

    resolved = resolved.clone()
    resolved[config.slot_index] = dissenter_delta
    return resolved, dissenter_idx


def dissent_trust(
    confidence: float,
    calibration_ece: float,
) -> float:
    """Weight dissent by historical reliability (§5.2).

    [SPECULATIVE] — calibration scoring is well-established, but applying it
    to dynamic multi-agent dissent weighting is novel.

    trust = confidence * (1 - calibration_ece)
    High confidence + low ECE = high trust.
    """
    return confidence * (1.0 - calibration_ece)
