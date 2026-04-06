"""TIES-Resolve Multi-Writer Mechanism (§2.3).

Adapts TIES-Merging from static weight merging to dynamic activation merging.
Three steps: TRIM → ELECT SIGN → DISJOINT MERGE.

[SPECULATIVE] — TIES-Merging for static weights is proven (NeurIPS 2023).
Adapting it to dynamic per-inference activations is speculative.
"""

from __future__ import annotations

import torch
from typing import List


def ties_resolve(deltas: List[torch.Tensor], density: float = 0.3) -> torch.Tensor:
    """TIES-Resolve: combine multiple agent deltas into a single resolved delta.

    Implements the three-step TIES-Merging process adapted for dynamic activations:
      1. TRIM — zero out low-magnitude components (keep top density%)
      2. ELECT SIGN — majority sign wins per dimension (magnitude-weighted)
      3. DISJOINT MERGE — average only sign-agreeing deltas

    Args:
        deltas: List of N tensors, each shape [S, D].
        density: Fraction of components to keep (0.0–1.0). Default 0.3 means
                 70% of each delta is zeroed.

    Returns:
        Resolved delta, shape [S, D].
    """
    if len(deltas) == 0:
        raise ValueError("Need at least one delta")
    if len(deltas) == 1:
        return deltas[0].clone()

    trimmed = _trim(deltas, density)
    stacked = torch.stack(trimmed)  # [N, S, D]
    elected_sign = _elect_sign(stacked)
    merged = _disjoint_merge(stacked, elected_sign)
    return merged


def _trim(deltas: List[torch.Tensor], density: float) -> List[torch.Tensor]:
    """STEP 1 — TRIM: Zero out low-magnitude components.

    Keep only top density% of each delta by absolute value.
    """
    trimmed = []
    for d in deltas:
        d = d.clone()
        if density >= 1.0:
            trimmed.append(d)
            continue
        abs_vals = d.abs()
        threshold = torch.quantile(abs_vals.flatten(), 1.0 - density)
        mask = abs_vals >= threshold
        trimmed.append(d * mask)
    return trimmed


def _elect_sign(stacked: torch.Tensor) -> torch.Tensor:
    """STEP 2 — ELECT SIGN: Majority sign wins per dimension.

    Uses magnitude-weighted sign votes so that larger-magnitude deltas
    have more influence on the elected sign.

    Args:
        stacked: [N, S, D] trimmed deltas.

    Returns:
        elected_sign: [S, D] tensor of {-1, 0, +1}.
    """
    sign_votes = torch.sign(stacked)  # {-1, 0, +1}
    magnitude_weighted_signs = (sign_votes * stacked.abs()).sum(dim=0)
    return torch.sign(magnitude_weighted_signs)


def _disjoint_merge(stacked: torch.Tensor, elected_sign: torch.Tensor) -> torch.Tensor:
    """STEP 3 — DISJOINT MERGE: Average only sign-agreeing deltas.

    For each (slot, dimension), only deltas whose sign matches the elected
    sign are included in the average. Disagreeing deltas are excluded, not
    averaged — this is cross-inhibition at the representation level.

    Args:
        stacked: [N, S, D] trimmed deltas.
        elected_sign: [S, D] elected signs.

    Returns:
        merged: [S, D] resolved delta.
    """
    sign_votes = torch.sign(stacked)
    agreement_mask = (sign_votes == elected_sign.unsqueeze(0))  # [N, S, D]
    agreement_count = agreement_mask.float().sum(dim=0).clamp(min=1)
    merged = (stacked * agreement_mask.float()).sum(dim=0) / agreement_count
    return merged
