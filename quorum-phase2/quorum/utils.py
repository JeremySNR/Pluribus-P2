"""Shared utilities for the Quorum Phase 2 system."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from enum import IntEnum
from typing import Optional


class SlotName(IntEnum):
    """Named slots in the Universal Latent Buffer (§2.1)."""
    FACTUAL_GROUNDING = 0
    REASONING_CHAIN = 1
    UNCERTAINTY_MAP = 2
    META_COORDINATION = 3
    COMPETITIVE_ARENA = 4
    DISSENTER_CHANNEL = 5


DEFAULT_NUM_SLOTS = 6
DEFAULT_BUFFER_DIM = 512  # 4096 for production; 512 for testing
DISSENTER_SLOT = SlotName.DISSENTER_CHANNEL


def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Compute L2 norm with clamped denominator for numerical stability."""
    return x.norm(dim=dim).clamp(min=eps)


def safe_cosine_similarity(
    a: torch.Tensor, b: torch.Tensor, dim: int = 0, eps: float = 1e-8
) -> torch.Tensor:
    """Cosine similarity with clamped norms to avoid division by zero."""
    return F.cosine_similarity(a, b, dim=dim, eps=eps)


def log_sum_exp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log-sum-exp (subtracts max before exp)."""
    x_max = x.max(dim=dim, keepdim=True).values
    return x_max.squeeze(dim) + (x - x_max).exp().sum(dim=dim).log()


def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy of a probability distribution."""
    probs = probs.clamp(min=eps)
    return -(probs * probs.log()).sum()
