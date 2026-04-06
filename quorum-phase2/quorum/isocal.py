"""Isotropy Calibration — All-but-the-Top anisotropy removal.

Transformer hidden states in upper layers occupy a narrow cone where random
pairs have >0.95 cosine similarity.  This module removes that shared direction
so downstream codecs operate in a space with real discriminative range.

Three-step transform (Mu et al., "All-but-the-Top"):
  1. Mean-centre
  2. Remove top-k principal components (the shared cone directions)
  3. Diagonal-whiten (per-dimension normalisation)
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field


@dataclass
class IsocalConfig:
    k: int = 16
    eps: float = 1e-4


class IsotropyCalibrator:
    """Non-parametric anisotropy removal computed from a calibration corpus."""

    def __init__(self, config: IsocalConfig | None = None):
        self.config = config or IsocalConfig()
        self.mean: torch.Tensor | None = None
        self.top_pcs: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self._calibrated = False

    def calibrate(self, states: torch.Tensor) -> None:
        """Compute calibration statistics from a corpus of hidden states.

        Args:
            states: [N, d] hidden states from one extraction layer.
        """
        k = min(self.config.k, states.shape[1], states.shape[0])

        self.mean = states.mean(dim=0)
        centred = states - self.mean

        _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
        self.top_pcs = Vt[:k]  # [k, d]

        projected = centred - (centred @ self.top_pcs.T) @ self.top_pcs
        self.std = projected.std(dim=0)
        self.std = self.std.clamp(min=self.config.eps)

        self._calibrated = True

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def transform(self, h: torch.Tensor) -> torch.Tensor:
        """Apply isotropy calibration to hidden states.

        Args:
            h: [..., d] hidden state(s).
        Returns:
            Calibrated hidden state(s), same shape.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before transform()")

        mean = self.mean.to(h.device)
        top_pcs = self.top_pcs.to(h.device)
        std = self.std.to(h.device)

        out = h - mean
        out = out - (out @ top_pcs.T) @ top_pcs
        out = out / std
        return out

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "top_pcs": self.top_pcs,
            "std": self.std,
            "config": self.config,
        }

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.top_pcs = d["top_pcs"]
        self.std = d["std"]
        self.config = d.get("config", IsocalConfig())
        self._calibrated = True
