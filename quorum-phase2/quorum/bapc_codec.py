"""Behaviourally-Aligned Prefix Codec (BAPC).

Linear slot-headed encoder/decoder that:
  - Extracts from a middle layer (not final) to avoid catastrophic anisotropy
  - Applies isotropy calibration (ABTT) before encoding
  - Encodes to S buffer slots via per-slot linear maps
  - Decodes buffer slots to prefix embeddings in the agent's embedding space
  - Uses per-slot sigmoid gates to control injection strength

The decoder outputs prefix embeddings (d_emb), not hidden-space vectors.
These are prepended to the agent's input sequence for generation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional

from .isocal import IsotropyCalibrator, IsocalConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class BAPCCodec(nn.Module):
    """Behaviourally-Aligned Prefix Codec for one agent.

    Encoder: IsoCal(h) -> S linear maps -> RMSNorm -> [S, D]
    Decoder: [S, D] -> S linear maps -> gate -> [S, d_emb]

    The encode/decode interface matches AgentCodec so this is a drop-in
    replacement for the loop infrastructure.
    """

    def __init__(
        self,
        agent_dim: int,
        buffer_dim: int,
        embed_dim: int | None = None,
        num_slots: int = 6,
        isocal_config: IsocalConfig | None = None,
    ):
        super().__init__()
        self.agent_dim = agent_dim
        self.buffer_dim = buffer_dim
        self.embed_dim = embed_dim or agent_dim
        self.num_slots = num_slots

        self.isocal = IsotropyCalibrator(isocal_config or IsocalConfig())

        self.encoders = nn.ModuleList([
            nn.Linear(agent_dim, buffer_dim) for _ in range(num_slots)
        ])
        self.enc_norms = nn.ModuleList([
            RMSNorm(buffer_dim) for _ in range(num_slots)
        ])

        self.decoders = nn.ModuleList([
            nn.Linear(buffer_dim, self.embed_dim) for _ in range(num_slots)
        ])
        self.gate_projectors = nn.ModuleList([
            nn.Linear(buffer_dim, 1) for _ in range(num_slots)
        ])

        self._init_weights()

    def _init_weights(self):
        for enc in self.encoders:
            nn.init.orthogonal_(enc.weight)
            nn.init.zeros_(enc.bias)
        for dec in self.decoders:
            nn.init.orthogonal_(dec.weight)
            nn.init.zeros_(dec.bias)
        for gate in self.gate_projectors:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, 1.0)

    def calibrate_isocal(self, states: torch.Tensor) -> None:
        """Compute isotropy calibration statistics from a corpus."""
        self.isocal.calibrate(states)

    def encode(self, h_raw: torch.Tensor) -> torch.Tensor:
        """Encode a hidden state to buffer space [S, D].

        Args:
            h_raw: [..., agent_dim] raw hidden state from extraction layer.
        Returns:
            [S, D] buffer-space encoding (or [..., S, D] if batched).
        """
        h = self.isocal.transform(h_raw) if self.isocal.is_calibrated else h_raw
        slots = []
        for s in range(self.num_slots):
            z_s = self.enc_norms[s](self.encoders[s](h))
            slots.append(z_s)
        return torch.stack(slots, dim=-2)  # [..., S, D]

    def decode(self, buffer: torch.Tensor) -> torch.Tensor:
        """Decode buffer state to prefix embeddings [S, d_emb].

        Args:
            buffer: [..., S, D] or [S, D] buffer state.
        Returns:
            [..., S, d_emb] prefix embeddings for injection.
        """
        prefixes = []
        for s in range(self.num_slots):
            b_s = buffer[..., s, :] if buffer.dim() > 1 and buffer.shape[-2] == self.num_slots else buffer
            gate = torch.sigmoid(self.gate_projectors[s](b_s))
            p_s = gate * self.decoders[s](b_s)
            prefixes.append(p_s)
        return torch.stack(prefixes, dim=-2)  # [..., S, d_emb]

    def encode_single_slot(self, h_raw: torch.Tensor, slot: int) -> torch.Tensor:
        """Encode to a single buffer slot [D]."""
        h = self.isocal.transform(h_raw) if self.isocal.is_calibrated else h_raw
        return self.enc_norms[slot](self.encoders[slot](h))

    def get_encoder_weights(self) -> list[torch.Tensor]:
        """Return encoder weight matrices for orthogonality regularisation."""
        return [enc.weight for enc in self.encoders]
