"""Per-Agent Encoder/Decoder Codecs (§2.2).

Each agent i has an encoder E_i: ℝ^{d_i} → ℝ^D and decoder D_i: ℝ^D → ℝ^{d_i}.
Encoder: E_i(h) = LayerNorm(W₂ · ReLU(W₁ · h + b₁) + b₂)
Decoder: D_i(z) = W₄ · ReLU(W₃ · z + b₃) + b₄
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class AgentEncoder(nn.Module):
    """Encoder E_i: ℝ^{d_i} → ℝ^D (§2.2).

    Two-layer MLP with LayerNorm to project agent hidden states into the
    shared buffer space. Output lies on the unit-norm hypersphere in ℝ^D.
    """

    def __init__(self, source_dim: int, buffer_dim: int):
        super().__init__()
        self.source_dim = source_dim
        self.buffer_dim = buffer_dim
        self.fc1 = nn.Linear(source_dim, buffer_dim)
        self.fc2 = nn.Linear(buffer_dim, buffer_dim)
        self.layer_norm = nn.LayerNorm(buffer_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Encode agent hidden state to buffer space.

        Args:
            h: Agent hidden state, shape [..., source_dim].
        Returns:
            Buffer-space encoding, shape [..., buffer_dim].
        """
        x = torch.relu(self.fc1(h))
        x = self.fc2(x)
        return self.layer_norm(x)


class AgentDecoder(nn.Module):
    """Decoder D_i: ℝ^D → ℝ^{d_i} (§2.2).

    Two-layer MLP that projects buffer vectors back into the agent's
    native hidden space.
    """

    def __init__(self, buffer_dim: int, target_dim: int):
        super().__init__()
        self.buffer_dim = buffer_dim
        self.target_dim = target_dim
        self.fc3 = nn.Linear(buffer_dim, target_dim)
        self.fc4 = nn.Linear(target_dim, target_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode buffer vector to agent hidden space.

        Args:
            z: Buffer-space vector, shape [..., buffer_dim].
        Returns:
            Agent-space vector, shape [..., target_dim].
        """
        x = torch.relu(self.fc3(z))
        return self.fc4(x)


class AgentCodec(nn.Module):
    """Combined encoder-decoder codec for a single agent (§2.2).

    Wraps AgentEncoder and AgentDecoder into a single module with convenience
    methods for roundtrip encoding and slot-level operations.
    """

    def __init__(self, agent_dim: int, buffer_dim: int):
        super().__init__()
        self.agent_dim = agent_dim
        self.buffer_dim = buffer_dim
        self.encoder = AgentEncoder(agent_dim, buffer_dim)
        self.decoder = AgentDecoder(buffer_dim, agent_dim)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Encode agent hidden state → buffer space."""
        return self.encoder(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode buffer space → agent hidden space."""
        return self.decoder(z)

    def roundtrip(self, h: torch.Tensor) -> torch.Tensor:
        """Encode then decode (for measuring reconstruction loss)."""
        return self.decode(self.encode(h))

    def encode_buffer(self, buffer_state: torch.Tensor) -> torch.Tensor:
        """Decode the full buffer [S, D] into agent space [S, d_i]."""
        return self.decoder(buffer_state)

    def decode_buffer(self, agent_states: torch.Tensor) -> torch.Tensor:
        """Encode agent states [S, d_i] into buffer space [S, D]."""
        return self.encoder(agent_states)
