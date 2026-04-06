"""Universal Latent Buffer (§2.1).

The buffer is a matrix B ∈ ℝ^{S × D} where S is the number of named slots
and D is the embedding dimension. Each slot is a D-dimensional vector
representing a distinct semantic role.
"""

from __future__ import annotations

import torch
from typing import Optional, List
from dataclasses import dataclass, field

from .utils import SlotName, DEFAULT_NUM_SLOTS, DEFAULT_BUFFER_DIM


@dataclass
class BufferSnapshot:
    """A point-in-time snapshot of the buffer and associated diagnostics."""
    state: torch.Tensor
    round_idx: int
    agent_deltas: Optional[List[torch.Tensor]] = None
    resolved_delta: Optional[torch.Tensor] = None
    energy: Optional[float] = None
    residual: Optional[float] = None


class UniversalLatentBuffer:
    """Universal Latent Buffer — shared state for multi-agent communication.

    Implements §2.1: matrix B ∈ ℝ^{S × D} with S named slots and D dimensions.
    Tracks full history for debugging as specified in §2.4.
    """

    def __init__(
        self,
        num_slots: int = DEFAULT_NUM_SLOTS,
        buffer_dim: int = DEFAULT_BUFFER_DIM,
        device: Optional[torch.device] = None,
    ):
        self.num_slots = num_slots
        self.buffer_dim = buffer_dim
        self.device = device or torch.device("cpu")
        self._data = torch.zeros(num_slots, buffer_dim, device=self.device)
        self._history: List[BufferSnapshot] = []
        self._round_idx = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_slots, self.buffer_dim)

    @property
    def data(self) -> torch.Tensor:
        """Return the current buffer state (read-only view)."""
        return self._data

    def initialize(self, values: Optional[torch.Tensor] = None) -> None:
        """Initialize buffer to zeros or provided values."""
        if values is not None:
            assert values.shape == (self.num_slots, self.buffer_dim), (
                f"Expected shape ({self.num_slots}, {self.buffer_dim}), got {values.shape}"
            )
            self._data = values.clone().to(self.device)
        else:
            self._data = torch.zeros(
                self.num_slots, self.buffer_dim, device=self.device
            )
        self._history.clear()
        self._round_idx = 0
        self._record_snapshot()

    def read(self, slot: Optional[SlotName] = None) -> torch.Tensor:
        """Read the full buffer or a specific slot.

        Args:
            slot: If provided, return only that slot's vector [D].
                  If None, return the full buffer [S, D].
        """
        if slot is not None:
            return self._data[slot].clone()
        return self._data.clone()

    def get_slot(self, slot: SlotName) -> torch.Tensor:
        """Return a specific slot's vector [D]."""
        return self._data[slot].clone()

    def write_delta(
        self,
        delta: torch.Tensor,
        slot: Optional[SlotName] = None,
        agent_deltas: Optional[List[torch.Tensor]] = None,
        resolved_delta: Optional[torch.Tensor] = None,
        energy: Optional[float] = None,
        residual: Optional[float] = None,
    ) -> None:
        """Apply a delta update to the buffer.

        Args:
            delta: If slot is None, shape [S, D] applied to full buffer.
                   If slot is given, shape [D] applied to that slot only.
            agent_deltas: Per-agent deltas for history tracking.
            resolved_delta: The resolved delta for history tracking.
            energy: Cross-inhibition energy for diagnostics.
            residual: Convergence residual for diagnostics.
        """
        if slot is not None:
            assert delta.shape == (self.buffer_dim,), (
                f"Expected shape ({self.buffer_dim},), got {delta.shape}"
            )
            self._data[slot] = self._data[slot] + delta
        else:
            assert delta.shape == (self.num_slots, self.buffer_dim), (
                f"Expected shape ({self.num_slots}, {self.buffer_dim}), got {delta.shape}"
            )
            self._data = self._data + delta

        self._round_idx += 1
        self._record_snapshot(agent_deltas, resolved_delta, energy, residual)

    def set_state(self, state: torch.Tensor) -> None:
        """Directly set the buffer state (used by contraction/Anderson)."""
        assert state.shape == (self.num_slots, self.buffer_dim)
        self._data = state.clone().to(self.device)

    def get_history(self) -> List[BufferSnapshot]:
        """Return the full trajectory for post-hoc analysis (§2.4)."""
        return list(self._history)

    def _record_snapshot(
        self,
        agent_deltas: Optional[List[torch.Tensor]] = None,
        resolved_delta: Optional[torch.Tensor] = None,
        energy: Optional[float] = None,
        residual: Optional[float] = None,
    ) -> None:
        self._history.append(
            BufferSnapshot(
                state=self._data.clone(),
                round_idx=self._round_idx,
                agent_deltas=[d.clone() for d in agent_deltas] if agent_deltas else None,
                resolved_delta=resolved_delta.clone() if resolved_delta is not None else None,
                energy=energy,
                residual=residual,
            )
        )

    def __repr__(self) -> str:
        return (
            f"UniversalLatentBuffer(slots={self.num_slots}, dim={self.buffer_dim}, "
            f"round={self._round_idx}, norm={self._data.norm().item():.4f})"
        )
