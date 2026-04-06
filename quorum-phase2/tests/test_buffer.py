"""Tests for Universal Latent Buffer (§2.1)."""

import torch
import pytest
from quorum.buffer import UniversalLatentBuffer, BufferSnapshot
from quorum.utils import SlotName


class TestUniversalLatentBuffer:
    def test_creation_default(self):
        buf = UniversalLatentBuffer()
        assert buf.shape == (6, 512)
        assert buf.data.shape == (6, 512)
        assert torch.allclose(buf.data, torch.zeros(6, 512))

    def test_creation_custom_dims(self):
        buf = UniversalLatentBuffer(num_slots=4, buffer_dim=128)
        assert buf.shape == (4, 128)

    def test_initialize_zeros(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        buf.initialize()
        assert torch.allclose(buf.data, torch.zeros(6, 64))
        assert len(buf.get_history()) == 1

    def test_initialize_with_values(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        values = torch.randn(6, 64)
        buf.initialize(values)
        assert torch.allclose(buf.data, values)

    def test_read_full_buffer(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        values = torch.randn(6, 64)
        buf.initialize(values)
        read = buf.read()
        assert torch.allclose(read, values)
        read[0, 0] = 999.0
        assert not torch.allclose(buf.data, read), "read() should return a copy"

    def test_read_specific_slot(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        values = torch.randn(6, 64)
        buf.initialize(values)
        slot = buf.read(SlotName.FACTUAL_GROUNDING)
        assert torch.allclose(slot, values[0])

    def test_get_slot(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        values = torch.randn(6, 64)
        buf.initialize(values)
        for s in SlotName:
            assert torch.allclose(buf.get_slot(s), values[s])

    def test_write_delta_full_buffer(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        buf.initialize()
        delta = torch.ones(6, 64)
        buf.write_delta(delta)
        assert torch.allclose(buf.data, delta)

    def test_write_delta_specific_slot(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        buf.initialize()
        delta = torch.ones(64) * 3.0
        buf.write_delta(delta, slot=SlotName.COMPETITIVE_ARENA)
        assert torch.allclose(buf.data[SlotName.COMPETITIVE_ARENA], delta)
        assert torch.allclose(buf.data[0], torch.zeros(64))

    def test_write_delta_accumulates(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        buf.initialize()
        d1 = torch.ones(6, 64)
        d2 = torch.ones(6, 64) * 2
        buf.write_delta(d1)
        buf.write_delta(d2)
        assert torch.allclose(buf.data, d1 + d2)

    def test_history_tracking(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        buf.initialize()
        assert len(buf.get_history()) == 1

        for i in range(5):
            delta = torch.randn(6, 64)
            buf.write_delta(delta)
        
        history = buf.get_history()
        assert len(history) == 6  # init + 5 writes
        for snap in history:
            assert isinstance(snap, BufferSnapshot)
            assert snap.state.shape == (6, 64)

    def test_history_with_diagnostics(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        buf.initialize()
        delta = torch.randn(6, 64)
        agent_deltas = [torch.randn(6, 64) for _ in range(3)]
        buf.write_delta(delta, agent_deltas=agent_deltas, energy=1.5, residual=0.03)
        snap = buf.get_history()[-1]
        assert snap.energy == 1.5
        assert snap.residual == 0.03
        assert len(snap.agent_deltas) == 3

    def test_set_state(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        buf.initialize()
        new_state = torch.randn(6, 64)
        buf.set_state(new_state)
        assert torch.allclose(buf.data, new_state)

    def test_slot_enum_values(self):
        assert SlotName.FACTUAL_GROUNDING == 0
        assert SlotName.REASONING_CHAIN == 1
        assert SlotName.UNCERTAINTY_MAP == 2
        assert SlotName.META_COORDINATION == 3
        assert SlotName.COMPETITIVE_ARENA == 4
        assert SlotName.DISSENTER_CHANNEL == 5
        assert len(SlotName) == 6

    def test_repr(self):
        buf = UniversalLatentBuffer(buffer_dim=64)
        r = repr(buf)
        assert "slots=6" in r
        assert "dim=64" in r
