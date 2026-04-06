"""Tests for Dissenter Channel (§5)."""

import torch
import pytest
from quorum.dissenter import (
    select_dissenter,
    compute_dissenter_delta,
    apply_dissenter_rules,
    dissent_trust,
    DissenterConfig,
)
from quorum.utils import DISSENTER_SLOT


class TestSelectDissenter:
    def test_single_agent(self):
        d = torch.randn(6, 128)
        assert select_dissenter([d]) == 0

    def test_most_divergent_selected(self):
        d1 = torch.ones(6, 128)
        d2 = torch.ones(6, 128) * 0.9
        d3 = -torch.ones(6, 128)  # most divergent from mean
        idx = select_dissenter([d1, d2, d3])
        assert idx == 2, f"Agent 3 is most divergent, but got {idx}"

    def test_all_identical(self):
        d = torch.randn(6, 128)
        idx = select_dissenter([d, d.clone(), d.clone()])
        assert idx in [0, 1, 2]  # any is valid

    def test_anti_correlated_pair(self):
        d1 = torch.randn(6, 128)
        d2 = -d1
        idx = select_dissenter([d1, d2])
        assert idx in [0, 1]


class TestComputeDissenterDelta:
    def test_output_shape(self):
        deltas = [torch.randn(6, 128) for _ in range(3)]
        qualities = torch.tensor([0.5, 0.5, 0.5])
        result = compute_dissenter_delta(deltas, qualities, 0)
        assert result.shape == (128,)

    def test_quality_gating(self):
        delta = torch.ones(6, 128)
        deltas = [delta]
        qualities = torch.tensor([0.8])
        result = compute_dissenter_delta(deltas, qualities, 0)
        expected = 0.8 * delta[DISSENTER_SLOT]
        assert torch.allclose(result, expected)

    def test_quality_floor(self):
        delta = torch.ones(6, 128)
        deltas = [delta]
        qualities = torch.tensor([0.001])
        result = compute_dissenter_delta(deltas, qualities, 0, epsilon_floor=0.01)
        expected = 0.01 * delta[DISSENTER_SLOT]
        assert torch.allclose(result, expected)


class TestApplyDissenterRules:
    def test_replaces_slot_5(self):
        resolved = torch.zeros(6, 128)
        deltas = [torch.ones(6, 128), -torch.ones(6, 128)]
        qualities = torch.tensor([0.5, 0.8])
        
        result, idx = apply_dissenter_rules(resolved, deltas, qualities)
        assert result[DISSENTER_SLOT].abs().sum() > 0
        assert (result[:DISSENTER_SLOT] == 0).all()

    def test_dissenter_immune_to_cross_inhibition(self):
        """Slot 5 should preserve the dissenter's contribution."""
        deltas = [torch.randn(6, 128) for _ in range(4)]
        qualities = torch.tensor([0.5, 0.5, 0.5, 0.5])
        
        resolved = torch.randn(6, 128)
        result, dissenter_idx = apply_dissenter_rules(resolved, deltas, qualities)
        
        expected_slot5 = max(qualities[dissenter_idx].item(), 0.01) * deltas[dissenter_idx][DISSENTER_SLOT]
        assert torch.allclose(result[DISSENTER_SLOT], expected_slot5, atol=1e-5)


class TestDissentTrust:
    def test_high_confidence_low_ece(self):
        trust = dissent_trust(0.9, 0.05)
        assert trust > 0.8

    def test_low_confidence(self):
        trust = dissent_trust(0.1, 0.05)
        assert trust < 0.15

    def test_high_ece_reduces_trust(self):
        trust = dissent_trust(0.9, 0.9)
        assert trust < 0.15

    def test_perfect_calibration(self):
        trust = dissent_trust(1.0, 0.0)
        assert trust == 1.0
