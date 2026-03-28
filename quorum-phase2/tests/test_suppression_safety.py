"""Tests for Suppression Safety / Collapse Prevention (§3.3)."""

import torch
import pytest
from quorum.suppression_safety import (
    apply_activity_floor,
    homeostatic_sigma,
    apply_inhibition_cap,
    apply_refractory_period,
    safe_bee_step,
    SuppressionSafetyConfig,
    SuppressionSafetyState,
)


class TestActivityFloor:
    def test_clamps_below_floor(self):
        strengths = torch.tensor([0.005, 0.005, 0.99])
        result = apply_activity_floor(strengths, epsilon_floor=0.01)
        assert (result >= 0.009).all(), f"Strengths {result} should be near floor after renorm"

    def test_normalised(self):
        strengths = torch.tensor([0.005, 0.005, 0.99])
        result = apply_activity_floor(strengths)
        assert abs(result.sum().item() - 1.0) < 1e-5

    def test_no_change_above_floor(self):
        strengths = torch.tensor([0.3, 0.3, 0.4])
        result = apply_activity_floor(strengths, epsilon_floor=0.01)
        assert torch.allclose(result, strengths, atol=1e-5)


class TestHomeostaticSigma:
    def test_high_activity_returns_sigma_base(self):
        strengths = torch.ones(4) / 4
        deltas = torch.randn(4, 6, 128) * 2.0
        sigma = homeostatic_sigma(strengths, deltas, sigma_base=0.3, A_min=0.1)
        assert sigma > 0.2, f"High activity should give sigma near base, got {sigma}"

    def test_low_activity_reduces_sigma(self):
        strengths = torch.ones(4) / 4
        deltas = torch.zeros(4, 6, 128)
        sigma = homeostatic_sigma(strengths, deltas, sigma_base=0.3, A_min=0.1)
        assert sigma < 0.2, f"Zero activity should reduce sigma, got {sigma}"


class TestInhibitionCap:
    def test_caps_at_strength(self):
        strengths = torch.tensor([0.1, 0.2, 0.3])
        inhibition = torch.tensor([0.5, 0.1, 1.0])
        capped = apply_inhibition_cap(strengths, inhibition)
        assert (capped <= strengths).all()

    def test_no_cap_when_low(self):
        strengths = torch.tensor([0.5, 0.5])
        inhibition = torch.tensor([0.1, 0.1])
        capped = apply_inhibition_cap(strengths, inhibition)
        assert torch.allclose(capped, inhibition)


class TestRefractoryPeriod:
    def test_refractory_set_on_drop(self):
        state = SuppressionSafetyState(3)
        strengths = torch.tensor([0.03, 0.5, 0.47])
        protected = apply_refractory_period(strengths, state)
        assert not protected.any(), "First round: no one should be protected yet"

    def test_refractory_protects_next_round(self):
        state = SuppressionSafetyState(3)
        s1 = torch.tensor([0.03, 0.5, 0.47])
        _ = apply_refractory_period(s1, state)
        s2 = torch.tensor([0.02, 0.5, 0.48])
        protected = apply_refractory_period(s2, state)
        assert protected[0], "Agent 0 should be protected after dropping below threshold"


class TestSafeBeeStep:
    def test_prevents_amplitude_death(self):
        """No agent should collapse to zero strength after many rounds."""
        config = SuppressionSafetyConfig(epsilon_floor=0.01, sigma_base=0.5)
        state = SuppressionSafetyState(4, config)
        strengths = torch.ones(4) / 4
        qualities = torch.tensor([0.8, 0.1, 0.1, 0.1])
        deltas = torch.randn(4, 6, 128)

        for _ in range(100):
            strengths = safe_bee_step(strengths, qualities, deltas, state)

        assert (strengths >= config.epsilon_floor - 0.001).all(), \
            f"All strengths should remain above floor, got {strengths}"

    def test_prevents_winner_lock(self):
        """With safety mechanisms, one agent should not permanently hold 1.0."""
        config = SuppressionSafetyConfig(epsilon_floor=0.01, sigma_base=0.3)
        state = SuppressionSafetyState(4, config)
        strengths = torch.ones(4) / 4
        qualities = torch.tensor([0.9, 0.3, 0.3, 0.3])
        deltas = torch.randn(4, 6, 128) * 0.5

        for _ in range(100):
            strengths = safe_bee_step(strengths, qualities, deltas, state)

        assert strengths.max() < 0.99, \
            f"No single agent should lock at 1.0, max={strengths.max():.3f}"
        assert (strengths >= 0.009).all(), \
            f"No agent should be completely dead: {strengths}"
