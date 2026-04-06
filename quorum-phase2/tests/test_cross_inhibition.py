"""Tests for Cross-Inhibition Engine (§3.1, §3.2)."""

import torch
import math
import pytest
from quorum.cross_inhibition import (
    compute_beta,
    hopfield_energy,
    hopfield_update,
    bee_inhibition_step,
    compute_agreement,
    CrossInhibitionConfig,
)


class TestComputeBeta:
    def test_starts_at_beta_min(self):
        beta = compute_beta(0, beta_min=0.1, beta_max=10.0, tau_beta=3.0)
        assert abs(beta - 0.1) < 0.01

    def test_approaches_beta_max(self):
        beta = compute_beta(100, beta_min=0.1, beta_max=10.0, tau_beta=3.0)
        assert abs(beta - 10.0) < 0.1

    def test_exponential_warmup(self):
        betas = [compute_beta(t) for t in range(20)]
        for i in range(len(betas) - 1):
            assert betas[i] <= betas[i + 1], "β should monotonically increase"

    def test_agreement_jump(self):
        beta = compute_beta(0, agreement=0.9, agreement_threshold=0.8)
        assert beta == 10.0

    def test_no_agreement_jump_below_threshold(self):
        beta = compute_beta(0, agreement=0.5, agreement_threshold=0.8)
        assert beta < 1.0


class TestHopfieldEnergy:
    def test_energy_finite(self):
        B_slot = torch.randn(128)
        deltas = torch.randn(4, 128)
        E = hopfield_energy(B_slot, deltas, beta=1.0)
        assert torch.isfinite(E)

    def test_energy_decreases_toward_attractor(self):
        """Energy should be lower when buffer is near a delta."""
        delta = torch.randn(128)
        delta = delta / delta.norm()
        deltas = delta.unsqueeze(0)  # single delta

        E_far = hopfield_energy(torch.zeros(128), deltas, beta=1.0)
        E_near = hopfield_energy(0.5 * delta, deltas, beta=1.0)
        assert E_near < E_far, "Energy should decrease toward the attractor"

    def test_repulsion_penalty_increases_with_alignment(self):
        d1 = torch.randn(128)
        d1 = d1 / d1.norm()
        d2 = d1.clone()  # perfectly aligned
        d3 = -d1  # anti-aligned

        B = torch.zeros(128)
        E_aligned = hopfield_energy(B, torch.stack([d1, d2]), beta=1.0, lambda_repel=1.0)
        E_anti = hopfield_energy(B, torch.stack([d1, d3]), beta=1.0, lambda_repel=1.0)
        assert E_aligned > E_anti, \
            "Aligned proposals should have higher energy due to repulsion"


class TestHopfieldUpdate:
    def test_output_shape(self):
        B_slot = torch.randn(128)
        deltas = torch.randn(4, 128)
        result = hopfield_update(B_slot, deltas, beta=1.0)
        assert result.shape == (128,)

    def test_high_beta_selects_winner(self):
        """Critical test: at high β, the update should be close to the best-matching delta."""
        B_slot = torch.randn(128)
        d1 = B_slot + torch.randn(128) * 0.01  # very close to B
        d2 = -B_slot  # opposite direction
        deltas = torch.stack([d1, d2])

        result = hopfield_update(B_slot, deltas, beta=100.0)
        sim_to_d1 = torch.cosine_similarity(result.unsqueeze(0), d1.unsqueeze(0)).item()
        sim_to_d2 = torch.cosine_similarity(result.unsqueeze(0), d2.unsqueeze(0)).item()
        assert sim_to_d1 > sim_to_d2, "High β should select the closer delta"

    def test_low_beta_averages(self):
        d1 = torch.ones(64)
        d2 = torch.ones(64) * 3
        deltas = torch.stack([d1, d2])
        B_slot = torch.zeros(64)

        result = hopfield_update(B_slot, deltas, beta=0.001)
        expected_avg = (d1 + d2) / 2
        assert torch.allclose(result, expected_avg, atol=0.1), \
            "Low β should approximate averaging"

    def test_competing_proposals_winner_not_average(self):
        """Critical test: two competing proposals with high β → winner, not blend."""
        torch.manual_seed(42)
        d1 = torch.randn(256)
        d1 = d1 / d1.norm()
        d2 = -d1

        B_slot = d1 * 0.1 + torch.randn(256) * 0.01
        deltas = torch.stack([d1, d2])

        result = hopfield_update(B_slot, deltas, beta=50.0)
        sim_d1 = torch.cosine_similarity(result.unsqueeze(0), d1.unsqueeze(0)).item()
        sim_avg = torch.cosine_similarity(
            result.unsqueeze(0), torch.zeros(1, 256)
        ).item()

        assert abs(sim_d1) > 0.5, "Should pick a winner, not average to zero"


class TestBeeInhibition:
    def test_output_shape(self):
        strengths = torch.ones(5) / 5
        qualities = torch.ones(5) * 0.5
        result = bee_inhibition_step(strengths, qualities)
        assert result.shape == (5,)

    def test_normalised(self):
        strengths = torch.ones(5) / 5
        qualities = torch.rand(5)
        result = bee_inhibition_step(strengths, qualities)
        assert abs(result.sum().item() - 1.0) < 1e-5

    def test_quality_dominance(self):
        """Higher quality agent should gain strength over time."""
        strengths = torch.ones(3) / 3
        qualities = torch.tensor([0.9, 0.1, 0.1])

        for _ in range(50):
            strengths = bee_inhibition_step(strengths, qualities)

        assert strengths[0] > strengths[1], \
            "Higher quality agent should dominate"
        assert strengths[0] > strengths[2]

    def test_convergence_to_winner(self):
        """With differing qualities, one agent should eventually dominate."""
        strengths = torch.ones(4) / 4
        qualities = torch.tensor([0.8, 0.3, 0.2, 0.1])

        for _ in range(200):
            strengths = bee_inhibition_step(strengths, qualities)

        assert strengths[0] > 0.4, \
            f"Best agent should dominate, got {strengths}"


class TestComputeAgreement:
    def test_identical_deltas(self):
        d = torch.randn(6, 128)
        agreement = compute_agreement([d, d.clone(), d.clone()])
        assert agreement > 0.99

    def test_opposite_deltas(self):
        d = torch.randn(6, 128)
        agreement = compute_agreement([d, -d])
        assert agreement < -0.9

    def test_single_delta(self):
        d = torch.randn(6, 128)
        assert compute_agreement([d]) == 1.0
