"""Tests for TIES-Resolve Multi-Writer Mechanism (§2.3)."""

import torch
import pytest
from quorum.ties_resolve import ties_resolve, _trim, _elect_sign, _disjoint_merge


class TestTrim:
    def test_zeros_correct_percentage(self):
        torch.manual_seed(42)
        delta = torch.randn(6, 512)
        trimmed = _trim([delta], density=0.3)[0]
        nonzero_frac = (trimmed != 0).float().mean().item()
        assert 0.25 < nonzero_frac < 0.40, \
            f"Expected ~30% nonzero, got {nonzero_frac:.2%}"

    def test_density_1_keeps_all(self):
        delta = torch.randn(6, 512)
        trimmed = _trim([delta], density=1.0)[0]
        assert torch.allclose(trimmed, delta)

    def test_preserves_large_values(self):
        delta = torch.zeros(1, 100)
        delta[0, :10] = 10.0  # large values
        delta[0, 10:] = 0.01  # small values
        trimmed = _trim([delta], density=0.1)[0]
        assert (trimmed[0, :10] != 0).all()


class TestElectSign:
    def test_majority_sign_wins(self):
        d1 = torch.ones(1, 10)
        d2 = torch.ones(1, 10)
        d3 = -torch.ones(1, 10)
        stacked = torch.stack([d1, d2, d3])
        sign = _elect_sign(stacked)
        assert (sign == 1).all(), "2/3 positive should elect positive"

    def test_magnitude_weighted(self):
        d1 = torch.ones(1, 10) * 0.1  # weak positive
        d2 = -torch.ones(1, 10) * 10.0  # strong negative
        stacked = torch.stack([d1, d2])
        sign = _elect_sign(stacked)
        assert (sign == -1).all(), "Stronger magnitude should win"


class TestDisjointMerge:
    def test_excludes_disagreeing_deltas(self):
        d1 = torch.ones(1, 4) * 2.0
        d2 = torch.ones(1, 4) * 3.0
        d3 = -torch.ones(1, 4) * 1.0  # weaker negative, so positive wins by magnitude
        stacked = torch.stack([d1, d2, d3])
        sign = _elect_sign(stacked)
        merged = _disjoint_merge(stacked, sign)
        assert (merged > 0).all(), "d3 (negative) should be excluded"
        expected = (d1 + d2) / 2
        assert torch.allclose(merged, expected, atol=1e-5)


class TestTiesResolve:
    def test_single_delta(self):
        delta = torch.randn(6, 512)
        result = ties_resolve([delta])
        assert torch.allclose(result, delta)

    def test_output_shape(self):
        deltas = [torch.randn(6, 512) for _ in range(5)]
        result = ties_resolve(deltas)
        assert result.shape == (6, 512)

    def test_opposing_deltas_picks_direction(self):
        """Critical test: opposing deltas should result in a direction, not average.
        
        With 3 agents where 2 agree, the majority should win decisively.
        With exactly 2 equal-magnitude opposing agents, TIES correctly produces
        zero (no majority). So we test with asymmetric groups.
        """
        torch.manual_seed(42)
        d1 = torch.ones(6, 128)
        d2 = torch.ones(6, 128) * 0.8
        d3 = -torch.ones(6, 128) * 0.5
        result = ties_resolve([d1, d2, d3], density=1.0)
        assert (result >= 0).all(), \
            "With 2 positive vs 1 weaker negative, positive direction should win"
        assert result.abs().mean() > 0.1, \
            "Result should be non-trivial, not averaged to near-zero"

    def test_aligned_deltas_merged(self):
        d1 = torch.ones(6, 64) * 2.0
        d2 = torch.ones(6, 64) * 4.0
        d3 = torch.ones(6, 64) * 6.0
        result = ties_resolve([d1, d2, d3], density=1.0)
        expected = (d1 + d2 + d3) / 3.0
        assert torch.allclose(result, expected, atol=1e-4)

    def test_sparsity_reduces_output(self):
        deltas = [torch.randn(6, 128) for _ in range(4)]
        result_dense = ties_resolve(deltas, density=0.8)
        result_sparse = ties_resolve(deltas, density=0.2)
        assert result_sparse.abs().mean() < result_dense.abs().mean()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ties_resolve([])
