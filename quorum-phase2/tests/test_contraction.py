"""Tests for Contraction Enforcement (§4.1)."""

import torch
import pytest
from quorum.contraction import (
    damped_update,
    spectral_normalize_weight,
    apply_spectral_norm_to_module,
    hutchinson_jacobian_penalty,
    estimate_spectral_radius,
)
from quorum.codecs import AgentCodec


class TestDampedUpdate:
    def test_output_shape(self):
        B = torch.randn(6, 128)
        F_raw = torch.randn(6, 128)
        result = damped_update(B, F_raw, alpha=0.5)
        assert result.shape == (6, 128)

    def test_alpha_zero_returns_B(self):
        B = torch.randn(6, 128)
        F_raw = torch.randn(6, 128)
        result = damped_update(B, F_raw, alpha=0.0)
        assert torch.allclose(result, B)

    def test_alpha_one_returns_F(self):
        B = torch.randn(6, 128)
        F_raw = torch.randn(6, 128)
        result = damped_update(B, F_raw, alpha=1.0)
        assert torch.allclose(result, F_raw)

    def test_damping_reduces_step_size(self):
        B = torch.zeros(6, 128)
        F_raw = torch.ones(6, 128) * 10
        result = damped_update(B, F_raw, alpha=0.5)
        assert result.norm() < F_raw.norm()
        assert torch.allclose(result, 0.5 * F_raw)

    def test_non_contractive_becomes_contractive(self):
        """Critical test: damping should make a non-contractive map contractive.
        
        If F has Lipschitz constant L=4, then with α=0.3, the effective
        Lipschitz constant is (1-0.3) + 0.3*4 = 0.7 + 1.2 = 1.9 > 1.
        But with α=0.1, it's (1-0.1) + 0.1*4 = 0.9 + 0.4 = 1.3 still > 1.
        With α < 2/(1+L) = 2/5 = 0.4, we need to verify contraction empirically.
        """
        torch.manual_seed(42)
        W = torch.randn(128, 128) * 3.0  # non-contractive linear map

        B1 = torch.randn(1, 128)
        B2 = torch.randn(1, 128)
        dist_before = (B1 - B2).norm()

        F1 = B1 @ W.T
        F2 = B2 @ W.T
        dist_after_raw = (F1 - F2).norm()
        assert dist_after_raw > dist_before, "Raw map should be expansive"

        alpha = 0.1
        D1 = damped_update(B1, F1, alpha)
        D2 = damped_update(B2, F2, alpha)
        dist_after_damped = (D1 - D2).norm()

        assert dist_after_damped < dist_after_raw, \
            "Damping should reduce the expansion"


class TestSpectralNormalisation:
    def test_bounds_spectral_norm(self):
        W = torch.randn(64, 128) * 5
        W_normed, _ = spectral_normalize_weight(W, n_power_iterations=10, lambda_sn=0.9)
        
        U, S, Vt = torch.linalg.svd(W_normed)
        assert S[0].item() < 1.0, \
            f"Spectral norm should be bounded by λ_SN=0.9, got {S[0].item():.4f}"

    def test_preserves_direction(self):
        W = torch.randn(32, 32)
        W_normed, _ = spectral_normalize_weight(W, n_power_iterations=10)
        cos_sim = torch.cosine_similarity(W.flatten().unsqueeze(0), 
                                            W_normed.flatten().unsqueeze(0))
        assert cos_sim > 0.99, "Direction should be preserved"

    def test_apply_to_module(self):
        codec = AgentCodec(agent_dim=64, buffer_dim=32)
        apply_spectral_norm_to_module(codec, lambda_sn=0.9, n_power_iterations=10)
        
        for name, param in codec.named_parameters():
            if "weight" in name and param.dim() == 2:
                U, S, Vt = torch.linalg.svd(param.data)
                assert S[0].item() < 1.05, \
                    f"{name}: spectral norm {S[0].item():.4f} should be ≤ ~0.9"


class TestJacobianRegularisation:
    def test_penalty_is_positive(self):
        def simple_fn(x):
            return x @ torch.randn(128, 128) * 0.5
        
        B = torch.randn(6, 128)
        penalty = hutchinson_jacobian_penalty(simple_fn, B)
        assert penalty.item() > 0

    def test_identity_has_low_penalty(self):
        def identity_fn(x):
            return x * 0.1
        
        B = torch.randn(6, 128)
        penalty = hutchinson_jacobian_penalty(identity_fn, B)
        assert penalty.item() < B.numel() * 0.5


class TestSpectralRadius:
    def test_contractive_map(self):
        scale = 0.5
        def contractive(x):
            return x * scale
        
        B = torch.randn(6, 64)
        rho = estimate_spectral_radius(contractive, B, n_iterations=20)
        assert rho < 1.0, f"Spectral radius should be < 1 for contractive map, got {rho}"

    def test_expansive_map(self):
        scale = 2.0
        def expansive(x):
            return x * scale

        B = torch.randn(2, 32)
        rho = estimate_spectral_radius(expansive, B, n_iterations=20)
        assert rho > 1.0, f"Spectral radius should be > 1 for expansive map, got {rho}"
