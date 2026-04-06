"""Tests for Anderson Acceleration (§4.2)."""

import torch
import pytest
from quorum.anderson import AndersonAccelerator, anderson_accelerated_loop


class TestAndersonAccelerator:
    def test_output_shape(self):
        acc = AndersonAccelerator(S=6, D=128)
        B = torch.randn(6, 128)
        F_B = torch.randn(6, 128)
        result = acc.step(B, F_B)
        assert result.shape == (6, 128)

    def test_first_step_returns_f(self):
        acc = AndersonAccelerator(S=6, D=64)
        B = torch.randn(6, 64)
        F_B = torch.randn(6, 64)
        result = acc.step(B, F_B)
        assert torch.allclose(result, F_B)

    def test_reset(self):
        acc = AndersonAccelerator(S=6, D=64)
        B = torch.randn(6, 64)
        F_B = torch.randn(6, 64)
        acc.step(B, F_B)
        acc.step(F_B, B)
        acc.reset()
        assert acc.t == 0

    def test_accelerates_contractive_map(self):
        """Anderson should converge faster than plain iteration on a contractive map."""
        torch.manual_seed(42)
        S, D = 2, 32
        fixed_point = torch.randn(S, D) * 0.5
        contraction_rate = 0.7

        def contractive_map(B):
            return fixed_point + contraction_rate * (B - fixed_point)

        B_plain = torch.randn(S, D)
        B_anderson = B_plain.clone()
        acc = AndersonAccelerator(S, D, m=5)

        plain_residuals = []
        anderson_residuals = []

        for t in range(30):
            F_plain = contractive_map(B_plain)
            plain_residuals.append((F_plain - B_plain).norm().item())
            B_plain = F_plain

            F_anderson = contractive_map(B_anderson)
            B_anderson = acc.step(B_anderson, F_anderson)
            anderson_residuals.append((B_anderson - fixed_point).norm().item())

        assert anderson_residuals[-1] < plain_residuals[-1], \
            "Anderson should converge faster than plain iteration"


class TestAndersonAcceleratedLoop:
    def test_converges_on_known_fixed_point(self):
        torch.manual_seed(42)
        S, D = 2, 32
        fixed_point = torch.randn(S, D) * 0.5

        def contractive_step(B):
            return fixed_point + 0.5 * (B - fixed_point)

        B0 = torch.randn(S, D) * 2
        result, rounds, residual = anderson_accelerated_loop(
            contractive_step, B0, max_rounds=50, tol=1e-4,
        )

        assert residual < 1e-3
        assert torch.allclose(result, fixed_point, atol=0.1)

    def test_fewer_rounds_than_plain(self):
        """Anderson should use fewer rounds than plain iteration."""
        torch.manual_seed(42)
        S, D = 2, 32
        fixed_point = torch.randn(S, D) * 0.5

        def contractive_step(B):
            return fixed_point + 0.7 * (B - fixed_point)

        B0 = torch.randn(S, D)
        _, anderson_rounds, anderson_res = anderson_accelerated_loop(
            contractive_step, B0, max_rounds=100, tol=1e-3,
        )

        B = B0.clone()
        plain_rounds = 0
        for t in range(100):
            B_new = contractive_step(B)
            r = (B_new - B).norm() / B_new.norm().clamp(min=1e-8)
            B = B_new
            plain_rounds = t + 1
            if r < 1e-3:
                break

        assert anderson_rounds <= plain_rounds, \
            f"Anderson ({anderson_rounds}) should be <= plain ({plain_rounds})"

    def test_handles_max_rounds(self):
        def non_converging(B):
            return B + torch.randn_like(B) * 0.1

        B0 = torch.randn(2, 16)
        result, rounds, _ = anderson_accelerated_loop(
            non_converging, B0, max_rounds=5, tol=1e-10,
        )
        assert rounds == 5
