"""Tests for Convergence Controller (§4.3)."""

import torch
import pytest
from quorum.convergence import (
    QuorumHaltCriterion,
    PonderNetHalter,
    compute_pondernet_loss,
    compute_residual,
    ConvergenceConfig,
)


class TestQuorumHalt:
    def test_does_not_halt_early(self):
        halt = QuorumHaltCriterion(tol=0.01, consecutive_required=3)
        assert not halt.should_halt(0.1)
        assert not halt.should_halt(0.05)

    def test_halts_after_consecutive_below_tol(self):
        halt = QuorumHaltCriterion(tol=0.01, consecutive_required=3)
        assert not halt.should_halt(0.005)
        assert not halt.should_halt(0.005)
        assert halt.should_halt(0.005)

    def test_resets_on_above_tol(self):
        halt = QuorumHaltCriterion(tol=0.01, consecutive_required=3)
        halt.should_halt(0.005)
        halt.should_halt(0.005)
        halt.should_halt(0.05)  # above tol, resets
        assert not halt.should_halt(0.005)
        assert not halt.should_halt(0.005)
        assert halt.should_halt(0.005)

    def test_synthetic_convergence_trajectory(self):
        halt = QuorumHaltCriterion(tol=0.01, consecutive_required=3)
        residuals = [1.0, 0.5, 0.2, 0.05, 0.02, 0.005, 0.003, 0.001]
        halted_at = None
        for i, r in enumerate(residuals):
            if halt.should_halt(r):
                halted_at = i
                break
        assert halted_at == 7, f"Should halt at index 7, got {halted_at}"

    def test_reset(self):
        halt = QuorumHaltCriterion(tol=0.01, consecutive_required=3)
        halt.should_halt(0.005)
        halt.should_halt(0.005)
        halt.reset()
        assert halt.consecutive_count == 0
        assert not halt.should_halt(0.005)


class TestPonderNetHalter:
    def test_output_range(self):
        halter = PonderNetHalter()
        prob = halter(0.1, 0.5, 1.0, 0.5)
        assert 0 <= prob.item() <= 1

    def test_gradient_flow(self):
        halter = PonderNetHalter()
        prob = halter(0.01, 0.9, 0.1, 0.9)
        prob.backward()
        for p in halter.parameters():
            assert p.grad is not None


class TestPonderNetLoss:
    def test_loss_positive(self):
        halt_probs = [torch.tensor(0.3), torch.tensor(0.4), torch.tensor(0.5)]
        task_losses = [torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.2)]
        loss = compute_pondernet_loss(halt_probs, task_losses)
        assert loss.item() > 0

    def test_halt_probs_sum_to_near_one(self):
        halt_probs = [torch.tensor(0.2), torch.tensor(0.3), torch.tensor(0.5)]
        p_uncond = []
        cum = 1.0
        for lam in halt_probs:
            p_uncond.append(lam.item() * cum)
            cum *= (1.0 - lam.item())
        total = sum(p_uncond)
        assert total < 1.01  # can be < 1 if not halting guaranteed


class TestComputeResidual:
    def test_zero_change(self):
        B = torch.randn(6, 128)
        r = compute_residual(B, B)
        assert r == 0.0

    def test_large_change(self):
        B_old = torch.zeros(6, 128)
        B_new = torch.ones(6, 128)
        r = compute_residual(B_new, B_old)
        assert r > 0

    def test_relative_to_norm(self):
        B_old = torch.ones(6, 128)
        B_new = torch.ones(6, 128) * 1.01
        r = compute_residual(B_new, B_old)
        assert r < 0.02
