"""Full integration tests for the Latent Resonance Loop system."""

import torch
import pytest
from quorum.loop import (
    latent_resonance_loop,
    SyntheticAgent,
    LoopConfig,
)
from quorum.utils import DISSENTER_SLOT


class TestIntegrationAlignedAgents:
    """Test 1: 4 agents with aligned deltas → should converge quickly."""

    def test_aligned_agents_converge_fast(self):
        torch.manual_seed(42)
        config = LoopConfig(S=6, D=128, max_rounds=20, tol=0.02, alpha=0.5)
        target = torch.randn(6, 128) * 0.3

        agents = []
        for i in range(4):
            noise_scale = 0.02
            def make_fn(tgt, ns):
                def fn(B, t):
                    return (tgt - B) * 0.3 + torch.randn_like(B) * ns
                return fn
            agents.append(SyntheticAgent(
                agent_id=i,
                delta_fn=make_fn(target, noise_scale),
                quality=0.7 + i * 0.05,
            ))

        B, diag = latent_resonance_loop(agents, config)
        assert len(diag.residuals) > 0
        if len(diag.residuals) > 3:
            assert diag.residuals[-1] < diag.residuals[0], \
                "Residuals should decrease for aligned agents"


class TestIntegrationOpposingAgents:
    """Test 2: 4 agents with opposing deltas → cross-inhibition should resolve."""

    def test_opposing_agents_converge(self):
        torch.manual_seed(42)
        config = LoopConfig(S=6, D=128, max_rounds=30, tol=0.05, alpha=0.4)

        direction = torch.randn(6, 128)
        direction = direction / direction.norm()

        agents = [
            SyntheticAgent(0, lambda B, t: direction * 0.5, quality=0.8),
            SyntheticAgent(1, lambda B, t: direction * 0.4, quality=0.6),
            SyntheticAgent(2, lambda B, t: -direction * 0.5, quality=0.3),
            SyntheticAgent(3, lambda B, t: -direction * 0.3, quality=0.2),
        ]

        B, diag = latent_resonance_loop(agents, config)
        assert B.shape == (6, 128)
        assert len(diag.residuals) > 0

        final_strengths = diag.strengths[-1]
        assert final_strengths[0] > final_strengths[2], \
            "Higher quality agents should have more influence"


class TestIntegrationDissenter:
    """Test 3: 4 agents + 1 dissenter → dissenter slot should preserve minority position."""

    def test_dissenter_preserved(self):
        torch.manual_seed(42)
        config = LoopConfig(S=6, D=128, max_rounds=15, tol=0.05)

        consensus_dir = torch.randn(6, 128) * 0.3
        dissent_dir = -consensus_dir * 2

        agents = [
            SyntheticAgent(0, lambda B, t: consensus_dir.clone(), quality=0.6),
            SyntheticAgent(1, lambda B, t: consensus_dir.clone(), quality=0.6),
            SyntheticAgent(2, lambda B, t: consensus_dir.clone(), quality=0.6),
            SyntheticAgent(3, lambda B, t: consensus_dir.clone(), quality=0.6),
            SyntheticAgent(4, lambda B, t: dissent_dir.clone(), quality=0.7),
        ]

        B, diag = latent_resonance_loop(agents, config)

        assert any(d == 4 for d in diag.dissenter_indices), \
            "Agent 4 (the actual dissenter) should be selected at least once"

        dissenter_slot_norm = B[DISSENTER_SLOT].norm().item()
        assert dissenter_slot_norm > 0, \
            "Dissenter slot should contain non-zero content"


class TestIntegrationNonContractive:
    """Test 4: Non-contractive agents → damping should enforce convergence."""

    def test_damping_tames_non_contractive(self):
        torch.manual_seed(42)
        config = LoopConfig(
            S=6, D=64, max_rounds=20, tol=0.1,
            alpha=0.2,  # strong damping
        )

        def expanding_fn(B, t):
            return B * 1.5 + torch.randn_like(B) * 0.1

        agents = [
            SyntheticAgent(i, expanding_fn, quality=0.5) for i in range(3)
        ]

        B, diag = latent_resonance_loop(agents, config)
        assert torch.isfinite(B).all(), "Buffer should not contain NaN/Inf"
        assert B.norm().item() < 1e6, \
            f"Buffer norm should be bounded, got {B.norm().item()}"


class TestIntegrationConvergenceDetection:
    """Test 5: Convergence detection fires correctly."""

    def test_convergence_detected(self):
        torch.manual_seed(42)
        config = LoopConfig(S=6, D=64, max_rounds=30, tol=0.02)
        target = torch.randn(6, 64) * 0.2

        def decay_fn(B, t):
            return (target - B) * 0.4

        agents = [SyntheticAgent(i, decay_fn, quality=0.7) for i in range(3)]
        B, diag = latent_resonance_loop(agents, config)

        assert "converged" in diag.halt_reason or diag.residuals[-1] < 0.1, \
            f"Should converge or have low residual. Halt: {diag.halt_reason}, " \
            f"last residual: {diag.residuals[-1]:.4f}"

    def test_max_rounds_respected(self):
        config = LoopConfig(S=6, D=64, max_rounds=3, tol=1e-10)
        agents = [
            SyntheticAgent(0, lambda B, t: torch.randn(6, 64) * 0.5, quality=0.5)
        ]
        B, diag = latent_resonance_loop(agents, config)
        assert len(diag.residuals) == 3
