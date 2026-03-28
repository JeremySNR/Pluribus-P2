"""Tests for Main Latent Resonance Loop (§6)."""

import torch
import pytest
from quorum.loop import (
    latent_resonance_loop,
    SyntheticAgent,
    LoopConfig,
    LoopDiagnostics,
)


def make_constant_agent(agent_id: int, delta: torch.Tensor, quality: float = 0.5):
    """Create a synthetic agent that always proposes the same delta."""
    return SyntheticAgent(
        agent_id=agent_id,
        delta_fn=lambda B, t, d=delta: d.clone(),
        quality=quality,
    )


def make_decaying_agent(agent_id: int, target: torch.Tensor, quality: float = 0.5):
    """Create an agent that proposes deltas toward a fixed target."""
    def delta_fn(B, t):
        return (target - B) * 0.3
    return SyntheticAgent(agent_id=agent_id, delta_fn=delta_fn, quality=quality)


class TestLatentResonanceLoop:
    def test_runs_and_returns(self):
        config = LoopConfig(S=6, D=64, max_rounds=5)
        agents = [
            make_constant_agent(i, torch.randn(6, 64) * 0.1)
            for i in range(3)
        ]
        B, diag = latent_resonance_loop(agents, config)
        assert B.shape == (6, 64)
        assert isinstance(diag, LoopDiagnostics)
        assert len(diag.residuals) > 0

    def test_aligned_agents_converge(self):
        """4 agents with aligned deltas should converge quickly."""
        config = LoopConfig(S=6, D=64, max_rounds=20, tol=0.05)
        target = torch.randn(6, 64) * 0.5
        agents = [
            make_decaying_agent(i, target, quality=0.7)
            for i in range(4)
        ]
        B, diag = latent_resonance_loop(agents, config)
        assert diag.residuals[-1] < 0.5 or "converged" in diag.halt_reason

    def test_diagnostics_collected(self):
        config = LoopConfig(S=6, D=64, max_rounds=5)
        agents = [make_constant_agent(i, torch.randn(6, 64) * 0.1) for i in range(3)]
        B, diag = latent_resonance_loop(agents, config)
        
        assert len(diag.trajectory) == len(diag.residuals)
        assert len(diag.strengths) == len(diag.residuals)
        assert len(diag.betas) == len(diag.residuals)
        assert len(diag.dissenter_indices) == len(diag.residuals)

    def test_without_anderson(self):
        config = LoopConfig(S=6, D=64, max_rounds=5, use_anderson=False)
        agents = [make_constant_agent(i, torch.randn(6, 64) * 0.1) for i in range(3)]
        B, diag = latent_resonance_loop(agents, config)
        assert B.shape == (6, 64)

    def test_strengths_normalised(self):
        config = LoopConfig(S=6, D=64, max_rounds=5)
        agents = [make_constant_agent(i, torch.randn(6, 64) * 0.1, quality=0.5)
                  for i in range(4)]
        B, diag = latent_resonance_loop(agents, config)
        for s in diag.strengths:
            assert abs(s.sum().item() - 1.0) < 0.01

    def test_beta_increases(self):
        config = LoopConfig(S=6, D=64, max_rounds=10)
        agents = [make_constant_agent(i, torch.randn(6, 64) * 0.1) for i in range(3)]
        _, diag = latent_resonance_loop(agents, config)
        if len(diag.betas) > 2:
            assert diag.betas[-1] >= diag.betas[0], "β should increase over time"

    def test_single_agent(self):
        config = LoopConfig(S=6, D=64, max_rounds=5)
        agent = make_constant_agent(0, torch.randn(6, 64) * 0.1)
        B, diag = latent_resonance_loop([agent], config)
        assert B.shape == (6, 64)
