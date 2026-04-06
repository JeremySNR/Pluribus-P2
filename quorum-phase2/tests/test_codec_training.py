"""Tests for Codec Training (§2.2)."""

import torch
import pytest
from quorum.codecs import AgentCodec
from quorum.codec_training import (
    procrustes_init,
    composite_loss,
    train_codec_pair,
    CodecTrainingConfig,
)


class TestProcrustesInit:
    def test_basic_init(self):
        codec = AgentCodec(agent_dim=128, buffer_dim=64)
        source = torch.randn(50, 128)
        target = torch.randn(50, 64)
        procrustes_init(codec, source, target)
        z = codec.encode(source[:5])
        assert torch.isfinite(z).all()

    def test_improves_alignment(self):
        """Procrustes init should improve alignment vs random init."""
        torch.manual_seed(42)
        dim_a, dim_b = 64, 64
        N = 100
        W_true = torch.randn(dim_b, dim_a)
        source = torch.randn(N, dim_a)
        target = source @ W_true.T

        codec_random = AgentCodec(agent_dim=dim_a, buffer_dim=dim_b)
        codec_procrustes = AgentCodec(agent_dim=dim_a, buffer_dim=dim_b)
        procrustes_init(codec_procrustes, source, target)

        with torch.no_grad():
            err_random = (codec_random.encode(source) - target).pow(2).mean().item()
            err_proc = (codec_procrustes.encode(source) - target).pow(2).mean().item()

        assert err_proc < err_random, \
            f"Procrustes ({err_proc:.4f}) should be better than random ({err_random:.4f})"


class TestCompositeLoss:
    def test_reconstruction_only(self):
        codec = AgentCodec(agent_dim=128, buffer_dim=64)
        h = torch.randn(16, 128)
        loss, components = composite_loss(codec, h)
        assert loss.item() > 0
        assert "reconstruction" in components
        assert "total" in components

    def test_with_alignment_and_cycle(self):
        codec_i = AgentCodec(agent_dim=128, buffer_dim=64)
        codec_j = AgentCodec(agent_dim=256, buffer_dim=64)
        h_i = torch.randn(16, 128)
        h_j = torch.randn(16, 256)
        loss, components = composite_loss(codec_i, h_i, codec_j, h_j)
        assert "reconstruction" in components
        assert "alignment" in components
        assert "cycle" in components
        assert loss.item() > 0

    def test_gradients_flow(self):
        codec_i = AgentCodec(agent_dim=64, buffer_dim=32)
        codec_j = AgentCodec(agent_dim=64, buffer_dim=32)
        h_i = torch.randn(8, 64)
        h_j = torch.randn(8, 64)
        loss, _ = composite_loss(codec_i, h_i, codec_j, h_j)
        loss.backward()
        for p in codec_i.parameters():
            assert p.grad is not None


class TestTrainCodecPair:
    def test_training_reduces_loss(self):
        torch.manual_seed(42)
        codec_i = AgentCodec(agent_dim=64, buffer_dim=32)
        codec_j = AgentCodec(agent_dim=64, buffer_dim=32)
        data_i = torch.randn(100, 64)
        data_j = torch.randn(100, 64)

        config = CodecTrainingConfig(lr=1e-3, num_steps=200, log_every=50)
        history = train_codec_pair(codec_i, codec_j, data_i, data_j, config)

        assert len(history) > 1
        assert history[-1]["total"] < history[0]["total"], \
            "Training should reduce the total loss"
