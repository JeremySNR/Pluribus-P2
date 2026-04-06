"""Tests for On-Manifold Projection (§2.3)."""

import torch
import pytest
from quorum.manifold import (
    layernorm_project,
    on_manifold_project,
    CleanupAutoencoder,
    train_cleanup_autoencoder,
)


class TestLayerNormProject:
    def test_output_shape(self):
        B = torch.randn(6, 512)
        result = layernorm_project(B)
        assert result.shape == (6, 512)

    def test_zero_mean_unit_var(self):
        B = torch.randn(6, 512) * 5 + 3
        result = layernorm_project(B)
        means = result.mean(dim=-1)
        vars_ = result.var(dim=-1, unbiased=False)
        assert torch.allclose(means, torch.zeros(6), atol=1e-5)
        assert torch.allclose(vars_, torch.ones(6), atol=1e-2)

    def test_idempotent_approximately(self):
        B = torch.randn(6, 512)
        r1 = layernorm_project(B)
        r2 = layernorm_project(r1)
        assert torch.allclose(r1, r2, atol=1e-5)


class TestCleanupAutoencoder:
    def test_output_shape(self):
        ae = CleanupAutoencoder(buffer_dim=128)
        x = torch.randn(10, 128)
        out = ae(x)
        assert out.shape == (10, 128)

    def test_reconstruction(self):
        ae = CleanupAutoencoder(buffer_dim=64)
        x = torch.randn(8, 64)
        out = ae(x)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        ae = CleanupAutoencoder(buffer_dim=64)
        x = torch.randn(4, 64, requires_grad=True)
        out = ae(x)
        out.sum().backward()
        assert x.grad is not None


class TestTrainCleanupAutoencoder:
    def test_training_reduces_loss(self):
        torch.manual_seed(42)
        ae = CleanupAutoencoder(buffer_dim=64)
        data = torch.randn(100, 6, 64)
        losses = train_cleanup_autoencoder(ae, data, num_steps=500, lr=1e-3)
        assert len(losses) > 1
        assert losses[-1] < losses[0], "Training should reduce loss"


class TestOnManifoldProject:
    def test_without_ae(self):
        B = torch.randn(6, 128)
        result = on_manifold_project(B, cleanup_ae=None)
        expected = layernorm_project(B)
        assert torch.allclose(result, expected)

    def test_with_ae(self):
        ae = CleanupAutoencoder(buffer_dim=128)
        B = torch.randn(6, 128)
        result = on_manifold_project(B, cleanup_ae=ae)
        assert result.shape == (6, 128)
        assert torch.isfinite(result).all()
