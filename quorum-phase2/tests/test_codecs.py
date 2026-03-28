"""Tests for Per-Agent Codecs (§2.2)."""

import torch
import pytest
from quorum.codecs import AgentEncoder, AgentDecoder, AgentCodec


class TestAgentEncoder:
    def test_output_shape(self):
        enc = AgentEncoder(source_dim=768, buffer_dim=512)
        h = torch.randn(10, 768)
        out = enc(h)
        assert out.shape == (10, 512)

    def test_single_input(self):
        enc = AgentEncoder(source_dim=256, buffer_dim=128)
        h = torch.randn(256)
        out = enc(h)
        assert out.shape == (128,)

    def test_layernorm_normalisation(self):
        enc = AgentEncoder(source_dim=256, buffer_dim=512)
        h = torch.randn(32, 256)
        out = enc(h)
        means = out.mean(dim=-1)
        vars_ = out.var(dim=-1, unbiased=False)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
        assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-2)

    def test_gradient_flow(self):
        enc = AgentEncoder(source_dim=128, buffer_dim=64)
        h = torch.randn(4, 128, requires_grad=True)
        out = enc(h)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        assert h.grad.shape == (4, 128)


class TestAgentDecoder:
    def test_output_shape(self):
        dec = AgentDecoder(buffer_dim=512, target_dim=768)
        z = torch.randn(10, 512)
        out = dec(z)
        assert out.shape == (10, 768)

    def test_single_input(self):
        dec = AgentDecoder(buffer_dim=128, target_dim=256)
        z = torch.randn(128)
        out = dec(z)
        assert out.shape == (256,)

    def test_gradient_flow(self):
        dec = AgentDecoder(buffer_dim=64, target_dim=128)
        z = torch.randn(4, 64, requires_grad=True)
        out = dec(z)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None


class TestAgentCodec:
    def test_roundtrip_shape(self):
        codec = AgentCodec(agent_dim=768, buffer_dim=512)
        h = torch.randn(8, 768)
        h_recon = codec.roundtrip(h)
        assert h_recon.shape == (8, 768)

    def test_encode_decode_shape(self):
        codec = AgentCodec(agent_dim=256, buffer_dim=128)
        h = torch.randn(4, 256)
        z = codec.encode(h)
        assert z.shape == (4, 128)
        h_recon = codec.decode(z)
        assert h_recon.shape == (4, 256)

    def test_roundtrip_with_random_inputs(self):
        """After random init, roundtrip is not identity but should have finite values."""
        codec = AgentCodec(agent_dim=128, buffer_dim=64)
        h = torch.randn(16, 128)
        h_recon = codec.roundtrip(h)
        assert torch.isfinite(h_recon).all()
        assert not torch.allclose(h, h_recon, atol=1e-2), \
            "Random codec should not perfectly reconstruct"

    def test_encode_buffer(self):
        codec = AgentCodec(agent_dim=256, buffer_dim=128)
        buffer_state = torch.randn(6, 128)
        native = codec.encode_buffer(buffer_state)
        assert native.shape == (6, 256)

    def test_decode_buffer(self):
        codec = AgentCodec(agent_dim=256, buffer_dim=128)
        agent_states = torch.randn(6, 256)
        encoded = codec.decode_buffer(agent_states)
        assert encoded.shape == (6, 128)

    def test_different_agent_dims(self):
        for agent_dim in [128, 256, 512, 768, 1024, 4096]:
            codec = AgentCodec(agent_dim=agent_dim, buffer_dim=512)
            h = torch.randn(2, agent_dim)
            z = codec.encode(h)
            assert z.shape == (2, 512)
            h_r = codec.decode(z)
            assert h_r.shape == (2, agent_dim)
