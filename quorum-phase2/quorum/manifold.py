"""On-Manifold Projection / Cleanup Autoencoder (§2.3).

After TIES-Resolve produces the merged delta, project the buffer state
back onto the learned data manifold.

Method 1: LayerNorm per slot (fast, always available).
Method 2: Cleanup autoencoder (better, requires training).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CleanupAutoencoder(nn.Module):
    """Shallow autoencoder for on-manifold projection (§2.3).

    Trained on buffer states observed during codec training, learning the
    manifold of "valid" buffer configurations. Single hidden layer with
    high capacity (hidden_dim = 2 * D).

    [PLAUSIBLE] — autoencoder-based manifold projection is standard in
    generative modeling but untested for multi-agent buffer cleanup.
    """

    def __init__(self, buffer_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.buffer_dim = buffer_dim
        self.hidden_dim = hidden_dim or 2 * buffer_dim

        self.enc = nn.Sequential(
            nn.Linear(buffer_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, buffer_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(buffer_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, buffer_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def layernorm_project(buffer_state: torch.Tensor) -> torch.Tensor:
    """Method 1: LayerNorm per slot — fast, always available.

    Normalises each slot vector to have zero mean and unit variance.

    Args:
        buffer_state: [S, D] buffer state.
    Returns:
        Normalised buffer state [S, D].
    """
    D = buffer_state.shape[-1]
    return F.layer_norm(buffer_state, [D])


def on_manifold_project(
    buffer_state: torch.Tensor,
    cleanup_ae: Optional[CleanupAutoencoder] = None,
) -> torch.Tensor:
    """Project buffer state back onto the learned data manifold (§2.3).

    Uses cleanup autoencoder if available, otherwise falls back to LayerNorm.

    Args:
        buffer_state: [S, D] buffer state.
        cleanup_ae: Optional trained cleanup autoencoder.
    Returns:
        Projected buffer state [S, D].
    """
    normed = layernorm_project(buffer_state)

    if cleanup_ae is not None:
        return cleanup_ae(normed)

    return normed


def train_cleanup_autoencoder(
    ae: CleanupAutoencoder,
    buffer_states: torch.Tensor,
    num_steps: int = 2000,
    lr: float = 1e-3,
    noise_std: float = 0.1,
) -> list[float]:
    """Train the cleanup autoencoder on observed buffer states.

    Uses reconstruction loss plus a contrastive term that pushes the encoding
    away from pathological states (all-zero, random noise).

    Args:
        ae: Cleanup autoencoder to train.
        buffer_states: [N, S, D] training buffer states.
        num_steps: Number of training steps.
        lr: Learning rate.
        noise_std: Std of noise for contrastive examples.

    Returns:
        List of loss values during training.
    """
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    N = buffer_states.shape[0]
    losses = []

    for step in range(num_steps):
        idx = torch.randint(0, N, (min(32, N),))
        batch = buffer_states[idx]
        S, D = batch.shape[-2], batch.shape[-1]
        flat = batch.reshape(-1, D)

        reconstructed = ae(flat)
        l_recon = (reconstructed - flat).pow(2).mean()

        noisy = flat + torch.randn_like(flat) * noise_std
        noisy_recon = ae(noisy)
        l_denoise = (noisy_recon - flat).pow(2).mean()

        loss = l_recon + 0.5 * l_denoise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            losses.append(loss.item())

    return losses
