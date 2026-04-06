"""Codec Training with Procrustes Initialization + Composite Loss (§2.2).

Training uses a composite loss:
  L_codec = L_reconstruction + λ_align * L_alignment + λ_cycle * L_cycle

Procrustes initialization: W* = UV^T where UΣV^T = SVD(H_j^T H_i).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .codecs import AgentCodec


@dataclass
class CodecTrainingConfig:
    """Configuration for codec training."""
    lr: float = 1e-4
    num_steps: int = 5000
    lambda_align: float = 1.0
    lambda_cycle: float = 0.5
    batch_size: int = 32
    log_every: int = 100


def procrustes_init(
    codec: AgentCodec,
    source_states: torch.Tensor,
    target_states: torch.Tensor,
) -> None:
    """Initialize encoder weights via Procrustes alignment (§2.2).

    Given paired hidden states, compute the optimal orthogonal alignment
    W* = UV^T where UΣV^T = SVD(target^T @ source).

    Args:
        codec: The codec to initialize.
        source_states: [N, d_i] hidden states from this agent.
        target_states: [N, D] target buffer representations.
    """
    with torch.no_grad():
        H_s = source_states - source_states.mean(dim=0, keepdim=True)
        H_t = target_states - target_states.mean(dim=0, keepdim=True)

        cross = H_t.T @ H_s  # [D, d_i]
        U, _, Vt = torch.linalg.svd(cross, full_matrices=False)
        W_opt = U @ Vt  # [D, min(D, d_i)]

        min_dim = min(codec.encoder.fc1.weight.shape[0], W_opt.shape[0])
        min_in = min(codec.encoder.fc1.weight.shape[1], W_opt.shape[1])
        codec.encoder.fc1.weight.data[:min_dim, :min_in] = W_opt[:min_dim, :min_in]
        nn.init.eye_(codec.encoder.fc2.weight)

        W_dec = W_opt.T  # [d_i, D] approximately
        min_dim_d = min(codec.decoder.fc3.weight.shape[0], W_dec.shape[0])
        min_in_d = min(codec.decoder.fc3.weight.shape[1], W_dec.shape[1])
        codec.decoder.fc3.weight.data[:min_dim_d, :min_in_d] = W_dec[:min_dim_d, :min_in_d]
        nn.init.eye_(codec.decoder.fc4.weight)


def composite_loss(
    codec_i: AgentCodec,
    h_i: torch.Tensor,
    codec_j: Optional[AgentCodec] = None,
    h_j: Optional[torch.Tensor] = None,
    lambda_align: float = 1.0,
    lambda_cycle: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """Compute the composite codec training loss (§2.2).

    L_codec = L_reconstruction + λ_align * L_alignment + λ_cycle * L_cycle

    Args:
        codec_i: Codec for agent i.
        h_i: [B, d_i] hidden states from agent i.
        codec_j: Optional codec for agent j (for alignment/cycle losses).
        h_j: Optional [B, d_j] hidden states from agent j (paired with h_i).
        lambda_align: Weight for alignment loss.
        lambda_cycle: Weight for cycle-consistency loss.

    Returns:
        (total_loss, loss_components_dict)
    """
    h_i_reconstructed = codec_i.roundtrip(h_i)
    l_recon = (h_i_reconstructed - h_i).pow(2).mean()

    losses = {"reconstruction": l_recon.item()}
    total = l_recon

    if codec_j is not None and h_j is not None:
        z_i = codec_i.encode(h_i)
        z_j = codec_j.encode(h_j)
        l_align = (z_i - z_j).pow(2).mean()
        losses["alignment"] = l_align.item()
        total = total + lambda_align * l_align

        h_j_from_i = codec_j.decode(z_i)
        l_cycle = (h_j_from_i - h_j).pow(2).mean()
        losses["cycle"] = l_cycle.item()
        total = total + lambda_cycle * l_cycle

    losses["total"] = total.item()
    return total, losses


def train_codec_pair(
    codec_i: AgentCodec,
    codec_j: AgentCodec,
    data_i: torch.Tensor,
    data_j: torch.Tensor,
    config: Optional[CodecTrainingConfig] = None,
) -> List[dict]:
    """Train a pair of codecs jointly with the composite loss.

    Args:
        codec_i: Codec for agent i.
        codec_j: Codec for agent j.
        data_i: [N, d_i] training hidden states from agent i.
        data_j: [N, d_j] training hidden states from agent j (paired).
        config: Training configuration.

    Returns:
        List of loss dictionaries from training.
    """
    if config is None:
        config = CodecTrainingConfig()

    all_params = list(codec_i.parameters()) + list(codec_j.parameters())
    optimizer = optim.Adam(all_params, lr=config.lr)

    N = data_i.shape[0]
    history = []

    for step in range(config.num_steps):
        idx = torch.randint(0, N, (min(config.batch_size, N),))
        h_i = data_i[idx]
        h_j = data_j[idx]

        loss_i, losses_i = composite_loss(
            codec_i, h_i, codec_j, h_j,
            config.lambda_align, config.lambda_cycle,
        )
        loss_j, losses_j = composite_loss(
            codec_j, h_j, codec_i, h_i,
            config.lambda_align, config.lambda_cycle,
        )

        total_loss = loss_i + loss_j
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        if step % config.log_every == 0:
            history.append({
                "step": step,
                "loss_i": losses_i,
                "loss_j": losses_j,
                "total": total_loss.item(),
            })

    return history
