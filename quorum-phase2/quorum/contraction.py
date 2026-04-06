"""Contraction Enforcement (§4.1).

Three layers:
  1. Damped iteration: B_{t+1} = (1-α)B_t + α·F(B_t), α=0.5
  2. Spectral normalisation of codec weights (power iteration, λ_SN=0.9)
  3. Jacobian regularisation via Hutchinson trace estimator

[PROVEN]: Damped iteration (textbook), spectral norm (Miyato et al., 2018),
Jacobian reg for DEQs (Bai et al., 2021). Composition is [PLAUSIBLE].
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple


def damped_update(
    B_current: torch.Tensor,
    F_raw: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Layer 1 — Damped iteration (§4.1).

    B_{t+1} = (1 − α) · B_t + α · F_raw(B_t)

    If F_raw has Lipschitz constant L, the damped iteration has effective
    Lipschitz constant (1−α) + α·L, which is < 1 when α < 2/(1+L).

    Args:
        B_current: [S, D] current buffer state.
        F_raw: [S, D] raw update from the resolve step.
        alpha: Damping factor in (0, 1). Default 0.5 handles L up to 3.0.
    Returns:
        Damped buffer state [S, D].
    """
    return (1.0 - alpha) * B_current + alpha * F_raw


def spectral_normalize_weight(
    weight: torch.Tensor,
    n_power_iterations: int = 1,
    lambda_sn: float = 0.9,
    u: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Layer 2 — Spectral normalisation (§4.1).

    Normalise W̃ = W / σ_max(W) * λ_SN, bounding the layer's Lipschitz
    constant to λ_SN.

    Args:
        weight: [out, in] weight matrix.
        n_power_iterations: Number of power iterations for estimating σ_max.
        lambda_sn: Target spectral norm bound (default 0.9 for contraction).
        u: Optional left singular vector estimate from previous call.

    Returns:
        (normalised_weight, updated_u) for stateful tracking.
    """
    h, w = weight.shape
    if u is None:
        u = torch.randn(h, device=weight.device)
        u = u / u.norm()

    with torch.no_grad():
        for _ in range(n_power_iterations):
            v = weight.T @ u
            v = v / v.norm().clamp(min=1e-12)
            u = weight @ v
            u = u / u.norm().clamp(min=1e-12)

    sigma_max = (u @ weight @ v).item()
    sigma_max = max(abs(sigma_max), 1e-12)

    return weight * (lambda_sn / sigma_max), u


def apply_spectral_norm_to_module(
    module: nn.Module,
    lambda_sn: float = 0.9,
    n_power_iterations: int = 3,
) -> None:
    """Apply spectral normalisation to all Linear layers in a module.

    Rescales weight matrices so their spectral norm ≤ λ_SN.
    """
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() == 2:
            with torch.no_grad():
                normed, _ = spectral_normalize_weight(
                    param.data, n_power_iterations, lambda_sn
                )
                param.data.copy_(normed)


def hutchinson_jacobian_penalty(
    F_func: Callable[[torch.Tensor], torch.Tensor],
    B: torch.Tensor,
    n_samples: int = 1,
) -> torch.Tensor:
    """Layer 3 — Jacobian regularisation via Hutchinson trace estimator (§4.1).

    Estimates ‖J_F(B)‖_F² using random Rademacher vectors:
      ‖J_F‖_F² ≈ E_v[‖J_F · v‖²] where v ~ Rademacher(D)

    Cost: one additional backward pass per sample.

    Args:
        F_func: The function F whose Jacobian we're regularising.
        B: [S, D] input buffer state (must require grad).
        n_samples: Number of random vectors for the estimate.

    Returns:
        Scalar estimate of ‖J_F(B)‖_F².
    """
    B_input = B.detach().clone().requires_grad_(True)
    F_out = F_func(B_input)

    penalty = torch.tensor(0.0, device=B.device)
    for _ in range(n_samples):
        v = torch.randint(0, 2, B.shape, device=B.device).float() * 2 - 1  # Rademacher
        Jv = torch.autograd.grad(
            outputs=F_out,
            inputs=B_input,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )[0]
        penalty = penalty + Jv.pow(2).sum()

    return penalty / n_samples


def estimate_spectral_radius(
    F_func: Callable[[torch.Tensor], torch.Tensor],
    B: torch.Tensor,
    n_iterations: int = 10,
) -> float:
    """Estimate the spectral radius of J_F via power iteration (§4.1 Monitoring).

    If ρ exceeds 0.98, the system should increase damping.

    Args:
        F_func: The map F.
        B: [S, D] current buffer state.
        n_iterations: Power iteration steps.
    Returns:
        Estimated spectral radius ρ(J_F).
    """
    B_input = B.detach().clone().requires_grad_(True)
    F_out = F_func(B_input)

    v = torch.randn_like(B)
    v = v / v.norm().clamp(min=1e-12)

    for _ in range(n_iterations):
        Jv = torch.autograd.grad(
            outputs=F_out,
            inputs=B_input,
            grad_outputs=v,
            create_graph=False,
            retain_graph=True,
        )[0]
        v_norm = Jv.norm().clamp(min=1e-12)
        v = Jv / v_norm

    Jv_final = torch.autograd.grad(
        outputs=F_out,
        inputs=B_input,
        grad_outputs=v,
        create_graph=False,
        retain_graph=False,
    )[0]
    spectral_radius = Jv_final.norm().item()
    return spectral_radius
