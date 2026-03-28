"""Anderson Acceleration (§4.2).

Uses a history of m previous iterates to extrapolate, achieving
superlinear convergence in practice (~10× faster than naive iteration
in DEQ benchmarks).

[PROVEN]: Anderson acceleration for DEQs is standard in TorchDEQ.
Application to multi-agent convergence with cross-inhibition is [PLAUSIBLE].
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple


class AndersonAccelerator:
    """Anderson acceleration for fixed-point iteration (§4.2).

    Maintains a history of m iterates and uses least-squares extrapolation
    to accelerate convergence.

    Hyperparameters:
      m = 5: history size (balancing acceleration vs. memory)
      lam = 1e-5: ridge regularisation (numerical stability)
    """

    def __init__(
        self,
        S: int,
        D: int,
        m: int = 5,
        lam: float = 1e-5,
        device: Optional[torch.device] = None,
    ):
        self.S = S
        self.D = D
        self.m = m
        self.lam = lam
        self.device = device or torch.device("cpu")

        self.X_hist = torch.zeros(m, S, D, device=self.device)
        self.F_hist = torch.zeros(m, S, D, device=self.device)
        self.t = 0

    def reset(self) -> None:
        """Reset the acceleration state."""
        self.X_hist.zero_()
        self.F_hist.zero_()
        self.t = 0

    def step(
        self,
        B_current: torch.Tensor,
        F_B: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one Anderson-accelerated step.

        Args:
            B_current: [S, D] current iterate.
            F_B: [S, D] result of applying the fixed-point map F to B_current.

        Returns:
            [S, D] accelerated next iterate.
        """
        idx = self.t % self.m
        self.X_hist[idx] = B_current.detach()
        self.F_hist[idx] = F_B.detach()
        self.t += 1

        n = min(self.t, self.m)
        if n < 2:
            return F_B.clone()

        G = self.F_hist[:n] - self.X_hist[:n]  # [n, S, D]
        G_flat = G.reshape(n, -1)  # [n, S*D]

        GTG = G_flat @ G_flat.T + self.lam * torch.eye(n, device=self.device)

        ones = torch.ones(n, 1, device=self.device)
        try:
            alpha_mix = torch.linalg.solve(GTG, ones)
        except torch.linalg.LinAlgError:
            return F_B.clone()

        alpha_sum = alpha_mix.sum()
        if alpha_sum.abs() < 1e-12:
            return F_B.clone()
        alpha_mix = alpha_mix / alpha_sum

        F_flat = self.F_hist[:n].reshape(n, -1)  # [n, S*D]
        result = (alpha_mix.T @ F_flat).reshape(self.S, self.D)

        return result


def anderson_accelerated_loop(
    loop_step_fn,
    buffer_init: torch.Tensor,
    m: int = 5,
    lam: float = 1e-5,
    max_rounds: int = 20,
    tol: float = 1e-3,
) -> Tuple[torch.Tensor, int, float]:
    """Run a fixed-point iteration with Anderson acceleration (§4.2).

    This is the standalone version for testing. The main loop in loop.py
    uses the AndersonAccelerator class directly.

    Args:
        loop_step_fn: Callable that takes buffer [S, D] → updated buffer [S, D].
        buffer_init: [S, D] initial buffer state.
        m: History size.
        lam: Ridge regularisation.
        max_rounds: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        (converged_buffer, num_rounds, final_residual)
    """
    S, D = buffer_init.shape
    acc = AndersonAccelerator(S, D, m=m, lam=lam, device=buffer_init.device)

    buffer = buffer_init.clone()
    final_residual = float("inf")

    for t in range(max_rounds):
        F_new = loop_step_fn(buffer)
        buffer_next = acc.step(buffer, F_new)

        residual = (buffer_next - buffer).norm() / buffer_next.norm().clamp(min=1e-8)
        final_residual = residual.item()

        buffer = buffer_next

        if final_residual < tol:
            return buffer, t + 1, final_residual

    return buffer, max_rounds, final_residual
