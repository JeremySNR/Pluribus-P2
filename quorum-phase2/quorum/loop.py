"""Main Latent Resonance Loop (§6).

Read → Propose → Resolve cycle integrating all subsystems:
  - Buffer reads and writes
  - Agent forward passes (simulated for testing)
  - Codec encode/decode
  - TIES-Resolve for conflict resolution
  - Cross-inhibition (Hopfield energy + bee dynamics)
  - Contraction enforcement (damped iteration)
  - Anderson acceleration
  - Convergence detection
  - Dissenter channel protection
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass, field

from .buffer import UniversalLatentBuffer
from .codecs import AgentCodec
from .ties_resolve import ties_resolve
from .manifold import on_manifold_project, CleanupAutoencoder
from .cross_inhibition import (
    hopfield_update,
    bee_inhibition_step,
    compute_beta,
    compute_agreement,
    CrossInhibitionConfig,
)
from .suppression_safety import (
    apply_activity_floor,
    homeostatic_sigma,
    SuppressionSafetyConfig,
    SuppressionSafetyState,
)
from .contraction import damped_update
from .anderson import AndersonAccelerator
from .convergence import (
    QuorumHaltCriterion,
    PonderNetHalter,
    compute_residual,
    ConvergenceConfig,
)
from .dissenter import (
    select_dissenter,
    compute_dissenter_delta,
    DissenterConfig,
)
from .utils import SlotName, DISSENTER_SLOT, entropy


@dataclass
class LoopConfig:
    """Configuration for the main latent resonance loop."""
    S: int = 6
    D: int = 512
    max_rounds: int = 20
    tol: float = 0.01
    alpha: float = 0.5
    ties_density: float = 0.3
    beta_min: float = 0.1
    beta_max: float = 10.0
    tau_beta: float = 3.0
    lambda_repel: float = 0.1
    sigma: float = 0.3
    decay_rate: float = 0.05
    recruit_rate: float = 0.1
    epsilon_floor: float = 0.01
    A_min: float = 0.1
    use_pondernet: bool = False
    use_anderson: bool = True
    anderson_m: int = 5
    anderson_lam: float = 1e-5
    use_cleanup_ae: bool = False


@dataclass
class SyntheticAgent:
    """A simulated agent for testing purposes.

    Each agent is defined by a delta-generating function and optional codec.
    In the real system, agents would be LLMs with codecs; here we simulate
    the forward pass by generating deltas directly.
    """
    agent_id: int
    delta_fn: Callable[[torch.Tensor, int], torch.Tensor]
    quality: float = 0.5
    codec: Optional[AgentCodec] = None

    def propose_delta(self, buffer_state: torch.Tensor, round_idx: int) -> torch.Tensor:
        """Generate a proposed delta for this round."""
        return self.delta_fn(buffer_state, round_idx)


@dataclass
class LoopDiagnostics:
    """Diagnostics collected during loop execution."""
    trajectory: list = field(default_factory=list)
    residuals: list = field(default_factory=list)
    strengths: list = field(default_factory=list)
    energies: list = field(default_factory=list)
    dissenter_indices: list = field(default_factory=list)
    betas: list = field(default_factory=list)
    halt_reason: str = "max_rounds"


def latent_resonance_loop(
    agents: List[SyntheticAgent],
    config: Optional[LoopConfig] = None,
    cleanup_ae: Optional[CleanupAutoencoder] = None,
    initial_buffer: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, LoopDiagnostics]:
    """Full Latent Resonance Loop execution (§6).

    Implements the Read → Propose → Resolve cycle with all subsystems.

    Args:
        agents: List of agents (real or synthetic).
        config: Loop configuration.
        cleanup_ae: Optional cleanup autoencoder for on-manifold projection.
        initial_buffer: Optional initial buffer state [S, D].

    Returns:
        (converged_buffer, diagnostics)
    """
    if config is None:
        config = LoopConfig()

    N = len(agents)
    diag = LoopDiagnostics()

    if initial_buffer is not None:
        B = initial_buffer.clone()
    else:
        B = torch.zeros(config.S, config.D)

    strengths = torch.ones(N) / N
    beta = config.beta_min

    anderson = None
    if config.use_anderson:
        anderson = AndersonAccelerator(
            config.S, config.D,
            m=config.anderson_m, lam=config.anderson_lam,
        )

    halt = QuorumHaltCriterion(tol=config.tol, consecutive_required=3)

    suppression_state = SuppressionSafetyState(
        N,
        SuppressionSafetyConfig(
            epsilon_floor=config.epsilon_floor,
            A_min=config.A_min,
            sigma_base=config.sigma,
        ),
    )

    for t in range(config.max_rounds):
        # === PHASE 1: READ ===
        buffer_state = B.clone()

        # === PHASE 2: PROPOSE ===
        deltas = []
        qualities = []
        for agent in agents:
            delta = agent.propose_delta(buffer_state, t)
            deltas.append(delta)
            qualities.append(agent.quality)

        qualities_tensor = torch.tensor(qualities)

        # === PHASE 3: RESOLVE ===

        # 3a. Compute agreement for adaptive β
        agreement = compute_agreement(deltas)

        # 3b. Update β with annealing
        beta = compute_beta(
            t, config.beta_min, config.beta_max, config.tau_beta,
            agreement=agreement,
        )
        diag.betas.append(beta)

        # 3c. Bee-inspired strength update with homeostatic regulation
        effective_deltas = torch.stack([s * d for s, d in zip(strengths, deltas)])
        sigma_eff = homeostatic_sigma(
            strengths, effective_deltas,
            sigma_base=config.sigma, A_min=config.A_min,
        )
        strengths = bee_inhibition_step(
            strengths, qualities_tensor,
            sigma=sigma_eff, alpha=config.decay_rate, gamma=config.recruit_rate,
        )
        strengths = apply_activity_floor(strengths, config.epsilon_floor)
        diag.strengths.append(strengths.clone())

        # 3d. Select dissenter and protect Slot 5
        dissenter_idx = select_dissenter(deltas)
        diag.dissenter_indices.append(dissenter_idx)

        # 3e. Apply strength weighting to deltas
        weighted_deltas = [s * d for s, d in zip(strengths, deltas)]

        # 3f. TIES-Resolve for Slots 0–4
        non_dissent_deltas = [d[:config.S - 1] for d in weighted_deltas]
        if len(non_dissent_deltas) > 0:
            resolved_main = ties_resolve(non_dissent_deltas, density=config.ties_density)
        else:
            resolved_main = torch.zeros(config.S - 1, config.D)

        # 3g. Dissenter writes to Slot 5 unimpeded
        dissenter_delta = compute_dissenter_delta(
            deltas, qualities_tensor, dissenter_idx,
            slot_index=DISSENTER_SLOT, epsilon_floor=config.epsilon_floor,
        )

        resolved = torch.cat([resolved_main, dissenter_delta.unsqueeze(0)], dim=0)

        # 3h. Hopfield energy-based competition weighting (Slots 0–4)
        for s in range(config.S - 1):
            slot_deltas = torch.stack([d[s] for d in deltas])  # [N, D]
            B_slot = B[s]
            if B_slot.norm() > 1e-8:
                updated_slot = hopfield_update(B_slot, slot_deltas, beta)
                resolved[s] = resolved[s] + 0.5 * (updated_slot - resolved[s])

        # 3i. Damped update
        F_raw = B + resolved
        B_new = damped_update(B, F_raw, alpha=config.alpha)

        # 3j. On-manifold projection
        ae = cleanup_ae if config.use_cleanup_ae else None
        B_new = on_manifold_project(B_new, ae)

        # 3k. Anderson acceleration
        if anderson is not None:
            B_new = anderson.step(B, B_new)

        # === CONVERGENCE CHECK ===
        residual = compute_residual(B_new, B)
        diag.trajectory.append(B_new.clone())
        diag.residuals.append(residual)

        B = B_new

        if halt.should_halt(residual):
            diag.halt_reason = f"converged at round {t}"
            break

        if t == config.max_rounds - 1:
            diag.halt_reason = f"max_rounds ({config.max_rounds})"

    return B, diag
