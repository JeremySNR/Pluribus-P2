"""Experiment: Test full loop convergence properties.

Runs the complete Latent Resonance Loop with synthetic agents generating
random deltas. Measures rounds to convergence, final residual, and
contraction rate estimates across varying conditions.

Varies: number of agents (3, 5, 7, 10), buffer dimensionality (128, 512, 2048),
        suppression strength.

Saves convergence curves and summary metrics to experiments/results/.
"""

from __future__ import annotations

import os
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.loop import latent_resonance_loop, SyntheticAgent, LoopConfig


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_target_seeking_agent(agent_id: int, target: torch.Tensor, quality: float):
    """Agent that proposes deltas moving toward a fixed target."""
    def delta_fn(B, t):
        return (target - B) * 0.3 + torch.randn_like(B) * 0.05
    return SyntheticAgent(agent_id=agent_id, delta_fn=delta_fn, quality=quality)


def make_random_agent(agent_id: int, S: int, D: int, quality: float):
    """Agent that proposes random deltas (stress test)."""
    def delta_fn(B, t):
        scale = 0.5 / (1 + t * 0.1)
        return torch.randn(S, D) * scale
    return SyntheticAgent(agent_id=agent_id, delta_fn=delta_fn, quality=quality)


def run_convergence_experiment(
    num_agents: int,
    D: int,
    sigma: float,
    seed: int = 42,
    max_rounds: int = 30,
    agent_type: str = "target_seeking",
) -> dict:
    """Run a single convergence experiment."""
    torch.manual_seed(seed)
    S = 6
    config = LoopConfig(
        S=S, D=D, max_rounds=max_rounds, tol=0.02,
        alpha=0.5, sigma=sigma, ties_density=0.3,
        use_anderson=True, anderson_m=5,
    )

    if agent_type == "target_seeking":
        target = torch.randn(S, D) * 0.5
        agents = [
            make_target_seeking_agent(i, target + torch.randn(S, D) * 0.1, 
                                       quality=0.5 + 0.1 * (i % 3))
            for i in range(num_agents)
        ]
    else:
        agents = [
            make_random_agent(i, S, D, quality=0.5)
            for i in range(num_agents)
        ]

    B, diag = latent_resonance_loop(agents, config)

    return {
        "num_agents": num_agents,
        "D": D,
        "sigma": sigma,
        "agent_type": agent_type,
        "rounds": len(diag.residuals),
        "final_residual": diag.residuals[-1] if diag.residuals else float("inf"),
        "halt_reason": diag.halt_reason,
        "residuals": diag.residuals,
        "buffer_norm": B.norm().item(),
        "converged": "converged" in diag.halt_reason,
    }


def plot_convergence_curves(results: list[dict], filename: str):
    """Plot convergence curves for a set of experiments."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for r in results:
        label = f"N={r['num_agents']}, D={r['D']}, σ={r['sigma']}"
        color = "green" if r["converged"] else "red"
        ax.plot(r["residuals"], label=label, alpha=0.7, color=color if len(results) <= 3 else None)
    
    ax.set_xlabel("Round")
    ax.set_ylabel("Relative Residual")
    ax.set_title("Convergence Trajectories")
    ax.set_yscale("log")
    ax.axhline(y=0.02, color="gray", linestyle="--", alpha=0.5, label="tol=0.02")
    if len(results) <= 10:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def main():
    print("=" * 60)
    print("SYNTHETIC CONVERGENCE EXPERIMENT")
    print("=" * 60)

    all_results = []

    # Experiment 1: Vary number of agents
    print("\n--- Varying number of agents (D=128) ---")
    agent_counts = [3, 5, 7, 10]
    agent_results = []
    for N in agent_counts:
        r = run_convergence_experiment(num_agents=N, D=128, sigma=0.3)
        agent_results.append(r)
        status = "CONVERGED" if r["converged"] else "DID NOT CONVERGE"
        print(f"  N={N}: {status} in {r['rounds']} rounds, residual={r['final_residual']:.6f}")
        all_results.append(r)

    plot_convergence_curves(agent_results, "convergence_vary_agents.png")

    # Experiment 2: Vary buffer dimensionality
    print("\n--- Varying dimensionality (N=5) ---")
    dims = [128, 512, 2048]
    dim_results = []
    for D in dims:
        r = run_convergence_experiment(num_agents=5, D=D, sigma=0.3)
        dim_results.append(r)
        status = "CONVERGED" if r["converged"] else "DID NOT CONVERGE"
        print(f"  D={D}: {status} in {r['rounds']} rounds, residual={r['final_residual']:.6f}")
        all_results.append(r)

    plot_convergence_curves(dim_results, "convergence_vary_dims.png")

    # Experiment 3: Vary suppression strength
    print("\n--- Varying suppression strength (N=5, D=128) ---")
    sigmas = [0.1, 0.3, 0.5, 0.8]
    sigma_results = []
    for sigma in sigmas:
        r = run_convergence_experiment(num_agents=5, D=128, sigma=sigma)
        sigma_results.append(r)
        status = "CONVERGED" if r["converged"] else "DID NOT CONVERGE"
        print(f"  σ={sigma}: {status} in {r['rounds']} rounds, residual={r['final_residual']:.6f}")
        all_results.append(r)

    plot_convergence_curves(sigma_results, "convergence_vary_sigma.png")

    # Experiment 4: Random agents (stress test)
    print("\n--- Random agents stress test (D=128) ---")
    random_results = []
    for N in [3, 5, 7]:
        r = run_convergence_experiment(num_agents=N, D=128, sigma=0.3, agent_type="random")
        random_results.append(r)
        status = "CONVERGED" if r["converged"] else "DID NOT CONVERGE"
        print(f"  Random N={N}: {status} in {r['rounds']} rounds, residual={r['final_residual']:.6f}")
        all_results.append(r)

    plot_convergence_curves(random_results, "convergence_random_agents.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    converged = sum(1 for r in all_results if r["converged"])
    total = len(all_results)
    print(f"Converged: {converged}/{total}")
    
    target_results = [r for r in all_results if r["agent_type"] == "target_seeking"]
    if target_results:
        avg_rounds = sum(r["rounds"] for r in target_results) / len(target_results)
        print(f"Average rounds (target-seeking): {avg_rounds:.1f}")

    summary = {
        "total_experiments": total,
        "converged": converged,
        "results": [
            {k: v for k, v in r.items() if k != "residuals"}
            for r in all_results
        ],
    }
    with open(RESULTS_DIR / "synthetic_convergence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to experiments/results/synthetic_convergence_summary.json")


if __name__ == "__main__":
    main()
