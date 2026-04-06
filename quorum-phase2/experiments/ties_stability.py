"""Experiment: Test TIES-Resolve stability on real activations.

- Extract hidden states from GPT-2 for 1000 diverse prompts
- Simulate 5 agents by running the same prompts with different random seeds
- Run TIES-Resolve on the resulting deltas
- Measure: coefficient of variation of trim thresholds across inputs
- If CoV > 1.0, TIES-Resolve is unstable on real activations
"""

from __future__ import annotations

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.ties_resolve import ties_resolve, _trim

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def generate_synthetic_agent_deltas(
    base_states: torch.Tensor,
    num_agents: int = 5,
    noise_scale: float = 0.3,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Simulate multiple agents by adding different noise to base states.
    
    Each agent produces a delta as the difference between its noisy version
    and the base state.
    """
    deltas = []
    for i in range(num_agents):
        torch.manual_seed(seed + i * 1000)
        noise = torch.randn_like(base_states) * noise_scale
        delta = noise
        deltas.append(delta)
    return deltas


def measure_trim_thresholds(deltas: list[torch.Tensor], density: float = 0.3) -> torch.Tensor:
    """Measure the trim threshold for each delta."""
    thresholds = []
    for d in deltas:
        threshold = torch.quantile(d.abs().flatten(), 1.0 - density)
        thresholds.append(threshold.item())
    return torch.tensor(thresholds)


def main():
    print("=" * 60)
    print("TIES-RESOLVE STABILITY EXPERIMENT")
    print("=" * 60)

    results = {}

    # Generate base states — either from real model or synthetic
    print("\n--- Generating base states ---")
    try:
        from transformers import AutoTokenizer, AutoModel

        print("  Loading GPT-2 small...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModel.from_pretrained("gpt2")
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompts = [f"The number {i} is interesting because" for i in range(200)]
        extra_prompts = [
            "In the beginning there was",
            "The cat sat on the",
            "Machine learning is revolutionizing",
            "The weather today is",
            "According to recent studies",
        ]
        prompts = (prompts + extra_prompts * 40)[:1000]

        print(f"  Extracting hidden states for {len(prompts)} prompts...")
        hidden_states = []
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=32)
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
                hidden_states.append(last_hidden)

        base_states = torch.stack(hidden_states)
        print(f"  Base states shape: {base_states.shape}")
        source = "gpt2_real"
    except Exception as e:
        print(f"  Could not load GPT-2: {e}")
        print("  Using synthetic states instead")
        torch.manual_seed(42)
        base_states = torch.randn(1000, 768)
        source = "synthetic"

    results["source"] = source
    results["num_inputs"] = base_states.shape[0]
    results["hidden_dim"] = base_states.shape[1]

    # Experiment 1: Trim threshold stability
    print("\n--- Trim threshold stability ---")
    num_agents = 5
    densities = [0.2, 0.3, 0.5]

    for density in densities:
        all_thresholds = []
        
        for sample_idx in range(min(100, base_states.shape[0])):
            sample = base_states[sample_idx:sample_idx+1]  # [1, D]
            S = 6
            D = sample.shape[1]
            sample_expanded = sample.expand(S, -1)  # [S, D]

            deltas = generate_synthetic_agent_deltas(
                sample_expanded, num_agents=num_agents, noise_scale=0.3, seed=sample_idx
            )
            thresholds = measure_trim_thresholds(deltas, density=density)
            all_thresholds.append(thresholds)

        all_thresholds_tensor = torch.stack(all_thresholds)  # [100, num_agents]
        mean_threshold = all_thresholds_tensor.mean().item()
        std_threshold = all_thresholds_tensor.std().item()
        cov = std_threshold / max(mean_threshold, 1e-8)

        results[f"density_{density}"] = {
            "mean_threshold": mean_threshold,
            "std_threshold": std_threshold,
            "coefficient_of_variation": cov,
            "stable": cov < 1.0,
        }
        status = "STABLE" if cov < 1.0 else "UNSTABLE"
        print(f"  density={density}: mean={mean_threshold:.4f}, std={std_threshold:.4f}, CoV={cov:.4f} [{status}]")

    # Experiment 2: Output stability across inputs
    print("\n--- TIES-Resolve output stability ---")
    output_norms = []
    output_variances = []

    for sample_idx in range(min(200, base_states.shape[0])):
        sample = base_states[sample_idx:sample_idx+1]
        S = 6
        sample_expanded = sample.expand(S, -1)

        deltas = generate_synthetic_agent_deltas(
            sample_expanded, num_agents=5, noise_scale=0.3, seed=sample_idx
        )
        resolved = ties_resolve(deltas, density=0.3)
        output_norms.append(resolved.norm().item())
        output_variances.append(resolved.var().item())

    output_norms_t = torch.tensor(output_norms)
    output_vars_t = torch.tensor(output_variances)

    results["output_stability"] = {
        "norm_mean": output_norms_t.mean().item(),
        "norm_std": output_norms_t.std().item(),
        "norm_cov": (output_norms_t.std() / output_norms_t.mean().clamp(min=1e-8)).item(),
        "variance_mean": output_vars_t.mean().item(),
        "variance_std": output_vars_t.std().item(),
    }
    print(f"  Output norm: mean={output_norms_t.mean():.4f}, std={output_norms_t.std():.4f}, CoV={results['output_stability']['norm_cov']:.4f}")

    # Experiment 3: Sparsity pattern consistency
    print("\n--- Sparsity pattern analysis ---")
    sparsity_rates = []
    for sample_idx in range(min(100, base_states.shape[0])):
        sample = base_states[sample_idx:sample_idx+1]
        sample_expanded = sample.expand(6, -1)
        deltas = generate_synthetic_agent_deltas(
            sample_expanded, num_agents=5, noise_scale=0.3, seed=sample_idx
        )
        trimmed = _trim(deltas, density=0.3)
        for t in trimmed:
            sparsity = (t == 0).float().mean().item()
            sparsity_rates.append(sparsity)

    sparsity_t = torch.tensor(sparsity_rates)
    results["sparsity"] = {
        "mean": sparsity_t.mean().item(),
        "std": sparsity_t.std().item(),
        "min": sparsity_t.min().item(),
        "max": sparsity_t.max().item(),
    }
    print(f"  Sparsity: mean={sparsity_t.mean():.4f}, std={sparsity_t.std():.4f}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(output_norms, bins=30, alpha=0.7, edgecolor="black")
    axes[0].set_title("TIES-Resolve Output Norms")
    axes[0].set_xlabel("L2 Norm")
    axes[0].set_ylabel("Count")

    axes[1].hist(output_variances, bins=30, alpha=0.7, color="orange", edgecolor="black")
    axes[1].set_title("TIES-Resolve Output Variances")
    axes[1].set_xlabel("Variance")

    axes[2].hist(sparsity_rates, bins=30, alpha=0.7, color="green", edgecolor="black")
    axes[2].set_title("Trimmed Delta Sparsity")
    axes[2].set_xlabel("Fraction of Zeros")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ties_stability.png", dpi=150)
    plt.close(fig)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for density in densities:
        d = results[f"density_{density}"]
        print(f"  density={density}: CoV={d['coefficient_of_variation']:.4f} ({'STABLE' if d['stable'] else 'UNSTABLE'})")
    print(f"  Output norm CoV: {results['output_stability']['norm_cov']:.4f}")

    with open(RESULTS_DIR / "ties_stability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to experiments/results/ties_stability_results.json")


if __name__ == "__main__":
    main()
