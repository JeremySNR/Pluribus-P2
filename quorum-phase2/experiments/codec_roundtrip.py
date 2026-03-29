"""Experiment: Test codecs with real hidden states from GPT-2 small.

- Load GPT-2 small (124M params) from HuggingFace — runs on CPU
- Extract hidden states from the last layer for 100 diverse prompts
- Train a codec (Procrustes initialisation + fine-tuning with composite loss)
- Measure roundtrip reconstruction loss
- Optionally load distilgpt2 and test cross-model alignment
"""

from __future__ import annotations

import os
import json
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.codecs import AgentCodec
from quorum.codec_training import (
    procrustes_init,
    composite_loss,
    CodecTrainingConfig,
    train_codec_pair,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DIVERSE_PROMPTS = [
    "The capital of France is",
    "In quantum mechanics, the wave function",
    "The recipe for chocolate cake requires",
    "According to the second law of thermodynamics",
    "The stock market today showed signs of",
    "Machine learning algorithms can be divided into",
    "The history of ancient Rome begins with",
    "In Python, a list comprehension is",
    "The human brain contains approximately",
    "Climate change is primarily caused by",
    "The fastest animal on Earth is",
    "Shakespeare wrote many famous plays including",
    "The periodic table organizes elements by",
    "Artificial intelligence was first conceptualized",
    "The Great Wall of China was built to",
    "In mathematics, a prime number is",
    "The solar system consists of eight",
    "DNA replication begins when the enzyme",
    "The Renaissance period was characterized by",
    "Photosynthesis converts sunlight into",
    "The speed of light in a vacuum is",
    "Democracy as a form of government originated",
    "Neural networks are inspired by the structure",
    "The Amazon rainforest produces approximately",
    "Einstein's theory of relativity states that",
    "The Internet was originally developed as",
    "Vaccines work by stimulating the immune",
    "The Mona Lisa was painted by",
    "Plate tectonics explains how the Earth's",
    "The concept of zero was invented by",
    "Black holes are regions of spacetime where",
    "The Industrial Revolution began in",
    "Bacteria can be classified as gram-positive",
    "The United Nations was established in",
    "Fibonacci numbers appear frequently in",
    "The human genome contains approximately",
    "Water molecules are held together by",
    "The French Revolution started in",
    "Convolutional neural networks are particularly",
    "The deepest point in the ocean is",
    "Gravity is described by Newton's law as",
    "The printing press was invented by",
    "Mitochondria are often called the powerhouse",
    "The Pythagorean theorem states that",
    "Global warming has led to rising sea",
    "The Wright brothers achieved the first",
    "In economics, supply and demand determine",
    "The Hubble Space Telescope has captured",
    "Antibiotics should not be used to treat",
    "The theory of evolution was proposed by",
    "Cryptocurrency relies on blockchain technology to",
    "The human heart beats approximately",
    "The Treaty of Versailles was signed in",
    "Quantum computing uses qubits instead of",
    "The largest planet in our solar system",
    "Insulin is produced by the pancreas and",
    "The Cold War lasted from approximately",
    "Machine translation has improved significantly with",
    "The Nile River is the longest river",
    "Magnetic resonance imaging uses strong magnetic",
    "The Declaration of Independence was signed",
    "Transformer models use self-attention mechanisms",
    "The boiling point of water at sea level",
    "The Renaissance began in Italy during",
    "Genetic engineering allows scientists to modify",
    "The Theory of Everything attempts to unify",
    "Social media platforms have transformed how",
    "The Great Barrier Reef is the largest",
    "Reinforcement learning agents learn through",
    "The Roman Empire fell in the year",
    "CRISPR-Cas9 is a revolutionary gene editing",
    "The Magna Carta was signed in",
    "Cloud computing provides on-demand access to",
    "The circulatory system transports blood throughout",
    "World War II ended in the year",
    "Natural language processing enables computers to",
    "The Antarctic ice sheet contains enough water",
    "The theory of plate tectonics explains",
    "Autonomous vehicles use sensors and AI to",
    "The Silk Road was an ancient trade",
    "Stem cells have the unique ability to",
    "The Enlightenment was an intellectual movement",
    "5G networks provide significantly faster data",
    "The human eye can distinguish approximately",
    "The space race between the US and USSR",
    "Nanotechnology manipulates matter at the atomic",
    "The Rosetta Stone helped scholars decipher",
    "Deep learning has achieved remarkable results in",
    "The Pacific Ocean is the largest and deepest",
    "Inflation occurs when the general price level",
    "The Apollo 11 mission successfully landed on",
    "Photovoltaic cells convert sunlight directly into",
    "The Ottoman Empire lasted for over six",
    "Graph neural networks can model relationships",
    "The Grand Canyon was formed by the Colorado",
    "Antibodies are proteins produced by the immune",
    "The Babylonians developed one of the earliest",
    "Edge computing processes data closer to where",
    "The Amazon River is the largest river by",
    "Epigenetics studies changes in gene expression",
]


def extract_hidden_states(model_name: str, prompts: list[str], device: str = "cpu") -> torch.Tensor:
    """Extract last-layer hidden states from a HuggingFace model."""
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_states = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.mean(dim=1).squeeze(0)
            hidden_states.append(pooled)
            if (i + 1) % 25 == 0:
                print(f"    Extracted {i + 1}/{len(prompts)} hidden states")

    stacked = torch.stack(hidden_states)
    print(f"  Hidden state stats: shape={stacked.shape}, "
          f"mean={stacked.mean():.4f}, std={stacked.std():.4f}, "
          f"min={stacked.min():.4f}, max={stacked.max():.4f}, "
          f"norm_mean={stacked.norm(dim=-1).mean():.4f}")
    return stacked


def train_and_evaluate_codec(
    hidden_states: torch.Tensor,
    agent_dim: int,
    buffer_dim: int = 512,
    num_steps: int = 3000,
) -> dict:
    """Train a codec and evaluate roundtrip reconstruction."""
    N = hidden_states.shape[0]
    train_n = int(0.8 * N)
    train_data = hidden_states[:train_n]
    test_data = hidden_states[train_n:]

    codec = AgentCodec(agent_dim=agent_dim, buffer_dim=buffer_dim)

    target_states = torch.randn(train_n, buffer_dim)
    target_states = target_states / target_states.norm(dim=-1, keepdim=True)
    procrustes_init(codec, train_data, target_states)

    optimizer = torch.optim.Adam(codec.parameters(), lr=1e-4)
    train_losses = []

    print(f"  Training codec: agent_dim={agent_dim}, buffer_dim={buffer_dim}, steps={num_steps}")
    print(f"  Train samples: {train_n}, Test samples: {N - train_n}")
    print(f"  Train data stats: mean={train_data.mean():.4f}, var={train_data.var():.4f}")
    for step in range(num_steps):
        idx = torch.randint(0, train_n, (min(32, train_n),))
        batch = train_data[idx]

        loss, components = composite_loss(codec, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0:
            train_losses.append(components["reconstruction"])
            print(f"    Step {step:5d}/{num_steps}: recon_loss={components['reconstruction']:.6f}")

    with torch.no_grad():
        test_recon = codec.roundtrip(test_data)
        test_loss = (test_recon - test_data).pow(2).mean().item()
        test_data_var = test_data.var().item()
        normalised_loss = test_loss / max(test_data_var, 1e-8)

        cos_sims = torch.cosine_similarity(test_recon, test_data, dim=-1)
        mean_cos_sim = cos_sims.mean().item()

        print(f"  --- Evaluation on {test_data.shape[0]} held-out samples ---")
        print(f"  Test data variance: {test_data_var:.6f}")
        print(f"  Raw MSE: {test_loss:.6f}")
        print(f"  Normalised loss (MSE / var): {normalised_loss:.6f}")
        print(f"  Per-sample cosine similarities: {cos_sims.tolist()}")
        print(f"  Cosine sim: mean={mean_cos_sim:.6f}, min={cos_sims.min():.6f}, max={cos_sims.max():.6f}")

    return {
        "train_losses": train_losses,
        "test_mse": test_loss,
        "test_normalised_loss": normalised_loss,
        "mean_cosine_similarity": mean_cos_sim,
        "data_variance": test_data_var,
    }


def main():
    print("=" * 60)
    print("CODEC ROUNDTRIP EXPERIMENT")
    print("=" * 60)

    prompts = DIVERSE_PROMPTS[:100]
    results = {}

    # Experiment 1: GPT-2 small codec
    print("\n--- GPT-2 small codec training ---")
    try:
        gpt2_states = extract_hidden_states("gpt2", prompts)
        print(f"  Hidden states shape: {gpt2_states.shape}")
        gpt2_dim = gpt2_states.shape[1]

        gpt2_results = train_and_evaluate_codec(gpt2_states, agent_dim=gpt2_dim)
        results["gpt2"] = {
            "agent_dim": gpt2_dim,
            "test_mse": gpt2_results["test_mse"],
            "normalised_loss": gpt2_results["test_normalised_loss"],
            "mean_cosine_similarity": gpt2_results["mean_cosine_similarity"],
            "data_variance": gpt2_results["data_variance"],
        }
        print(f"  Test MSE: {gpt2_results['test_mse']:.6f}")
        print(f"  Normalised loss: {gpt2_results['test_normalised_loss']:.6f}")
        print(f"  Mean cosine similarity: {gpt2_results['mean_cosine_similarity']:.4f}")
        success = gpt2_results["test_normalised_loss"] < 0.1
        print(f"  Target (normalised < 0.1): {'PASS' if success else 'FAIL'}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["gpt2"] = {"error": str(e)}

    # Experiment 2: DistilGPT-2 codec
    print("\n--- DistilGPT-2 codec training ---")
    try:
        distil_states = extract_hidden_states("distilgpt2", prompts)
        print(f"  Hidden states shape: {distil_states.shape}")
        distil_dim = distil_states.shape[1]

        distil_results = train_and_evaluate_codec(distil_states, agent_dim=distil_dim)
        results["distilgpt2"] = {
            "agent_dim": distil_dim,
            "test_mse": distil_results["test_mse"],
            "normalised_loss": distil_results["test_normalised_loss"],
            "mean_cosine_similarity": distil_results["mean_cosine_similarity"],
        }
        print(f"  Test MSE: {distil_results['test_mse']:.6f}")
        print(f"  Normalised loss: {distil_results['test_normalised_loss']:.6f}")
        print(f"  Mean cosine similarity: {distil_results['mean_cosine_similarity']:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["distilgpt2"] = {"error": str(e)}

    # Experiment 3: Cross-model alignment
    if "error" not in results.get("gpt2", {}) and "error" not in results.get("distilgpt2", {}):
        print("\n--- Cross-model alignment test ---")
        try:
            buffer_dim = 512
            codec_gpt2 = AgentCodec(agent_dim=gpt2_dim, buffer_dim=buffer_dim)
            codec_distil = AgentCodec(agent_dim=distil_dim, buffer_dim=buffer_dim)

            config = CodecTrainingConfig(lr=1e-4, num_steps=2000, log_every=200)
            history = train_codec_pair(
                codec_gpt2, codec_distil,
                gpt2_states[:80], distil_states[:80],
                config,
            )

            with torch.no_grad():
                z_gpt2 = codec_gpt2.encode(gpt2_states[80:])
                z_distil = codec_distil.encode(distil_states[80:])
                alignment_sim = torch.cosine_similarity(z_gpt2, z_distil, dim=-1).mean().item()

                cross_decoded = codec_distil.decode(z_gpt2)
                cross_mse = (cross_decoded - distil_states[80:]).pow(2).mean().item()

            results["cross_model"] = {
                "alignment_cosine_similarity": alignment_sim,
                "cross_decode_mse": cross_mse,
                "training_steps": len(history),
                "final_loss": history[-1]["total"] if history else None,
            }
            print(f"  Alignment cosine similarity: {alignment_sim:.4f}")
            print(f"  Cross-decode MSE: {cross_mse:.6f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results["cross_model"] = {"error": str(e)}

    # Save results
    print("\n" + "=" * 60)
    with open(RESULTS_DIR / "codec_roundtrip_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to experiments/results/codec_roundtrip_results.json")


if __name__ == "__main__":
    main()
