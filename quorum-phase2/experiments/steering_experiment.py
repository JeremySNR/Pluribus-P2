"""Experiment C: Test whether the converged buffer improves downstream generation.

Uses the converged latent buffer as a steering signal for GPT-2 generation,
comparing steered vs unsteered outputs across multiple prompts.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.codecs import AgentCodec
from quorum.codec_training import procrustes_init, composite_loss, train_codec_pair, CodecTrainingConfig
from quorum.loop import latent_resonance_loop, SyntheticAgent, LoopConfig

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_MODEL_CACHE: dict = {}
_HIDDEN_CACHE: dict = {}

TRAINING_PROMPTS = [
    "The capital of France is", "In quantum mechanics, the wave function",
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


def _load(model_name):
    if model_name not in _MODEL_CACHE:
        from transformers import AutoTokenizer, AutoModel
        print(f"  Loading {model_name} ...")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name); mdl.eval()
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        _MODEL_CACHE[model_name] = (mdl, tok)
    return _MODEL_CACHE[model_name]


def _load_lm(model_name):
    key = model_name + "_lm"
    if key not in _MODEL_CACHE:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"  Loading {model_name} (causal LM head) ...")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name); mdl.eval()
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        _MODEL_CACHE[key] = (mdl, tok)
    return _MODEL_CACHE[key]


def _extract(model_name, prompts):
    key = (model_name, tuple(prompts))
    if key not in _HIDDEN_CACHE:
        mdl, tok = _load(model_name)
        hs = []
        with torch.no_grad():
            for p in prompts:
                inp = tok(p, return_tensors="pt", truncation=True, max_length=64)
                out = mdl(**inp, output_hidden_states=True)
                hs.append(out.hidden_states[-1][0, -1, :])
        _HIDDEN_CACHE[key] = torch.stack(hs)
    return _HIDDEN_CACHE[key]


def _train_pair(states_a, states_b, buffer_dim=512, num_steps=3000):
    da, db = states_a.shape[1], states_b.shape[1]
    ca = AgentCodec(agent_dim=da, buffer_dim=buffer_dim)
    cb = AgentCodec(agent_dim=db, buffer_dim=buffer_dim)
    cfg = CodecTrainingConfig(lr=1e-4, num_steps=num_steps, log_every=500)
    train_codec_pair(ca, cb, states_a, states_b, cfg)
    return ca, cb


def run_convergence_loop(task_prompt, codec_gpt2, codec_distil, D=512, S=6):
    """Run the latent resonance loop and return the converged buffer."""
    gpt2_task = _extract("gpt2", [task_prompt])
    distil_task = _extract("distilgpt2", [task_prompt])

    agents = []
    for i, (name, h_raw, codec) in enumerate([
        ("gpt2_A", gpt2_task.squeeze(0), codec_gpt2),
        ("gpt2_B", gpt2_task.squeeze(0), codec_gpt2),
        ("distil_A", distil_task.squeeze(0), codec_distil),
        ("distil_B", distil_task.squeeze(0), codec_distil),
    ]):
        quality = 0.7 if "gpt2" in name else 0.6

        def make_fn(agent_codec, agent_h):
            def fn(B, t):
                with torch.no_grad():
                    fresh_z = agent_codec.encode(agent_h.unsqueeze(0)).squeeze(0)
                fresh_enc = fresh_z.expand(B.shape[0], -1)
                return (fresh_enc - B) * 0.3 + torch.randn_like(B) * 0.01
            return fn

        agents.append(SyntheticAgent(
            agent_id=i, delta_fn=make_fn(codec, h_raw), quality=quality
        ))

    cfg = LoopConfig(S=S, D=D, max_rounds=30, tol=0.02, alpha=0.5)
    B_final, diag = latent_resonance_loop(agents, cfg)
    return B_final, diag


def generate_text(lm_model, tokenizer, prompt, max_new_tokens=50):
    """Standard unsteered generation."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = lm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


def generate_reranked(lm_model, tokenizer, prompt, steering_hidden, n_candidates=10, max_new_tokens=50):
    """Generate N candidates and pick the one closest to the steering hidden state."""
    inputs = tokenizer(prompt, return_tensors="pt")
    candidates = []
    candidate_hiddens = []

    base_model_name = "gpt2"
    base_model, _ = _load(base_model_name)

    for _ in range(n_candidates):
        with torch.no_grad():
            output_ids = lm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        candidates.append(gen_text)

        with torch.no_grad():
            gen_inputs = tokenizer(gen_text, return_tensors="pt", truncation=True, max_length=128)
            gen_out = base_model(**gen_inputs, output_hidden_states=True)
            gen_hidden = gen_out.hidden_states[-1][0, -1, :]
            candidate_hiddens.append(gen_hidden)

    steering_flat = steering_hidden.flatten()
    best_idx = -1
    best_sim = -1.0
    sims = []
    for idx, ch in enumerate(candidate_hiddens):
        sim = F.cosine_similarity(ch.unsqueeze(0), steering_flat.unsqueeze(0)).item()
        sims.append(sim)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    return candidates[best_idx], best_sim, candidates, sims


def main():
    print("=" * 70)
    print("EXPERIMENT C: Buffer as downstream steering object")
    print("=" * 70)

    prompts = TRAINING_PROMPTS[:100]
    gpt2_states = _extract("gpt2", prompts)
    distil_states = _extract("distilgpt2", prompts)

    print("\n  Training aligned codec pair ...")
    codec_gpt2, codec_distil = _train_pair(
        gpt2_states[:80], distil_states[:80], buffer_dim=512, num_steps=3000
    )

    task_prompts = [
        "What are the benefits and risks of remote work?",
        "Explain the causes and consequences of climate change.",
        "What are the key differences between capitalism and socialism?",
    ]

    lm_model, tokenizer = _load_lm("gpt2")
    results = {}

    for task_prompt in task_prompts:
        print(f"\n{'='*60}")
        print(f"  Task: '{task_prompt}'")
        print(f"{'='*60}")

        print(f"  Running convergence loop ...")
        B_final, diag = run_convergence_loop(
            task_prompt, codec_gpt2, codec_distil
        )
        print(f"  Converged in {len(diag.residuals)} rounds, residual={diag.residuals[-1]:.6f}")

        with torch.no_grad():
            decoded_h = codec_gpt2.decode(B_final[0].unsqueeze(0)).squeeze(0)
        print(f"  Decoded buffer hidden norm: {decoded_h.norm():.4f}")

        print(f"\n  --- Unsteered generation ---")
        unsteered = generate_text(lm_model, tokenizer, task_prompt)
        print(f"  Output: {unsteered[:200]}...")

        print(f"\n  --- Steered generation (reranking, N=10) ---")
        steered, best_sim, all_candidates, all_sims = generate_reranked(
            lm_model, tokenizer, task_prompt, decoded_h, n_candidates=10
        )
        print(f"  Best candidate similarity to buffer: {best_sim:.4f}")
        print(f"  Similarity range: [{min(all_sims):.4f}, {max(all_sims):.4f}]")
        print(f"  Output: {steered[:200]}...")

        with torch.no_grad():
            un_inputs = tokenizer(task_prompt + unsteered, return_tensors="pt", truncation=True, max_length=128)
            base_model, _ = _load("gpt2")
            un_out = base_model(**un_inputs, output_hidden_states=True)
            un_hidden = un_out.hidden_states[-1][0, -1, :]
            unsteered_sim = F.cosine_similarity(
                un_hidden.unsqueeze(0), decoded_h.unsqueeze(0)
            ).item()

        print(f"\n  Unsteered output similarity to buffer: {unsteered_sim:.4f}")
        print(f"  Steered output similarity to buffer:   {best_sim:.4f}")
        print(f"  Improvement: {best_sim - unsteered_sim:+.4f}")

        results[task_prompt] = {
            "rounds_to_converge": len(diag.residuals),
            "final_residual": diag.residuals[-1],
            "decoded_norm": decoded_h.norm().item(),
            "unsteered_text": unsteered[:500],
            "steered_text": steered[:500],
            "unsteered_sim_to_buffer": unsteered_sim,
            "steered_sim_to_buffer": best_sim,
            "sim_improvement": best_sim - unsteered_sim,
            "all_candidate_sims": all_sims,
        }

    with open(RESULTS_DIR / "steering_experiment.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to steering_experiment.json")

    print(f"\n{'='*70}")
    print(f"STEERING EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for prompt, r in results.items():
        print(f"\n  Task: {prompt[:60]}...")
        print(f"    Unsteered sim: {r['unsteered_sim_to_buffer']:.4f}")
        print(f"    Steered sim:   {r['steered_sim_to_buffer']:.4f}")
        print(f"    Improvement:   {r['sim_improvement']:+.4f}")


if __name__ == "__main__":
    main()
