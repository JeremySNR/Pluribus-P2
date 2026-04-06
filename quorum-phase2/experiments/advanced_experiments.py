"""Advanced experiments for Latent Resonance Loop validation.

Experiments 1-6: Cross-model full loop, semantic coherence, adversarial
alignment, TIES with real competing representations, anchor pair scaling,
buffer dimensionality sensitivity.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.codecs import AgentCodec
from quorum.codec_training import (
    procrustes_init, composite_loss, train_codec_pair, CodecTrainingConfig,
)
from quorum.ties_resolve import ties_resolve
from quorum.loop import latent_resonance_loop, SyntheticAgent, LoopConfig

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── shared model infrastructure ──────────────────────────────────────

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
                hs.append(out.hidden_states[-1].mean(dim=1).squeeze(0))
        _HIDDEN_CACHE[key] = torch.stack(hs)
    return _HIDDEN_CACHE[key]


def _train_codec(states, buffer_dim=512, num_steps=3000, verbose=False):
    agent_dim = states.shape[1]
    codec = AgentCodec(agent_dim=agent_dim, buffer_dim=buffer_dim)
    n = states.shape[0]
    tgt = torch.randn(n, buffer_dim)
    tgt = tgt / tgt.norm(dim=-1, keepdim=True)
    procrustes_init(codec, states, tgt)
    opt = torch.optim.Adam(codec.parameters(), lr=1e-4)
    losses = []
    for step in range(num_steps):
        idx = torch.randint(0, n, (min(32, n),))
        loss, comp = composite_loss(codec, states[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        opt.step()
        if step % max(num_steps // 10, 1) == 0:
            losses.append(comp["reconstruction"])
            if verbose:
                print(f"    Step {step:5d}/{num_steps}: recon={comp['reconstruction']:.6f}")
    return codec, losses


def _train_pair(states_a, states_b, buffer_dim=512, num_steps=3000):
    da, db = states_a.shape[1], states_b.shape[1]
    ca = AgentCodec(agent_dim=da, buffer_dim=buffer_dim)
    cb = AgentCodec(agent_dim=db, buffer_dim=buffer_dim)
    cfg = CodecTrainingConfig(lr=1e-4, num_steps=num_steps, log_every=500)
    train_codec_pair(ca, cb, states_a, states_b, cfg)
    return ca, cb


# =====================================================================
# EXPERIMENT 1 — Cross-model full loop
# =====================================================================
def experiment_1():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Cross-model full loop with real GPT-2 / DistilGPT-2")
    print("=" * 70)

    prompts = TRAINING_PROMPTS[:100]
    gpt2_states = _extract("gpt2", prompts)
    distil_states = _extract("distilgpt2", prompts)
    print(f"  GPT-2 hidden states: {gpt2_states.shape}")
    print(f"  DistilGPT-2 hidden states: {distil_states.shape}")

    D = 512
    print(f"\n  Training aligned codec pair (D={D}, 3000 steps) ...")
    codec_gpt2, codec_distil = _train_pair(
        gpt2_states[:80], distil_states[:80], buffer_dim=D, num_steps=3000
    )

    task_prompt = "What are the benefits and risks of remote work?"
    print(f"\n  Task prompt: '{task_prompt}'")

    gpt2_task = _extract("gpt2", [task_prompt])    # [1, 768]
    distil_task = _extract("distilgpt2", [task_prompt])  # [1, 768]

    with torch.no_grad():
        z_gpt2  = codec_gpt2.encode(gpt2_task)      # [1, D]
        z_distil = codec_distil.encode(distil_task)  # [1, D]

    print(f"  GPT-2 encoding norm:   {z_gpt2.norm():.4f}")
    print(f"  DistilGPT-2 encoding norm: {z_distil.norm():.4f}")
    print(f"  Cross-model cosine sim of task encodings: "
          f"{F.cosine_similarity(z_gpt2, z_distil).item():.4f}")

    S = 6
    initial_encodings = {}
    agents = []
    for i, (name, z) in enumerate([
        ("gpt2_A", z_gpt2), ("gpt2_B", z_gpt2),
        ("distil_A", z_distil), ("distil_B", z_distil),
    ]):
        enc = z.squeeze(0).expand(S, -1).clone()
        enc = enc + torch.randn_like(enc) * 0.02 * enc.norm()
        initial_encodings[name] = enc.clone()
        quality = 0.7 if "gpt2" in name else 0.6

        def make_fn(target_enc):
            def fn(B, t):
                return (target_enc - B) * 0.3 + torch.randn_like(B) * 0.01
            return fn

        agents.append(SyntheticAgent(
            agent_id=i, delta_fn=make_fn(enc), quality=quality
        ))

    cfg = LoopConfig(S=S, D=D, max_rounds=30, tol=0.02, alpha=0.5)
    print(f"\n  Running loop with 4 agents (2x GPT-2, 2x DistilGPT-2) ...")
    B_final, diag = latent_resonance_loop(agents, cfg)

    print(f"\n  --- Results ---")
    print(f"  Rounds: {len(diag.residuals)}")
    print(f"  Halt reason: {diag.halt_reason}")
    print(f"  Final residual: {diag.residuals[-1]:.6f}")
    print(f"  Residual trajectory: {[f'{r:.4f}' for r in diag.residuals]}")

    print(f"\n  Cosine similarity of converged buffer to each agent's initial encoding:")
    sims = {}
    for name, enc in initial_encodings.items():
        sim = F.cosine_similarity(B_final.flatten().unsqueeze(0),
                                   enc.flatten().unsqueeze(0)).item()
        sims[name] = sim
        print(f"    {name}: {sim:.4f}")

    sim_values = list(sims.values())
    spread = max(sim_values) - min(sim_values)
    print(f"\n  Similarity spread (max - min): {spread:.4f}")
    if spread < 0.01:
        print("  VERDICT: Averaging — converged state equidistant from all agents")
    else:
        closest = max(sims, key=sims.get)
        print(f"  VERDICT: Competitive selection — closest to {closest}")

    final_strengths = diag.strengths[-1] if diag.strengths else None
    if final_strengths is not None:
        print(f"  Final agent strengths: {final_strengths.tolist()}")

    results = {
        "rounds": len(diag.residuals),
        "halt_reason": diag.halt_reason,
        "final_residual": diag.residuals[-1],
        "residual_trajectory": diag.residuals,
        "cosine_sims_to_initial": sims,
        "similarity_spread": spread,
        "final_strengths": final_strengths.tolist() if final_strengths is not None else None,
    }
    with open(RESULTS_DIR / "exp1_crossmodel_loop.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to exp1_crossmodel_loop.json")
    return B_final, codec_gpt2, codec_distil


# =====================================================================
# EXPERIMENT 2 — Semantic coherence of converged buffer
# =====================================================================
def experiment_2(B_final, codec_gpt2):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Semantic coherence — decode buffer to token logits")
    print("=" * 70)

    lm_model, tokenizer = _load_lm("gpt2")

    slot_names = [
        "FACTUAL_GROUNDING", "REASONING_CHAIN", "UNCERTAINTY_MAP",
        "META_COORDINATION", "COMPETITIVE_ARENA", "DISSENTER_CHANNEL",
    ]

    results = {}
    for slot_idx, slot_name in enumerate(slot_names):
        buffer_slot = B_final[slot_idx]  # [D]
        with torch.no_grad():
            decoded_h = codec_gpt2.decode(buffer_slot.unsqueeze(0))  # [1, 768]

        # Use GPT-2's lm_head to get logits from this hidden state
        with torch.no_grad():
            logits = lm_model.lm_head(decoded_h)  # [1, vocab_size]
            probs = F.softmax(logits, dim=-1)
            top_probs, top_ids = probs.topk(10, dim=-1)

        top_tokens = [tokenizer.decode([tid]) for tid in top_ids[0]]
        top_probs_list = top_probs[0].tolist()

        print(f"\n  Slot {slot_idx} ({slot_name}):")
        print(f"    Decoded hidden state norm: {decoded_h.norm():.4f}")
        for rank, (tok, prob) in enumerate(zip(top_tokens, top_probs_list)):
            print(f"    #{rank+1}: '{tok}' (p={prob:.4f})")

        results[slot_name] = {
            "decoded_norm": decoded_h.norm().item(),
            "top_tokens": top_tokens,
            "top_probs": top_probs_list,
        }

    # Also decode an actual GPT-2 hidden state for comparison
    print(f"\n  --- Baseline: actual GPT-2 hidden state for reference ---")
    task_prompt = "What are the benefits and risks of remote work?"
    mdl, tok = _load("gpt2")
    with torch.no_grad():
        inp = tok(task_prompt, return_tensors="pt", truncation=True, max_length=64)
        out = mdl(**inp, output_hidden_states=True)
        real_h = out.hidden_states[-1][0, -1, :]  # last token hidden state [768]
        logits = lm_model.lm_head(real_h.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(10, dim=-1)
    top_tokens_real = [tok.decode([tid]) for tid in top_ids[0]]
    print(f"    Real hidden state norm: {real_h.norm():.4f}")
    for rank, (t, p) in enumerate(zip(top_tokens_real, top_probs[0].tolist())):
        print(f"    #{rank+1}: '{t}' (p={p:.4f})")
    results["baseline_real_hidden"] = {
        "norm": real_h.norm().item(),
        "top_tokens": top_tokens_real,
        "top_probs": top_probs[0].tolist(),
    }

    with open(RESULTS_DIR / "exp2_semantic_coherence.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to exp2_semantic_coherence.json")


# =====================================================================
# EXPERIMENT 3 — Adversarial codec alignment
# =====================================================================
def experiment_3():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Adversarial codec alignment — noise robustness")
    print("=" * 70)

    prompts = TRAINING_PROMPTS[:100]
    gpt2_s = _extract("gpt2", prompts)
    distil_s = _extract("distilgpt2", prompts)

    print(f"  Training aligned codec pair ...")
    ca, cb = _train_pair(gpt2_s[:80], distil_s[:80], buffer_dim=512, num_steps=3000)

    test_gpt2 = gpt2_s[80:]
    test_distil = distil_s[80:]
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.5]

    print(f"\n  Testing cross-decode with Gaussian noise on {test_gpt2.shape[0]} samples")
    results = {}
    for noise_std_mult in noise_levels:
        cos_sims = []
        mses = []
        with torch.no_grad():
            z_gpt2 = ca.encode(test_gpt2)  # [N, D]
            enc_norm = z_gpt2.norm(dim=-1, keepdim=True).mean()
            noise = torch.randn_like(z_gpt2) * noise_std_mult * enc_norm
            z_noisy = z_gpt2 + noise
            decoded = cb.decode(z_noisy)  # cross-decode to distilgpt2 space
            for i in range(test_distil.shape[0]):
                cs = F.cosine_similarity(
                    decoded[i:i+1], test_distil[i:i+1]
                ).item()
                mse = (decoded[i] - test_distil[i]).pow(2).mean().item()
                cos_sims.append(cs)
                mses.append(mse)

        mean_sim = sum(cos_sims) / len(cos_sims)
        mean_mse = sum(mses) / len(mses)
        print(f"  noise={noise_std_mult:.2f}x enc_norm: "
              f"cos_sim={mean_sim:.4f}, MSE={mean_mse:.4f}")
        results[f"noise_{noise_std_mult}"] = {
            "noise_multiplier": noise_std_mult,
            "mean_cosine_similarity": mean_sim,
            "mean_mse": mean_mse,
            "per_sample_cosine": cos_sims,
        }

    # Plot degradation curve
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = noise_levels
    ys = [results[f"noise_{n}"]["mean_cosine_similarity"] for n in xs]
    ax.plot(xs, ys, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Noise level (× encoding norm)")
    ax.set_ylabel("Cross-decode cosine similarity")
    ax.set_title("Codec Alignment Robustness to Buffer Perturbation")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_noise_robustness.png", dpi=150)
    plt.close(fig)

    with open(RESULTS_DIR / "exp3_adversarial_alignment.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to exp3_adversarial_alignment.json + exp3_noise_robustness.png")


# =====================================================================
# EXPERIMENT 4 — TIES-Resolve with real competing representations
# =====================================================================
def experiment_4():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: TIES-Resolve with real competing semantic reps")
    print("=" * 70)

    optimistic = "The economy is growing strongly and unemployment is falling"
    pessimistic = "The economy is in recession and inflation is rising"

    print(f"  Optimistic: '{optimistic}'")
    print(f"  Pessimistic: '{pessimistic}'")

    gpt2_opt = _extract("gpt2", [optimistic])   # [1, 768]
    gpt2_pes = _extract("gpt2", [pessimistic])  # [1, 768]

    raw_sim = F.cosine_similarity(gpt2_opt, gpt2_pes).item()
    print(f"  Raw cosine similarity between prompts: {raw_sim:.4f}")

    train_states = _extract("gpt2", TRAINING_PROMPTS[:100])
    codec, _ = _train_codec(train_states[:80], buffer_dim=512, num_steps=3000)

    with torch.no_grad():
        z_opt = codec.encode(gpt2_opt)    # [1, 512]
        z_pes = codec.encode(gpt2_pes)    # [1, 512]
    encoded_sim = F.cosine_similarity(z_opt, z_pes).item()
    print(f"  Encoded cosine similarity: {encoded_sim:.4f}")

    # 5 agents: 3 optimistic, 2 pessimistic
    S = 6
    D = 512
    deltas = []
    for i in range(3):
        d = z_opt.squeeze(0).expand(S, -1).clone()
        d = d + torch.randn_like(d) * 0.02
        deltas.append(d)
    for i in range(2):
        d = z_pes.squeeze(0).expand(S, -1).clone()
        d = d + torch.randn_like(d) * 0.02
        deltas.append(d)

    print(f"\n  Running TIES-Resolve with 3 optimistic + 2 pessimistic agents ...")
    resolved = ties_resolve(deltas, density=0.3)

    sim_to_opt = F.cosine_similarity(
        resolved.flatten().unsqueeze(0),
        deltas[0].flatten().unsqueeze(0)
    ).item()
    sim_to_pes = F.cosine_similarity(
        resolved.flatten().unsqueeze(0),
        deltas[3].flatten().unsqueeze(0)
    ).item()
    avg_of_all = torch.stack(deltas).mean(dim=0)
    sim_to_avg = F.cosine_similarity(
        resolved.flatten().unsqueeze(0),
        avg_of_all.flatten().unsqueeze(0)
    ).item()

    print(f"  Resolved sim to optimistic: {sim_to_opt:.4f}")
    print(f"  Resolved sim to pessimistic: {sim_to_pes:.4f}")
    print(f"  Resolved sim to naive average: {sim_to_avg:.4f}")

    if sim_to_opt > sim_to_pes:
        print(f"  VERDICT: Majority (optimistic) wins — TIES sign-election working")
    else:
        print(f"  VERDICT: Minority won — unexpected")

    # Decode resolved back to token logits
    print(f"\n  Decoding resolved state to token logits via GPT-2 LM head ...")
    lm_model, tokenizer = _load_lm("gpt2")
    with torch.no_grad():
        resolved_slot0 = resolved[0]  # [512] — factual grounding slot
        decoded_h = codec.decode(resolved_slot0.unsqueeze(0))  # [1, 768]
        logits = lm_model.lm_head(decoded_h)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(10, dim=-1)
    top_tokens = [tokenizer.decode([tid]) for tid in top_ids[0]]
    print(f"  Top 10 tokens from resolved state:")
    for rank, (tok, prob) in enumerate(zip(top_tokens, top_probs[0].tolist())):
        print(f"    #{rank+1}: '{tok}' (p={prob:.4f})")

    # For comparison, decode the optimistic and pessimistic directly
    for label, z in [("optimistic", z_opt), ("pessimistic", z_pes)]:
        with torch.no_grad():
            h = codec.decode(z)
            logits = lm_model.lm_head(h)
            probs = F.softmax(logits, dim=-1)
            tp, ti = probs.topk(5, dim=-1)
        toks = [tokenizer.decode([t]) for t in ti[0]]
        print(f"  Top 5 from {label}: {toks}")

    results = {
        "raw_cosine_sim": raw_sim,
        "encoded_cosine_sim": encoded_sim,
        "resolved_sim_to_optimistic": sim_to_opt,
        "resolved_sim_to_pessimistic": sim_to_pes,
        "resolved_sim_to_average": sim_to_avg,
        "majority_wins": sim_to_opt > sim_to_pes,
        "top_tokens_resolved": top_tokens,
        "top_probs_resolved": top_probs[0].tolist(),
    }
    with open(RESULTS_DIR / "exp4_ties_competing.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to exp4_ties_competing.json")


# =====================================================================
# EXPERIMENT 5 — Anchor pair scaling
# =====================================================================
def experiment_5():
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Codec quality vs number of anchor pairs")
    print("=" * 70)

    all_states = _extract("gpt2", TRAINING_PROMPTS[:100])
    test_data = all_states[80:]  # 20 test samples
    test_var = test_data.var().item()

    anchor_counts = [10, 25, 50, 75, 100]
    # We can only go up to 100 with our prompt list, but that covers the range
    results = {}

    for n_anchors in anchor_counts:
        train_data = all_states[:n_anchors]
        codec, losses = _train_codec(train_data, buffer_dim=512, num_steps=3000)

        with torch.no_grad():
            recon = codec.roundtrip(test_data)
            mse = (recon - test_data).pow(2).mean().item()
            norm_loss = mse / max(test_var, 1e-8)
            cos_sims = F.cosine_similarity(recon, test_data, dim=-1)
            mean_cos = cos_sims.mean().item()

        print(f"  n_anchors={n_anchors:4d}: MSE={mse:.4f}, "
              f"norm_loss={norm_loss:.6f}, cos_sim={mean_cos:.6f}")
        results[str(n_anchors)] = {
            "n_anchors": n_anchors,
            "test_mse": mse,
            "normalised_loss": norm_loss,
            "mean_cosine_sim": mean_cos,
            "training_losses": losses,
        }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ns = anchor_counts
    mses = [results[str(n)]["test_mse"] for n in ns]
    coss = [results[str(n)]["mean_cosine_sim"] for n in ns]

    ax1.plot(ns, mses, "ro-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of anchor pairs")
    ax1.set_ylabel("Test MSE")
    ax1.set_title("Reconstruction Loss vs Anchor Pairs")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ns, coss, "bo-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of anchor pairs")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("Reconstruction Quality vs Anchor Pairs")
    ax2.set_ylim(0.99, 1.001)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_anchor_scaling.png", dpi=150)
    plt.close(fig)

    # Find the knee
    best_cos = coss[-1]
    for i, (n, c) in enumerate(zip(ns, coss)):
        if c >= 0.9 * best_cos + 0.1:
            print(f"\n  Knee of curve: ~{n} anchors gives {c:.6f} cos_sim "
                  f"({(c/best_cos)*100:.1f}% of best)")
            break

    with open(RESULTS_DIR / "exp5_anchor_scaling.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to exp5_anchor_scaling.json + exp5_anchor_scaling.png")


# =====================================================================
# EXPERIMENT 6 — Buffer dimensionality sensitivity
# =====================================================================
def experiment_6():
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Buffer dimensionality sensitivity")
    print("=" * 70)

    gpt2_states = _extract("gpt2", TRAINING_PROMPTS[:100])
    distil_states = _extract("distilgpt2", TRAINING_PROMPTS[:100])
    test_gpt2 = gpt2_states[80:]
    test_distil = distil_states[80:]
    test_var = test_gpt2.var().item()

    dims = [64, 128, 256, 512, 768, 1024, 2048]
    results = {}

    for D in dims:
        print(f"\n  D={D}:")

        # Single-model roundtrip
        codec, _ = _train_codec(gpt2_states[:80], buffer_dim=D, num_steps=2000)
        with torch.no_grad():
            recon = codec.roundtrip(test_gpt2)
            mse = (recon - test_gpt2).pow(2).mean().item()
            norm_loss = mse / max(test_var, 1e-8)
            cos_sim = F.cosine_similarity(recon, test_gpt2, dim=-1).mean().item()
        print(f"    Roundtrip: MSE={mse:.4f}, norm={norm_loss:.6f}, cos_sim={cos_sim:.6f}")

        # Cross-model alignment
        ca, cb = _train_pair(gpt2_states[:80], distil_states[:80],
                             buffer_dim=D, num_steps=2000)
        with torch.no_grad():
            za = ca.encode(test_gpt2)
            zb = cb.encode(test_distil)
            align_sim = F.cosine_similarity(za, zb, dim=-1).mean().item()
            cross_decoded = cb.decode(za)
            cross_mse = (cross_decoded - test_distil).pow(2).mean().item()
        print(f"    Alignment: cos_sim={align_sim:.4f}, cross_MSE={cross_mse:.4f}")

        results[str(D)] = {
            "D": D,
            "roundtrip_mse": mse,
            "roundtrip_normalised": norm_loss,
            "roundtrip_cosine": cos_sim,
            "alignment_cosine": align_sim,
            "cross_decode_mse": cross_mse,
        }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ds = dims
    rt_cos = [results[str(d)]["roundtrip_cosine"] for d in ds]
    al_cos = [results[str(d)]["alignment_cosine"] for d in ds]
    rt_mse = [results[str(d)]["roundtrip_mse"] for d in ds]

    axes[0].plot(ds, rt_cos, "bo-", linewidth=2, markersize=8)
    axes[0].axhline(y=0.95, color="r", linestyle="--", alpha=0.5, label="95% threshold")
    axes[0].set_xlabel("Buffer dimension D")
    axes[0].set_ylabel("Roundtrip cosine similarity")
    axes[0].set_title("Roundtrip Quality vs D")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ds, al_cos, "go-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Buffer dimension D")
    axes[1].set_ylabel("Cross-model alignment cosine")
    axes[1].set_title("Alignment Quality vs D")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ds, rt_mse, "ro-", linewidth=2, markersize=8)
    axes[2].set_xlabel("Buffer dimension D")
    axes[2].set_ylabel("Roundtrip MSE")
    axes[2].set_title("Reconstruction Loss vs D")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp6_dim_sensitivity.png", dpi=150)
    plt.close(fig)

    # Find minimum D for >95% cosine sim
    for d in ds:
        if results[str(d)]["roundtrip_cosine"] >= 0.95:
            print(f"\n  Minimum D for >95% cosine sim: {d}")
            break

    with open(RESULTS_DIR / "exp6_dim_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to exp6_dim_sensitivity.json + exp6_dim_sensitivity.png")
    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("ADVANCED EXPERIMENTS — Latent Resonance Loop Validation")
    print("=" * 70)

    B_final, codec_gpt2, codec_distil = experiment_1()
    experiment_2(B_final, codec_gpt2)
    experiment_3()
    experiment_4()
    experiment_5()
    experiment_6()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
