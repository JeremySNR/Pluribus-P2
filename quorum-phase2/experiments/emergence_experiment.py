"""Round 4: Emergence Test — Does the collective attractor produce emergent output?

Tests whether the converged buffer produces downstream output that is detectably
different from and better than what any individual agent produces alone.

Three measurements per prompt:
  a. Distinctness — cosine similarity of collective vs each individual hidden state
  b. Diversity — variance of candidate similarities under collective vs individual steering
  c. Novel tokens — tokens in collective top-10 absent from ALL individual top-10s
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.codecs import AgentCodec
from quorum.codec_training import (
    procrustes_init, composite_loss, train_codec_pair, CodecTrainingConfig,
)
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

TASK_PROMPTS = [
    "What are the benefits and risks of remote work?",
    "Should governments regulate artificial intelligence?",
    "The economy is growing strongly but inequality is rising. Advise.",
    "A company must choose between short-term profit and long-term sustainability.",
    "Explain why some scientific discoveries are initially rejected by the mainstream.",
]

AGENT_NAMES = ["gpt2_A", "gpt2_B", "distil_A", "distil_B"]


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


def decode_to_tokens(hidden_state, lm_model, tokenizer, top_k=10):
    """Decode a hidden state through the lm_head to get top-k token distribution."""
    with torch.no_grad():
        logits = lm_model.lm_head(hidden_state.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(top_k, dim=-1)
    tokens = [tokenizer.decode([tid]) for tid in top_ids[0]]
    probs_list = top_probs[0].tolist()
    return tokens, probs_list


def generate_reranked(lm_model, tokenizer, prompt, steering_hidden,
                      n_candidates=10, max_new_tokens=50):
    """Generate N candidates and pick the one closest to the steering hidden state."""
    base_model, _ = _load("gpt2")
    inputs = tokenizer(prompt, return_tensors="pt")
    candidates = []
    candidate_hiddens = []

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
            gen_inputs = tokenizer(gen_text, return_tensors="pt",
                                   truncation=True, max_length=128)
            gen_out = base_model(**gen_inputs, output_hidden_states=True)
            gen_hidden = gen_out.hidden_states[-1][0, -1, :]
            candidate_hiddens.append(gen_hidden)

    steering_flat = steering_hidden.flatten()
    sims = []
    for ch in candidate_hiddens:
        sim = F.cosine_similarity(ch.unsqueeze(0), steering_flat.unsqueeze(0)).item()
        sims.append(sim)

    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    return candidates[best_idx], sims[best_idx], candidates, sims


def run_loop_for_prompt(task_prompt, codec_gpt2, codec_distil, D=512, S=6):
    """Run the latent resonance loop with 4 agents using delta-based encoding."""
    gpt2_task = _extract("gpt2", [task_prompt])
    distil_task = _extract("distilgpt2", [task_prompt])

    raw_hiddens = {
        "gpt2_A": gpt2_task.squeeze(0),
        "gpt2_B": gpt2_task.squeeze(0),
        "distil_A": distil_task.squeeze(0),
        "distil_B": distil_task.squeeze(0),
    }
    codecs = {
        "gpt2_A": codec_gpt2, "gpt2_B": codec_gpt2,
        "distil_A": codec_distil, "distil_B": codec_distil,
    }

    agents = []
    for i, name in enumerate(AGENT_NAMES):
        codec = codecs[name]
        h_raw = raw_hiddens[name]
        quality = 0.7 if "gpt2" in name else 0.6

        def make_fn(agent_codec, agent_h, seed=i):
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
    return B_final, diag, raw_hiddens, codecs


def get_individual_hidden(h_raw, source_codec, target_codec):
    """Encode individual agent state through its codec, decode via GPT-2 codec."""
    with torch.no_grad():
        z = source_codec.encode(h_raw.unsqueeze(0)).squeeze(0)
        h_decoded = target_codec.decode(z.unsqueeze(0)).squeeze(0)
    return h_decoded


def run_emergence_experiment():
    print("=" * 70)
    print("ROUND 4: EMERGENCE TEST")
    print("Does the collective attractor produce emergent output?")
    print("=" * 70)

    # Train codecs
    prompts = TRAINING_PROMPTS[:100]
    gpt2_states = _extract("gpt2", prompts)
    distil_states = _extract("distilgpt2", prompts)

    print("\n  Training aligned codec pair (D=512, 3000 steps) ...")
    codec_gpt2, codec_distil = _train_pair(
        gpt2_states[:80], distil_states[:80], buffer_dim=512, num_steps=3000
    )

    lm_model, tokenizer = _load_lm("gpt2")
    all_results = []

    for prompt_idx, task_prompt in enumerate(TASK_PROMPTS):
        print(f"\n{'=' * 70}")
        print(f"  PROMPT {prompt_idx + 1}/5: '{task_prompt}'")
        print(f"{'=' * 70}")

        # Step 1: Run the full loop
        print(f"\n  Step 1: Running convergence loop ...")
        B_final, diag, raw_hiddens, codecs_map = run_loop_for_prompt(
            task_prompt, codec_gpt2, codec_distil
        )
        n_rounds = len(diag.residuals)
        final_residual = diag.residuals[-1]
        print(f"    Converged in {n_rounds} rounds, residual={final_residual:.6f}")

        # Step 2: Decode collective buffer through GPT-2 codec (use slot 0)
        print(f"\n  Step 2: Decoding collective buffer ...")
        with torch.no_grad():
            h_collective = codec_gpt2.decode(B_final[0].unsqueeze(0)).squeeze(0)
        print(f"    Collective hidden norm: {h_collective.norm():.4f}")

        collective_tokens, collective_probs = decode_to_tokens(
            h_collective, lm_model, tokenizer
        )
        print(f"    Collective top-5: {list(zip(collective_tokens[:5], [f'{p:.4f}' for p in collective_probs[:5]]))}")

        # Step 3: Get individual hidden states
        print(f"\n  Step 3: Getting individual agent hidden states ...")
        individual_hiddens = {}
        individual_tokens_map = {}

        for name in AGENT_NAMES:
            h_ind = get_individual_hidden(
                raw_hiddens[name], codecs_map[name], codec_gpt2
            )
            individual_hiddens[name] = h_ind
            ind_tokens, ind_probs = decode_to_tokens(h_ind, lm_model, tokenizer)
            individual_tokens_map[name] = {
                "top_tokens": ind_tokens,
                "top_probs": ind_probs,
            }
            print(f"    {name} top-3: {list(zip(ind_tokens[:3], [f'{p:.4f}' for p in ind_probs[:3]]))}")

        # Step 4: Generate steered text
        print(f"\n  Step 4: Generating steered text ...")

        print(f"    Collective-steered (10 candidates) ...")
        coll_text, coll_sim, _, coll_sims = generate_reranked(
            lm_model, tokenizer, task_prompt, h_collective
        )
        coll_variance = torch.tensor(coll_sims).var().item()
        print(f"    Best sim: {coll_sim:.4f}, variance: {coll_variance:.6f}")
        print(f"    Text: {coll_text[len(task_prompt):len(task_prompt)+150]}...")

        individual_steered = {}
        individual_variances = {}
        for name in AGENT_NAMES:
            print(f"    {name}-steered ...")
            ind_text, ind_sim, _, ind_sims = generate_reranked(
                lm_model, tokenizer, task_prompt, individual_hiddens[name]
            )
            ind_var = torch.tensor(ind_sims).var().item()
            individual_steered[name] = ind_text
            individual_variances[name] = ind_var
            print(f"      Best sim: {ind_sim:.4f}, variance: {ind_var:.6f}")

        print(f"    Unsteered ...")
        inputs = tokenizer(task_prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = lm_model.generate(
                **inputs, max_new_tokens=50, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        unsteered_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Step 5a: Distinctness — collective vs each individual
        print(f"\n  Step 5a: Distinctness ...")
        distinctness = {}
        for name in AGENT_NAMES:
            sim = F.cosine_similarity(
                h_collective.unsqueeze(0),
                individual_hiddens[name].unsqueeze(0)
            ).item()
            distinctness[f"collective_vs_{name}"] = sim
            print(f"    collective vs {name}: {sim:.4f}")

        sim_values = list(distinctness.values())
        sim_spread = max(sim_values) - min(sim_values)
        equidistant = sim_spread < 0.05
        max_sim = max(sim_values)
        winner_picked = max_sim > 0.95
        print(f"    Spread: {sim_spread:.4f} (equidistant={equidistant})")
        print(f"    Max sim: {max_sim:.4f} (winner_picked={winner_picked})")

        # Step 5b: Diversity — variance comparison
        print(f"\n  Step 5b: Candidate diversity ...")
        mean_ind_var = sum(individual_variances.values()) / len(individual_variances)
        print(f"    Collective variance: {coll_variance:.6f}")
        print(f"    Mean individual variance: {mean_ind_var:.6f}")
        print(f"    Ratio (collective/individual): {coll_variance / max(mean_ind_var, 1e-10):.4f}")

        # Step 5c: Novel tokens
        print(f"\n  Step 5c: Novel token analysis ...")
        collective_token_set = set(collective_tokens)
        all_individual_tokens = set()
        for name in AGENT_NAMES:
            all_individual_tokens.update(individual_tokens_map[name]["top_tokens"])

        novel_tokens = list(collective_token_set - all_individual_tokens)
        print(f"    Collective top-10: {collective_tokens}")
        print(f"    Union of individual top-10s: {sorted(all_individual_tokens)}")
        print(f"    Novel tokens: {novel_tokens}")
        print(f"    Novel count: {len(novel_tokens)}")

        subset_of_union = collective_token_set.issubset(all_individual_tokens)
        print(f"    Collective is strict subset of union: {subset_of_union}")

        result = {
            "task_prompt": task_prompt,
            "loop_rounds": n_rounds,
            "final_residual": final_residual,
            "collective_tokens": {
                "top_tokens": collective_tokens,
                "top_probs": collective_probs,
            },
            "individual_tokens": individual_tokens_map,
            "collective_steered_text": coll_text[:500],
            "individual_steered_texts": {
                k: v[:500] for k, v in individual_steered.items()
            },
            "unsteered_text": unsteered_text[:500],
            "distinctness": distinctness,
            "distinctness_spread": sim_spread,
            "equidistant": equidistant,
            "max_individual_sim": max_sim,
            "winner_picked": winner_picked,
            "collective_candidate_variance": coll_variance,
            "individual_candidate_variances": individual_variances,
            "novel_tokens": novel_tokens,
            "novel_token_count": len(novel_tokens),
            "collective_is_subset_of_union": subset_of_union,
        }
        all_results.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"EMERGENCE TEST SUMMARY")
    print(f"{'=' * 70}")

    novel_counts = [r["novel_token_count"] for r in all_results]
    equidistant_flags = [r["equidistant"] for r in all_results]
    winner_flags = [r["winner_picked"] for r in all_results]
    subset_flags = [r["collective_is_subset_of_union"] for r in all_results]

    prompts_with_novel = sum(1 for c in novel_counts if c > 0)
    prompts_equidistant = sum(equidistant_flags)
    prompts_winner_picked = sum(winner_flags)
    prompts_subset = sum(subset_flags)

    print(f"\n  Pass criteria (need 3+ of 5 prompts):")
    print(f"    Novel tokens present: {prompts_with_novel}/5 prompts")
    print(f"    Equidistant (spread < 0.05): {prompts_equidistant}/5 prompts")

    print(f"\n  Fail criteria:")
    print(f"    Collective is strict subset of union: {prompts_subset}/5 prompts")
    print(f"    Winner picked (max sim > 0.95): {prompts_winner_picked}/5 prompts")

    emergence_novel = prompts_with_novel >= 3
    emergence_equidistant = prompts_equidistant >= 3
    emergence_found = emergence_novel or emergence_equidistant

    fail_subset = prompts_subset >= 3
    fail_winner = prompts_winner_picked >= 3

    print(f"\n  VERDICT:")
    if emergence_found:
        if emergence_novel:
            print(f"    EMERGENCE DETECTED via novel tokens ({prompts_with_novel}/5)")
        if emergence_equidistant:
            print(f"    EMERGENCE DETECTED via equidistance ({prompts_equidistant}/5)")
    elif fail_subset and fail_winner:
        print(f"    NO EMERGENCE — collective is subset + winner picking")
    elif fail_subset:
        print(f"    NO EMERGENCE — collective is strict subset of union")
    elif fail_winner:
        print(f"    NO EMERGENCE — buffer just picks a winner")
    else:
        print(f"    INCONCLUSIVE — neither pass nor fail criteria fully met")

    summary = {
        "prompts_with_novel_tokens": prompts_with_novel,
        "prompts_equidistant": prompts_equidistant,
        "prompts_winner_picked": prompts_winner_picked,
        "prompts_subset_of_union": prompts_subset,
        "emergence_via_novel": emergence_novel,
        "emergence_via_equidistant": emergence_equidistant,
        "emergence_found": emergence_found,
        "fail_subset": fail_subset,
        "fail_winner": fail_winner,
    }

    output = {"summary": summary, "per_prompt": all_results}
    with open(RESULTS_DIR / "emergence_experiment.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to emergence_experiment.json")

    return output


if __name__ == "__main__":
    run_emergence_experiment()
