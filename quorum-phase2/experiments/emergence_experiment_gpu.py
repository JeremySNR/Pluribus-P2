"""Round 4b: GPU Emergence Test with heterogeneous 7B models.

Adapted from emergence_experiment.py (CPU/GPT-2) to use genuinely different
model architectures: Qwen2.5-7B, Mistral-7B-v0.3, Falcon-7B.

Loads models one at a time in fp16 to stay within 24GB VRAM.
Qwen2.5-7B is the reference model for decoding and generation.
"""

from __future__ import annotations

import gc
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.codecs import AgentCodec
from quorum.codec_training import procrustes_init, composite_loss, CodecTrainingConfig
from quorum.loop import latent_resonance_loop, SyntheticAgent, LoopConfig

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BUFFER_DIM = 1024
SLOTS = 6

MODELS = {
    "qwen": "Qwen/Qwen3-8B-Base",
    "mistral": "mistralai/Mistral-7B-v0.3",
    "olmo": "allenai/OLMo-2-1124-7B",
}

REFERENCE_MODEL = "qwen"

AGENT_DEFS = [
    ("qwen_A", "qwen"),
    ("mistral_B", "mistral"),
    ("olmo_C", "olmo"),
    ("qwen_D", "qwen"),
]

TASK_PROMPTS = [
    "What are the benefits and risks of remote work?",
    "Should governments regulate artificial intelligence?",
    "The economy is growing strongly but inequality is rising. Advise.",
    "A company must choose between short-term profit and long-term sustainability.",
    "Explain why some scientific discoveries are initially rejected by the mainstream.",
]

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


# ── Model loading / hidden state extraction ──────────────────────────

def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_hidden_states(model_key, prompts):
    """Load model in fp16, extract last-token hidden states, unload. Returns CPU tensors."""
    model_name = MODELS[model_key]
    print(f"  Loading {model_name} for hidden state extraction ...")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True,
        device_map={"": 0},
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    states = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
            out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float().cpu()
            states.append(h)
            del out, inp
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(prompts)} extracted")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    stacked = torch.stack(states)
    print(f"    Done: {stacked.shape}, hidden_dim={hidden_dim}")
    return stacked, hidden_dim


def load_reference_lm():
    """Load the reference model as CausalLM for token decoding and generation."""
    model_name = MODELS[REFERENCE_MODEL]
    print(f"  Loading {model_name} (CausalLM) for generation ...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True,
        device_map={"": 0},
    )
    model.eval()
    return model, tokenizer


# ── Codec training ───────────────────────────────────────────────────

def train_reference_codec(states, hidden_dim, num_steps=3000):
    """Train the reference model's codec with Procrustes init + reconstruction loss."""
    print(f"  Training reference codec (dim={hidden_dim} -> {BUFFER_DIM}) ...", flush=True)
    codec = AgentCodec(agent_dim=hidden_dim, buffer_dim=BUFFER_DIM)
    n = states.shape[0]
    tgt = torch.randn(n, BUFFER_DIM)
    tgt = tgt / tgt.norm(dim=-1, keepdim=True)
    procrustes_init(codec, states, tgt)

    codec = codec.to(DEVICE)
    states_dev = states.to(DEVICE)

    opt = torch.optim.Adam(codec.parameters(), lr=1e-4)
    for step in range(num_steps):
        idx = torch.randint(0, n, (min(32, n),))
        loss, comp = composite_loss(codec, states_dev[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"    Step {step}/{num_steps}: recon={comp['reconstruction']:.6f}", flush=True)

    with torch.no_grad():
        recon = codec.roundtrip(states_dev)
        cos = F.cosine_similarity(recon, states_dev, dim=-1).mean().item()
    print(f"    Reference codec cosine sim: {cos:.6f}", flush=True)
    return codec.cpu()


def train_aligned_codec(new_states, ref_states, ref_codec, new_hidden_dim, num_steps=3000):
    """Train a new codec aligned to a frozen reference codec's buffer space."""
    print(f"  Training aligned codec (dim={new_hidden_dim} -> {BUFFER_DIM}) ...", flush=True)
    new_codec = AgentCodec(agent_dim=new_hidden_dim, buffer_dim=BUFFER_DIM)

    with torch.no_grad():
        ref_targets = ref_codec.encode(ref_states)
    procrustes_init(new_codec, new_states, ref_targets.detach())

    new_codec = new_codec.to(DEVICE)
    ref_codec_dev = ref_codec.to(DEVICE)
    new_states_dev = new_states.to(DEVICE)
    ref_states_dev = ref_states.to(DEVICE)

    for p in ref_codec_dev.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(new_codec.parameters(), lr=1e-4)
    n = new_states.shape[0]

    for step in range(num_steps):
        idx = torch.randint(0, n, (min(32, n),))
        h_new = new_states_dev[idx]
        h_ref = ref_states_dev[idx]

        l_recon = (new_codec.roundtrip(h_new) - h_new).pow(2).mean()

        z_new = new_codec.encode(h_new)
        with torch.no_grad():
            z_ref = ref_codec_dev.encode(h_ref)
        l_align = (z_new - z_ref).pow(2).mean()

        h_ref_from_new = ref_codec_dev.decode(z_new)
        l_cycle = (h_ref_from_new - h_ref).pow(2).mean()

        loss = l_recon + l_align + 0.5 * l_cycle
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new_codec.parameters(), 1.0)
        opt.step()

        if step % 1000 == 0:
            print(f"    Step {step}/{num_steps}: recon={l_recon.item():.6f} "
                  f"align={l_align.item():.6f} cycle={l_cycle.item():.6f}", flush=True)

    for p in ref_codec_dev.parameters():
        p.requires_grad = True

    with torch.no_grad():
        z_new = new_codec.encode(new_states_dev)
        z_ref = ref_codec_dev.encode(ref_states_dev)
        align_cos = F.cosine_similarity(z_new, z_ref, dim=-1).mean().item()
    print(f"    Cross-model alignment cosine: {align_cos:.6f}", flush=True)

    ref_codec_cpu = ref_codec_dev.cpu()
    return new_codec.cpu()


# ── Experiment logic (reused from CPU version) ───────────────────────

def decode_to_tokens(hidden_state, lm_model, tokenizer, top_k=10):
    with torch.no_grad():
        h = hidden_state.to(DEVICE).half()
        logits = lm_model.lm_head(h.unsqueeze(0))
        probs = F.softmax(logits.float(), dim=-1)
        top_probs, top_ids = probs.topk(top_k, dim=-1)
    tokens = [tokenizer.decode([tid]) for tid in top_ids[0]]
    return tokens, top_probs[0].tolist()


def generate_reranked(lm_model, tokenizer, prompt, steering_hidden,
                      ref_base_model, n_candidates=10, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    candidates = []
    candidate_hiddens = []

    for _ in range(n_candidates):
        with torch.no_grad():
            output_ids = lm_model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.9, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        candidates.append(gen_text)

        with torch.no_grad():
            gen_inp = tokenizer(gen_text, return_tensors="pt",
                                truncation=True, max_length=128).to(DEVICE)
            gen_out = ref_base_model(**gen_inp, output_hidden_states=True)
            gen_h = gen_out.hidden_states[-1][0, -1, :].float().cpu()
            candidate_hiddens.append(gen_h)

    steering_flat = steering_hidden.flatten()
    sims = [
        F.cosine_similarity(ch.unsqueeze(0), steering_flat.unsqueeze(0)).item()
        for ch in candidate_hiddens
    ]
    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    return candidates[best_idx], sims[best_idx], candidates, sims


def run_loop_for_prompt(task_hiddens, codecs, D=BUFFER_DIM, S=SLOTS):
    """Run the latent resonance loop with 4 heterogeneous agents."""
    agents = []
    for i, (name, model_key) in enumerate(AGENT_DEFS):
        codec = codecs[model_key]
        h_raw = task_hiddens[model_key]
        quality = 0.7 if model_key == REFERENCE_MODEL else 0.6

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


def get_individual_hidden(h_raw, source_codec, ref_codec):
    with torch.no_grad():
        z = source_codec.encode(h_raw.unsqueeze(0)).squeeze(0)
        h_decoded = ref_codec.decode(z.unsqueeze(0)).squeeze(0)
    return h_decoded


# ── Main experiment ──────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 70)
    print("ROUND 4b: GPU EMERGENCE TEST — Heterogeneous 7B Models")
    print(f"Device: {DEVICE}")
    print(f"Models: {list(MODELS.values())}")
    print(f"Buffer: D={BUFFER_DIM}, S={SLOTS}")
    print("=" * 70)

    all_prompts = TRAINING_PROMPTS[:100] + TASK_PROMPTS

    # Phase 1: Extract hidden states from each model (load/unload one at a time)
    print(f"\n{'='*70}")
    print("PHASE 1: Hidden state extraction")
    print(f"{'='*70}")

    all_states = {}
    hidden_dims = {}
    for model_key in MODELS:
        states, hdim = extract_hidden_states(model_key, all_prompts)
        all_states[model_key] = states
        hidden_dims[model_key] = hdim

    train_states = {k: v[:80] for k, v in all_states.items()}
    task_states = {k: v[100:] for k, v in all_states.items()}

    print(f"\n  Hidden dimensions: {hidden_dims}")

    # Phase 2: Train codecs
    print(f"\n{'='*70}")
    print("PHASE 2: Codec training")
    print(f"{'='*70}")

    ref_key = REFERENCE_MODEL
    ref_codec = train_reference_codec(
        train_states[ref_key], hidden_dims[ref_key]
    )

    codecs = {ref_key: ref_codec}
    for model_key in MODELS:
        if model_key == ref_key:
            continue
        codecs[model_key] = train_aligned_codec(
            train_states[model_key], train_states[ref_key],
            ref_codec, hidden_dims[model_key],
        )

    # Phase 3: Load reference CausalLM (kept for rest of experiment)
    print(f"\n{'='*70}")
    print("PHASE 3: Loading reference model for decoding & generation")
    print(f"{'='*70}")

    lm_model, tokenizer = load_reference_lm()

    # We also need a base model (AutoModel) for candidate hidden state extraction.
    # The CausalLM's base model is accessible via lm_model.model
    ref_base_model = lm_model.model

    # Phase 4: Run emergence test
    print(f"\n{'='*70}")
    print("PHASE 4: Emergence test")
    print(f"{'='*70}")

    agent_names = [name for name, _ in AGENT_DEFS]
    all_results = []

    for prompt_idx, task_prompt in enumerate(TASK_PROMPTS):
        print(f"\n{'='*70}")
        print(f"  PROMPT {prompt_idx + 1}/5: '{task_prompt}'")
        print(f"{'='*70}")

        prompt_task_hiddens = {k: task_states[k][prompt_idx] for k in MODELS}

        # Step 1: Run loop
        print(f"\n  Step 1: Convergence loop ...")
        B_final, diag = run_loop_for_prompt(prompt_task_hiddens, codecs)
        n_rounds = len(diag.residuals)
        final_residual = diag.residuals[-1]
        print(f"    Converged in {n_rounds} rounds, residual={final_residual:.6f}")

        # Step 2: Decode collective buffer (slot 0) through reference codec
        print(f"\n  Step 2: Collective hidden state ...")
        with torch.no_grad():
            h_collective = ref_codec.decode(B_final[0].unsqueeze(0)).squeeze(0)
        print(f"    Norm: {h_collective.norm():.4f}")

        collective_tokens, collective_probs = decode_to_tokens(
            h_collective, lm_model, tokenizer
        )
        print(f"    Top-5: {list(zip(collective_tokens[:5], [f'{p:.4f}' for p in collective_probs[:5]]))}")

        # Step 3: Individual hidden states
        print(f"\n  Step 3: Individual hidden states ...")
        individual_hiddens = {}
        individual_tokens_map = {}

        for name, model_key in AGENT_DEFS:
            h_ind = get_individual_hidden(
                prompt_task_hiddens[model_key], codecs[model_key], ref_codec
            )
            individual_hiddens[name] = h_ind
            ind_tokens, ind_probs = decode_to_tokens(h_ind, lm_model, tokenizer)
            individual_tokens_map[name] = {
                "top_tokens": ind_tokens, "top_probs": ind_probs,
            }
            print(f"    {name} top-3: {list(zip(ind_tokens[:3], [f'{p:.4f}' for p in ind_probs[:3]]))}")

        # Step 4: Steered generation
        print(f"\n  Step 4: Steered generation ...")

        print(f"    Collective-steered ...")
        coll_text, coll_sim, _, coll_sims = generate_reranked(
            lm_model, tokenizer, task_prompt, h_collective, ref_base_model
        )
        coll_variance = torch.tensor(coll_sims).var().item()
        print(f"    Best sim: {coll_sim:.4f}, variance: {coll_variance:.6f}")

        individual_steered = {}
        individual_variances = {}
        for name, _ in AGENT_DEFS:
            print(f"    {name}-steered ...")
            ind_text, ind_sim, _, ind_sims = generate_reranked(
                lm_model, tokenizer, task_prompt,
                individual_hiddens[name], ref_base_model
            )
            individual_steered[name] = ind_text
            individual_variances[name] = torch.tensor(ind_sims).var().item()
            print(f"      Best sim: {ind_sim:.4f}")

        print(f"    Unsteered ...")
        inputs = tokenizer(task_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = lm_model.generate(
                **inputs, max_new_tokens=50, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        unsteered_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Step 5a: Distinctness
        print(f"\n  Step 5a: Distinctness ...")
        distinctness = {}
        for name, _ in AGENT_DEFS:
            sim = F.cosine_similarity(
                h_collective.unsqueeze(0),
                individual_hiddens[name].unsqueeze(0)
            ).item()
            distinctness[f"collective_vs_{name}"] = sim
            print(f"    collective vs {name}: {sim:.4f}")

        sim_values = list(distinctness.values())
        sim_spread = max(sim_values) - min(sim_values)
        max_sim = max(sim_values)
        equidistant = sim_spread < 0.05
        winner_picked = max_sim > 0.95
        print(f"    Spread: {sim_spread:.4f}, Max sim: {max_sim:.4f}")

        # Step 5c: Novel tokens
        print(f"\n  Step 5c: Novel tokens ...")
        collective_token_set = set(collective_tokens)
        all_individual_tokens = set()
        for name, _ in AGENT_DEFS:
            all_individual_tokens.update(individual_tokens_map[name]["top_tokens"])

        novel_tokens = list(collective_token_set - all_individual_tokens)
        subset_of_union = collective_token_set.issubset(all_individual_tokens)
        print(f"    Novel: {novel_tokens}")
        print(f"    Subset of union: {subset_of_union}")

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
    print(f"\n{'='*70}")
    print("EMERGENCE TEST SUMMARY (GPU — Heterogeneous 7B)")
    print(f"{'='*70}")

    novel_counts = [r["novel_token_count"] for r in all_results]
    equidistant_flags = [r["equidistant"] for r in all_results]
    winner_flags = [r["winner_picked"] for r in all_results]
    subset_flags = [r["collective_is_subset_of_union"] for r in all_results]

    prompts_with_novel = sum(1 for c in novel_counts if c > 0)
    prompts_equidistant = sum(equidistant_flags)
    prompts_winner_picked = sum(winner_flags)
    prompts_subset = sum(subset_flags)

    print(f"\n  Models: {list(MODELS.values())}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Buffer: D={BUFFER_DIM}, S={SLOTS}")

    print(f"\n  Pass criteria (need 3+ of 5):")
    print(f"    Novel tokens: {prompts_with_novel}/5")
    print(f"    Equidistant (spread < 0.05): {prompts_equidistant}/5")

    print(f"\n  Fail criteria:")
    print(f"    Subset of union: {prompts_subset}/5")
    print(f"    Winner picked (max sim > 0.95): {prompts_winner_picked}/5")

    emergence_novel = prompts_with_novel >= 3
    emergence_equidistant = prompts_equidistant >= 3
    emergence_found = emergence_novel or emergence_equidistant
    fail_subset = prompts_subset >= 3
    fail_winner = prompts_winner_picked >= 3

    print(f"\n  VERDICT:")
    if emergence_found and not (fail_subset and fail_winner):
        if emergence_novel:
            print(f"    EMERGENCE DETECTED via novel tokens ({prompts_with_novel}/5)")
        if emergence_equidistant:
            print(f"    EMERGENCE DETECTED via equidistance ({prompts_equidistant}/5)")
    elif fail_subset and fail_winner:
        print(f"    NO EMERGENCE — subset + winner picking")
    else:
        print(f"    INCONCLUSIVE")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    summary = {
        "models": MODELS,
        "hidden_dims": hidden_dims,
        "buffer_dim": BUFFER_DIM,
        "device": DEVICE,
        "elapsed_seconds": elapsed,
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
    with open(RESULTS_DIR / "emergence_experiment_gpu.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to emergence_experiment_gpu.json")

    return output


if __name__ == "__main__":
    main()
