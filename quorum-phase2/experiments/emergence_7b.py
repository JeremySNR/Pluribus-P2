"""Emergence Test + Reranking for 7B Models with BAPC Codecs.

Tests whether the latent resonance loop produces emergent collective
representations at 7B scale, and whether those representations select
meaningfully different (hopefully better) text through reranking.

Two modes:
  1. Single-model (default): 4 agents from one 7B model with different role
     prompts. Keeps the model loaded for the entire loop — no swapping needed.
     This is the direct 7B equivalent of Round 7 (GPT-2).

  2. Cross-model: 4 agents from 3 model families with model swapping per round.
     Loads/unloads each model family per loop round. Slower but tests whether
     cross-architecture emergence occurs at 7B scale.

Loads BAPC codec checkpoints from bapc_7b_training.py.
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

from quorum.isocal import IsotropyCalibrator, IsocalConfig
from quorum.bapc_codec import BAPCCodec
from quorum.bapc_training import js_divergence
from quorum.ties_resolve import ties_resolve
from quorum.contraction import damped_update
from quorum.convergence import compute_residual

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BUFFER_DIM = 1024
NUM_SLOTS = 6
MAX_ROUNDS = 25
TOL = 0.02
ALPHA = 0.5
TIES_DENSITY = 0.3
NUM_RERANK_CANDIDATES = 20

MODELS = {
    "qwen": "Qwen/Qwen3-8B-Base",
    "mistral": "mistralai/Mistral-7B-v0.3",
    "olmo": "allenai/OLMo-2-1124-7B",
}

LOCAL_MODELS = {
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
}

REFERENCE_MODEL = "qwen"
LOCAL_REFERENCE_MODEL = "gpt2"

TASK_PROMPTS = [
    "What are the benefits and risks of remote work?",
    "Should governments regulate artificial intelligence?",
    "The economy is growing strongly but inequality is rising. Advise.",
    "A company must choose between short-term profit and long-term sustainability.",
    "Explain why some scientific discoveries are initially rejected by the mainstream.",
]

ROLE_PROMPTS = {
    "neutral": "",
    "supportive": "Argue strongly in favor. ",
    "critical": "Find every flaw and risk. ",
    "carol": "Challenge the obvious answer and defend the opposite position. ",
}

def get_agent_defs(mode, local_mode=False):
    ref = LOCAL_REFERENCE_MODEL if local_mode else REFERENCE_MODEL
    if mode == "single":
        return [
            ("neutral_A", "neutral", ref),
            ("supportive_B", "supportive", ref),
            ("critical_C", "critical", ref),
            ("carol_D", "carol", ref),
        ]
    else:
        if local_mode:
            return [
                ("gpt2_neutral_A", "neutral", "gpt2"),
                ("distilgpt2_support_B", "supportive", "distilgpt2"),
                ("gpt2_critical_C", "critical", "gpt2"),
                ("distilgpt2_carol_D", "carol", "distilgpt2"),
            ]
        else:
            return [
                ("qwen_neutral_A", "neutral", "qwen"),
                ("mistral_support_B", "supportive", "mistral"),
                ("olmo_critical_C", "critical", "olmo"),
                ("qwen_carol_D", "carol", "qwen"),
            ]


# ── GPU memory management ────────────────────────────────────────────

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── Codec checkpoint loading ─────────────────────────────────────────

def load_codec(model_key, local_mode=False):
    """Load a trained BAPC codec from checkpoint."""
    prefix = "bapc_local" if local_mode else "bapc_7b"
    ckpt_path = CHECKPOINT_DIR / f"{prefix}_{model_key}.pt"
    print(f"  Loading codec: {ckpt_path.name} ...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hidden_dim = ckpt["hidden_dim"]
    extraction_layer = ckpt["extraction_layer"]
    buffer_dim = ckpt["buffer_dim"]
    num_slots = ckpt["num_slots"]

    codec = BAPCCodec(
        agent_dim=hidden_dim, buffer_dim=buffer_dim,
        embed_dim=hidden_dim, num_slots=num_slots,
    )
    codec.load_state_dict(ckpt["codec_state"])

    isocal = IsotropyCalibrator()
    isocal.load_state_dict(ckpt["isocal_state"])
    codec.isocal = isocal

    codec.eval()
    print(f"    dim={hidden_dim}, buffer={buffer_dim}, layer={extraction_layer}",
          flush=True)

    return codec, extraction_layer


def load_model_7b(model_key, local_mode=False):
    """Load a CausalLM. fp16 for 7B, float32 for local GPT-2."""
    model_table = LOCAL_MODELS if local_mode else MODELS
    model_name = model_table[model_key]
    print(f"  Loading {model_name} ...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if local_mode:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(DEVICE)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True,
            device_map={"": 0},
        )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    free_gpu()


# ── Single-model emergence loop ──────────────────────────────────────

def run_single_model_emergence(task_prompt, model, tokenizer, codec,
                               extraction_layer, agent_defs, local_mode=False):
    """Run the emergence loop with all agents sharing one loaded model.

    Each round, every agent:
      1. Decodes the buffer to prefix embeddings
      2. Runs a forward pass: prefix + role_prompt + task_prompt
      3. Extracts hidden state from extraction_layer
      4. Encodes through codec -> proposed buffer update
    """
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()
    S = codec.num_slots
    D = codec.buffer_dim

    agent_prompt_embeds = {}
    for name, role_key, _ in agent_defs:
        full_prompt = ROLE_PROMPTS[role_key] + task_prompt
        ids = tokenizer(full_prompt, return_tensors="pt",
                        truncation=True, max_length=64)
        with torch.no_grad():
            emb = embed_layer(ids["input_ids"][0].to(device))
        agent_prompt_embeds[name] = emb

    B = torch.zeros(S, D)
    residuals = []
    converged = False

    codec_dev = codec.to(device)

    for t in range(MAX_ROUNDS):
        deltas = []
        B_dev = B.to(device).float()

        for name, role_key, _ in agent_defs:
            with torch.no_grad():
                prefix = codec_dev.decode(B_dev)
                if not local_mode:
                    prefix = prefix.half()

            combined = torch.cat([prefix, agent_prompt_embeds[name]], dim=0).unsqueeze(0)
            with torch.no_grad():
                out = model(inputs_embeds=combined, output_hidden_states=True)
                h = out.hidden_states[extraction_layer][0, -1, :].float()

            z_new = codec_dev.encode(h).cpu()
            delta = (z_new - B) * 0.3
            deltas.append(delta)

        non_dissent = [d[:S - 1] for d in deltas]
        resolved_main = ties_resolve(non_dissent, density=TIES_DENSITY)
        dissent_delta = deltas[-1][S - 1:S]
        resolved = torch.cat([resolved_main, dissent_delta], dim=0)

        B_new = damped_update(B, B + resolved, alpha=ALPHA)
        residual = compute_residual(B_new, B)
        residuals.append(residual)
        B = B_new

        if t % 5 == 0:
            print(f"      Round {t}: residual={residual:.6f}", flush=True)

        if residual < TOL and t >= 3:
            print(f"      Converged at round {t}", flush=True)
            converged = True
            break

    codec.cpu()
    return B, residuals, converged


# ── Cross-model emergence loop ───────────────────────────────────────

def run_cross_model_emergence(task_prompt, codecs, extraction_layers,
                              agent_defs, local_mode=False):
    """Run emergence loop with model swapping per model family per round.

    Groups agents by model family. Each round: load model -> forward passes
    for all agents of that family -> unload -> next family.
    """
    S = NUM_SLOTS
    ref_key = list(codecs.keys())[0]
    D = codecs[ref_key].buffer_dim

    model_to_agents = {}
    for i, (name, role_key, model_key) in enumerate(agent_defs):
        model_to_agents.setdefault(model_key, []).append((i, name, role_key))

    B = torch.zeros(S, D)
    residuals = []
    converged = False

    for t in range(MAX_ROUNDS):
        deltas = [None] * len(agent_defs)

        for model_key, agents_info in model_to_agents.items():
            model, tokenizer = load_model_7b(model_key, local_mode=local_mode)
            device = next(model.parameters()).device
            embed_layer = model.get_input_embeddings()
            codec = codecs[model_key].to(device)
            ext_layer = extraction_layers[model_key]

            for agent_idx, name, role_key in agents_info:
                full_prompt = ROLE_PROMPTS[role_key] + task_prompt
                ids = tokenizer(full_prompt, return_tensors="pt",
                                truncation=True, max_length=64)

                with torch.no_grad():
                    prompt_emb = embed_layer(ids["input_ids"][0].to(device))
                    prefix = codec.decode(B.to(device).float())
                    if not local_mode:
                        prefix = prefix.half()
                    combined = torch.cat([prefix, prompt_emb], dim=0).unsqueeze(0)
                    out = model(inputs_embeds=combined, output_hidden_states=True)
                    h = out.hidden_states[ext_layer][0, -1, :].float().cpu()

                codec_cpu = codec.cpu()
                z_new = codec_cpu.encode(h)
                delta = (z_new - B) * 0.3
                deltas[agent_idx] = delta

            codec.cpu()
            unload_model(model, tokenizer)

        non_dissent = [d[:S - 1] for d in deltas]
        resolved_main = ties_resolve(non_dissent, density=TIES_DENSITY)
        dissent_delta = deltas[-1][S - 1:S]
        resolved = torch.cat([resolved_main, dissent_delta], dim=0)

        B_new = damped_update(B, B + resolved, alpha=ALPHA)
        residual = compute_residual(B_new, B)
        residuals.append(residual)
        B = B_new

        if t % 3 == 0:
            print(f"      Round {t}: residual={residual:.6f}", flush=True)

        if residual < TOL and t >= 3:
            print(f"      Converged at round {t}", flush=True)
            converged = True
            break

    return B, residuals, converged


# ── Individual encoding (no loop, single forward pass) ───────────────

def get_individual_encoding(model, tokenizer, codec, extraction_layer,
                            role_prefix, task_prompt):
    """Encode one agent's hidden state without the loop."""
    device = next(model.parameters()).device
    full_prompt = role_prefix + task_prompt
    inp = tokenizer(full_prompt, return_tensors="pt",
                    truncation=True, max_length=64)
    inp = {k: v.to(device) for k, v in inp.items()}

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[extraction_layer][0, -1, :].float().cpu()

    codec_cpu = codec.cpu()
    return codec_cpu.encode(h)


# ── Reranking evaluation ─────────────────────────────────────────────

def reranking_test(model, tokenizer, codec, extraction_layer,
                   collective_buffer, individual_encodings,
                   task_prompt, n_candidates=NUM_RERANK_CANDIDATES):
    """Generate candidates and rerank by buffer similarity.

    Compares collective-reranked vs individual-reranked vs random selection.
    """
    device = next(model.parameters()).device

    candidates = []
    candidate_hiddens = []

    inp = tokenizer(task_prompt, return_tensors="pt",
                    truncation=True, max_length=64)
    inp = {k: v.to(device) for k, v in inp.items()}

    print(f"      Generating {n_candidates} candidates ...", flush=True)
    for c in range(n_candidates):
        with torch.no_grad():
            gen_ids = model.generate(
                **inp, max_new_tokens=80,
                do_sample=True, temperature=0.9, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen_ids[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        candidates.append(text)

        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        gen_inp = tokenizer(full_text, return_tensors="pt",
                            truncation=True, max_length=128)
        gen_inp = {k: v.to(device) for k, v in gen_inp.items()}
        with torch.no_grad():
            gen_out = model(**gen_inp, output_hidden_states=True)
            h = gen_out.hidden_states[extraction_layer][0, -1, :].float().cpu()
        candidate_hiddens.append(h)

    codec_cpu = codec.cpu()
    candidate_encodings = []
    for h in candidate_hiddens:
        z = codec_cpu.encode(h)
        candidate_encodings.append(z[0])

    def rank_by_buffer(buffer_state):
        sims = [F.cosine_similarity(enc.unsqueeze(0),
                                    buffer_state[0].unsqueeze(0)).item()
                for enc in candidate_encodings]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        return ranked, sims

    coll_ranked, coll_sims = rank_by_buffer(collective_buffer)
    coll_pick = coll_ranked[0]

    ind_picks = {}
    for name, z_ind in individual_encodings.items():
        ind_ranked, ind_sims = rank_by_buffer(z_ind)
        ind_picks[name] = {
            "pick": ind_ranked[0],
            "sims": ind_sims,
            "top3": ind_ranked[:3],
        }

    coll_unique = coll_pick not in [v["pick"] for v in ind_picks.values()]
    top3_overlap = len(set(coll_ranked[:3]) &
                       set().union(*[set(v["top3"]) for v in ind_picks.values()]))

    return {
        "n_candidates": n_candidates,
        "collective_pick": coll_pick,
        "collective_pick_text": candidates[coll_pick][:300],
        "collective_sims": coll_sims,
        "collective_sim_spread": max(coll_sims) - min(coll_sims),
        "individual_picks": {k: v["pick"] for k, v in ind_picks.items()},
        "collective_pick_unique": coll_unique,
        "top3_overlap": top3_overlap,
        "all_candidates": [c[:200] for c in candidates],
    }


# ── Token analysis ───────────────────────────────────────────────────

def analyze_tokens(model, tokenizer, codec, buffer_state,
                   individual_encodings, task_prompt, top_k=10):
    """Decode buffer and individual encodings to tokens, find novel tokens."""
    device = next(model.parameters()).device
    lm_head = model.lm_head

    def decode_buffer_to_tokens(z):
        codec_dev = codec.to(device)
        model_dtype = next(model.parameters()).dtype
        with torch.no_grad():
            h = codec_dev.decode(z.to(device)).to(model_dtype)
            logits = lm_head(h[0].unsqueeze(0))
            probs = F.softmax(logits.float(), dim=-1)
            top_probs, top_ids = probs.topk(top_k, dim=-1)
        codec.cpu()
        tokens = [tokenizer.decode([tid]) for tid in top_ids[0]]
        return tokens, top_probs[0].tolist()

    coll_tokens, coll_probs = decode_buffer_to_tokens(buffer_state)

    all_individual_tokens = set()
    individual_top = {}
    for name, z_ind in individual_encodings.items():
        ind_tokens, ind_probs = decode_buffer_to_tokens(z_ind)
        individual_top[name] = {"tokens": ind_tokens, "probs": ind_probs}
        all_individual_tokens.update(ind_tokens)

    novel = [t for t in coll_tokens if t not in all_individual_tokens]
    is_subset = set(coll_tokens).issubset(all_individual_tokens)

    return {
        "collective_tokens": coll_tokens,
        "collective_probs": coll_probs,
        "individual_top": individual_top,
        "novel_tokens": novel,
        "novel_count": len(novel),
        "collective_is_subset": is_subset,
    }


# ── Main experiment ──────────────────────────────────────────────────

def run_experiment(mode="single", local_mode=False):
    """Run the full emergence + reranking experiment.

    Args:
        mode: "single" for single-model, "cross" for cross-model.
        local_mode: Use GPT-2/DistilGPT-2 for local GPU testing.
    """
    t_start = time.time()
    agent_defs = get_agent_defs(mode, local_mode=local_mode)
    ref_model_key = LOCAL_REFERENCE_MODEL if local_mode else REFERENCE_MODEL

    label = "LOCAL TEST" if local_mode else "7B MODELS"
    print(f"{'='*70}")
    print(f"EMERGENCE TEST — {label} ({mode.upper()} MODE)")
    print(f"Device: {DEVICE}")
    print(f"Agents: {[a[0] for a in agent_defs]}")
    print(f"{'='*70}", flush=True)

    # Load codecs
    needed_models = set(m for _, _, m in agent_defs)
    codecs = {}
    extraction_layers = {}
    for mk in needed_models:
        codec, ext_layer = load_codec(mk, local_mode=local_mode)
        codecs[mk] = codec
        extraction_layers[mk] = ext_layer

    if mode == "single":
        ref_model, ref_tokenizer = load_model_7b(ref_model_key,
                                                   local_mode=local_mode)

    all_results = []

    for prompt_idx, task_prompt in enumerate(TASK_PROMPTS):
        print(f"\n{'='*70}")
        print(f"  PROMPT {prompt_idx + 1}/5: '{task_prompt}'")
        print(f"{'='*70}", flush=True)

        # Step 1: Individual encodings (no loop)
        print(f"\n  Step 1: Individual encodings ...", flush=True)
        individual_encodings = {}

        if mode == "single":
            for name, role_key, model_key in agent_defs:
                z = get_individual_encoding(
                    ref_model, ref_tokenizer, codecs[model_key],
                    extraction_layers[model_key],
                    ROLE_PROMPTS[role_key], task_prompt
                )
                individual_encodings[name] = z
                print(f"    {name}: norm={z.norm():.4f}", flush=True)
        else:
            for name, role_key, model_key in agent_defs:
                model, tok = load_model_7b(model_key, local_mode=local_mode)
                z = get_individual_encoding(
                    model, tok, codecs[model_key],
                    extraction_layers[model_key],
                    ROLE_PROMPTS[role_key], task_prompt
                )
                individual_encodings[name] = z
                print(f"    {name}: norm={z.norm():.4f}", flush=True)
                unload_model(model, tok)

        # Pairwise similarity between individuals
        ind_names = list(individual_encodings.keys())
        ind_sims = {}
        for i in range(len(ind_names)):
            for j in range(i + 1, len(ind_names)):
                a, b = ind_names[i], ind_names[j]
                sim = F.cosine_similarity(
                    individual_encodings[a][0].unsqueeze(0),
                    individual_encodings[b][0].unsqueeze(0)
                ).item()
                ind_sims[f"{a}_vs_{b}"] = sim
        avg_ind_sim = sum(ind_sims.values()) / max(len(ind_sims), 1)
        print(f"    Avg pairwise sim: {avg_ind_sim:.4f}", flush=True)

        # Step 2: Convergence loop
        print(f"\n  Step 2: Running emergence loop ...", flush=True)
        if mode == "single":
            B_final, residuals, converged = run_single_model_emergence(
                task_prompt, ref_model, ref_tokenizer,
                codecs[ref_model_key], extraction_layers[ref_model_key],
                agent_defs, local_mode=local_mode
            )
        else:
            B_final, residuals, converged = run_cross_model_emergence(
                task_prompt, codecs, extraction_layers, agent_defs,
                local_mode=local_mode
            )

        n_rounds = len(residuals)
        final_residual = residuals[-1] if residuals else float("inf")
        print(f"    Rounds: {n_rounds}, final residual: {final_residual:.6f}, "
              f"converged: {converged}", flush=True)

        # Step 3: Distinctness analysis
        print(f"\n  Step 3: Distinctness ...", flush=True)
        distinctness = {}
        for name in ind_names:
            sim = F.cosine_similarity(
                B_final[0].unsqueeze(0),
                individual_encodings[name][0].unsqueeze(0)
            ).item()
            distinctness[f"collective_vs_{name}"] = sim
            print(f"    collective vs {name}: {sim:.4f}", flush=True)

        sim_values = list(distinctness.values())
        spread = max(sim_values) - min(sim_values) if sim_values else 0
        max_sim = max(sim_values) if sim_values else 0
        print(f"    Spread: {spread:.4f}, Max sim: {max_sim:.4f}", flush=True)

        # Step 4: Token analysis
        print(f"\n  Step 4: Token analysis ...", flush=True)
        if mode == "single":
            token_result = analyze_tokens(
                ref_model, ref_tokenizer, codecs[ref_model_key],
                B_final, individual_encodings, task_prompt
            )
        else:
            model, tok = load_model_7b(ref_model_key, local_mode=local_mode)
            token_result = analyze_tokens(
                model, tok, codecs[ref_model_key],
                B_final, individual_encodings, task_prompt
            )
            unload_model(model, tok)

        print(f"    Collective top-5: {token_result['collective_tokens'][:5]}", flush=True)
        print(f"    Novel tokens: {token_result['novel_tokens']}", flush=True)

        # Step 5: JS divergence
        print(f"\n  Step 5: JS divergence ...", flush=True)
        if mode == "single":
            device = next(ref_model.parameters()).device
            lm_head = ref_model.lm_head
            codec_dev = codecs[ref_model_key].to(device)

            model_dtype = next(ref_model.parameters()).dtype
            with torch.no_grad():
                coll_h = codec_dev.decode(B_final.to(device)).to(model_dtype)
                coll_logits = lm_head(coll_h[0].unsqueeze(0)).float()
                p_collective = F.softmax(coll_logits[0], dim=-1)

            js_results = {}
            for name in ind_names:
                with torch.no_grad():
                    ind_h = codec_dev.decode(individual_encodings[name].to(device)).to(model_dtype)
                    ind_logits = lm_head(ind_h[0].unsqueeze(0)).float()
                    p_ind = F.softmax(ind_logits[0], dim=-1)
                js_val = js_divergence(p_collective.unsqueeze(0),
                                       p_ind.unsqueeze(0)).item()
                js_results[name] = js_val
                print(f"    JS(collective vs {name}): {js_val:.4f}", flush=True)

            codecs[ref_model_key].cpu()
            avg_js = sum(js_results.values()) / max(len(js_results), 1)
        else:
            js_results = {}
            avg_js = 0.0

        # Step 6: Reranking
        print(f"\n  Step 6: Reranking ({NUM_RERANK_CANDIDATES} candidates) ...",
              flush=True)
        if mode == "single":
            rerank_result = reranking_test(
                ref_model, ref_tokenizer, codecs[ref_model_key],
                extraction_layers[ref_model_key],
                B_final, individual_encodings, task_prompt
            )
        else:
            model, tok = load_model_7b(ref_model_key, local_mode=local_mode)
            rerank_result = reranking_test(
                model, tok, codecs[ref_model_key],
                extraction_layers[ref_model_key],
                B_final, individual_encodings, task_prompt
            )
            unload_model(model, tok)

        print(f"    Collective pick: #{rerank_result['collective_pick']}", flush=True)
        print(f"    Individual picks: {rerank_result['individual_picks']}", flush=True)
        print(f"    Collective unique: {rerank_result['collective_pick_unique']}",
              flush=True)
        print(f"    Sim spread: {rerank_result['collective_sim_spread']:.4f}",
              flush=True)

        result = {
            "task_prompt": task_prompt,
            "loop_rounds": n_rounds,
            "final_residual": final_residual,
            "converged": converged,
            "individual_pairwise_sim": avg_ind_sim,
            "distinctness": distinctness,
            "distinctness_spread": spread,
            "max_sim_to_individual": max_sim,
            "token_analysis": token_result,
            "js_divergence": js_results,
            "avg_js": avg_js,
            "reranking": rerank_result,
        }
        all_results.append(result)

    if mode == "single":
        unload_model(ref_model, ref_tokenizer)

    # Summary
    print(f"\n{'='*70}")
    print(f"EMERGENCE TEST SUMMARY ({mode.upper()} MODE)")
    print(f"{'='*70}", flush=True)

    novel_counts = [r["token_analysis"]["novel_count"] for r in all_results]
    subset_flags = [r["token_analysis"]["collective_is_subset"] for r in all_results]
    max_sims = [r["max_sim_to_individual"] for r in all_results]
    rerank_unique = [r["reranking"]["collective_pick_unique"] for r in all_results]

    prompts_with_novel = sum(1 for c in novel_counts if c > 0)
    prompts_not_subset = sum(1 for s in subset_flags if not s)
    prompts_distinct = sum(1 for s in max_sims if s < 0.95)
    prompts_rerank_unique = sum(rerank_unique)

    print(f"\n  Emergence indicators:")
    print(f"    Novel tokens: {prompts_with_novel}/5 prompts")
    print(f"    Collective NOT subset: {prompts_not_subset}/5")
    print(f"    Distinct from individuals (sim < 0.95): {prompts_distinct}/5")
    print(f"    Max sim range: {min(max_sims):.3f} - {max(max_sims):.3f}")

    print(f"\n  Reranking indicators:")
    print(f"    Unique collective picks: {prompts_rerank_unique}/5")

    emergence_novel = prompts_with_novel >= 3
    emergence_distinct = prompts_distinct >= 3

    print(f"\n  VERDICT:")
    if emergence_novel and emergence_distinct:
        print(f"    EMERGENCE DETECTED at 7B scale")
    elif emergence_novel:
        print(f"    PARTIAL — novel tokens but close to individuals")
    elif emergence_distinct:
        print(f"    PARTIAL — distinct collective but no novel tokens")
    else:
        scale = "local" if local_mode else "7B"
        print(f"    NO EMERGENCE at {scale} scale with current configuration")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed / 60:.1f} minutes", flush=True)

    summary = {
        "mode": mode,
        "local_mode": local_mode,
        "agents": [a[0] for a in agent_defs],
        "prompts_with_novel": prompts_with_novel,
        "prompts_not_subset": prompts_not_subset,
        "prompts_distinct": prompts_distinct,
        "prompts_rerank_unique": prompts_rerank_unique,
        "emergence_novel": emergence_novel,
        "emergence_distinct": emergence_distinct,
        "elapsed_seconds": elapsed,
    }

    output = {"summary": summary, "per_prompt": all_results}
    suffix = "_local" if local_mode else ""
    filename = f"emergence_7b_{mode}{suffix}.json"
    with open(RESULTS_DIR / filename, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {filename}", flush=True)

    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "cross"], default="single",
                        help="single = one model + role prompts, "
                             "cross = 3 model families with swapping")
    parser.add_argument("--local", action="store_true",
                        help="Local test mode: GPT-2/DistilGPT-2 on RTX 4070")
    args = parser.parse_args()
    run_experiment(mode=args.mode, local_mode=args.local)


if __name__ == "__main__":
    main()
