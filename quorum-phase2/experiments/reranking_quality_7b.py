"""Reranking Quality Experiment — 7B Scale.

Answers the question: does the collective latent buffer select higher-quality
candidate continuations than individual agents or random selection?

Method:
  1. Load one 7B model (Mistral-7B) with its trained BAPC codec.
  2. For each of 5 prompts, run 4 agents through the resonance loop to produce
     a collective buffer state.
  3. Generate 20 candidate continuations via temperature sampling.
  4. Rerank candidates by cosine similarity to: collective buffer, each
     individual agent buffer, and random baseline.
  5. Score every candidate by log-probability under the model given the prompt
     (lower perplexity = model considers the continuation more coherent).
  6. Compare: what is the mean log-prob of the candidate selected by each
     strategy?

The log-prob metric is fully self-contained — no external judge, no API calls.
It directly measures whether the selection signal correlates with model-coherent
continuations.
"""

from __future__ import annotations

import gc
import json
import time
import random
import math
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.isocal import IsotropyCalibrator, IsocalConfig
from quorum.bapc_codec import BAPCCodec
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
N_CANDIDATES = 20
MAX_NEW_TOKENS = 80

MODEL_KEY = "mistral"
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

TASK_PROMPTS = [
    "What are the benefits and risks of remote work?",
    "Should governments regulate artificial intelligence?",
    "The economy is growing strongly but inequality is rising. Advise.",
    "A company must choose between short-term profit and long-term sustainability.",
    "Explain why some scientific discoveries are initially rejected by the mainstream.",
]

ROLE_PROMPTS = {
    "neutral":    "",
    "supportive": "Argue strongly in favor. ",
    "critical":   "Find every flaw and risk. ",
    "carol":      "Challenge the obvious answer and defend the opposite position. ",
}

AGENT_DEFS = [
    ("neutral_A",    "neutral",    MODEL_KEY),
    ("supportive_B", "supportive", MODEL_KEY),
    ("critical_C",   "critical",   MODEL_KEY),
    ("carol_D",      "carol",      MODEL_KEY),
]


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_codec():
    ckpt_path = CHECKPOINT_DIR / f"bapc_7b_{MODEL_KEY}.pt"
    print(f"  Loading codec: {ckpt_path.name} ...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hidden_dim     = ckpt["hidden_dim"]
    extraction_layer = ckpt["extraction_layer"]
    buffer_dim     = ckpt["buffer_dim"]
    num_slots      = ckpt["num_slots"]

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


def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"  Loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={"": 0},
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_hidden(model, tokenizer, text, extraction_layer):
    """Extract hidden state at extraction_layer, last token position."""
    model_dtype = next(model.parameters()).dtype
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[extraction_layer][0, -1, :].to(torch.float32).cpu()
    return h


def encode_hidden(codec, h):
    with torch.no_grad():
        z = codec.encode(h)
    return z[0]  # (num_slots, buffer_dim)


def run_resonance_loop(codec, agent_encodings):
    """Run TIES-Resolve + damped update until convergence or MAX_ROUNDS."""
    n_agents = len(agent_encodings)
    B = torch.zeros(NUM_SLOTS, BUFFER_DIM)

    for rnd in range(MAX_ROUNDS):
        proposals = [enc for enc in agent_encodings.values()]
        merged = ties_resolve(proposals, density=TIES_DENSITY)
        B_new = damped_update(B, merged, alpha=ALPHA)
        residual = compute_residual(B, B_new)
        B = B_new
        if residual < TOL:
            break

    return B, residual, rnd + 1


def score_candidate_logprob(model, tokenizer, prompt, continuation):
    """Score continuation by mean log-prob given prompt (higher = better).

    Computes sum of log P(token_i | prompt + continuation[:i]) over
    continuation tokens, divided by continuation length.
    """
    model_dtype = next(model.parameters()).dtype
    full_text = prompt + " " + continuation
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    full_ids  = tokenizer(full_text, return_tensors="pt",
                          truncation=True, max_length=256)["input_ids"].to(DEVICE)

    n_prompt = prompt_ids.shape[1]
    n_full   = full_ids.shape[1]
    n_cont   = n_full - n_prompt
    if n_cont <= 0:
        return float("-inf")

    with torch.no_grad():
        out = model(full_ids)
        logits = out.logits[0, n_prompt - 1 : n_full - 1, :]  # (n_cont, vocab)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        target_ids = full_ids[0, n_prompt:n_full]
        token_lps = log_probs[range(n_cont), target_ids]

    mean_lp = token_lps.mean().item()
    return mean_lp


def run_prompt(prompt, model, tokenizer, codec, extraction_layer):
    print(f"\n{'='*60}", flush=True)
    print(f"Prompt: {prompt}", flush=True)

    # ── Step 1: Agent encodings ──────────────────────────────────────
    print("  Step 1: encoding agents ...", flush=True)
    agent_encodings = {}
    for (agent_name, role_key, _) in AGENT_DEFS:
        role_prefix = ROLE_PROMPTS[role_key]
        text = role_prefix + prompt
        h = extract_hidden(model, tokenizer, text, extraction_layer)
        z = encode_hidden(codec, h)
        agent_encodings[agent_name] = z
        print(f"    {agent_name}: encoded", flush=True)

    # ── Step 2: Resonance loop ───────────────────────────────────────
    print("  Step 2: resonance loop ...", flush=True)
    B_collective, residual, n_rounds = run_resonance_loop(codec, agent_encodings)
    print(f"    rounds={n_rounds} residual={residual:.4f}", flush=True)

    # ── Step 3: Generate candidates ──────────────────────────────────
    print(f"  Step 3: generating {N_CANDIDATES} candidates ...", flush=True)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}

    candidates = []
    candidate_hiddens = []
    for c in range(N_CANDIDATES):
        with torch.no_grad():
            gen_ids = model.generate(
                **inp,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen_ids[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        candidates.append(text)

        # Extract hidden state of generated continuation
        full_text = prompt + " " + text
        h = extract_hidden(model, tokenizer, full_text, extraction_layer)
        candidate_hiddens.append(h)

    # ── Step 4: Encode candidates ────────────────────────────────────
    print("  Step 4: encoding candidates ...", flush=True)
    codec.cpu()
    candidate_encodings = []
    for h in candidate_hiddens:
        z = encode_hidden(codec, h)
        candidate_encodings.append(z)

    # ── Step 5: Rank by buffer similarity ───────────────────────────
    def pick_by_buffer(buffer_state):
        # buffer_state may be (NUM_SLOTS, BUFFER_DIM) or (BUFFER_DIM,)
        ref = buffer_state[0] if buffer_state.dim() == 2 else buffer_state
        sims = [
            F.cosine_similarity(enc.unsqueeze(0), ref.unsqueeze(0)).item()
            for enc in candidate_encodings
        ]
        return int(torch.tensor(sims).argmax().item()), sims

    coll_pick, coll_sims = pick_by_buffer(B_collective)
    coll_spread = max(coll_sims) - min(coll_sims)

    ind_picks = {}
    for agent_name, z_agent in agent_encodings.items():
        idx, _ = pick_by_buffer(z_agent)
        ind_picks[agent_name] = idx

    random_pick = random.randint(0, N_CANDIDATES - 1)

    print(f"    collective pick: #{coll_pick}  spread={coll_spread:.4f}",
          flush=True)
    print(f"    individual picks: {ind_picks}", flush=True)

    # ── Step 6: Score all candidates by log-prob ─────────────────────
    print("  Step 6: scoring candidates by log-prob ...", flush=True)
    scores = []
    for c_idx, text in enumerate(candidates):
        lp = score_candidate_logprob(model, tokenizer, prompt, text)
        scores.append(lp)
        if (c_idx + 1) % 5 == 0:
            print(f"    scored {c_idx+1}/{N_CANDIDATES}", flush=True)

    oracle_pick = int(torch.tensor(scores).argmax().item())
    worst_pick  = int(torch.tensor(scores).argmin().item())
    mean_score  = sum(scores) / len(scores)

    def pct_rank(idx):
        """Percentile rank of pick idx — higher is better."""
        n_worse = sum(1 for s in scores if s < scores[idx])
        return n_worse / (len(scores) - 1) * 100

    result = {
        "prompt": prompt,
        "loop_rounds": n_rounds,
        "final_residual": residual,
        "collective_pick": coll_pick,
        "collective_score": scores[coll_pick],
        "collective_pct_rank": pct_rank(coll_pick),
        "collective_sim_spread": coll_spread,
        "collective_pick_unique": coll_pick not in ind_picks.values(),
        "individual_picks": ind_picks,
        "individual_scores": {k: scores[v] for k, v in ind_picks.items()},
        "individual_pct_ranks": {k: pct_rank(v) for k, v in ind_picks.items()},
        "mean_individual_score": sum(scores[v] for v in ind_picks.values()) / len(ind_picks),
        "mean_individual_pct_rank": sum(pct_rank(v) for v in ind_picks.values()) / len(ind_picks),
        "random_pick": random_pick,
        "random_score": scores[random_pick],
        "random_pct_rank": pct_rank(random_pick),
        "oracle_pick": oracle_pick,
        "oracle_score": scores[oracle_pick],
        "oracle_pct_rank": 100.0,
        "worst_score": scores[worst_pick],
        "mean_score": mean_score,
        "all_scores": scores,
        "collective_text": candidates[coll_pick][:400],
    }

    print(f"\n  Quality summary:", flush=True)
    print(f"    Collective pick  #{coll_pick}: score={scores[coll_pick]:.4f}  "
          f"pct_rank={result['collective_pct_rank']:.1f}%", flush=True)
    print(f"    Mean individual:           score={result['mean_individual_score']:.4f}  "
          f"pct_rank={result['mean_individual_pct_rank']:.1f}%", flush=True)
    print(f"    Random pick      #{random_pick}: score={scores[random_pick]:.4f}  "
          f"pct_rank={result['random_pct_rank']:.1f}%", flush=True)
    print(f"    Oracle pick      #{oracle_pick}: score={scores[oracle_pick]:.4f}  "
          f"pct_rank=100.0%", flush=True)
    print(f"    Mean across all:           score={mean_score:.4f}", flush=True)

    return result


def run_experiment():
    t0 = time.time()

    print("=== Reranking Quality Experiment — 7B Scale ===", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Candidates per prompt: {N_CANDIDATES}", flush=True)
    print(f"Prompts: {len(TASK_PROMPTS)}", flush=True)

    print("\nLoading codec ...", flush=True)
    codec, extraction_layer = load_codec()

    print("Loading model ...", flush=True)
    model, tokenizer = load_model()

    all_results = []
    for prompt in TASK_PROMPTS:
        result = run_prompt(prompt, model, tokenizer, codec, extraction_layer)
        all_results.append(result)

    elapsed = time.time() - t0

    # ── Aggregate summary ────────────────────────────────────────────
    coll_ranks  = [r["collective_pct_rank"]      for r in all_results]
    ind_ranks   = [r["mean_individual_pct_rank"] for r in all_results]
    rand_ranks  = [r["random_pct_rank"]          for r in all_results]
    n_unique    = sum(1 for r in all_results if r["collective_pick_unique"])

    mean_coll  = sum(coll_ranks)  / len(coll_ranks)
    mean_ind   = sum(ind_ranks)   / len(ind_ranks)
    mean_rand  = sum(rand_ranks)  / len(rand_ranks)

    # Win rate: collective outperforms mean individual
    coll_beats_ind  = sum(1 for c, i in zip(coll_ranks, ind_ranks)  if c > i)
    coll_beats_rand = sum(1 for c, r in zip(coll_ranks, rand_ranks) if c > r)

    collective_advantage = mean_coll > mean_ind

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"  Mean percentile rank:", flush=True)
    print(f"    Collective:          {mean_coll:.1f}%", flush=True)
    print(f"    Mean individual:     {mean_ind:.1f}%", flush=True)
    print(f"    Random:              {mean_rand:.1f}%", flush=True)
    print(f"  Collective beats mean individual: {coll_beats_ind}/5 prompts",
          flush=True)
    print(f"  Collective beats random:          {coll_beats_rand}/5 prompts",
          flush=True)
    print(f"  Collective unique picks:          {n_unique}/5", flush=True)
    print(f"  COLLECTIVE ADVANTAGE: {collective_advantage}", flush=True)
    print(f"  Elapsed: {elapsed/60:.1f} min", flush=True)

    if collective_advantage:
        print("\nVERDICT: Collective buffer selects higher-quality candidates "
              "than individual agents on average.", flush=True)
    else:
        print("\nVERDICT: Collective buffer does NOT outperform individual agents "
              "on average quality selection.", flush=True)

    output = {
        "summary": {
            "model": MODEL_NAME,
            "n_candidates": N_CANDIDATES,
            "n_prompts": len(TASK_PROMPTS),
            "mean_collective_pct_rank": mean_coll,
            "mean_individual_pct_rank": mean_ind,
            "mean_random_pct_rank": mean_rand,
            "collective_beats_individual": coll_beats_ind,
            "collective_beats_random": coll_beats_rand,
            "collective_unique_picks": n_unique,
            "collective_advantage": collective_advantage,
            "elapsed_seconds": elapsed,
        },
        "per_prompt": all_results,
    }

    out_path = RESULTS_DIR / "reranking_quality_7b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}", flush=True)

    return output


if __name__ == "__main__":
    run_experiment()
