"""Round 9: Reranking — does the collective buffer select better text?

Generate 20 candidates from plain GPT-2 (no prefix injection).
Rerank by cosine similarity between each candidate's hidden state and:
  1. The collective buffer (after multi-agent loop)
  2. Each individual agent's buffer (no loop)
  3. Random selection (baseline)

Compare the selected outputs qualitatively and quantitatively.
"""

from __future__ import annotations

import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.isocal import IsotropyCalibrator
from quorum.bapc_codec import BAPCCodec
from quorum.loop import latent_resonance_loop, SyntheticAgent, LoopConfig

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACTION_LAYER = 7
S = 6
D = 512

_MODEL_CACHE: dict = {}

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

AGENT_DEFS = [
    ("neutral_A", "neutral"),
    ("supportive_B", "supportive"),
    ("critical_C", "critical"),
    ("carol_D", "carol"),
]


def _load_lm(model_name):
    key = model_name + "_lm"
    if key not in _MODEL_CACHE:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"  Loading {model_name} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        mdl.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _MODEL_CACHE[key] = (mdl, tok)
    return _MODEL_CACHE[key]


def load_codec():
    ckpt = torch.load(CHECKPOINT_DIR / "bapc_codec_final.pt",
                       map_location="cpu", weights_only=False)
    codec = BAPCCodec(agent_dim=768, buffer_dim=D, embed_dim=768, num_slots=S)
    codec.load_state_dict(ckpt["codec_state"])
    isocal = IsotropyCalibrator()
    isocal.load_state_dict(ckpt["isocal_state"])
    codec.isocal = isocal
    codec.eval()
    return codec


def make_live_agent_fn(model, tokenizer, codec, role_prefix, task_prompt):
    embed_layer = model.get_input_embeddings()
    full_prompt = role_prefix + task_prompt
    prompt_ids = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=64)
    prompt_ids_tensor = prompt_ids["input_ids"][0].to(next(model.parameters()).device)
    with torch.no_grad():
        prompt_embeds = embed_layer(prompt_ids_tensor)

    def delta_fn(B, t):
        device = next(model.parameters()).device
        B_cpu = B.cpu() if B.device != torch.device("cpu") else B
        with torch.no_grad():
            prefix = codec.decode(B_cpu).to(device)
        combined = torch.cat([prefix, prompt_embeds], dim=0).unsqueeze(0)
        with torch.no_grad():
            out = model(inputs_embeds=combined, output_hidden_states=True)
            h = out.hidden_states[EXTRACTION_LAYER][0, -1, :].cpu().float()
        z_new = codec.encode(h)
        return (z_new - B_cpu) * 0.3
    return delta_fn


def get_individual_encoding(model, tokenizer, codec, role_prefix, task_prompt):
    device = next(model.parameters()).device
    full_prompt = role_prefix + task_prompt
    inp = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=64)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[EXTRACTION_LAYER][0, -1, :].cpu().float()
    return codec.encode(h)


def generate_candidates(model, tokenizer, task_prompt, n=20, max_new_tokens=100):
    """Generate n candidate responses from plain GPT-2 (no prefix)."""
    device = next(model.parameters()).device
    inp = tokenizer(task_prompt, return_tensors="pt").to(device)
    candidates = []

    for _ in range(n):
        with torch.no_grad():
            gen_ids = model.generate(
                **inp, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.9, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen_ids[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        candidates.append(text)

    return candidates


def get_candidate_hiddens(model, tokenizer, task_prompt, candidates):
    """Extract layer-7 hidden state for each candidate (prompt + response)."""
    device = next(model.parameters()).device
    hiddens = []
    for text in candidates:
        full = task_prompt + text
        inp = tokenizer(full, return_tensors="pt", truncation=True, max_length=200).to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[EXTRACTION_LAYER][0, -1, :].cpu().float()
        hiddens.append(h)
    return hiddens


def rerank_by_buffer(candidates, candidate_hiddens, codec, buffer_state):
    """Rerank candidates by cosine similarity between their encoded state and the buffer."""
    sims = []
    buffer_slot0 = buffer_state[0]  # use slot 0 (factual grounding)

    for h in candidate_hiddens:
        z = codec.encode(h)  # [S, D]
        z_slot0 = z[0]       # [D] — same space as buffer
        sim = F.cosine_similarity(z_slot0.unsqueeze(0), buffer_slot0.unsqueeze(0)).item()
        sims.append(sim)

    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    return ranked, sims


def main():
    t_start = time.time()
    print("=" * 70)
    print("ROUND 9: RERANKING — DOES THE COLLECTIVE SELECT BETTER TEXT?")
    print("=" * 70, flush=True)

    model, tok = _load_lm("gpt2")
    model = model.to(DEVICE)
    codec = load_codec()

    all_results = []

    for prompt_idx, task_prompt in enumerate(TASK_PROMPTS):
        print(f"\n{'='*70}")
        print(f"  PROMPT {prompt_idx+1}/5: '{task_prompt}'")
        print(f"{'='*70}", flush=True)

        # Run convergence loop
        print(f"\n  Running convergence loop ...", flush=True)
        agents = []
        for i, (name, role_key) in enumerate(AGENT_DEFS):
            delta_fn = make_live_agent_fn(
                model, tok, codec, ROLE_PROMPTS[role_key], task_prompt
            )
            quality = 0.5 if role_key == "carol" else 0.7
            agents.append(SyntheticAgent(agent_id=i, delta_fn=delta_fn, quality=quality))

        cfg = LoopConfig(S=S, D=D, max_rounds=20, tol=0.02, alpha=0.5)
        B_final, diag = latent_resonance_loop(agents, cfg)
        print(f"    Converged in {len(diag.residuals)} rounds", flush=True)

        # Individual encodings
        individual_encodings = {}
        for name, role_key in AGENT_DEFS:
            individual_encodings[name] = get_individual_encoding(
                model, tok, codec, ROLE_PROMPTS[role_key], task_prompt
            )

        # Generate 20 candidates from plain GPT-2
        print(f"\n  Generating 20 candidates ...", flush=True)
        candidates = generate_candidates(model, tok, task_prompt, n=20)
        candidate_hiddens = get_candidate_hiddens(model, tok, task_prompt, candidates)
        print(f"    Generated {len(candidates)} candidates", flush=True)

        # Rerank by collective buffer
        coll_ranked, coll_sims = rerank_by_buffer(
            candidates, candidate_hiddens, codec, B_final
        )

        # Rerank by each individual
        ind_rankings = {}
        ind_sims_all = {}
        for name in ["neutral_A", "supportive_B", "critical_C", "carol_D"]:
            ranked, sims = rerank_by_buffer(
                candidates, candidate_hiddens, codec, individual_encodings[name]
            )
            ind_rankings[name] = ranked
            ind_sims_all[name] = sims

        # Random baseline
        import random
        random_pick = random.randint(0, len(candidates) - 1)

        # Results
        coll_best_idx = coll_ranked[0]
        coll_worst_idx = coll_ranked[-1]

        print(f"\n  --- COLLECTIVE BEST (rank 1/{len(candidates)}, sim={coll_sims[coll_best_idx]:.4f}) ---", flush=True)
        print(f"  {candidates[coll_best_idx][:400]}", flush=True)

        print(f"\n  --- COLLECTIVE WORST (rank {len(candidates)}/{len(candidates)}, sim={coll_sims[coll_worst_idx]:.4f}) ---", flush=True)
        print(f"  {candidates[coll_worst_idx][:400]}", flush=True)

        print(f"\n  --- RANDOM PICK (sim={coll_sims[random_pick]:.4f}) ---", flush=True)
        print(f"  {candidates[random_pick][:400]}", flush=True)

        # Do collective and individuals pick different candidates?
        print(f"\n  Selection comparison:", flush=True)
        print(f"    Collective picks candidate #{coll_best_idx}", flush=True)
        for name in ["neutral_A", "carol_D"]:
            ind_best = ind_rankings[name][0]
            print(f"    {name} picks candidate #{ind_best}"
                  f" (same={ind_best == coll_best_idx})", flush=True)
        print(f"    Random picks candidate #{random_pick}", flush=True)

        # Similarity distribution
        coll_sim_mean = sum(coll_sims) / len(coll_sims)
        coll_sim_spread = max(coll_sims) - min(coll_sims)
        print(f"\n  Collective similarity distribution:", flush=True)
        print(f"    Mean: {coll_sim_mean:.4f}, Spread: {coll_sim_spread:.4f}", flush=True)
        print(f"    Best: {coll_sims[coll_best_idx]:.4f}, Worst: {coll_sims[coll_worst_idx]:.4f}", flush=True)

        # Agreement matrix: how often do different selectors agree?
        top3_coll = set(coll_ranked[:3])
        agreements = {}
        for name in ["neutral_A", "supportive_B", "critical_C", "carol_D"]:
            top3_ind = set(ind_rankings[name][:3])
            overlap = len(top3_coll & top3_ind)
            agreements[name] = overlap
        print(f"\n  Top-3 agreement (collective vs individual):", flush=True)
        for name, overlap in agreements.items():
            print(f"    {name}: {overlap}/3 shared", flush=True)

        # All candidate lengths for context
        lengths = [len(c.split()) for c in candidates]
        coll_best_len = len(candidates[coll_best_idx].split())
        print(f"\n  Candidate lengths: {min(lengths)}-{max(lengths)} words "
              f"(collective pick: {coll_best_len})", flush=True)

        result = {
            "task_prompt": task_prompt,
            "loop_rounds": len(diag.residuals),
            "n_candidates": len(candidates),
            "collective_best_idx": coll_best_idx,
            "collective_best_text": candidates[coll_best_idx],
            "collective_worst_text": candidates[coll_worst_idx],
            "random_pick_text": candidates[random_pick],
            "collective_best_sim": coll_sims[coll_best_idx],
            "collective_worst_sim": coll_sims[coll_worst_idx],
            "collective_sim_mean": coll_sim_mean,
            "collective_sim_spread": coll_sim_spread,
            "individual_picks": {n: ind_rankings[n][0] for n in ind_rankings},
            "top3_agreement": agreements,
            "all_candidates": candidates,
            "all_sims": coll_sims,
        }
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}", flush=True)

    print(f"\n  Does the collective pick DIFFERENT candidates than individuals?", flush=True)
    total_different = 0
    for r in all_results:
        coll_pick = r["collective_best_idx"]
        ind_picks = list(r["individual_picks"].values())
        is_unique = coll_pick not in ind_picks
        total_different += int(is_unique)
        print(f"    '{r['task_prompt'][:50]}...': "
              f"collective=#{coll_pick}, "
              f"individuals={[r['individual_picks'][n] for n in ['neutral_A','carol_D']]}, "
              f"unique={'YES' if is_unique else 'no'}", flush=True)

    print(f"\n  Collective picks unique candidate: {total_different}/5 prompts", flush=True)

    avg_spread = sum(r["collective_sim_spread"] for r in all_results) / len(all_results)
    print(f"  Avg similarity spread across candidates: {avg_spread:.4f}", flush=True)
    print(f"  (Higher = collective is more discriminating)", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} minutes", flush=True)

    output = {"results": all_results}
    with open(RESULTS_DIR / "reranking_test.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to reranking_test.json", flush=True)


if __name__ == "__main__":
    main()
