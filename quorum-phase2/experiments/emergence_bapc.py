"""Round 7: Emergence test with BAPC codec and live agent forward passes.

The real test: agents actually think each round. They read the decoded buffer
as prefix embeddings, run a forward pass, and re-encode their new hidden state.
The buffer co-evolves with the agents instead of settling between fixed points.

4 agents (all GPT-2, different role prompts) compete through the loop.
After convergence, compare collective-prefixed vs individual-prefixed generation.
"""

from __future__ import annotations

import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.isocal import IsotropyCalibrator, IsocalConfig
from quorum.bapc_codec import BAPCCodec
from quorum.loop import latent_resonance_loop, SyntheticAgent, LoopConfig
from quorum.bapc_training import js_divergence

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
    """Load the trained BAPC codec from checkpoint."""
    ckpt_path = CHECKPOINT_DIR / "bapc_codec_final.pt"
    print(f"  Loading codec from {ckpt_path.name} ...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    codec = BAPCCodec(agent_dim=768, buffer_dim=D, embed_dim=768, num_slots=S)
    codec.load_state_dict(ckpt["codec_state"])

    isocal = IsotropyCalibrator()
    isocal.load_state_dict(ckpt["isocal_state"])
    codec.isocal = isocal

    codec.eval()
    return codec


def make_live_agent_fn(model, tokenizer, codec, role_prefix, task_prompt):
    """Create a delta function where the agent actually thinks each round.

    Each round:
      1. Decode buffer → prefix embeddings
      2. Forward pass: prefix + role_prompt + task_prompt
      3. Extract hidden state from layer 7
      4. Encode through codec → proposed buffer state
      5. Return delta (proposed - current buffer)
    """
    embed_layer = model.get_input_embeddings()
    full_prompt = role_prefix + task_prompt
    prompt_ids = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=64)
    prompt_ids_tensor = prompt_ids["input_ids"][0].to(next(model.parameters()).device)

    with torch.no_grad():
        prompt_embeds = embed_layer(prompt_ids_tensor)

    def delta_fn(B, t):
        device = next(model.parameters()).device

        # Decode current buffer to prefix embeddings
        B_cpu = B.cpu() if B.device != torch.device("cpu") else B
        with torch.no_grad():
            prefix = codec.decode(B_cpu)  # [S, d_emb]

        prefix_dev = prefix.to(device)
        combined = torch.cat([prefix_dev, prompt_embeds], dim=0).unsqueeze(0)

        with torch.no_grad():
            out = model(inputs_embeds=combined, output_hidden_states=True)
            h = out.hidden_states[EXTRACTION_LAYER][0, -1, :]

        # Encode new hidden state through codec
        h_cpu = h.cpu().float()
        z_new = codec.encode(h_cpu)  # [S, D]

        return (z_new - B_cpu) * 0.3

    return delta_fn


def get_individual_encoding(model, tokenizer, codec, role_prefix, task_prompt):
    """Get an agent's buffer encoding WITHOUT the loop (single forward pass, no prefix)."""
    full_prompt = role_prefix + task_prompt
    inp = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=64)
    inp = {k: v.to(next(model.parameters()).device) for k, v in inp.items()}

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[EXTRACTION_LAYER][0, -1, :]

    h_cpu = h.cpu().float()
    return codec.encode(h_cpu)  # [S, D]


def generate_with_prefix(model, tokenizer, codec, buffer_state, task_prompt,
                         max_new_tokens=50):
    """Generate text with buffer state decoded as prefix embeddings."""
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()

    prefix = codec.decode(buffer_state)  # [S, d_emb]
    prompt_ids = tokenizer(task_prompt, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        prompt_embeds = embed_layer(prompt_ids["input_ids"][0].to(device))

    prefix_dev = prefix.to(device)
    combined = torch.cat([prefix_dev, prompt_embeds], dim=0).unsqueeze(0)

    with torch.no_grad():
        out = model(inputs_embeds=combined, output_hidden_states=True)
        last_logits = out.logits[0, -1]

    # Generate by continuing from the combined embeddings
    # Use the logits to get the first token, then continue autoregressively
    inp_for_gen = tokenizer(task_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inp_for_gen, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.8, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(gen_ids[0][inp_for_gen["input_ids"].shape[1]:],
                            skip_special_tokens=True)

    return text, last_logits.cpu()


def main():
    t_start = time.time()
    print("=" * 70)
    print("ROUND 7: EMERGENCE TEST WITH LIVE AGENT FORWARD PASSES")
    print(f"Device: {DEVICE}")
    print("=" * 70, flush=True)

    model, tok = _load_lm("gpt2")
    model = model.to(DEVICE)
    codec = load_codec()

    all_results = []

    for prompt_idx, task_prompt in enumerate(TASK_PROMPTS):
        print(f"\n{'='*70}")
        print(f"  PROMPT {prompt_idx+1}/5: '{task_prompt}'")
        print(f"{'='*70}", flush=True)

        # Step 1: Get individual encodings (no loop, no prefix)
        print(f"\n  Step 1: Individual encodings (no loop) ...", flush=True)
        individual_encodings = {}
        for name, role_key in AGENT_DEFS:
            z = get_individual_encoding(model, tok, codec,
                                        ROLE_PROMPTS[role_key], task_prompt)
            individual_encodings[name] = z
            print(f"    {name}: norm={z.norm():.4f}", flush=True)

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
        avg_ind_sim = sum(ind_sims.values()) / len(ind_sims)
        print(f"    Avg pairwise sim between individuals: {avg_ind_sim:.4f}", flush=True)

        # Step 2: Run the convergence loop with live agents
        print(f"\n  Step 2: Running convergence loop with live agents ...", flush=True)
        agents = []
        for i, (name, role_key) in enumerate(AGENT_DEFS):
            delta_fn = make_live_agent_fn(
                model, tok, codec, ROLE_PROMPTS[role_key], task_prompt
            )
            quality = 0.5 if role_key == "carol" else 0.7
            agents.append(SyntheticAgent(agent_id=i, delta_fn=delta_fn, quality=quality))

        cfg = LoopConfig(S=S, D=D, max_rounds=20, tol=0.02, alpha=0.5)
        B_final, diag = latent_resonance_loop(agents, cfg)

        n_rounds = len(diag.residuals)
        final_residual = diag.residuals[-1]
        print(f"    Converged in {n_rounds} rounds, residual={final_residual:.6f}", flush=True)

        # Step 3: Compare collective to individuals
        print(f"\n  Step 3: Collective vs individual analysis ...", flush=True)

        # Distinctness: cosine similarity between collective and each individual
        distinctness = {}
        for name in ind_names:
            sim = F.cosine_similarity(
                B_final[0].unsqueeze(0),
                individual_encodings[name][0].unsqueeze(0)
            ).item()
            distinctness[f"collective_vs_{name}"] = sim
            print(f"    collective vs {name}: {sim:.4f}", flush=True)

        sim_values = list(distinctness.values())
        spread = max(sim_values) - min(sim_values)
        max_sim = max(sim_values)
        print(f"    Spread: {spread:.4f}, Max sim: {max_sim:.4f}", flush=True)

        # Step 4: Generate with collective vs individual prefixes
        print(f"\n  Step 4: Generation comparison ...", flush=True)

        # Collective-prefixed logits
        _, collective_logits = generate_with_prefix(
            model, tok, codec, B_final, task_prompt
        )

        # Individual-prefixed logits
        individual_logits = {}
        for name in ind_names:
            _, ind_logits = generate_with_prefix(
                model, tok, codec, individual_encodings[name], task_prompt
            )
            individual_logits[name] = ind_logits

        # No-prefix baseline logits
        inp = tok(task_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            baseline_logits = model(**inp).logits[0, -1].cpu()

        # JS divergence: collective vs each individual
        print(f"\n  Step 5: JS divergence analysis ...", flush=True)
        p_collective = F.softmax(collective_logits, dim=-1)
        p_baseline = F.softmax(baseline_logits, dim=-1)

        kl_collective_vs_baseline = F.kl_div(
            p_baseline.log(), p_collective, reduction="sum"
        ).item()

        js_collective_vs_individual = {}
        for name in ind_names:
            p_ind = F.softmax(individual_logits[name], dim=-1)
            js = js_divergence(p_collective.unsqueeze(0), p_ind.unsqueeze(0)).item()
            js_collective_vs_individual[name] = js
            print(f"    JS(collective vs {name}): {js:.4f}", flush=True)

        avg_js = sum(js_collective_vs_individual.values()) / len(js_collective_vs_individual)
        print(f"    Avg JS(collective vs individual): {avg_js:.4f}", flush=True)
        print(f"    KL(collective vs no-prefix): {kl_collective_vs_baseline:.4f}", flush=True)

        # Top tokens comparison
        print(f"\n  Step 6: Top token comparison ...", flush=True)
        top_k = 10
        _, top_coll_ids = p_collective.topk(top_k)
        collective_tokens = [tok.decode([tid]) for tid in top_coll_ids]
        print(f"    Collective top-{top_k}: {collective_tokens}", flush=True)

        all_individual_tokens = set()
        individual_top_tokens = {}
        for name in ind_names:
            p_ind = F.softmax(individual_logits[name], dim=-1)
            _, top_ids = p_ind.topk(top_k)
            ind_tokens = [tok.decode([tid]) for tid in top_ids]
            individual_top_tokens[name] = ind_tokens
            all_individual_tokens.update(ind_tokens)
            print(f"    {name} top-5: {ind_tokens[:5]}", flush=True)

        novel_tokens = [t for t in collective_tokens if t not in all_individual_tokens]
        subset = set(collective_tokens).issubset(all_individual_tokens)
        print(f"    Novel tokens in collective: {novel_tokens}", flush=True)
        print(f"    Collective is subset of union: {subset}", flush=True)

        result = {
            "task_prompt": task_prompt,
            "loop_rounds": n_rounds,
            "final_residual": final_residual,
            "individual_pairwise_sim": avg_ind_sim,
            "distinctness": distinctness,
            "distinctness_spread": spread,
            "max_sim_to_individual": max_sim,
            "avg_js_collective_vs_individual": avg_js,
            "kl_collective_vs_baseline": kl_collective_vs_baseline,
            "collective_top_tokens": collective_tokens,
            "individual_top_tokens": individual_top_tokens,
            "novel_tokens": novel_tokens,
            "novel_count": len(novel_tokens),
            "collective_is_subset": subset,
        }
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("EMERGENCE TEST SUMMARY (BAPC + Live Agents)")
    print(f"{'='*70}", flush=True)

    novel_counts = [r["novel_count"] for r in all_results]
    subset_flags = [r["collective_is_subset"] for r in all_results]
    spreads = [r["distinctness_spread"] for r in all_results]
    max_sims = [r["max_sim_to_individual"] for r in all_results]
    avg_js_vals = [r["avg_js_collective_vs_individual"] for r in all_results]
    ind_sims = [r["individual_pairwise_sim"] for r in all_results]

    prompts_with_novel = sum(1 for c in novel_counts if c > 0)
    prompts_subset = sum(subset_flags)

    print(f"\n  Individual agent pairwise similarity: {min(ind_sims):.3f} - {max(ind_sims):.3f}")
    print(f"  (Lower = agents are more diverse in buffer space)")

    print(f"\n  Emergence indicators:")
    print(f"    Novel tokens: {prompts_with_novel}/5 prompts")
    print(f"    Collective NOT subset of union: {5 - prompts_subset}/5")
    print(f"    Avg JS(collective vs individual): {sum(avg_js_vals)/len(avg_js_vals):.4f}")
    print(f"    Distinctness spread: {min(spreads):.3f} - {max(spreads):.3f}")
    print(f"    Max sim to any individual: {min(max_sims):.3f} - {max(max_sims):.3f}")

    emergence_novel = prompts_with_novel >= 3
    emergence_distinct = sum(1 for s in max_sims if s < 0.95) >= 3

    print(f"\n  VERDICT:")
    if emergence_novel and emergence_distinct:
        print(f"    EMERGENCE DETECTED — novel tokens + distinct collective state")
    elif emergence_novel:
        print(f"    PARTIAL EMERGENCE — novel tokens but collective close to individuals")
    elif emergence_distinct:
        print(f"    PARTIAL EMERGENCE — distinct collective but no novel tokens")
    else:
        print(f"    NO EMERGENCE — collective does not contain novel representations")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} minutes", flush=True)

    output = {
        "summary": {
            "prompts_with_novel": prompts_with_novel,
            "prompts_subset": prompts_subset,
            "emergence_novel": emergence_novel,
            "emergence_distinct": emergence_distinct,
            "avg_individual_sim": sum(ind_sims) / len(ind_sims),
        },
        "per_prompt": all_results,
    }
    with open(RESULTS_DIR / "emergence_bapc.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to emergence_bapc.json", flush=True)


if __name__ == "__main__":
    main()
