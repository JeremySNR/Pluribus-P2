"""Round 8: Is the emergence useful? Compare generation quality.

Three conditions per prompt:
  1. Baseline — GPT-2 answers alone (no prefix)
  2. Individual-prefixed — GPT-2 answers with one agent's latent (no loop)
  3. Collective-prefixed — GPT-2 answers with the converged buffer (after loop)

Generates full text responses and compares them.
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


def generate_with_prefix(model, tokenizer, codec, buffer_state, task_prompt,
                         n_samples=5, max_new_tokens=80):
    """Generate multiple text samples with buffer decoded as prefix."""
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()

    prefix = codec.decode(buffer_state).to(device)
    prompt_ids = tokenizer(task_prompt, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        prompt_embeds = embed_layer(prompt_ids["input_ids"][0].to(device))
    combined = torch.cat([prefix, prompt_embeds], dim=0).unsqueeze(0)

    # Get logits for distribution analysis
    with torch.no_grad():
        out = model(inputs_embeds=combined)
        logits = out.logits[0, -1]

    # Generate text samples (use standard generation from prompt text)
    texts = []
    inp = tokenizer(task_prompt, return_tensors="pt").to(device)
    for _ in range(n_samples):
        with torch.no_grad():
            gen_ids = model.generate(
                **inp, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.8, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen_ids[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        texts.append(text)

    return texts, logits.cpu()


def generate_baseline(model, tokenizer, task_prompt, n_samples=5, max_new_tokens=80):
    """Generate without any prefix."""
    device = next(model.parameters()).device
    inp = tokenizer(task_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inp)
        logits = out.logits[0, -1]

    texts = []
    for _ in range(n_samples):
        with torch.no_grad():
            gen_ids = model.generate(
                **inp, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.8, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen_ids[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        texts.append(text)

    return texts, logits.cpu()


def get_individual_encoding(model, tokenizer, codec, role_prefix, task_prompt):
    device = next(model.parameters()).device
    full_prompt = role_prefix + task_prompt
    inp = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=64)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[EXTRACTION_LAYER][0, -1, :].cpu().float()
    return codec.encode(h)


def diversity_metrics(texts):
    """Compute lexical diversity of a set of texts."""
    all_words = []
    for t in texts:
        all_words.extend(t.lower().split())
    if not all_words:
        return {"unique_words": 0, "total_words": 0, "type_token_ratio": 0}
    unique = len(set(all_words))
    return {
        "unique_words": unique,
        "total_words": len(all_words),
        "type_token_ratio": unique / len(all_words),
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("ROUND 8: IS THE EMERGENCE USEFUL?")
    print("Comparing collective vs individual vs baseline generation")
    print("=" * 70, flush=True)

    model, tok = _load_lm("gpt2")
    model = model.to(DEVICE)
    codec = load_codec()

    all_results = []

    for prompt_idx, task_prompt in enumerate(TASK_PROMPTS):
        print(f"\n{'='*70}")
        print(f"  PROMPT {prompt_idx+1}/5: '{task_prompt}'")
        print(f"{'='*70}", flush=True)

        # Run convergence loop with live agents
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

        # Get individual encodings
        individual_encodings = {}
        for name, role_key in AGENT_DEFS:
            individual_encodings[name] = get_individual_encoding(
                model, tok, codec, ROLE_PROMPTS[role_key], task_prompt
            )

        # Condition 1: Baseline (no prefix)
        print(f"\n  Condition 1: Baseline (no prefix) ...", flush=True)
        baseline_texts, baseline_logits = generate_baseline(model, tok, task_prompt)
        print(f"    Sample: {baseline_texts[0][:150]}...", flush=True)

        # Condition 2: Individual-prefixed (best individual — neutral agent)
        print(f"\n  Condition 2: Individual-prefixed (neutral agent) ...", flush=True)
        ind_texts, ind_logits = generate_with_prefix(
            model, tok, codec, individual_encodings["neutral_A"], task_prompt
        )
        print(f"    Sample: {ind_texts[0][:150]}...", flush=True)

        # Condition 3: Collective-prefixed (after loop)
        print(f"\n  Condition 3: Collective-prefixed (after loop) ...", flush=True)
        coll_texts, coll_logits = generate_with_prefix(
            model, tok, codec, B_final, task_prompt
        )
        print(f"    Sample: {coll_texts[0][:150]}...", flush=True)

        # Also generate from each role's individual prefix
        print(f"\n  Additional: Each role's individual prefix ...", flush=True)
        role_texts = {}
        for name, role_key in AGENT_DEFS:
            rtexts, _ = generate_with_prefix(
                model, tok, codec, individual_encodings[name], task_prompt, n_samples=2
            )
            role_texts[name] = rtexts[0]
            print(f"    {name}: {rtexts[0][:100]}...", flush=True)

        # Distribution analysis
        p_baseline = F.softmax(baseline_logits, dim=-1)
        p_ind = F.softmax(ind_logits, dim=-1)
        p_coll = F.softmax(coll_logits, dim=-1)

        kl_ind_vs_base = F.kl_div(p_baseline.log(), p_ind, reduction="sum").item()
        kl_coll_vs_base = F.kl_div(p_baseline.log(), p_coll, reduction="sum").item()
        kl_coll_vs_ind = F.kl_div(p_ind.log(), p_coll, reduction="sum").item()

        # Top token analysis
        _, top_base_ids = p_baseline.topk(5)
        _, top_ind_ids = p_ind.topk(5)
        _, top_coll_ids = p_coll.topk(5)

        top_base = [tok.decode([t]) for t in top_base_ids]
        top_ind = [tok.decode([t]) for t in top_ind_ids]
        top_coll = [tok.decode([t]) for t in top_coll_ids]

        print(f"\n  Distribution analysis:", flush=True)
        print(f"    Baseline top-5:    {top_base}", flush=True)
        print(f"    Individual top-5:  {top_ind}", flush=True)
        print(f"    Collective top-5:  {top_coll}", flush=True)
        print(f"    KL(individual vs baseline): {kl_ind_vs_base:.2f}", flush=True)
        print(f"    KL(collective vs baseline): {kl_coll_vs_base:.2f}", flush=True)
        print(f"    KL(collective vs individual): {kl_coll_vs_ind:.2f}", flush=True)

        # Diversity metrics
        base_div = diversity_metrics(baseline_texts)
        ind_div = diversity_metrics(ind_texts)
        coll_div = diversity_metrics(coll_texts)

        print(f"\n  Lexical diversity (5 samples each):", flush=True)
        print(f"    Baseline:   TTR={base_div['type_token_ratio']:.3f} ({base_div['unique_words']} unique / {base_div['total_words']} total)", flush=True)
        print(f"    Individual: TTR={ind_div['type_token_ratio']:.3f} ({ind_div['unique_words']} unique / {ind_div['total_words']} total)", flush=True)
        print(f"    Collective: TTR={coll_div['type_token_ratio']:.3f} ({coll_div['unique_words']} unique / {coll_div['total_words']} total)", flush=True)

        # Average response length
        avg_base = sum(len(t.split()) for t in baseline_texts) / len(baseline_texts)
        avg_ind = sum(len(t.split()) for t in ind_texts) / len(ind_texts)
        avg_coll = sum(len(t.split()) for t in coll_texts) / len(coll_texts)
        print(f"\n  Avg response length:", flush=True)
        print(f"    Baseline: {avg_base:.0f} words", flush=True)
        print(f"    Individual: {avg_ind:.0f} words", flush=True)
        print(f"    Collective: {avg_coll:.0f} words", flush=True)

        # Full text output for manual comparison
        print(f"\n  --- FULL OUTPUTS (best of 5) ---", flush=True)
        print(f"\n  [BASELINE]:", flush=True)
        print(f"  {baseline_texts[0][:400]}", flush=True)
        print(f"\n  [INDIVIDUAL - neutral]:", flush=True)
        print(f"  {ind_texts[0][:400]}", flush=True)
        print(f"\n  [COLLECTIVE - after loop]:", flush=True)
        print(f"  {coll_texts[0][:400]}", flush=True)

        result = {
            "task_prompt": task_prompt,
            "loop_rounds": len(diag.residuals),
            "baseline_texts": baseline_texts,
            "individual_texts": ind_texts,
            "collective_texts": coll_texts,
            "role_texts": role_texts,
            "top_tokens_baseline": top_base,
            "top_tokens_individual": top_ind,
            "top_tokens_collective": top_coll,
            "kl_ind_vs_base": kl_ind_vs_base,
            "kl_coll_vs_base": kl_coll_vs_base,
            "kl_coll_vs_ind": kl_coll_vs_ind,
            "diversity_baseline": base_div,
            "diversity_individual": ind_div,
            "diversity_collective": coll_div,
            "avg_length_baseline": avg_base,
            "avg_length_individual": avg_ind,
            "avg_length_collective": avg_coll,
        }
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}", flush=True)

    print(f"\n  KL divergence from baseline (higher = more influence):", flush=True)
    print(f"  {'Prompt':<55} {'Individual':>12} {'Collective':>12}", flush=True)
    print(f"  {'-'*55} {'-'*12} {'-'*12}", flush=True)
    for r in all_results:
        print(f"  {r['task_prompt'][:55]:<55} {r['kl_ind_vs_base']:>12.2f} {r['kl_coll_vs_base']:>12.2f}",
              flush=True)

    avg_kl_ind = sum(r["kl_ind_vs_base"] for r in all_results) / len(all_results)
    avg_kl_coll = sum(r["kl_coll_vs_base"] for r in all_results) / len(all_results)
    print(f"\n  Average KL from baseline:", flush=True)
    print(f"    Individual: {avg_kl_ind:.2f}", flush=True)
    print(f"    Collective: {avg_kl_coll:.2f}", flush=True)

    if avg_kl_coll > avg_kl_ind * 1.2:
        print(f"    -> Collective steers MORE than individual ({avg_kl_coll/avg_kl_ind:.1f}x)", flush=True)
    elif avg_kl_coll > avg_kl_ind * 0.8:
        print(f"    -> Collective steers SIMILARLY to individual", flush=True)
    else:
        print(f"    -> Collective steers LESS than individual", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} minutes", flush=True)

    output = {"results": all_results}
    with open(RESULTS_DIR / "quality_comparison.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to quality_comparison.json", flush=True)


if __name__ == "__main__":
    main()
