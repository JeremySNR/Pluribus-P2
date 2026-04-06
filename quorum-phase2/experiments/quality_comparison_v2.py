"""Round 8b: Quality comparison with custom autoregressive generation.

Fixes Round 8's limitation: generates text token-by-token from the
prefix-injected embedding sequence so the collective's steering signal
actually propagates through the full output.
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


def generate_from_embeds(model, tokenizer, input_embeds, max_new_tokens=80,
                         temperature=0.8, top_p=0.95):
    """Custom autoregressive generation starting from an embedding sequence.

    Unlike HuggingFace generate(), this actually continues from the
    prefix embeddings so the steering signal propagates.
    """
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()
    eos_id = tokenizer.eos_token_id

    current_embeds = input_embeds.to(device)
    if current_embeds.dim() == 2:
        current_embeds = current_embeds.unsqueeze(0)

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(inputs_embeds=current_embeds)
            next_logits = out.logits[0, -1, :] / temperature

            # Top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            probs = F.softmax(sorted_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            next_token_id = sorted_indices[next_idx].item()

            if next_token_id == eos_id:
                break

            generated_ids.append(next_token_id)

            next_embed = embed_layer(
                torch.tensor([[next_token_id]], device=device)
            )
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text


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


def generate_prefixed(model, tokenizer, codec, buffer_state, task_prompt,
                      n_samples=3, max_new_tokens=80):
    """Generate text with prefix embeddings using custom autoregressive loop."""
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()

    prefix = codec.decode(buffer_state).to(device)
    prompt_ids = tokenizer(task_prompt, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        prompt_embeds = embed_layer(prompt_ids["input_ids"][0].to(device))
    combined = torch.cat([prefix, prompt_embeds], dim=0)

    texts = []
    for _ in range(n_samples):
        text = generate_from_embeds(model, tokenizer, combined, max_new_tokens)
        texts.append(text)
    return texts


def generate_baseline(model, tokenizer, task_prompt, n_samples=3, max_new_tokens=80):
    """Generate without prefix using the same custom loop for fair comparison."""
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()

    prompt_ids = tokenizer(task_prompt, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        prompt_embeds = embed_layer(prompt_ids["input_ids"][0].to(device))

    texts = []
    for _ in range(n_samples):
        text = generate_from_embeds(model, tokenizer, prompt_embeds, max_new_tokens)
        texts.append(text)
    return texts


def main():
    t_start = time.time()
    print("=" * 70)
    print("ROUND 8b: QUALITY COMPARISON WITH PREFIX-STEERED GENERATION")
    print("=" * 70, flush=True)

    model, tok = _load_lm("gpt2")
    model = model.to(DEVICE)
    codec = load_codec()

    all_results = []

    for prompt_idx, task_prompt in enumerate(TASK_PROMPTS):
        print(f"\n{'='*70}")
        print(f"  PROMPT {prompt_idx+1}/5: '{task_prompt}'")
        print(f"{'='*70}", flush=True)

        # Run loop
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

        # Generate: baseline
        print(f"\n  Generating baseline (no prefix) ...", flush=True)
        baseline_texts = generate_baseline(model, tok, task_prompt)

        # Generate: each individual prefix
        print(f"  Generating individual-prefixed ...", flush=True)
        individual_texts = {}
        for name, role_key in AGENT_DEFS:
            texts = generate_prefixed(
                model, tok, codec, individual_encodings[name], task_prompt, n_samples=2
            )
            individual_texts[name] = texts

        # Generate: collective prefix
        print(f"  Generating collective-prefixed ...", flush=True)
        collective_texts = generate_prefixed(
            model, tok, codec, B_final, task_prompt
        )

        # Display results
        print(f"\n  --- BASELINE (no prefix) ---", flush=True)
        print(f"  {baseline_texts[0][:300]}", flush=True)

        print(f"\n  --- INDIVIDUAL PREFIXED ---", flush=True)
        for name in ["neutral_A", "carol_D"]:
            print(f"  [{name}]: {individual_texts[name][0][:200]}", flush=True)

        print(f"\n  --- COLLECTIVE PREFIXED (after loop) ---", flush=True)
        print(f"  {collective_texts[0][:300]}", flush=True)

        # Compare all three best outputs side by side
        print(f"\n  --- SIDE BY SIDE (best of each) ---", flush=True)
        print(f"\n  BASELINE:", flush=True)
        print(f"  {baseline_texts[0][:400]}", flush=True)
        print(f"\n  COLLECTIVE:", flush=True)
        print(f"  {collective_texts[0][:400]}", flush=True)

        # Length and diversity
        def avg_len(texts):
            return sum(len(t.split()) for t in texts) / max(len(texts), 1)

        def ttr(texts):
            words = []
            for t in texts:
                words.extend(t.lower().split())
            return len(set(words)) / max(len(words), 1)

        print(f"\n  Metrics:", flush=True)
        print(f"    Baseline:   {avg_len(baseline_texts):.0f} words, TTR={ttr(baseline_texts):.3f}", flush=True)
        all_ind = [t for ts in individual_texts.values() for t in ts]
        print(f"    Individual: {avg_len(all_ind):.0f} words, TTR={ttr(all_ind):.3f}", flush=True)
        print(f"    Collective: {avg_len(collective_texts):.0f} words, TTR={ttr(collective_texts):.3f}", flush=True)

        result = {
            "task_prompt": task_prompt,
            "loop_rounds": len(diag.residuals),
            "baseline_texts": baseline_texts,
            "individual_texts": {k: v for k, v in individual_texts.items()},
            "collective_texts": collective_texts,
            "avg_len_baseline": avg_len(baseline_texts),
            "avg_len_individual": avg_len(all_ind),
            "avg_len_collective": avg_len(collective_texts),
            "ttr_baseline": ttr(baseline_texts),
            "ttr_individual": ttr(all_ind),
            "ttr_collective": ttr(collective_texts),
        }
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}", flush=True)

    print(f"\n  {'Prompt':<55} {'Base TTR':>10} {'Ind TTR':>10} {'Coll TTR':>10}", flush=True)
    print(f"  {'-'*55} {'-'*10} {'-'*10} {'-'*10}", flush=True)
    for r in all_results:
        print(f"  {r['task_prompt'][:55]:<55} {r['ttr_baseline']:>10.3f} {r['ttr_individual']:>10.3f} {r['ttr_collective']:>10.3f}",
              flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} minutes", flush=True)

    output = {"results": all_results}
    with open(RESULTS_DIR / "quality_comparison_v2.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to quality_comparison_v2.json", flush=True)


if __name__ == "__main__":
    main()
