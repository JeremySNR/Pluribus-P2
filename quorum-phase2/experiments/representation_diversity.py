"""Round 5: Where does LLM diversity live — representation or generation?

Three experiments:
  A. Same hidden state -> 50 completions. Do they diverge in stance?
  B. Within-model vs between-model text diversity. Is cross-model diversity
     just sampling noise?
  C. Do role prompts (neutral/supportive/critical/Carol) change hidden states
     or just generation distributions?
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_MODEL_CACHE: dict = {}

TASK_PROMPTS = [
    "What are the benefits and risks of remote work?",
    "Should governments regulate artificial intelligence?",
    "The economy is growing strongly but inequality is rising. Advise.",
    "A company must choose between short-term profit and long-term sustainability.",
    "Explain why some scientific discoveries are initially rejected by the mainstream.",
]


def _load_lm(model_name):
    key = model_name + "_lm"
    if key not in _MODEL_CACHE:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"  Loading {model_name} (causal LM) ...")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        mdl.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _MODEL_CACHE[key] = (mdl, tok)
    return _MODEL_CACHE[key]


def _load_base(model_name):
    if model_name not in _MODEL_CACHE:
        from transformers import AutoTokenizer, AutoModel
        print(f"  Loading {model_name} (base) ...")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        mdl.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _MODEL_CACHE[model_name] = (mdl, tok)
    return _MODEL_CACHE[model_name]


def extract_hidden(model_name, text):
    mdl, tok = _load_base(model_name)
    if not text or not text.strip():
        return torch.zeros(mdl.config.hidden_size)
    with torch.no_grad():
        inp = tok(text, return_tensors="pt", truncation=True, max_length=128)
        if inp["input_ids"].shape[1] == 0:
            return torch.zeros(mdl.config.hidden_size)
        out = mdl(**inp, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :]


def generate_completions(model_name, prompt, n=20, max_new_tokens=60, temperature=0.9):
    mdl, tok = _load_lm(model_name)
    inputs = tok(prompt, return_tensors="pt")
    completions = []
    for _ in range(n):
        with torch.no_grad():
            ids = mdl.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append(text)
    return completions


def distinct_n(texts, n=2):
    all_ngrams = []
    for t in texts:
        words = t.lower().split()
        all_ngrams.extend(zip(*[words[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def pairwise_semantic_sim(texts, model_name="gpt2"):
    hiddens = [extract_hidden(model_name, t) for t in texts]
    sims = []
    for i in range(len(hiddens)):
        for j in range(i + 1, len(hiddens)):
            sim = F.cosine_similarity(hiddens[i].unsqueeze(0), hiddens[j].unsqueeze(0)).item()
            sims.append(sim)
    return sims


# =====================================================================
# EXPERIMENT A: Same state, many outputs
# =====================================================================
def experiment_a():
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Same hidden state -> diverse outputs?")
    print("=" * 70)

    results = {}
    for prompt in TASK_PROMPTS:
        print(f"\n  Prompt: '{prompt[:60]}...'")

        h = extract_hidden("gpt2", prompt)
        print(f"    Hidden state norm: {h.norm():.4f}")

        completions = generate_completions("gpt2", prompt, n=50, temperature=0.9)

        d1 = distinct_n(completions, 1)
        d2 = distinct_n(completions, 2)
        d3 = distinct_n(completions, 3)
        print(f"    Distinct-1: {d1:.4f}, Distinct-2: {d2:.4f}, Distinct-3: {d3:.4f}")

        lengths = [len(c.split()) for c in completions]
        print(f"    Avg length: {sum(lengths)/len(lengths):.1f} words")

        sem_sims = pairwise_semantic_sim(completions[:20])
        avg_sem_sim = sum(sem_sims) / len(sem_sims) if sem_sims else 0
        min_sem_sim = min(sem_sims) if sem_sims else 0
        print(f"    Semantic sim (first 20): avg={avg_sem_sim:.4f}, min={min_sem_sim:.4f}")

        unique_first_words = len(set(c.split()[0] if c.split() else "" for c in completions))
        print(f"    Unique first words: {unique_first_words}/50")

        sample = completions[:5]
        print(f"    Sample outputs:")
        for i, s in enumerate(sample):
            print(f"      [{i+1}] {s[:120]}...")

        results[prompt] = {
            "distinct_1": d1,
            "distinct_2": d2,
            "distinct_3": d3,
            "avg_semantic_sim": avg_sem_sim,
            "min_semantic_sim": min_sem_sim,
            "unique_first_words": unique_first_words,
            "avg_length": sum(lengths) / len(lengths),
            "sample_completions": completions[:10],
        }

    return results


# =====================================================================
# EXPERIMENT B: Within-model vs between-model diversity
# =====================================================================
def experiment_b():
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Within-model vs between-model text diversity")
    print("=" * 70)

    models = ["gpt2", "distilgpt2"]
    results = {}

    for prompt in TASK_PROMPTS:
        print(f"\n  Prompt: '{prompt[:60]}...'")

        h_gpt2 = extract_hidden("gpt2", prompt)
        h_distil = extract_hidden("distilgpt2", prompt)
        rep_sim = F.cosine_similarity(h_gpt2.unsqueeze(0), h_distil.unsqueeze(0)).item()
        print(f"    Representation similarity (raw, no codec): {rep_sim:.4f}")

        completions = {}
        for model_name in models:
            completions[model_name] = generate_completions(model_name, prompt, n=20, temperature=0.9)

        within_sims = {}
        for model_name in models:
            sims = pairwise_semantic_sim(completions[model_name][:10])
            within_sims[model_name] = sum(sims) / len(sims) if sims else 0
            print(f"    Within-{model_name} semantic sim: {within_sims[model_name]:.4f}")

        between_texts_a = completions["gpt2"][:10]
        between_texts_b = completions["distilgpt2"][:10]
        between_sims = []
        for ta in between_texts_a:
            ha = extract_hidden("gpt2", ta)
            for tb in between_texts_b:
                hb = extract_hidden("gpt2", tb)
                sim = F.cosine_similarity(ha.unsqueeze(0), hb.unsqueeze(0)).item()
                between_sims.append(sim)
        avg_between = sum(between_sims) / len(between_sims) if between_sims else 0
        print(f"    Between-model semantic sim: {avg_between:.4f}")

        avg_within = sum(within_sims.values()) / len(within_sims)
        ratio = avg_within / avg_between if avg_between > 0 else float("inf")
        print(f"    Ratio (within/between): {ratio:.4f}")
        if 0.8 < ratio < 1.2:
            print(f"    -> SIMILAR: diversity is from sampling, not model differences")
        elif ratio < 0.8:
            print(f"    -> WITHIN is LOWER: same model outputs are MORE diverse (unexpected)")
        else:
            print(f"    -> BETWEEN is LOWER: different models ARE more diverse")

        within_d2 = {m: distinct_n(completions[m], 2) for m in models}
        between_d2 = distinct_n(completions["gpt2"][:10] + completions["distilgpt2"][:10], 2)
        print(f"    Distinct-2 within gpt2: {within_d2['gpt2']:.4f}")
        print(f"    Distinct-2 within distilgpt2: {within_d2['distilgpt2']:.4f}")
        print(f"    Distinct-2 between (pooled): {between_d2:.4f}")

        results[prompt] = {
            "representation_similarity": rep_sim,
            "within_model_semantic_sim": within_sims,
            "between_model_semantic_sim": avg_between,
            "within_over_between_ratio": ratio,
            "within_distinct2": within_d2,
            "between_distinct2": between_d2,
        }

    return results


# =====================================================================
# EXPERIMENT C: Do role prompts change representations?
# =====================================================================
def experiment_c():
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Do role prompts change hidden states?")
    print("=" * 70)

    roles = {
        "neutral": "",
        "supportive": "Argue strongly in favor of this position. ",
        "critical": "Find every flaw, risk, and weakness in this idea. ",
        "carol": "You are a contrarian devil's advocate. Challenge the obvious answer and defend the opposite position. ",
    }

    results = {}
    for prompt in TASK_PROMPTS:
        print(f"\n  Prompt: '{prompt[:60]}...'")

        hiddens = {}
        for role_name, role_prefix in roles.items():
            full_prompt = role_prefix + prompt
            h = extract_hidden("gpt2", full_prompt)
            hiddens[role_name] = h

        print(f"    Pairwise cosine similarity of hidden states:")
        role_sims = {}
        role_names = list(roles.keys())
        for i in range(len(role_names)):
            for j in range(i + 1, len(role_names)):
                r1, r2 = role_names[i], role_names[j]
                sim = F.cosine_similarity(
                    hiddens[r1].unsqueeze(0), hiddens[r2].unsqueeze(0)
                ).item()
                key = f"{r1}_vs_{r2}"
                role_sims[key] = sim
                print(f"      {r1} vs {r2}: {sim:.6f}")

        avg_role_sim = sum(role_sims.values()) / len(role_sims)
        min_role_sim = min(role_sims.values())
        print(f"    Average: {avg_role_sim:.6f}, Min: {min_role_sim:.6f}")

        if min_role_sim > 0.98:
            print(f"    -> Role prompts barely change representations")
        elif min_role_sim > 0.95:
            print(f"    -> Role prompts create small representational differences")
        else:
            print(f"    -> Role prompts create SIGNIFICANT representational differences")

        print(f"\n    Sample generations per role (3 each):")
        for role_name, role_prefix in roles.items():
            full_prompt = role_prefix + prompt
            gens = generate_completions("gpt2", full_prompt, n=3, max_new_tokens=40, temperature=0.7)
            print(f"      [{role_name}]")
            for g in gens:
                print(f"        {g[:100]}...")

        cross_prompt_sims = []
        for other_prompt in TASK_PROMPTS:
            if other_prompt == prompt:
                continue
            h_other = extract_hidden("gpt2", other_prompt)
            sim = F.cosine_similarity(
                hiddens["neutral"].unsqueeze(0), h_other.unsqueeze(0)
            ).item()
            cross_prompt_sims.append(sim)
        avg_cross_prompt = sum(cross_prompt_sims) / len(cross_prompt_sims)
        print(f"\n    Cross-PROMPT similarity (different prompts, same role): {avg_cross_prompt:.6f}")
        print(f"    Cross-ROLE similarity (same prompt, different roles): {avg_role_sim:.6f}")

        if avg_role_sim > avg_cross_prompt:
            print(f"    -> Changing the ROLE changes representation LESS than changing the PROMPT")
        else:
            print(f"    -> Changing the ROLE changes representation MORE than changing the PROMPT")

        results[prompt] = {
            "role_pairwise_sims": role_sims,
            "avg_role_sim": avg_role_sim,
            "min_role_sim": min_role_sim,
            "avg_cross_prompt_sim": avg_cross_prompt,
            "role_changes_less_than_prompt": avg_role_sim > avg_cross_prompt,
        }

    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("ROUND 5: WHERE DOES LLM DIVERSITY LIVE?")
    print("Representation vs Generation")
    print("=" * 70)

    results_a = experiment_a()
    results_b = experiment_b()
    results_c = experiment_c()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nExperiment A — Same state, many outputs:")
    for prompt, r in results_a.items():
        print(f"  '{prompt[:50]}...': distinct-2={r['distinct_2']:.3f}, "
              f"unique_first_words={r['unique_first_words']}/50, "
              f"semantic_sim={r['avg_semantic_sim']:.3f}")

    print("\nExperiment B — Within vs between model diversity:")
    for prompt, r in results_b.items():
        print(f"  '{prompt[:50]}...': rep_sim={r['representation_similarity']:.3f}, "
              f"within/between={r['within_over_between_ratio']:.3f}")

    print("\nExperiment C — Role prompt effect on representations:")
    for prompt, r in results_c.items():
        tag = "ROLE < PROMPT" if r["role_changes_less_than_prompt"] else "ROLE > PROMPT"
        print(f"  '{prompt[:50]}...': avg_role_sim={r['avg_role_sim']:.4f}, "
              f"cross_prompt_sim={r['avg_cross_prompt_sim']:.4f} ({tag})")

    output = {
        "experiment_a": results_a,
        "experiment_b": results_b,
        "experiment_c": results_c,
    }
    with open(RESULTS_DIR / "representation_diversity.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to representation_diversity.json")


if __name__ == "__main__":
    main()
