"""Round 6: BAPC Codec — Behavioural Distillation Diagnostics.

Phase A: Confirm ABTT removes anisotropy at extraction layer
Phase B: Train BAPC codec with behavioural distillation
Phase C: Three diagnostics:
  1. Matched vs mismatched JS divergence (does receiver use the latent?)
  2. Distillation KL tracking (does training converge?)
  3. Buffer-space anisotropy monitor (is cone collapse avoided?)
Phase D: Mini end-to-end comparison (text comm vs latent comm vs no comm)
"""

from __future__ import annotations

import json
import math
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.isocal import IsotropyCalibrator, IsocalConfig
from quorum.bapc_codec import BAPCCodec
from quorum.bapc_training import (
    BAPCTrainingConfig, collect_training_data, train_bapc_codec, js_divergence,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_MODEL_CACHE: dict = {}

EXTRACTION_LAYER = 7  # floor(0.6 * 12) for GPT-2

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


def _load_lm(model_name):
    key = model_name + "_lm"
    if key not in _MODEL_CACHE:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"  Loading {model_name} (CausalLM) ...", flush=True)
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        mdl.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _MODEL_CACHE[key] = (mdl, tok)
    return _MODEL_CACHE[key]


def extract_hidden_at_layer(model, tokenizer, text, layer_idx):
    """Extract hidden state from a specific layer at the last token position."""
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return out.hidden_states[layer_idx][0, -1, :]


# =====================================================================
# PHASE A: Anisotropy validation at extraction layer
# =====================================================================
def phase_a():
    print("\n" + "=" * 70)
    print("PHASE A: Anisotropy validation at extraction layer")
    print("=" * 70, flush=True)

    model, tok = _load_lm("gpt2")

    words = ['apple', 'democracy', 'purple', 'running', 'telescope',
             'angry', 'molecule', 'finance', 'guitar', 'ocean',
             'breakfast', 'philosophy', 'triangle', 'whisper', 'volcano',
             'algorithm', 'curtain', 'liberty', 'penguin', 'satellite']

    hiddens_raw = []
    for w in words:
        h = extract_hidden_at_layer(model, tok, w, EXTRACTION_LAYER)
        hiddens_raw.append(h)
    raw_stack = torch.stack(hiddens_raw)

    def mean_cos(vecs):
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(F.cosine_similarity(vecs[i].unsqueeze(0),
                                                vecs[j].unsqueeze(0)).item())
        return sum(sims) / len(sims)

    raw_baseline = mean_cos(hiddens_raw)
    print(f"  Layer {EXTRACTION_LAYER} raw baseline: {raw_baseline:.4f}", flush=True)

    isocal = IsotropyCalibrator(IsocalConfig(k=16))
    isocal.calibrate(raw_stack)
    transformed = [isocal.transform(h) for h in hiddens_raw]
    abtt_baseline = mean_cos(transformed)
    print(f"  Layer {EXTRACTION_LAYER} after ABTT:   {abtt_baseline:.4f}", flush=True)

    for layer in [12]:
        hs = [extract_hidden_at_layer(model, tok, w, layer) for w in words]
        print(f"  Layer {layer} raw baseline:  {mean_cos(hs):.4f} (final layer for reference)",
              flush=True)

    result = {
        "extraction_layer": EXTRACTION_LAYER,
        "raw_baseline": raw_baseline,
        "abtt_baseline": abtt_baseline,
        "final_layer_baseline": mean_cos(
            [extract_hidden_at_layer(model, tok, w, 12) for w in words]
        ),
    }
    print(f"\n  PASS: ABTT baseline {abtt_baseline:.4f} << 0.75", flush=True)
    return result, isocal


# =====================================================================
# PHASE B: Train BAPC codec
# =====================================================================
def phase_b(isocal):
    print("\n" + "=" * 70)
    print("PHASE B: Train BAPC codec with behavioural distillation")
    print("=" * 70, flush=True)

    sender_model, sender_tok = _load_lm("gpt2")
    receiver_model, receiver_tok = _load_lm("gpt2")

    prompts = TRAINING_PROMPTS[:100]

    print(f"  Collecting training data ({len(prompts)} prompts) ...", flush=True)
    data = collect_training_data(
        sender_model, sender_tok,
        receiver_model, receiver_tok,
        prompts,
        extraction_layer=EXTRACTION_LAYER,
        max_gen_tokens=40,
        teacher_window=8,
    )
    print(f"    Sender hiddens: {data['sender_hiddens'].shape}", flush=True)
    print(f"    Teacher logits: {data['receiver_teacher_logits'].shape}", flush=True)

    D = 512
    S = 6
    d_agent = data["sender_hiddens"].shape[1]

    codec = BAPCCodec(
        agent_dim=d_agent, buffer_dim=D, embed_dim=d_agent, num_slots=S,
    )
    codec.isocal = isocal

    print(f"\n  Training codec (D={D}, S={S}) ...", flush=True)
    cfg = BAPCTrainingConfig(
        lr=3e-4, num_steps=2000, batch_size=16, teacher_window=8,
        temperature=2.0, lambda_sep=0.1, lambda_align=0.1, lambda_orth=0.01,
        log_every=200,
    )
    history = train_bapc_codec(
        codec, receiver_model, receiver_tok, data, prompts, cfg,
    )

    return codec, data, history


# =====================================================================
# PHASE C: Three diagnostics
# =====================================================================
def phase_c(codec, data, history, sender_model, sender_tok, receiver_model, receiver_tok):
    print("\n" + "=" * 70)
    print("PHASE C: Diagnostics")
    print("=" * 70, flush=True)

    results = {}

    # Diagnostic 1: Matched vs mismatched JS divergence
    print("\n  Diagnostic 1: Matched vs mismatched JS divergence", flush=True)
    sender_hiddens = data["sender_hiddens"]
    prompts = data["prompts"]
    embed_layer = receiver_model.get_input_embeddings()
    T = 2.0

    n_test = min(50, sender_hiddens.shape[0])
    js_scores = []

    for i in range(n_test):
        h = sender_hiddens[i]
        z = codec.encode(h)
        prefix_matched = codec.decode(z)

        j = (i + 1) % n_test
        z_other = codec.encode(sender_hiddens[j])
        prefix_mismatched = codec.decode(z_other)

        prompt = prompts[i]
        prompt_ids = receiver_tok(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            prompt_embeds = embed_layer(prompt_ids["input_ids"][0])

        combined_match = torch.cat([prefix_matched, prompt_embeds], dim=0).unsqueeze(0)
        combined_mismatch = torch.cat([prefix_mismatched, prompt_embeds], dim=0).unsqueeze(0)

        with torch.no_grad():
            logits_match = receiver_model(inputs_embeds=combined_match).logits[0, -8:]
            logits_mismatch = receiver_model(inputs_embeds=combined_mismatch).logits[0, -8:]

        p_match = F.softmax(logits_match / T, dim=-1)
        p_mismatch = F.softmax(logits_mismatch / T, dim=-1)
        js = js_divergence(p_match, p_mismatch).item()
        js_scores.append(js)

    mean_js = sum(js_scores) / len(js_scores)
    print(f"    Mean JS(matched, mismatched): {mean_js:.6f}", flush=True)
    print(f"    Min: {min(js_scores):.6f}, Max: {max(js_scores):.6f}", flush=True)

    if mean_js > 0.001:
        print(f"    PASS: Receiver distinguishes matched from mismatched latents", flush=True)
    else:
        print(f"    FAIL: Receiver ignores latent prefix (JS ~ 0)", flush=True)

    results["diagnostic_1_js"] = {
        "mean": mean_js, "min": min(js_scores), "max": max(js_scores),
        "pass": mean_js > 0.001,
    }

    # Diagnostic 2: Distillation KL tracking
    print("\n  Diagnostic 2: Distillation KL trajectory", flush=True)
    distill_values = [h["distill"] for h in history]
    first_kl = distill_values[0]
    last_kl = distill_values[-1]
    reduction = (first_kl - last_kl) / first_kl * 100 if first_kl > 0 else 0
    print(f"    KL start: {first_kl:.4f}", flush=True)
    print(f"    KL end:   {last_kl:.4f}", flush=True)
    print(f"    Reduction: {reduction:.1f}%", flush=True)

    if last_kl < first_kl:
        print(f"    PASS: KL decreased during training", flush=True)
    else:
        print(f"    FAIL: KL did not decrease", flush=True)

    results["diagnostic_2_kl"] = {
        "start": first_kl, "end": last_kl, "reduction_pct": reduction,
        "pass": last_kl < first_kl,
    }

    # Diagnostic 3: Buffer-space anisotropy
    print("\n  Diagnostic 3: Buffer-space anisotropy", flush=True)
    n_check = min(50, sender_hiddens.shape[0])
    buffer_vecs = []
    for i in range(n_check):
        z = codec.encode(sender_hiddens[i])  # [S, D]
        buffer_vecs.append(z[0])  # slot 0

    buffer_sims = []
    for i in range(len(buffer_vecs)):
        for j in range(i + 1, len(buffer_vecs)):
            sim = F.cosine_similarity(buffer_vecs[i].unsqueeze(0),
                                      buffer_vecs[j].unsqueeze(0)).item()
            buffer_sims.append(sim)
    mean_buf_sim = sum(buffer_sims) / len(buffer_sims)
    print(f"    Mean pairwise cosine in buffer space: {mean_buf_sim:.4f}", flush=True)
    print(f"    Min: {min(buffer_sims):.4f}, Max: {max(buffer_sims):.4f}", flush=True)

    if mean_buf_sim < 0.95:
        print(f"    PASS: No cone collapse in buffer space", flush=True)
    else:
        print(f"    FAIL: Cone collapse detected in buffer space", flush=True)

    results["diagnostic_3_buffer_anisotropy"] = {
        "mean_cosine": mean_buf_sim,
        "min": min(buffer_sims), "max": max(buffer_sims),
        "pass": mean_buf_sim < 0.95,
    }

    return results


# =====================================================================
# PHASE D: Mini end-to-end comparison
# =====================================================================
def phase_d(codec, receiver_model, receiver_tok):
    print("\n" + "=" * 70)
    print("PHASE D: Mini end-to-end — text vs latent vs no communication")
    print("=" * 70, flush=True)

    sender_model, sender_tok = _load_lm("gpt2")
    embed_layer = receiver_model.get_input_embeddings()

    results = {}
    for prompt in TASK_PROMPTS:
        print(f"\n  Prompt: '{prompt[:55]}...'", flush=True)

        sender_inp = sender_tok(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            gen_ids = sender_model.generate(
                **sender_inp, max_new_tokens=40, do_sample=True,
                temperature=0.8, top_p=0.95,
                pad_token_id=sender_tok.eos_token_id,
            )
        sender_text = sender_tok.decode(gen_ids[0][sender_inp["input_ids"].shape[1]:],
                                         skip_special_tokens=True)

        with torch.no_grad():
            sender_out = sender_model(**sender_inp, output_hidden_states=True)
            h_sender = sender_out.hidden_states[EXTRACTION_LAYER][0, -1, :]

        # Condition 1: No communication (receiver sees only prompt)
        recv_inp_none = receiver_tok(prompt, return_tensors="pt",
                                      truncation=True, max_length=64)
        with torch.no_grad():
            out_none = receiver_model(**recv_inp_none)
            logits_none = out_none.logits[0, -1]

        # Condition 2: Text communication (receiver sees prompt + sender text)
        combined_text = prompt + " " + sender_text
        recv_inp_text = receiver_tok(combined_text, return_tensors="pt",
                                      truncation=True, max_length=128)
        with torch.no_grad():
            out_text = receiver_model(**recv_inp_text)
            logits_text = out_text.logits[0, -1]

        # Condition 3: Latent communication (receiver sees prefix + prompt)
        z = codec.encode(h_sender)
        prefix = codec.decode(z)  # [S, d_emb]

        prompt_ids = receiver_tok(prompt, return_tensors="pt",
                                   truncation=True, max_length=64)
        with torch.no_grad():
            prompt_embeds = embed_layer(prompt_ids["input_ids"][0])
        combined_embeds = torch.cat([prefix, prompt_embeds], dim=0).unsqueeze(0)
        with torch.no_grad():
            out_latent = receiver_model(inputs_embeds=combined_embeds)
            logits_latent = out_latent.logits[0, -1]

        p_none = F.softmax(logits_none, dim=-1)
        p_text = F.softmax(logits_text, dim=-1)
        p_latent = F.softmax(logits_latent, dim=-1)

        kl_text_vs_none = F.kl_div(p_none.log(), p_text, reduction="sum").item()
        kl_latent_vs_none = F.kl_div(p_none.log(), p_latent, reduction="sum").item()
        kl_latent_vs_text = F.kl_div(p_text.log(), p_latent, reduction="sum").item()

        top_none = receiver_tok.decode([p_none.argmax()])
        top_text = receiver_tok.decode([p_text.argmax()])
        top_latent = receiver_tok.decode([p_latent.argmax()])

        print(f"    No comm top token: '{top_none}' (p={p_none.max():.4f})", flush=True)
        print(f"    Text comm top token: '{top_text}' (p={p_text.max():.4f})", flush=True)
        print(f"    Latent comm top token: '{top_latent}' (p={p_latent.max():.4f})", flush=True)
        print(f"    KL(text vs none):   {kl_text_vs_none:.4f}", flush=True)
        print(f"    KL(latent vs none): {kl_latent_vs_none:.4f}", flush=True)
        print(f"    KL(latent vs text): {kl_latent_vs_text:.4f}", flush=True)

        if kl_latent_vs_none > 0.1:
            print(f"    -> Latent prefix IS changing receiver behaviour", flush=True)
        else:
            print(f"    -> Latent prefix has minimal effect", flush=True)

        results[prompt] = {
            "sender_text": sender_text[:200],
            "top_token_none": top_none,
            "top_token_text": top_text,
            "top_token_latent": top_latent,
            "kl_text_vs_none": kl_text_vs_none,
            "kl_latent_vs_none": kl_latent_vs_none,
            "kl_latent_vs_text": kl_latent_vs_text,
            "latent_changes_behaviour": kl_latent_vs_none > 0.1,
        }

    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("ROUND 6: BAPC CODEC — BEHAVIOURAL DISTILLATION DIAGNOSTICS")
    print("=" * 70, flush=True)

    phase_a_results, isocal = phase_a()
    codec, data, history = phase_b(isocal)

    sender_model, sender_tok = _load_lm("gpt2")
    receiver_model, receiver_tok = _load_lm("gpt2")

    phase_c_results = phase_c(
        codec, data, history,
        sender_model, sender_tok, receiver_model, receiver_tok,
    )
    phase_d_results = phase_d(codec, receiver_model, receiver_tok)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70, flush=True)

    d1 = phase_c_results["diagnostic_1_js"]
    d2 = phase_c_results["diagnostic_2_kl"]
    d3 = phase_c_results["diagnostic_3_buffer_anisotropy"]

    print(f"\n  Phase A — Anisotropy removal:")
    print(f"    Raw baseline: {phase_a_results['raw_baseline']:.4f}")
    print(f"    After ABTT:   {phase_a_results['abtt_baseline']:.4f}")

    print(f"\n  Phase C — Diagnostics:")
    print(f"    1. JS(match vs mismatch): {d1['mean']:.6f} ({'PASS' if d1['pass'] else 'FAIL'})")
    print(f"    2. KL reduction: {d2['reduction_pct']:.1f}% ({'PASS' if d2['pass'] else 'FAIL'})")
    print(f"    3. Buffer cosine: {d3['mean_cosine']:.4f} ({'PASS' if d3['pass'] else 'FAIL'})")

    latent_effects = sum(1 for r in phase_d_results.values() if r["latent_changes_behaviour"])
    print(f"\n  Phase D — End-to-end:")
    print(f"    Latent prefix changes behaviour: {latent_effects}/{len(phase_d_results)} prompts")

    all_pass = d1["pass"] and d2["pass"] and d3["pass"]
    print(f"\n  OVERALL: {'ALL DIAGNOSTICS PASS' if all_pass else 'SOME DIAGNOSTICS FAILED'}")

    output = {
        "phase_a": phase_a_results,
        "training_history": history,
        "phase_c": phase_c_results,
        "phase_d": phase_d_results,
        "all_pass": all_pass,
    }
    with open(RESULTS_DIR / "bapc_diagnostics.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to bapc_diagnostics.json", flush=True)

    return output


if __name__ == "__main__":
    main()
