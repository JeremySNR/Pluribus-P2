"""Round 6c: BAPC codec with fixed gradient flow + batched forward passes.

Fixes from 6b:
  1. Gradients flow through receiver to codec decoder (standard prefix tuning)
  2. Batched forward passes instead of per-sample loop (16x speedup)
  3. Separation loss is differentiable w.r.t. matched prefix
"""

from __future__ import annotations

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.isocal import IsotropyCalibrator, IsocalConfig
from quorum.bapc_codec import BAPCCodec
from quorum.bapc_training import collect_training_data, js_divergence

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACTION_LAYER = 7

_MODEL_CACHE: dict = {}

PROMPTS = [
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
    "The primary function of the liver is",
    "Beethoven composed his famous symphonies during",
    "The principle of superposition states that",
    "Renewable energy sources include solar and",
    "The United States Constitution was ratified in",
    "Artificial neural networks consist of layers",
    "The Sahara Desert is the largest hot",
    "Quantum entanglement occurs when particles become",
    "The invention of the telephone revolutionized",
    "Mitosis is the process by which cells",
    "The gross domestic product measures the",
    "Charles Darwin published On the Origin of",
    "The ozone layer protects Earth from",
    "Blockchain technology ensures data integrity through",
    "The fall of the Berlin Wall occurred in",
    "Supervised learning requires labeled training",
    "The Amazon basin contains the largest tropical",
    "General relativity describes gravity as curvature",
    "The Suez Canal connects the Mediterranean",
    "Recurrent neural networks are designed to process",
    "The human skeletal system contains approximately",
    "Supply chain management involves coordinating",
    "The Hubble constant measures the rate of",
    "Object-oriented programming organizes code into",
    "The Great Depression began with the stock",
    "Stem cell therapy has potential applications in",
    "The electromagnetic spectrum ranges from radio",
    "Feudalism was a social system that dominated",
    "Docker containers provide lightweight virtualization",
    "The discovery of penicillin by Alexander Fleming",
    "Dark matter makes up approximately twenty-seven",
    "The Doppler effect explains how the frequency",
    "Ancient Egyptian civilization developed along the",
    "Gradient descent is an optimization algorithm",
    "The greenhouse effect traps heat in Earth's",
    "The Renaissance spread from Italy to northern",
    "Kubernetes orchestrates containerized applications across",
    "The human immune system fights pathogens through",
    "Mercantilism was an economic theory that promoted",
    "Attention mechanisms in neural networks allow",
    "The Mariana Trench reaches depths of nearly",
    "Keynesian economics argues that government spending",
    "The discovery of DNA structure by Watson and",
    "Version control systems like Git track changes",
    "The Cambrian explosion was a period of rapid",
    "Elastic computing allows resources to scale",
    "Photosynthetic organisms convert carbon dioxide and",
    "The Protestant Reformation began when Martin Luther",
    "Bayesian inference updates probability estimates based",
    "The Earth's magnetic field protects the planet",
    "Microservices architecture decomposes applications into",
    "The Industrial Revolution transformed manufacturing through",
    "Compiler optimization techniques include loop unrolling",
    "The Cretaceous-Paleogene extinction event eliminated",
    "Distributed computing systems coordinate multiple",
    "The circulatory system delivers oxygen and nutrients",
    "Functional programming treats computation as evaluation",
    "The Scientific Revolution challenged traditional views",
    "Load balancing distributes network traffic across",
    "Cellular respiration converts glucose into adenosine",
    "The Age of Exploration led European nations",
    "Convolutional layers in neural networks detect",
    "The endocrine system regulates hormones throughout",
    "API design principles include consistency and",
    "The Meiji Restoration modernized Japan's political",
    "Transfer learning applies knowledge from one task",
    "Tectonic plates move due to convection currents",
    "Database normalization reduces data redundancy by",
    "The Enlightenment emphasized reason and individual",
    "Generative adversarial networks consist of a generator",
    "The water cycle involves evaporation condensation",
    "Event-driven architecture processes actions triggered by",
    "The Monroe Doctrine established American foreign policy",
    "Principal component analysis reduces dimensionality by",
    "The human digestive system breaks down food",
    "Continuous integration automatically builds and tests",
    "The Treaty of Westphalia established the concept",
    "Regularization techniques prevent overfitting by adding",
    "The respiratory system exchanges oxygen and carbon",
    "Message queues enable asynchronous communication between",
    "The Congress of Vienna aimed to restore",
    "Batch normalization stabilizes neural network training",
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
        print(f"  Loading {model_name} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        mdl.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _MODEL_CACHE[key] = (mdl, tok)
    return _MODEL_CACHE[key]


def orthogonality_loss(weights):
    loss = torch.tensor(0.0, device=weights[0].device)
    for W in weights:
        WWT = W @ W.T
        I = torch.eye(WWT.shape[0], device=W.device)
        loss = loss + (WWT - I).pow(2).mean()
    return loss


def precompute_prompt_embeddings(model, tokenizer, prompts, device):
    """Pre-compute and pad prompt embeddings for batched training."""
    embed_layer = model.get_input_embeddings()
    all_embeds = []
    all_lengths = []

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            emb = embed_layer(ids["input_ids"][0].to(device))
        all_embeds.append(emb)
        all_lengths.append(emb.shape[0])

    max_len = max(all_lengths)
    d_emb = all_embeds[0].shape[1]
    padded = torch.zeros(len(prompts), max_len, d_emb, device=device)
    mask = torch.zeros(len(prompts), max_len, device=device)

    for i, (emb, length) in enumerate(zip(all_embeds, all_lengths)):
        padded[i, :length] = emb
        mask[i, :length] = 1.0

    return padded, mask, all_lengths


def train_batched(codec, model, tokenizer, sender_hiddens, teacher_logits,
                  prompt_embeds_padded, prompt_mask, prompt_lengths, prompts,
                  num_steps=2000, batch_size=16, lr=3e-4, T=2.0,
                  lambda_sep=0.1, lambda_align=0.1, lambda_orth=0.01,
                  log_every=100):
    """Train BAPC codec with proper gradient flow and batched forward passes."""

    device = next(model.parameters()).device
    N = sender_hiddens.shape[0]
    S = codec.num_slots

    for p in model.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(codec.parameters(), lr=lr)
    history = []

    teacher_window = teacher_logits.shape[1]
    max_prompt_len = prompt_embeds_padded.shape[1]

    for step in range(num_steps):
        idx = torch.randint(0, N, (min(batch_size, N),))
        h_batch = sender_hiddens[idx]
        teacher_batch = teacher_logits[idx]
        B = h_batch.shape[0]

        # Encode + decode (WITH gradients — this is the codec we're training)
        z_batch = codec.encode(h_batch)            # [B, S, D]
        prefix_batch = codec.decode(z_batch)        # [B, S, d_emb]

        # Build batched input: [prefix | prompt | padding]
        prompt_embs = prompt_embeds_padded[idx]     # [B, max_prompt_len, d_emb]
        prompt_msk = prompt_mask[idx]               # [B, max_prompt_len]

        combined = torch.cat([prefix_batch, prompt_embs], dim=1)  # [B, S+max_prompt_len, d_emb]
        prefix_mask = torch.ones(B, S, device=device)
        attn_mask = torch.cat([prefix_mask, prompt_msk], dim=1)   # [B, S+max_prompt_len]

        # Forward WITH gradients (standard prefix tuning — frozen backbone, differentiable path)
        out = model(inputs_embeds=combined, attention_mask=attn_mask)

        # Extract logits at the last real token position per sample
        student_logits_list = []
        for b in range(B):
            real_len = S + prompt_lengths[idx[b]]
            window = min(teacher_window, real_len)
            student_logits_list.append(out.logits[b, real_len - window:real_len])
        student_logits_all = torch.stack(student_logits_list)  # [B, window, vocab]

        # L_distill: KL(teacher || student) — core behavioural distillation
        p_teacher = F.softmax(teacher_batch / T, dim=-1).detach()
        log_p_student = F.log_softmax(student_logits_all / T, dim=-1)
        l_distill = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

        # L_separation: JS(matched, mismatched) — force receiver to use the prefix
        perm = torch.randperm(B)
        z_shuffled = z_batch[perm].detach()
        prefix_shuffled = codec.decode(z_shuffled)

        combined_mm = torch.cat([prefix_shuffled, prompt_embs], dim=1)
        with torch.no_grad():
            out_mm = model(inputs_embeds=combined_mm, attention_mask=attn_mask)

        mismatch_logits_list = []
        for b in range(B):
            real_len = S + prompt_lengths[idx[b]]
            window = min(teacher_window, real_len)
            mismatch_logits_list.append(out_mm.logits[b, real_len - window:real_len])
        mismatch_logits_all = torch.stack(mismatch_logits_list)

        p_matched = F.softmax(student_logits_all / T, dim=-1)
        p_mismatched = F.softmax(mismatch_logits_all / T, dim=-1).detach()
        l_sep = -js_divergence(p_matched, p_mismatched)

        # L_align: cosine alignment between student and teacher logits
        l_align = 1.0 - F.cosine_similarity(
            student_logits_all.reshape(-1, student_logits_all.shape[-1]),
            teacher_batch.reshape(-1, teacher_batch.shape[-1]),
            dim=-1
        ).mean()

        # L_orth: keep encoder weights orthogonal
        l_orth = orthogonality_loss(codec.get_encoder_weights())

        loss = l_distill + lambda_sep * l_sep + lambda_align * l_align + lambda_orth * l_orth

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0:
            entry = {
                "step": step,
                "loss": loss.item(),
                "distill": l_distill.item(),
                "separation": l_sep.item(),
                "align": l_align.item(),
                "orth": l_orth.item(),
            }
            history.append(entry)
            print(f"    Step {step:4d}/{num_steps}: "
                  f"distill={l_distill.item():.2f} sep={l_sep.item():.4f} "
                  f"align={l_align.item():.4f} orth={l_orth.item():.4f}",
                  flush=True)

    return history


def main():
    t_start = time.time()
    print("=" * 70)
    print(f"ROUND 6c: FIXED BAPC TRAINING (gradient flow + batched, {DEVICE})")
    print("=" * 70, flush=True)

    model, tok = _load_lm("gpt2")

    # IsoCal calibration
    print(f"\n  Calibrating IsoCal from layer {EXTRACTION_LAYER} ...", flush=True)
    cal_states = []
    with torch.no_grad():
        for i, p in enumerate(PROMPTS):
            inp = tok(p, return_tensors="pt", truncation=True, max_length=64)
            out = model(**inp, output_hidden_states=True)
            cal_states.append(out.hidden_states[EXTRACTION_LAYER][0, -1, :])
    cal_stack = torch.stack(cal_states)
    isocal = IsotropyCalibrator(IsocalConfig(k=16))
    isocal.calibrate(cal_stack)
    print(f"    Calibrated on {len(PROMPTS)} states", flush=True)

    # Collect training data
    print(f"\n  Collecting training data ...", flush=True)
    data = collect_training_data(
        model, tok, model, tok, PROMPTS,
        extraction_layer=EXTRACTION_LAYER,
        max_gen_tokens=40, teacher_window=8,
    )
    N = data["sender_hiddens"].shape[0]
    print(f"    {N} samples, teacher logits: {data['receiver_teacher_logits'].shape}", flush=True)

    # Move to GPU
    model = model.to(DEVICE)
    sender_hiddens = data["sender_hiddens"].to(DEVICE)
    teacher_logits = data["receiver_teacher_logits"].to(DEVICE)

    # Create codec on GPU
    d_agent = sender_hiddens.shape[1]
    codec = BAPCCodec(agent_dim=d_agent, buffer_dim=512, embed_dim=d_agent, num_slots=6)
    codec.isocal = isocal
    codec = codec.to(DEVICE)

    # Pre-compute padded prompt embeddings
    print(f"\n  Pre-computing prompt embeddings ...", flush=True)
    prompt_embeds, prompt_mask, prompt_lengths = precompute_prompt_embeddings(
        model, tok, PROMPTS[:N], DEVICE
    )
    print(f"    Padded shape: {prompt_embeds.shape}", flush=True)

    # Train with fixed gradient flow
    print(f"\n  Training (2000 steps, batched, with gradient flow) ...", flush=True)
    history = train_batched(
        codec, model, tok, sender_hiddens, teacher_logits,
        prompt_embeds, prompt_mask, prompt_lengths, PROMPTS[:N],
        num_steps=2000, batch_size=16, lr=3e-4, T=2.0,
        lambda_sep=0.1, lambda_align=0.1, lambda_orth=0.01,
        log_every=100,
    )

    # ── Diagnostics ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DIAGNOSTICS")
    print(f"{'='*70}", flush=True)

    # KL trajectory
    distill_vals = [h["distill"] for h in history]
    first_kl = distill_vals[0]
    last_kl = distill_vals[-1]
    min_kl = min(distill_vals)
    reduction = (first_kl - last_kl) / first_kl * 100 if first_kl > 0 else 0
    print(f"\n  KL: {first_kl:.2f} -> {last_kl:.2f} (min: {min_kl:.2f}, reduction: {reduction:.1f}%)",
          flush=True)

    # JS matched vs mismatched
    print(f"\n  JS divergence (100 samples) ...", flush=True)
    codec_cpu = codec.cpu()
    model_cpu = model.cpu()
    embed_layer = model_cpu.get_input_embeddings()
    sh_cpu = sender_hiddens.cpu()

    n_test = min(100, N)
    js_scores = []
    for i in range(n_test):
        z = codec_cpu.encode(sh_cpu[i])
        prefix_match = codec_cpu.decode(z)
        j = (i + 7) % n_test
        prefix_mismatch = codec_cpu.decode(codec_cpu.encode(sh_cpu[j]))

        prompt = PROMPTS[i]
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            pe = embed_layer(ids["input_ids"][0])

        cm = torch.cat([prefix_match, pe], dim=0).unsqueeze(0)
        cmm = torch.cat([prefix_mismatch, pe], dim=0).unsqueeze(0)

        with torch.no_grad():
            lm = model_cpu(inputs_embeds=cm).logits[0, -8:]
            lmm = model_cpu(inputs_embeds=cmm).logits[0, -8:]

        pm = F.softmax(lm / 2.0, dim=-1)
        pmm = F.softmax(lmm / 2.0, dim=-1)
        js_scores.append(js_divergence(pm, pmm).item())

    mean_js = sum(js_scores) / len(js_scores)
    print(f"    Mean JS: {mean_js:.6f} (was 0.045 without gradient flow)", flush=True)

    # Buffer anisotropy
    buf_vecs = [codec_cpu.encode(sh_cpu[i])[0] for i in range(n_test)]
    buf_sims = []
    for i in range(len(buf_vecs)):
        for j in range(i + 1, min(i + 20, len(buf_vecs))):
            buf_sims.append(F.cosine_similarity(
                buf_vecs[i].unsqueeze(0), buf_vecs[j].unsqueeze(0)).item())
    mean_buf = sum(buf_sims) / len(buf_sims)
    print(f"    Buffer cosine: {mean_buf:.4f}", flush=True)

    # End-to-end on task prompts
    print(f"\n  End-to-end comparison:", flush=True)
    e2e_results = {}
    for prompt in TASK_PROMPTS:
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            sender_out = model_cpu(**inp, output_hidden_states=True)
            h_sender = sender_out.hidden_states[EXTRACTION_LAYER][0, -1, :]
            logits_none = model_cpu(**inp).logits[0, -1]

            gen_ids = model_cpu.generate(**inp, max_new_tokens=40, do_sample=True,
                                         temperature=0.8, top_p=0.95,
                                         pad_token_id=tok.eos_token_id)
        sender_text = tok.decode(gen_ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        recv_inp = tok(prompt + " " + sender_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits_text = model_cpu(**recv_inp).logits[0, -1]

        z = codec_cpu.encode(h_sender)
        prefix = codec_cpu.decode(z)
        with torch.no_grad():
            pe = embed_layer(inp["input_ids"][0])
        combined = torch.cat([prefix, pe], dim=0).unsqueeze(0)
        with torch.no_grad():
            logits_latent = model_cpu(inputs_embeds=combined).logits[0, -1]

        p_none = F.softmax(logits_none, dim=-1)
        p_text = F.softmax(logits_text, dim=-1)
        p_latent = F.softmax(logits_latent, dim=-1)

        kl_text = F.kl_div(p_none.log(), p_text, reduction="sum").item()
        kl_latent = F.kl_div(p_none.log(), p_latent, reduction="sum").item()
        ratio = kl_latent / kl_text * 100 if kl_text > 0 else 0

        top_none = tok.decode([p_none.argmax()])
        top_text = tok.decode([p_text.argmax()])
        top_latent = tok.decode([p_latent.argmax()])

        print(f"\n    '{prompt[:50]}...'", flush=True)
        print(f"      KL(text vs none):   {kl_text:.2f}", flush=True)
        print(f"      KL(latent vs none): {kl_latent:.2f} ({ratio:.0f}% of text)", flush=True)
        print(f"      Tokens: none='{top_none.strip()}' text='{top_text.strip()}' latent='{top_latent.strip()}'",
              flush=True)

        e2e_results[prompt] = {
            "kl_text_vs_none": kl_text,
            "kl_latent_vs_none": kl_latent,
            "latent_pct_of_text": ratio,
            "top_none": top_none, "top_text": top_text, "top_latent": top_latent,
        }

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed / 60:.1f} minutes", flush=True)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}", flush=True)
    print(f"  KL reduction: {first_kl:.1f} -> {last_kl:.1f} ({reduction:.1f}%)", flush=True)
    print(f"  JS (match vs mismatch): {mean_js:.6f}", flush=True)
    print(f"  Buffer cosine: {mean_buf:.4f}", flush=True)
    latent_pcts = [r["latent_pct_of_text"] for r in e2e_results.values()]
    print(f"  Latent as % of text effect: {min(latent_pcts):.0f}-{max(latent_pcts):.0f}%", flush=True)
    print(f"  Time: {elapsed/60:.1f} minutes", flush=True)

    output = {
        "config": {"prompts": N, "steps": 2000, "device": DEVICE, "gradient_flow": True, "batched": True},
        "training_history": history,
        "diagnostics": {
            "kl_start": first_kl, "kl_end": last_kl, "kl_min": min_kl,
            "kl_reduction_pct": reduction,
            "js_matched_mismatched": mean_js,
            "buffer_cosine": mean_buf,
        },
        "end_to_end": e2e_results,
    }
    with open(RESULTS_DIR / "bapc_fixed_training.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to bapc_fixed_training.json", flush=True)


if __name__ == "__main__":
    main()
