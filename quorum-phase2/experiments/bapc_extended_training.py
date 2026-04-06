"""Round 6d: Extended BAPC training — 10,000 steps with checkpoint saving.

Find the ceiling: does the KL keep dropping or does it plateau?
Saves codec checkpoints every 2000 steps for reuse.
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
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACTION_LAYER = 7
NUM_STEPS = 10000

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


def save_checkpoint(codec, isocal, optimizer, step, path):
    torch.save({
        "codec_state": codec.state_dict(),
        "isocal_state": isocal.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
    }, path)


def main():
    t_start = time.time()
    print("=" * 70)
    print(f"ROUND 6d: EXTENDED BAPC TRAINING ({NUM_STEPS} steps, {DEVICE})")
    print(f"Checkpoint saving every 2000 steps")
    print("=" * 70, flush=True)

    model, tok = _load_lm("gpt2")

    # IsoCal
    print(f"\n  Calibrating IsoCal ...", flush=True)
    cal_states = []
    with torch.no_grad():
        for p in PROMPTS:
            inp = tok(p, return_tensors="pt", truncation=True, max_length=64)
            out = model(**inp, output_hidden_states=True)
            cal_states.append(out.hidden_states[EXTRACTION_LAYER][0, -1, :])
    cal_stack = torch.stack(cal_states)
    isocal = IsotropyCalibrator(IsocalConfig(k=16))
    isocal.calibrate(cal_stack)
    print(f"    Calibrated on {len(PROMPTS)} states", flush=True)

    # Training data
    print(f"\n  Collecting training data ...", flush=True)
    data = collect_training_data(
        model, tok, model, tok, PROMPTS,
        extraction_layer=EXTRACTION_LAYER,
        max_gen_tokens=40, teacher_window=8,
    )
    N = data["sender_hiddens"].shape[0]
    print(f"    {N} samples", flush=True)

    # Move to GPU
    model = model.to(DEVICE)
    sender_hiddens = data["sender_hiddens"].to(DEVICE)
    teacher_logits = data["receiver_teacher_logits"].to(DEVICE)

    # Codec
    d_agent = sender_hiddens.shape[1]
    codec = BAPCCodec(agent_dim=d_agent, buffer_dim=512, embed_dim=d_agent, num_slots=6)
    codec.isocal = isocal
    codec = codec.to(DEVICE)

    # Pre-compute prompt embeddings
    prompt_embeds, prompt_mask, prompt_lengths = precompute_prompt_embeddings(
        model, tok, PROMPTS[:N], DEVICE
    )

    # Freeze receiver
    for p in model.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(codec.parameters(), lr=3e-4)
    S = 6
    T = 2.0
    B_SIZE = 16
    teacher_window = teacher_logits.shape[1]
    history = []

    print(f"\n  Training {NUM_STEPS} steps ...", flush=True)
    for step in range(NUM_STEPS):
        idx = torch.randint(0, N, (min(B_SIZE, N),))
        h_batch = sender_hiddens[idx]
        teacher_batch = teacher_logits[idx]
        B = h_batch.shape[0]

        z_batch = codec.encode(h_batch)
        prefix_batch = codec.decode(z_batch)

        prompt_embs = prompt_embeds[idx]
        prompt_msk = prompt_mask[idx]

        combined = torch.cat([prefix_batch, prompt_embs], dim=1)
        pfx_mask = torch.ones(B, S, device=DEVICE)
        attn_mask = torch.cat([pfx_mask, prompt_msk], dim=1)

        out = model(inputs_embeds=combined, attention_mask=attn_mask)

        student_logits_list = []
        for b in range(B):
            real_len = S + prompt_lengths[idx[b]]
            window = min(teacher_window, real_len)
            student_logits_list.append(out.logits[b, real_len - window:real_len])
        student_logits_all = torch.stack(student_logits_list)

        p_teacher = F.softmax(teacher_batch / T, dim=-1).detach()
        log_p_student = F.log_softmax(student_logits_all / T, dim=-1)
        l_distill = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

        perm = torch.randperm(B)
        z_shuffled = z_batch[perm].detach()
        prefix_shuffled = codec.decode(z_shuffled)
        combined_mm = torch.cat([prefix_shuffled, prompt_embs], dim=1)
        with torch.no_grad():
            out_mm = model(inputs_embeds=combined_mm, attention_mask=attn_mask)
        mm_list = []
        for b in range(B):
            real_len = S + prompt_lengths[idx[b]]
            window = min(teacher_window, real_len)
            mm_list.append(out_mm.logits[b, real_len - window:real_len])
        mm_all = torch.stack(mm_list)

        p_matched = F.softmax(student_logits_all / T, dim=-1)
        p_mismatched = F.softmax(mm_all / T, dim=-1).detach()
        l_sep = -js_divergence(p_matched, p_mismatched)

        l_align = 1.0 - F.cosine_similarity(
            student_logits_all.reshape(-1, student_logits_all.shape[-1]),
            teacher_batch.reshape(-1, teacher_batch.shape[-1]),
            dim=-1).mean()

        l_orth = orthogonality_loss(codec.get_encoder_weights())

        loss = l_distill + 0.1 * l_sep + 0.1 * l_align + 0.01 * l_orth

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0:
            entry = {"step": step, "distill": l_distill.item(), "sep": l_sep.item(),
                     "align": l_align.item(), "orth": l_orth.item()}
            history.append(entry)
            print(f"    Step {step:5d}/{NUM_STEPS}: distill={l_distill.item():.2f} "
                  f"sep={l_sep.item():.3f} align={l_align.item():.4f}", flush=True)

        if (step + 1) % 2000 == 0:
            ckpt_path = CHECKPOINT_DIR / f"bapc_codec_step{step+1}.pt"
            save_checkpoint(codec, isocal, optimizer, step + 1, ckpt_path)
            print(f"    >>> Checkpoint saved: {ckpt_path.name}", flush=True)

    # Final checkpoint
    final_path = CHECKPOINT_DIR / "bapc_codec_final.pt"
    save_checkpoint(codec, isocal, optimizer, NUM_STEPS, final_path)
    print(f"\n  Final checkpoint saved: {final_path}", flush=True)

    # ── Diagnostics ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DIAGNOSTICS")
    print(f"{'='*70}", flush=True)

    distill_vals = [h["distill"] for h in history]
    print(f"\n  KL trajectory:", flush=True)
    for h in history:
        print(f"    Step {h['step']:5d}: {h['distill']:.2f}", flush=True)

    first_kl = distill_vals[0]
    last_kl = distill_vals[-1]
    min_kl = min(distill_vals)
    reduction = (first_kl - last_kl) / first_kl * 100
    print(f"\n  KL: {first_kl:.2f} -> {last_kl:.2f} (min: {min_kl:.2f}, reduction: {reduction:.1f}%)", flush=True)

    # Plateau detection
    last_5 = distill_vals[-5:]
    plateau_range = max(last_5) - min(last_5)
    print(f"  Last 5 checkpoints range: {plateau_range:.2f} "
          f"({'PLATEAUED' if plateau_range < 1.0 else 'STILL DESCENDING'})", flush=True)

    # JS + end-to-end (on CPU)
    codec_cpu = codec.cpu()
    model_cpu = model.cpu()
    embed_layer = model_cpu.get_input_embeddings()
    sh_cpu = sender_hiddens.cpu()

    n_test = min(100, N)
    js_scores = []
    for i in range(n_test):
        z = codec_cpu.encode(sh_cpu[i])
        pfx_m = codec_cpu.decode(z)
        j = (i + 7) % n_test
        pfx_mm = codec_cpu.decode(codec_cpu.encode(sh_cpu[j]))

        ids = tok(PROMPTS[i], return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            pe = embed_layer(ids["input_ids"][0])
        cm = torch.cat([pfx_m, pe], dim=0).unsqueeze(0)
        cmm = torch.cat([pfx_mm, pe], dim=0).unsqueeze(0)
        with torch.no_grad():
            lm = model_cpu(inputs_embeds=cm).logits[0, -8:]
            lmm = model_cpu(inputs_embeds=cmm).logits[0, -8:]
        pm = F.softmax(lm / 2.0, dim=-1)
        pmm = F.softmax(lmm / 2.0, dim=-1)
        js_scores.append(js_divergence(pm, pmm).item())
    mean_js = sum(js_scores) / len(js_scores)
    print(f"\n  JS (match vs mismatch): {mean_js:.6f}", flush=True)

    print(f"\n  End-to-end:", flush=True)
    e2e = {}
    for prompt in TASK_PROMPTS:
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            s_out = model_cpu(**inp, output_hidden_states=True)
            h_s = s_out.hidden_states[EXTRACTION_LAYER][0, -1, :]
            logits_none = model_cpu(**inp).logits[0, -1]
            gen_ids = model_cpu.generate(**inp, max_new_tokens=40, do_sample=True,
                                         temperature=0.8, top_p=0.95,
                                         pad_token_id=tok.eos_token_id)
        s_text = tok.decode(gen_ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        r_inp = tok(prompt + " " + s_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits_text = model_cpu(**r_inp).logits[0, -1]

        z = codec_cpu.encode(h_s)
        prefix = codec_cpu.decode(z)
        with torch.no_grad():
            pe = embed_layer(inp["input_ids"][0])
        comb = torch.cat([prefix, pe], dim=0).unsqueeze(0)
        with torch.no_grad():
            logits_lat = model_cpu(inputs_embeds=comb).logits[0, -1]

        p_n = F.softmax(logits_none, dim=-1)
        p_t = F.softmax(logits_text, dim=-1)
        p_l = F.softmax(logits_lat, dim=-1)
        kl_t = F.kl_div(p_n.log(), p_t, reduction="sum").item()
        kl_l = F.kl_div(p_n.log(), p_l, reduction="sum").item()
        pct = kl_l / kl_t * 100 if kl_t > 0 else 0

        print(f"    '{prompt[:50]}...': latent={kl_l:.2f} text={kl_t:.2f} ({pct:.0f}%)", flush=True)
        e2e[prompt] = {"kl_text": kl_t, "kl_latent": kl_l, "pct": pct}

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} minutes", flush=True)

    output = {
        "config": {"steps": NUM_STEPS, "device": DEVICE},
        "history": history,
        "kl_start": first_kl, "kl_end": last_kl, "kl_min": min_kl,
        "kl_reduction_pct": reduction, "plateau_range": plateau_range,
        "js": mean_js, "end_to_end": e2e,
    }
    with open(RESULTS_DIR / "bapc_extended_training.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to bapc_extended_training.json", flush=True)


if __name__ == "__main__":
    main()
