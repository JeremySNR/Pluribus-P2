"""BAPC Codec Training for 7B Models on AWS GPU (A10G 24GB).

Adapts the validated Round 6c training pipeline (bapc_fixed_training.py) for 7B
scale models: Qwen3-8B-Base, Mistral-7B-v0.3, OLMo-2-1124-7B.

Key changes from GPT-2 training:
  - fp16 model loading with device_map={"": 0} to fit in 24GB VRAM
  - Extraction layer: floor(0.6 * num_layers) per model
  - IsoCal k=32 for 4096-dim hidden states
  - Buffer dim: 1024 (was 512 for GPT-2)
  - Cross-model alignment loss for non-reference codecs
  - Explicit GPU memory management (load one model at a time)

Trains a reference codec for Qwen3-8B, then aligned codecs for the others.
Saves checkpoints to checkpoints/ for the emergence test.
"""

from __future__ import annotations

import gc
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
from quorum.bapc_training import js_divergence

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Defaults — overridden by --local flag for RTX 4070 testing
BUFFER_DIM = 1024
NUM_SLOTS = 6
ISOCAL_K = 32
NUM_STEPS = 3000
BATCH_SIZE = 4
LR = 1e-4
TEACHER_WINDOW = 8
TEMPERATURE = 2.0
LOG_EVERY = 200
MAX_GEN_TOKENS = 20

MODELS = {
    "qwen": "Qwen/Qwen3-8B-Base",
    "mistral": "mistralai/Mistral-7B-v0.3",
    "olmo": "allenai/OLMo-2-1124-7B",
}

REFERENCE_MODEL = "qwen"

LOCAL_MODELS = {
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
}

LOCAL_REFERENCE_MODEL = "gpt2"

LOCAL_OVERRIDES = {
    "buffer_dim": 512,
    "isocal_k": 16,
    "num_steps": 500,
    "batch_size": 16,
    "log_every": 50,
    "num_prompts": 50,
}

# ~300 diverse training prompts covering science, history, technology, etc.
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
    # Additional prompts for broader coverage
    "Cognitive dissonance occurs when a person holds",
    "The trolley problem is a thought experiment",
    "Maslow's hierarchy of needs places physiological",
    "The Turing test evaluates whether a machine",
    "Osmosis is the movement of water across",
    "The Heisenberg uncertainty principle states that",
    "Nuclear fusion powers the Sun by combining",
    "Tidal forces are caused by gravitational",
    "The Richter scale measures the magnitude of",
    "Coral reefs support approximately one quarter of",
    "The Dopamine system in the brain is involved",
    "Descartes argued that the mind and body",
    "Utilitarianism judges the morality of actions based",
    "The categorical imperative requires that one act",
    "Existentialism emphasizes individual freedom and the",
    "Game theory studies strategic interactions between",
    "The Nash equilibrium is a state where no",
    "Comparative advantage explains why countries benefit",
    "Monetary policy is implemented by central banks",
    "The Laffer curve illustrates the relationship between",
    "The Coriolis effect causes moving objects to",
    "Hadley cells are large-scale atmospheric circulation",
    "El Nino events are characterized by warming",
    "Continental drift was first proposed by Alfred",
    "Pangaea was a supercontinent that existed approximately",
    "The pH scale ranges from zero to fourteen",
    "Redox reactions involve the transfer of electrons",
    "Chemical equilibrium occurs when forward and reverse",
    "Le Chatelier's principle predicts how a system",
    "Entropy in thermodynamics measures the disorder of",
    "Maxwell's equations describe the behaviour of electric",
    "Superconductivity occurs when certain materials reach",
    "Semiconductors have electrical conductivity between that",
    "The Hardy-Weinberg principle describes allele frequencies",
    "Natural selection acts on phenotypic variation within",
    "Genetic drift causes random changes in allele",
    "The central dogma of molecular biology describes",
    "Ribosomes translate messenger RNA into protein",
    "Homeostasis is the maintenance of stable internal",
    "The blood-brain barrier selectively restricts substances",
    "Neuroplasticity refers to the brain's ability to",
    "The Standard Model of particle physics classifies",
    "Quarks combine to form hadrons such as",
    "The Higgs boson gives particles their mass",
    "Antimatter consists of particles with opposite charge",
    "String theory proposes that fundamental particles are",
    "The cosmic microwave background radiation is a",
    "Dark energy is thought to be responsible",
    "Hubble's law states that the velocity of",
    "The Drake equation estimates the number of",
    "Fermat's last theorem was proven by Andrew",
    "The Riemann hypothesis concerns the distribution of",
    "Goedel's incompleteness theorems demonstrate that any",
    "Chaos theory studies systems that are highly",
    "Fractals are geometric shapes that exhibit self",
    "The travelling salesman problem asks for the",
    "Public key cryptography relies on the difficulty",
    "Hashing algorithms convert input data into fixed",
    "The Byzantine generals problem illustrates the challenge",
    "CAP theorem states that distributed systems cannot",
    "MapReduce is a programming model for processing",
    "The lambda calculus provides a formal system",
    "Garbage collection automatically manages memory by",
    "Agroforestry combines trees with crops to improve",
    "Desalination removes salt from seawater to produce",
    "Carbon capture technology aims to reduce atmospheric",
    "The Green Revolution increased agricultural production through",
    "Gene therapy introduces genetic material into cells",
    "Immunotherapy treats cancer by enhancing the body's",
    "The placebo effect demonstrates that patient expectations",
    "Epidemiology studies the distribution and determinants of",
    "The Hippocratic oath is traditionally taken by",
    "Cognitive behavioral therapy treats mental disorders by",
    "Operant conditioning shapes behaviour through reinforcement",
    "The bystander effect describes the tendency for",
    "Social constructionism argues that knowledge is created",
    "Weber's iron cage metaphor describes how rationalization",
    "The tragedy of the commons describes how",
    "Pareto efficiency occurs when no one can",
    "Information asymmetry in markets leads to adverse",
    "The Phillips curve shows an inverse relationship",
    "Fiscal multipliers measure how government spending changes",
    "The Sapir-Whorf hypothesis suggests that language",
    "Chomsky's universal grammar theory proposes that the",
    "Phonemes are the smallest units of sound",
    "The printing revolution transformed European society by",
    "Impressionism originated in Paris during the eighteen",
    "The Bauhaus school unified crafts and fine",
    "Gothic architecture is characterized by pointed arches",
    "The golden ratio appears frequently in art",
    "Jazz evolved from African American musical traditions",
    "Sonata form consists of an exposition development",
    "The Gutenberg Bible was the first major",
    "Stream of consciousness narration presents a character's",
    "Haiku is a form of Japanese poetry",
    "The theory of comparative literature examines texts",
    "Roman concrete allowed the construction of structures",
    "The aqueducts of ancient Rome transported water",
    "Feudal Japan was organized around samurai warriors",
    "The Mongol Empire was the largest contiguous",
    "The transatlantic slave trade forcibly transported millions",
    "The partition of India in 1947 created",
    "Decolonization movements in Africa accelerated after",
    "The Bretton Woods system established fixed exchange",
    "The European Union began as an economic",
    "Nuclear deterrence theory relies on the concept",
]

TASK_PROMPTS = [
    "What are the benefits and risks of remote work?",
    "Should governments regulate artificial intelligence?",
    "The economy is growing strongly but inequality is rising. Advise.",
    "A company must choose between short-term profit and long-term sustainability.",
    "Explain why some scientific discoveries are initially rejected by the mainstream.",
]


# ── GPU memory management ────────────────────────────────────────────

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model(model_key, local_mode=False):
    """Load a CausalLM. Uses fp16 + device_map for 7B, float32 for local GPT-2."""
    model_table = LOCAL_MODELS if local_mode else MODELS
    model_name = model_table[model_key]
    print(f"  Loading {model_name} {'(local)' if local_mode else 'in fp16'} ...",
          flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if local_mode:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(DEVICE)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map={"": 0},
        )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    extraction_layer = int(0.6 * num_layers)

    print(f"    layers={num_layers}, hidden={hidden_dim}, "
          f"extraction_layer={extraction_layer}, vocab={model.config.vocab_size}",
          flush=True)

    return model, tokenizer, {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "extraction_layer": extraction_layer,
        "vocab_size": model.config.vocab_size,
    }


def unload_model(*objects):
    """Delete model objects and free GPU memory."""
    for obj in objects:
        del obj
    free_gpu()
    print("    Model unloaded, GPU freed.", flush=True)


# ── Orthogonality loss ───────────────────────────────────────────────

def orthogonality_loss(weights):
    loss = torch.tensor(0.0, device=weights[0].device)
    for W in weights:
        WWT = W @ W.T
        I = torch.eye(WWT.shape[0], device=W.device)
        loss = loss + (WWT - I).pow(2).mean()
    return loss


# ── Data collection ──────────────────────────────────────────────────

def collect_training_data_7b(model, tokenizer, prompts, extraction_layer,
                              max_gen_tokens=40, teacher_window=8):
    """Extract (sender_hidden, teacher_logits) pairs from a 7B model.

    Sender and receiver are the same model. For each prompt:
      1. Generate text continuation (sender)
      2. Extract hidden state from extraction_layer at last token
      3. Feed prompt + generated text to model (receiver), record teacher logits
    """
    device = next(model.parameters()).device
    sender_hiddens = []
    teacher_logits_list = []

    for i, prompt in enumerate(prompts):
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        inp = {k: v.to(device) for k, v in inp.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inp, max_new_tokens=max_gen_tokens,
                do_sample=True, temperature=0.8, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(gen_ids[0][inp["input_ids"].shape[1]:],
                                    skip_special_tokens=True)

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[extraction_layer][0, -1, :].float().cpu()
        sender_hiddens.append(h)

        combined_text = prompt + " " + gen_text
        recv_inp = tokenizer(combined_text, return_tensors="pt",
                             truncation=True, max_length=128)
        recv_inp = {k: v.to(device) for k, v in recv_inp.items()}
        with torch.no_grad():
            recv_out = model(**recv_inp)
            logits = recv_out.logits[0].float().cpu()
            window = min(teacher_window, logits.shape[0])
            teacher_logits_list.append(logits[-window:])

        del out, recv_out, gen_ids

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(prompts)}", flush=True)

    max_window = min(teacher_window, min(t.shape[0] for t in teacher_logits_list))
    teacher_logits = torch.stack([t[-max_window:] for t in teacher_logits_list])
    sender_hiddens = torch.stack(sender_hiddens)

    print(f"    Collected {sender_hiddens.shape[0]} samples, "
          f"hiddens={sender_hiddens.shape}, teacher_logits={teacher_logits.shape}",
          flush=True)

    return sender_hiddens, teacher_logits


# ── Prompt embedding pre-computation ─────────────────────────────────

def precompute_prompt_embeddings(model, tokenizer, prompts, device, local_mode=False):
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
    dtype = torch.float32 if local_mode else torch.bfloat16
    padded = torch.zeros(len(prompts), max_len, d_emb, device=device, dtype=dtype)
    mask = torch.zeros(len(prompts), max_len, device=device)

    for i, (emb, length) in enumerate(zip(all_embeds, all_lengths)):
        padded[i, :length] = emb
        mask[i, :length] = 1.0

    return padded, mask, all_lengths


# ── BAPC training with gradient flow ─────────────────────────────────

def train_bapc_7b(codec, model, tokenizer, sender_hiddens, teacher_logits,
                  prompts, prompt_embeds_padded, prompt_mask, prompt_lengths,
                  ref_encodings=None, local_mode=False,
                  num_steps=NUM_STEPS, batch_size=BATCH_SIZE, lr=LR,
                  T=TEMPERATURE, lambda_sep=0.1, lambda_align=0.1,
                  lambda_orth=0.01, lambda_cross=0.5, log_every=LOG_EVERY):
    """Train BAPC codec with correct gradient flow through frozen 7B receiver.

    Based on train_batched from bapc_fixed_training.py. Receiver parameters are
    frozen (requires_grad=False) but the forward pass IS differentiable so
    gradients flow: loss -> logits -> embeddings -> prefix -> codec decoder.

    If ref_encodings is provided, adds a cross-model alignment loss that pulls
    this codec's buffer encodings toward the reference codec's.
    """
    device = next(model.parameters()).device
    N = sender_hiddens.shape[0]
    S = codec.num_slots

    for p in model.parameters():
        p.requires_grad = False

    codec = codec.to(device)
    optimizer = optim.AdamW(codec.parameters(), lr=lr)
    history = []

    teacher_window = teacher_logits.shape[1]

    for step in range(num_steps):
        idx = torch.randint(0, N, (min(batch_size, N),))
        h_batch = sender_hiddens[idx].to(device)
        teacher_batch = teacher_logits[idx].to(device)
        B = h_batch.shape[0]

        z_batch = codec.encode(h_batch)
        prefix_batch = codec.decode(z_batch)
        if not local_mode:
            prefix_batch = prefix_batch.to(torch.bfloat16)

        prompt_embs = prompt_embeds_padded[idx]
        prompt_msk = prompt_mask[idx]

        combined = torch.cat([prefix_batch, prompt_embs], dim=1)
        prefix_msk = torch.ones(B, S, device=device)
        attn_mask = torch.cat([prefix_msk, prompt_msk], dim=1)

        out = model(inputs_embeds=combined, attention_mask=attn_mask)

        student_logits_list = []
        for b in range(B):
            real_len = S + prompt_lengths[idx[b]]
            window = min(teacher_window, real_len)
            student_logits_list.append(out.logits[b, real_len - window:real_len].float())
        student_logits_all = torch.stack(student_logits_list)

        p_teacher = F.softmax(teacher_batch / T, dim=-1).detach()
        log_p_student = F.log_softmax(student_logits_all / T, dim=-1)
        l_distill = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

        perm = torch.randperm(B)
        z_shuffled = z_batch[perm].detach()
        prefix_shuffled = codec.decode(z_shuffled)
        if not local_mode:
            prefix_shuffled = prefix_shuffled.to(torch.bfloat16)
        combined_mm = torch.cat([prefix_shuffled, prompt_embs], dim=1)
        with torch.no_grad():
            out_mm = model(inputs_embeds=combined_mm, attention_mask=attn_mask)

        mismatch_logits_list = []
        for b in range(B):
            real_len = S + prompt_lengths[idx[b]]
            window = min(teacher_window, real_len)
            mismatch_logits_list.append(out_mm.logits[b, real_len - window:real_len].float())
        mismatch_logits_all = torch.stack(mismatch_logits_list)

        p_matched = F.softmax(student_logits_all / T, dim=-1)
        p_mismatched = F.softmax(mismatch_logits_all / T, dim=-1).detach()
        l_sep = -js_divergence(p_matched, p_mismatched)

        l_align_cos = 1.0 - F.cosine_similarity(
            student_logits_all.reshape(-1, student_logits_all.shape[-1]),
            teacher_batch.reshape(-1, teacher_batch.shape[-1]),
            dim=-1
        ).mean()

        l_orth = orthogonality_loss(codec.get_encoder_weights())

        l_cross = torch.tensor(0.0, device=device)
        if ref_encodings is not None:
            ref_batch = ref_encodings[idx].to(device)
            l_cross = F.mse_loss(z_batch, ref_batch)

        loss = (l_distill
                + lambda_sep * l_sep
                + lambda_align * l_align_cos
                + lambda_orth * l_orth
                + lambda_cross * l_cross)

        # Skip step entirely if loss is NaN — prevents codec weights being poisoned
        if torch.isnan(loss) or torch.isinf(loss):
            if step % log_every == 0:
                print(f"    Step {step:5d}/{num_steps}: NaN/Inf loss, skipping", flush=True)
            continue

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
                "align_cos": l_align_cos.item(),
                "orth": l_orth.item(),
                "cross": l_cross.item(),
            }
            history.append(entry)
            cross_str = f" cross={l_cross.item():.4f}" if ref_encodings is not None else ""
            print(f"    Step {step:5d}/{num_steps}: "
                  f"distill={l_distill.item():.2f} sep={l_sep.item():.4f} "
                  f"align={l_align_cos.item():.4f} orth={l_orth.item():.4f}"
                  f"{cross_str}", flush=True)

    return history


# ── Checkpoint save/load ─────────────────────────────────────────────

def save_codec_checkpoint(codec, isocal, model_key, model_info,
                          history, ref_encodings=None, local_mode=False):
    prefix = "bapc_local" if local_mode else "bapc_7b"
    path = CHECKPOINT_DIR / f"{prefix}_{model_key}.pt"
    model_table = LOCAL_MODELS if local_mode else MODELS
    state = {
        "codec_state": codec.cpu().state_dict(),
        "isocal_state": isocal.state_dict(),
        "model_key": model_key,
        "model_name": model_table[model_key],
        "hidden_dim": model_info["hidden_dim"],
        "extraction_layer": model_info["extraction_layer"],
        "buffer_dim": codec.buffer_dim,
        "num_slots": codec.num_slots,
        "isocal_k": isocal.config.k,
        "training_history": history,
    }
    if ref_encodings is not None:
        state["ref_encodings"] = ref_encodings.cpu()
    torch.save(state, path)
    print(f"    Saved checkpoint: {path.name}", flush=True)
    return path


# ── Main training pipeline ───────────────────────────────────────────

def train_single_model(model_key, ref_encodings=None, local_mode=False):
    """Train a BAPC codec for one model. Loads model, trains, saves, unloads."""
    t_start = time.time()
    ref_model = LOCAL_REFERENCE_MODEL if local_mode else REFERENCE_MODEL
    is_reference = (model_key == ref_model)
    label = "REFERENCE" if is_reference else "ALIGNED"

    print(f"\n{'='*70}")
    print(f"  Training {label} codec for {model_key}")
    print(f"{'='*70}", flush=True)

    model, tokenizer, model_info = load_model(model_key, local_mode=local_mode)
    extraction_layer = model_info["extraction_layer"]
    hidden_dim = model_info["hidden_dim"]
    device = next(model.parameters()).device

    # Effective config (local overrides for quick testing)
    buf_dim = LOCAL_OVERRIDES["buffer_dim"] if local_mode else BUFFER_DIM
    k = LOCAL_OVERRIDES["isocal_k"] if local_mode else ISOCAL_K
    steps = LOCAL_OVERRIDES["num_steps"] if local_mode else NUM_STEPS
    bs = LOCAL_OVERRIDES["batch_size"] if local_mode else BATCH_SIZE
    log_freq = LOCAL_OVERRIDES["log_every"] if local_mode else LOG_EVERY
    n_prompts = LOCAL_OVERRIDES["num_prompts"] if local_mode else min(150, len(PROMPTS))
    train_prompts = PROMPTS[:n_prompts]

    # IsoCal calibration
    n_cal = min(100, n_prompts)
    print(f"\n  Phase 1: IsoCal calibration (k={k}) ...", flush=True)
    cal_states = []
    with torch.no_grad():
        for i, p in enumerate(train_prompts[:n_cal]):
            inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
            inp = {kk: v.to(device) for kk, v in inp.items()}
            out = model(**inp, output_hidden_states=True)
            cal_states.append(out.hidden_states[extraction_layer][0, -1, :].float().cpu())
            del out
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{n_cal} calibration states", flush=True)

    cal_stack = torch.stack(cal_states)
    isocal = IsotropyCalibrator(IsocalConfig(k=k))
    isocal.calibrate(cal_stack)
    del cal_states, cal_stack

    raw_sims = []
    cal_sims = []
    sample_states = []
    n_sample = min(20, n_prompts)
    with torch.no_grad():
        for p in train_prompts[:n_sample]:
            inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
            inp = {kk: v.to(device) for kk, v in inp.items()}
            out = model(**inp, output_hidden_states=True)
            sample_states.append(out.hidden_states[extraction_layer][0, -1, :].float().cpu())
    for i in range(len(sample_states)):
        for j in range(i + 1, len(sample_states)):
            raw_sims.append(F.cosine_similarity(
                sample_states[i].unsqueeze(0), sample_states[j].unsqueeze(0)).item())
            ci = isocal.transform(sample_states[i])
            cj = isocal.transform(sample_states[j])
            cal_sims.append(F.cosine_similarity(ci.unsqueeze(0), cj.unsqueeze(0)).item())
    raw_mean = sum(raw_sims) / len(raw_sims)
    cal_mean = sum(cal_sims) / len(cal_sims)
    print(f"    Raw pairwise cosine ({n_sample} prompts): {raw_mean:.4f}", flush=True)
    print(f"    After IsoCal pairwise cosine:     {cal_mean:.4f}", flush=True)
    del sample_states

    # Data collection
    print(f"\n  Phase 2: Collecting training data ({len(train_prompts)} prompts) ...",
          flush=True)
    sender_hiddens, teacher_logits = collect_training_data_7b(
        model, tokenizer, train_prompts, extraction_layer,
        max_gen_tokens=MAX_GEN_TOKENS, teacher_window=TEACHER_WINDOW,
    )
    N = sender_hiddens.shape[0]

    # Create codec
    codec = BAPCCodec(
        agent_dim=hidden_dim, buffer_dim=buf_dim,
        embed_dim=hidden_dim, num_slots=NUM_SLOTS,
    )
    codec.isocal = isocal

    # Pre-compute prompt embeddings
    print(f"\n  Phase 3: Pre-computing prompt embeddings ...", flush=True)
    prompt_embeds, prompt_mask, prompt_lengths = precompute_prompt_embeddings(
        model, tokenizer, train_prompts[:N], device, local_mode=local_mode
    )
    print(f"    Padded shape: {prompt_embeds.shape}", flush=True)

    # Train
    lambda_cross = 0.5 if ref_encodings is not None else 0.0
    print(f"\n  Phase 4: Training ({steps} steps, batch={bs}, "
          f"cross_align={'YES' if ref_encodings is not None else 'NO'}) ...", flush=True)
    history = train_bapc_7b(
        codec, model, tokenizer, sender_hiddens, teacher_logits,
        train_prompts[:N], prompt_embeds, prompt_mask, prompt_lengths,
        ref_encodings=ref_encodings, local_mode=local_mode,
        num_steps=steps, batch_size=bs, lr=LR,
        lambda_cross=lambda_cross, log_every=log_freq,
    )

    # Compute reference encodings for alignment of subsequent models
    my_ref_encodings = None
    if is_reference:
        print(f"\n  Phase 5: Computing reference buffer encodings ...", flush=True)
        codec_cpu = codec.cpu()
        with torch.no_grad():
            my_ref_encodings = codec_cpu.encode(sender_hiddens)
        print(f"    Shape: {my_ref_encodings.shape}", flush=True)
        codec = codec_cpu

    # Training summary
    distill_vals = [h["distill"] for h in history]
    first_kl = distill_vals[0]
    last_kl = distill_vals[-1]
    min_kl = min(distill_vals)
    reduction = (first_kl - last_kl) / first_kl * 100 if first_kl > 0 else 0

    print(f"\n  SUMMARY ({model_key}):")
    print(f"    KL: {first_kl:.2f} -> {last_kl:.2f} (min: {min_kl:.2f}, "
          f"reduction: {reduction:.1f}%)", flush=True)

    # Save checkpoint
    save_codec_checkpoint(codec, isocal, model_key, model_info,
                          history, ref_encodings=my_ref_encodings,
                          local_mode=local_mode)

    elapsed = time.time() - t_start
    print(f"    Time: {elapsed / 60:.1f} minutes", flush=True)

    # Unload
    del model, tokenizer, prompt_embeds, prompt_mask
    del sender_hiddens, teacher_logits
    free_gpu()
    print(f"    Unloaded {model_key}", flush=True)

    return {
        "model_key": model_key,
        "kl_start": first_kl,
        "kl_end": last_kl,
        "kl_min": min_kl,
        "kl_reduction_pct": reduction,
        "hidden_dim": hidden_dim,
        "extraction_layer": model_info["extraction_layer"],
        "elapsed_seconds": elapsed,
    }, my_ref_encodings


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="Local test mode: GPT-2/DistilGPT-2, 512 buffer, "
                             "500 steps, 50 prompts (~5 min on RTX 4070)")
    args = parser.parse_args()

    local_mode = args.local
    model_table = LOCAL_MODELS if local_mode else MODELS
    ref_model = LOCAL_REFERENCE_MODEL if local_mode else REFERENCE_MODEL

    buf_dim = LOCAL_OVERRIDES["buffer_dim"] if local_mode else BUFFER_DIM
    k = LOCAL_OVERRIDES["isocal_k"] if local_mode else ISOCAL_K
    steps = LOCAL_OVERRIDES["num_steps"] if local_mode else NUM_STEPS
    bs = LOCAL_OVERRIDES["batch_size"] if local_mode else BATCH_SIZE
    n_prompts = LOCAL_OVERRIDES["num_prompts"] if local_mode else min(150, len(PROMPTS))

    t_start = time.time()
    print("=" * 70)
    label = "LOCAL TEST" if local_mode else "BAPC 7B CODEC TRAINING"
    print(label)
    print(f"Device: {DEVICE}")
    print(f"Models: {list(model_table.values())}")
    print(f"Buffer: D={buf_dim}, S={NUM_SLOTS}, IsoCal k={k}")
    print(f"Training: {steps} steps, batch={bs}, lr={LR}")
    print(f"Prompts: {n_prompts}")
    print("=" * 70, flush=True)

    all_results = {}

    ref_result, ref_encodings = train_single_model(
        ref_model, local_mode=local_mode)
    all_results[ref_model] = ref_result

    for model_key in model_table:
        if model_key == ref_model:
            continue
        result, _ = train_single_model(
            model_key, ref_encodings=ref_encodings, local_mode=local_mode)
        all_results[model_key] = result

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    for mk, r in all_results.items():
        print(f"  {mk}: KL {r['kl_start']:.1f} -> {r['kl_end']:.1f} "
              f"({r['kl_reduction_pct']:.1f}% reduction, "
              f"{r['elapsed_seconds']/60:.1f} min)", flush=True)
    print(f"\n  Total time: {elapsed / 60:.1f} minutes", flush=True)

    output = {
        "config": {
            "local_mode": local_mode,
            "buffer_dim": buf_dim,
            "num_slots": NUM_SLOTS,
            "isocal_k": k,
            "num_steps": steps,
            "batch_size": bs,
            "num_prompts": n_prompts,
            "device": DEVICE,
            "reference_model": ref_model,
        },
        "models": all_results,
        "elapsed_seconds": elapsed,
    }
    suffix = "_local" if local_mode else ""
    with open(RESULTS_DIR / f"bapc_7b_training{suffix}.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved results to bapc_7b_training{suffix}.json", flush=True)


if __name__ == "__main__":
    main()
