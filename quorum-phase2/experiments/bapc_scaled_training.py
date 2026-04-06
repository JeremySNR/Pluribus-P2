"""Round 6b: Scaled BAPC codec training on local GPU.

500 prompts, 5000 steps, on RTX 4070 Laptop GPU.
Tests whether more data and longer training closes the distillation gap.
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
from quorum.bapc_training import (
    BAPCTrainingConfig, collect_training_data, train_bapc_codec, js_divergence,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACTION_LAYER = 7

_MODEL_CACHE: dict = {}

PROMPTS_500 = [
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
    # Extended prompts for more training data
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
    "Symbiotic relationships between organisms can be",
    "Cryptographic hash functions produce fixed-length outputs",
    "The Green Revolution increased agricultural productivity",
    "Dropout randomly deactivates neurons during training",
    "The nervous system transmits electrical signals through",
    "RESTful APIs use HTTP methods to perform",
    "The Marshall Plan provided economic aid to",
    "Word embeddings represent words as dense vectors",
    "The carbon cycle describes how carbon moves",
    "OAuth provides secure delegated access through",
    "The Columbian Exchange transferred plants and animals",
    "Recurrent connections allow networks to maintain",
    "The nitrogen cycle converts atmospheric nitrogen into",
    "WebSocket enables full-duplex communication between",
    "The French and Indian War was fought",
    "Self-supervised learning creates labels from the",
    "The phosphorus cycle moves phosphorus through",
    "GraphQL provides a query language for APIs",
    "The Scramble for Africa divided the continent",
    "Few-shot learning trains models with very",
    "The rock cycle transforms igneous sedimentary and",
    "Containerization packages applications with their dependencies",
    "The Spanish Inquisition was established to maintain",
    "Knowledge distillation trains smaller models to mimic",
    "The sulfur cycle involves volcanic emissions and",
    "Infrastructure as code manages computing resources through",
    "The Hundred Years War was a series",
    "Data augmentation increases training set diversity",
    "The hydrological cycle redistributes water across",
    "Service mesh manages communication between microservices",
    "The Opium Wars were fought between China",
    "Ensemble methods combine multiple models to improve",
    "The oxygen cycle maintains atmospheric oxygen through",
    "Rate limiting controls the number of requests",
    "The Crimean War was fought primarily on",
    "Hyperparameter tuning optimizes model performance by",
    "Ecosystems consist of biotic and abiotic components",
    "Circuit breaker patterns prevent cascading failures in",
    "The Thirty Years War devastated Central Europe",
    "Cross-validation estimates model performance on unseen",
    "Food webs illustrate energy flow through",
    "Content delivery networks distribute content across",
    "The Peloponnesian War was fought between Athens",
    "Feature engineering creates new input variables from",
    "Biodiversity measures the variety of life forms",
    "Serverless computing executes code without managing",
    "The Punic Wars were fought between Rome",
    "Activation functions introduce nonlinearity into neural",
    "Conservation biology aims to protect endangered species",
    "Blue-green deployment reduces downtime during software",
    "The Reconquista was the centuries-long effort to",
    "Semantic segmentation classifies each pixel in an",
    "Natural selection drives evolution through differential",
    "Chaos engineering intentionally introduces failures to",
    "The War of the Roses was a civil",
    "Instance segmentation identifies and delineates individual",
    "Genetic drift causes random changes in allele",
    "Observability combines logging metrics and tracing",
    "The Boxer Rebellion was an anti-foreign uprising",
    "Object detection locates and classifies objects within",
    "Speciation occurs when populations become reproductively",
    "Idempotent operations produce the same result regardless",
    "The Glorious Revolution established parliamentary supremacy",
    "Pose estimation determines the position of body",
    "Mutualism is a symbiotic relationship where both",
    "Eventual consistency guarantees that distributed systems",
    "The Sepoy Mutiny was a major uprising",
    "Optical character recognition converts images of text",
    "Parasitism is a relationship where one organism",
    "Sharding distributes database records across multiple",
    "The Zulu Wars were fought between the",
    "Named entity recognition identifies and classifies",
    "Commensalism is a relationship where one organism",
    "Consensus algorithms ensure agreement among distributed",
    "The Taiping Rebellion was one of the",
    "Sentiment analysis determines the emotional tone of",
    "Predation is an ecological interaction where one",
    "Write-ahead logging ensures database transaction durability",
    "The Russo-Japanese War was the first major",
    "Text summarization condenses documents while preserving",
    "Decomposition breaks down organic matter into simpler",
    "Vector clocks track causality in distributed systems",
    "The Mexican-American War resulted in the Treaty",
    "Machine reading comprehension answers questions based on",
    "Primary succession occurs on newly formed substrates",
    "Two-phase commit coordinates transactions across multiple",
    "The Boer Wars were fought in South",
    "Dependency parsing analyzes grammatical structure of",
    "Secondary succession occurs after a disturbance removes",
    "Raft consensus provides a more understandable alternative",
    "The Spanish Civil War was fought between",
    "Coreference resolution identifies expressions that refer",
    "Ecological niches describe how organisms interact with",
    "MapReduce processes large datasets across distributed",
    "The Korean War was fought between North",
    "Relation extraction identifies semantic relationships between",
    "Trophic levels describe the position of organisms",
    "Bloom filters provide space-efficient probabilistic set",
    "The Vietnam War was a prolonged conflict",
    "Question answering systems extract answers from",
    "Keystone species have disproportionate effects on their",
    "Merkle trees efficiently verify data integrity in",
    "The Six-Day War was fought between Israel",
    "Dialogue systems generate conversational responses based on",
    "Invasive species threaten native ecosystems by competing",
    "Gossip protocols disseminate information across distributed",
    "The Iran-Iraq War lasted for eight years",
    "Language modeling predicts the probability of token",
    "Biomes are large ecological areas defined by",
    "Paxos provides a family of protocols for",
    "The Falklands War was fought between Argentina",
    "Speech recognition converts spoken language into text",
    "Ecological footprint measures human demand on natural",
    "Distributed hash tables provide decentralized key-value",
    "The Gulf War was triggered by Iraq's invasion",
    "Neural machine translation uses encoder-decoder architectures",
    "Carrying capacity is the maximum population size",
    "Consistent hashing minimizes redistribution when nodes",
    "The Rwandan genocide resulted in the deaths",
    "Abstractive summarization generates novel sentences to",
    "Population dynamics study how populations change over",
    "Leader election algorithms select a coordinator among",
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


def main():
    t_start = time.time()
    print("=" * 70)
    print(f"ROUND 6b: SCALED BAPC TRAINING ({DEVICE.upper()})")
    print(f"500 prompts, 5000 steps")
    print("=" * 70, flush=True)

    model, tok = _load_lm("gpt2")

    # Calibrate IsoCal from 500 prompts at extraction layer
    print(f"\n  Extracting calibration states from layer {EXTRACTION_LAYER} ...", flush=True)
    cal_states = []
    with torch.no_grad():
        for i, p in enumerate(PROMPTS_500):
            inp = tok(p, return_tensors="pt", truncation=True, max_length=64)
            out = model(**inp, output_hidden_states=True)
            cal_states.append(out.hidden_states[EXTRACTION_LAYER][0, -1, :])
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(PROMPTS_500)}", flush=True)
    cal_stack = torch.stack(cal_states)
    print(f"    Done: {cal_stack.shape}", flush=True)

    isocal = IsotropyCalibrator(IsocalConfig(k=16))
    isocal.calibrate(cal_stack)

    # Collect training data (500 prompts)
    print(f"\n  Collecting training data (500 prompts) ...", flush=True)
    data = collect_training_data(
        model, tok, model, tok,
        PROMPTS_500,
        extraction_layer=EXTRACTION_LAYER,
        max_gen_tokens=40,
        teacher_window=8,
    )
    print(f"    Sender hiddens: {data['sender_hiddens'].shape}", flush=True)
    print(f"    Teacher logits: {data['receiver_teacher_logits'].shape}", flush=True)

    # Create codec and move to GPU
    D = 512
    S = 6
    d_agent = data["sender_hiddens"].shape[1]
    codec = BAPCCodec(agent_dim=d_agent, buffer_dim=D, embed_dim=d_agent, num_slots=S)
    codec.isocal = isocal

    if DEVICE == "cuda":
        codec = codec.to(DEVICE)
        data["sender_hiddens"] = data["sender_hiddens"].to(DEVICE)
        data["receiver_teacher_logits"] = data["receiver_teacher_logits"].to(DEVICE)
        model = model.to(DEVICE)

    # Train (5000 steps)
    print(f"\n  Training codec (5000 steps, device={DEVICE}) ...", flush=True)
    cfg = BAPCTrainingConfig(
        lr=3e-4, num_steps=5000, batch_size=16, teacher_window=8,
        temperature=2.0, lambda_sep=0.1, lambda_align=0.1, lambda_orth=0.01,
        log_every=500,
    )
    history = train_bapc_codec(codec, model, tok, data, PROMPTS_500, cfg)

    # Diagnostics
    print(f"\n{'='*70}")
    print("DIAGNOSTICS")
    print(f"{'='*70}", flush=True)

    # KL trajectory
    distill_values = [h["distill"] for h in history]
    print(f"\n  KL trajectory:", flush=True)
    for h in history:
        print(f"    Step {h['step']:5d}: distill={h['distill']:.4f} sep={h['separation']:.4f}", flush=True)
    first_kl = distill_values[0]
    last_kl = distill_values[-1]
    min_kl = min(distill_values)
    reduction = (first_kl - last_kl) / first_kl * 100
    print(f"\n  KL: {first_kl:.2f} -> {last_kl:.2f} (min: {min_kl:.2f}, reduction: {reduction:.1f}%)", flush=True)

    # JS matched vs mismatched
    print(f"\n  JS divergence (matched vs mismatched) ...", flush=True)
    codec_cpu = codec.cpu()
    model_cpu = model.cpu()
    embed_layer = model_cpu.get_input_embeddings()
    sender_hiddens = data["sender_hiddens"].cpu()
    T = 2.0

    n_test = min(100, sender_hiddens.shape[0])
    js_scores = []
    for i in range(n_test):
        h = sender_hiddens[i]
        z = codec_cpu.encode(h)
        prefix_match = codec_cpu.decode(z)

        j = (i + 7) % n_test
        z_other = codec_cpu.encode(sender_hiddens[j])
        prefix_mismatch = codec_cpu.decode(z_other)

        prompt = PROMPTS_500[i]
        prompt_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            prompt_embeds = embed_layer(prompt_ids["input_ids"][0])

        combined_match = torch.cat([prefix_match, prompt_embeds], dim=0).unsqueeze(0)
        combined_mismatch = torch.cat([prefix_mismatch, prompt_embeds], dim=0).unsqueeze(0)

        with torch.no_grad():
            logits_match = model_cpu(inputs_embeds=combined_match).logits[0, -8:]
            logits_mismatch = model_cpu(inputs_embeds=combined_mismatch).logits[0, -8:]

        p_match = F.softmax(logits_match / T, dim=-1)
        p_mismatch = F.softmax(logits_mismatch / T, dim=-1)
        js = js_divergence(p_match, p_mismatch).item()
        js_scores.append(js)

    mean_js = sum(js_scores) / len(js_scores)
    print(f"    Mean JS: {mean_js:.6f} (was 0.040 at 100 prompts/2000 steps)", flush=True)

    # Buffer space anisotropy
    buffer_vecs = []
    for i in range(n_test):
        z = codec_cpu.encode(sender_hiddens[i])
        buffer_vecs.append(z[0])
    buf_sims = []
    for i in range(len(buffer_vecs)):
        for j in range(i + 1, min(i + 20, len(buffer_vecs))):
            sim = F.cosine_similarity(buffer_vecs[i].unsqueeze(0),
                                      buffer_vecs[j].unsqueeze(0)).item()
            buf_sims.append(sim)
    mean_buf = sum(buf_sims) / len(buf_sims)
    print(f"    Buffer cosine: {mean_buf:.4f} (was 0.378 at 100/2000)", flush=True)

    # End-to-end on task prompts
    print(f"\n  End-to-end comparison:", flush=True)
    for prompt in TASK_PROMPTS:
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model_cpu(**inp, output_hidden_states=True)
            h_sender = out.hidden_states[EXTRACTION_LAYER][0, -1, :]

        # No comm
        with torch.no_grad():
            logits_none = model_cpu(**inp).logits[0, -1]

        # Text comm (sender generates, receiver reads)
        with torch.no_grad():
            gen_ids = model_cpu.generate(**inp, max_new_tokens=40, do_sample=True,
                                         temperature=0.8, top_p=0.95,
                                         pad_token_id=tok.eos_token_id)
        sender_text = tok.decode(gen_ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        combined_text = prompt + " " + sender_text
        recv_inp = tok(combined_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits_text = model_cpu(**recv_inp).logits[0, -1]

        # Latent comm
        z = codec_cpu.encode(h_sender)
        prefix = codec_cpu.decode(z)
        with torch.no_grad():
            prompt_embeds = embed_layer(inp["input_ids"][0])
        combined_embeds = torch.cat([prefix, prompt_embeds], dim=0).unsqueeze(0)
        with torch.no_grad():
            logits_latent = model_cpu(inputs_embeds=combined_embeds).logits[0, -1]

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
        print(f"      KL(latent vs none): {kl_latent:.2f} ({ratio:.0f}% of text effect)", flush=True)
        print(f"      Tokens: none='{top_none}' text='{top_text}' latent='{top_latent}'", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} minutes", flush=True)

    output = {
        "config": {"prompts": len(PROMPTS_500), "steps": 5000, "device": DEVICE},
        "training_history": history,
        "diagnostics": {
            "kl_start": first_kl, "kl_end": last_kl, "kl_min": min_kl,
            "kl_reduction_pct": reduction,
            "js_matched_mismatched": mean_js,
            "buffer_cosine": mean_buf,
        },
    }
    with open(RESULTS_DIR / "bapc_scaled_training.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to bapc_scaled_training.json", flush=True)


if __name__ == "__main__":
    main()
