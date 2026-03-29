"""Shared model loading and codec training infrastructure for experiments.

Caches loaded models and trained codecs so multiple experiments can share them
without redundant loading/training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.codecs import AgentCodec
from quorum.codec_training import procrustes_init, composite_loss, train_codec_pair, CodecTrainingConfig

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_MODEL_CACHE: dict = {}
_HIDDEN_STATE_CACHE: dict = {}
_CODEC_CACHE: dict = {}

TRAINING_PROMPTS = [
    "The capital of France is",
    "In quantum mechanics, the wave function",
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


def load_model_and_tokenizer(model_name: str):
    """Load and cache a HuggingFace model + tokenizer."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    from transformers import AutoTokenizer, AutoModel
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer


def load_lm_head_model(model_name: str):
    """Load the full causal LM (with lm_head) for token decoding."""
    cache_key = model_name + "_lm"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"  Loading {model_name} (with LM head)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def extract_hidden_states(model_name: str, prompts: list[str]) -> torch.Tensor:
    """Extract last-layer hidden states, cached."""
    cache_key = (model_name, tuple(prompts))
    if cache_key in _HIDDEN_STATE_CACHE:
        return _HIDDEN_STATE_CACHE[cache_key]

    model, tokenizer = load_model_and_tokenizer(model_name)
    hidden_states = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                               truncation=True, max_length=64)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.mean(dim=1).squeeze(0)
            hidden_states.append(pooled)

    stacked = torch.stack(hidden_states)
    _HIDDEN_STATE_CACHE[cache_key] = stacked
    return stacked


def extract_per_token_hidden(model_name: str, prompt: str) -> torch.Tensor:
    """Extract per-token last-layer hidden states (not pooled). Returns [seq_len, hidden_dim]."""
    model, tokenizer = load_model_and_tokenizer(model_name)
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].squeeze(0)  # [seq_len, hidden_dim]


def train_codec(model_name: str, buffer_dim: int = 512,
                num_prompts: int = 100, num_steps: int = 3000) -> AgentCodec:
    """Train and cache a codec for a given model."""
    cache_key = (model_name, buffer_dim, num_prompts, num_steps)
    if cache_key in _CODEC_CACHE:
        return _CODEC_CACHE[cache_key]

    prompts = TRAINING_PROMPTS[:num_prompts]
    states = extract_hidden_states(model_name, prompts)
    agent_dim = states.shape[1]

    codec = AgentCodec(agent_dim=agent_dim, buffer_dim=buffer_dim)

    train_n = int(0.8 * len(prompts))
    train_data = states[:train_n]
    target_states = torch.randn(train_n, buffer_dim)
    target_states = target_states / target_states.norm(dim=-1, keepdim=True)
    procrustes_init(codec, train_data, target_states)

    optimizer = torch.optim.Adam(codec.parameters(), lr=1e-4)
    for step in range(num_steps):
        idx = torch.randint(0, train_n, (min(32, train_n),))
        batch = train_data[idx]
        loss, _ = composite_loss(codec, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        optimizer.step()

    _CODEC_CACHE[cache_key] = codec
    return codec


def train_codec_pair_aligned(
    model_a: str, model_b: str, buffer_dim: int = 512,
    num_prompts: int = 100, num_steps: int = 3000,
) -> tuple[AgentCodec, AgentCodec]:
    """Train a pair of codecs with cross-model alignment."""
    prompts = TRAINING_PROMPTS[:num_prompts]
    states_a = extract_hidden_states(model_a, prompts)
    states_b = extract_hidden_states(model_b, prompts)
    dim_a, dim_b = states_a.shape[1], states_b.shape[1]

    codec_a = AgentCodec(agent_dim=dim_a, buffer_dim=buffer_dim)
    codec_b = AgentCodec(agent_dim=dim_b, buffer_dim=buffer_dim)

    config = CodecTrainingConfig(lr=1e-4, num_steps=num_steps, log_every=500)
    train_n = int(0.8 * num_prompts)
    train_codec_pair(codec_a, codec_b, states_a[:train_n], states_b[:train_n], config)

    return codec_a, codec_b
