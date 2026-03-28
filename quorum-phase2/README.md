# Quorum Phase 2 — Latent Resonance Loop

A system where multiple LLM agents communicate through a shared continuous-valued
latent buffer (activations, not text), using Hopfield-energy cross-inhibition and
contraction-enforced convergence.

## Architecture

The loop comprises four subsystems:

1. **Universal Latent Buffer** — shared state matrix B ∈ ℝ^{S×D} with S=6 named slots
2. **Per-Agent Codecs** — encoder/decoder MLPs translating between agent hidden spaces and buffer space
3. **Cross-Inhibition Engine** — Hopfield energy competition + bee-inspired stop-signal dynamics
4. **Convergence Controller** — DEQ-style residual monitoring with Anderson acceleration

Each round: **Read** (agents attend to buffer) → **Propose** (agents generate deltas) → **Resolve** (cross-inhibition + contraction produce next state).

## Quick start

```bash
pip install -e .
pytest tests/ -v
```

## Running experiments

```bash
python experiments/synthetic_convergence.py
python experiments/codec_roundtrip.py
python experiments/ties_stability.py
```

Results are saved to `experiments/results/`.

## Technical spec

See `docs/latent_resonance_loop_spec.md` for the full technical specification.

## Project structure

```
quorum/
├── buffer.py              # Universal Latent Buffer (§2.1)
├── codecs.py              # Per-agent encoder/decoder MLPs (§2.2)
├── codec_training.py      # Procrustes + composite loss training (§2.2)
├── ties_resolve.py        # TIES-Resolve multi-writer mechanism (§2.3)
├── manifold.py            # On-manifold projection / cleanup autoencoder (§2.3)
├── cross_inhibition.py    # Hopfield energy + bee dynamics (§3.1, §3.2)
├── suppression_safety.py  # Collapse prevention mechanisms (§3.3)
├── contraction.py         # Damped iteration, spectral norm, Jacobian reg (§4.1)
├── anderson.py            # Anderson acceleration (§4.2)
├── convergence.py         # Halting criteria, PonderNet (§4.3)
├── dissenter.py           # Carol's protected channel (§5)
├── loop.py                # Main latent_resonance_loop function (§6)
└── utils.py               # Shared utilities
```
