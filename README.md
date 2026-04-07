# Pluribus Phase 2 — Latent Resonance Loop

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen.svg)](#tests)

> *"What if agents didn't just exchange words — they shared the thought itself?"*

---

## The origin story

[Phase 1](https://github.com/JeremySNR/Pluribus) built a blackboard hive. Agents debated in text, challenged each other, and converged on a shared answer. Carol dissented regardless. It worked.

But something about it bothered me. The agents were thinking in continuous high-dimensional space, then translating those thoughts into tokens, passing them around as words, and translating back. That translation is lossy in both directions. A transformer's final hidden state encodes roughly 40,000 bits of information per position. A discrete token encodes about 15.

The text channel was throwing away almost everything the model actually computed.

Phase 2 asks what happens if you skip the translation entirely. Instead of writing messages to a blackboard, agents encode their hidden states directly into a shared continuous buffer. The buffer resolves conflicts using a cross-inhibition mechanism adapted from the model-merging literature. The loop converges by contraction theory. Carol still can't be silenced — she gets her own buffer slot that nothing else can overwrite.

The honest result: collective representations emerge that are geometrically distinct from any individual agent's contribution. Whether that distinctness reliably translates into better outputs is a more qualified story, which the paper tells honestly.

---

## What it does

Multiple LLM agents share a continuous-valued buffer `B ∈ ℝ^(S×D)` instead of a text blackboard. Each round:

1. **Propose** — each agent encodes its hidden state into buffer space via a per-agent BAPC codec
2. **Resolve** — TIES-Resolve merges competing proposals using sign-election-based cross-inhibition (majority wins at each coordinate; minority is excluded, not diluted)
3. **Update** — the buffer updates by damped fixed-point iteration; convergence is monitored by the relative residual

Slot 5 is structurally immune to cross-inhibition. Carol writes there. Nothing overwrites it.

---

## How it works

```
  Agents (propose)                Buffer B ∈ ℝ^(S×D)         Agents (read)
  ┌─────────────┐                 ┌──────────────┐            ┌─────────────┐
  │ Neutral     │ ──── E_i ────►  │ factual      │ ◄── Ê_i ── │ Neutral     │
  │ Supportive  │ ──── E_i ────►  │ reasoning    │ ◄── Ê_i ── │ Supportive  │
  │ Critical    │ ──── E_i ────►  │ uncertainty  │ ◄── Ê_i ── │ Critical    │
  │ Carol       │ ──── E_i ────►  │ meta         │ ◄── Ê_i ── │ Carol       │
  └─────────────┘                 │ competition  │            └─────────────┘
                                  │ dissenter 🔒 │
                                  └──────┬───────┘
                                         │ {δ_i}
                                  ┌──────▼───────┐
                                  │ TIES-Resolve │
                                  └──────┬───────┘
                                         │ Δ
                                  ┌──────▼────────────────┐
                                  │ r_t = ‖ΔB‖/‖B‖ < τ   │
                                  │ halt or continue       │
                                  └───────────────────────┘
```

**BAPC codec** — encoder/decoder trained via KL distillation rather than cosine reconstruction. Preserves the model's output distribution, not just geometric similarity.

**Isotropy calibration (ABTT)** — removes the shared cone direction that makes upper-layer transformer representations cluster near a single centroid. Without this, 0.998 cosine similarity is a meaningless number. Random word pairs already score 0.985 in GPT-2's final layer.

**TIES-Resolve** — trim low-magnitude components, elect signs by majority, average only sign-agreeing proposals. Cross-inhibition at the representation level, not through textual argument.

**Damped iteration** — `B_{t+1} = (1-α)B_t + α(B_t + Δ_t)`, α=0.5. Contraction guaranteed by the Banach fixed-point theorem.

**Carol's protected slot** — Slot 5 is immune to cross-inhibition. The dissenter's contribution cannot be suppressed regardless of how much everyone else agrees. This is structural immunity, not a strongly-worded system prompt.

---

## The anisotropy problem

The first version of Phase 2 looked good on paper. Codec roundtrip cosine similarity of 0.998. I was quite pleased with this. Then I noticed the decoded tokens were function words — "the", commas, periods — with near-flat probability distributions.

This is a known failure mode. In GPT-2's upper layers, all hidden states occupy a narrow cone in embedding space. Arbitrary random word pairs already share cosine similarity above 0.95. A codec achieving 0.998 is preserving the shared cone direction, a trivial accomplishment, while destroying the small angular residuals that carry all the task-specific information. I had built a very precise instrument for measuring something that does not matter.

The fix is All-But-the-Top (ABTT) calibration: remove the mean and the top principal components before encoding. After the fix, pairwise cosine similarity between random word pairs drops from 0.985 to below 0.15. The maximum decoded token probability jumps from 10.7% to 63.7%.

The paper covers this in detail, including the progression of encoding strategies that led to the fix.

---

## Results

Emergence confirmed at GPT-2 scale and 7B scale. On all five test prompts, the collective buffer state decodes to tokens absent from every individual agent's top-10 distribution. The collective sits at cosine similarity 0.61–0.75 from all individuals — well below identity.

| Prompt | Rounds | Residual | Max sim to any individual | Novel tokens | JS divergence |
|--------|--------|----------|--------------------------|--------------|---------------|
| Remote work | 25 | 0.061 | 0.609 | 4 | 0.111 |
| AI regulation | 25 | 0.092 | 0.630 | 4 | 0.124 |
| Economy | 25 | 0.043 | 0.753 | 2 | 0.080 |
| Profit vs sustainability | 25 | 0.079 | 0.651 | 3 | 0.115 |
| Rejected discoveries | 25 | 0.066 | 0.675 | 4 | 0.128 |

Reranking quality is mixed. The collective shows a clear advantage on high-tension judgment prompts (89.5th percentile vs 65.8th on prompt 4) but only a marginal overall advantage (+1.1pp). The paper is honest about this.

---

## Quick start

```bash
git clone https://github.com/JeremySNR/Pluribus-P2.git
cd Pluribus-P2
pip install -e quorum-phase2/
```

Run the anisotropy diagnostic first — it takes minutes and tells you immediately whether the codec problem exists for your model:

```bash
python quorum-phase2/experiments/anisotropy_baseline.py
```

If average pairwise cosine similarity in the final layer is above 0.95, the codec needs ABTT calibration. It almost certainly will be.

### Core experiments

```bash
# Codec roundtrip fidelity (with and without ABTT)
python quorum-phase2/experiments/codec_roundtrip.py

# Emergence at GPT-2 scale
python quorum-phase2/experiments/emergence_experiment.py

# Cross-model alignment, noise robustness, anchor scaling, dimensionality sensitivity
python quorum-phase2/experiments/advanced_experiments.py
```

All results write to `quorum-phase2/experiments/results/`. Committed results cover every experiment reported in the paper.

### 7B experiments (GPU required)

Tested on an A10G (24GB). Codec training takes roughly 75 minutes per model:

```bash
python quorum-phase2/experiments/bapc_7b_training.py
python quorum-phase2/experiments/emergence_7b.py
python quorum-phase2/experiments/reranking_quality_7b.py
```

---

## Project structure

```
quorum-phase2/
├── quorum/                      # Core library
│   ├── buffer.py                # Shared latent buffer B ∈ ℝ^(S×D)
│   ├── codecs.py                # Per-agent encoder/decoder MLPs
│   ├── bapc_codec.py            # BAPC codec with ABTT isotropy calibration
│   ├── bapc_training.py         # KL distillation training loop
│   ├── isocal.py                # All-But-the-Top calibration
│   ├── ties_resolve.py          # Trim → sign-elect → disjoint merge
│   ├── cross_inhibition.py      # Hopfield energy + bee-inspired stop signals
│   ├── contraction.py           # Damped iteration and spectral norm enforcement
│   ├── convergence.py           # Residual monitoring and halting criteria
│   ├── dissenter.py             # Carol's protected channel
│   └── loop.py                  # Main latent_resonance_loop()
├── experiments/
│   ├── anisotropy_baseline.py   # Diagnose the cone problem before anything else
│   ├── codec_roundtrip.py       # Fidelity with/without calibration
│   ├── emergence_experiment.py  # GPT-2 scale emergence
│   ├── emergence_7b.py          # 7B scale emergence (GPU)
│   ├── advanced_experiments.py  # Cross-model, noise, scaling, dimensionality
│   ├── reranking_quality_7b.py  # Candidate selection quality (GPU)
│   └── results/                 # All committed experiment outputs
├── tests/                       # 142 tests
├── docs/
│   └── latent_resonance_loop_spec.md  # Full technical specification
└── pyproject.toml
```

---

## Tests

```bash
pip install -e "quorum-phase2/[dev]"
pytest quorum-phase2/tests/ -v
```

142 tests covering buffer operations, codec training, TIES-Resolve, cross-inhibition, contraction, convergence, and integration.

---

## Paper

*Don't Merge Carol: Cross-Inhibition, Protected Dissent, and Emergent Latent Consensus in Multi-Agent Language Systems* — full paper in [`paper/`](paper/). LaTeX source and diagram included. Covers both phases and is honest about what is and isn't established.

---

## Phase 1

The text-level blackboard hive that preceded this: [github.com/JeremySNR/Pluribus](https://github.com/JeremySNR/Pluribus)

---

## License

[MIT](LICENSE)
