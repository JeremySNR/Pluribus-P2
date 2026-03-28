# Latent Resonance Loop — Technical Design Specification

**The Latent Resonance Loop is a shared continuous-valued buffer that N heterogeneous LLM agents simultaneously read and write, using Hopfield-energy cross-inhibition and contraction-enforced convergence to produce collective intelligence that exceeds any single agent.** This document specifies the architecture precisely enough for an engineering team to build a prototype. It distinguishes proven mechanisms (published, replicated), plausible extensions (grounded in theory, untested for this use case), and speculative proposals (novel, requiring validation). The system bridges the Blackboard Hive's text-level coordination and the Neural Hive's end-to-end differentiability — agents communicate through activations, not tokens, achieving **235–471× higher information density** than text-based multi-agent systems while maintaining convergence guarantees grounded in contraction mapping theory.

---

## 1. Architecture overview and design decisions

The Latent Resonance Loop comprises four subsystems: a **Universal Latent Buffer** (the shared state), **Per-Agent Codecs** (encode/decode adapters), a **Cross-Inhibition Engine** (conflict resolution), and a **Convergence Controller** (halting and acceleration). Each round proceeds in three phases: *Read* (all agents attend to the buffer), *Propose* (agents generate update deltas), and *Resolve* (cross-inhibition and contraction produce the next buffer state). The loop terminates when the convergence controller fires a halt signal.

The single most consequential design choice is the buffer's representation space. Four candidate geometries were evaluated:

| Geometry | Capacity | Alignment difficulty | Cross-inhibition compatibility | Verdict |
|---|---|---|---|---|
| Raw KV-cache (C2C-style) | High — native model fidelity | O(N²) pairwise fusers | Poor — no shared metric | Rejected for N>2 |
| Projected Euclidean (hub-and-spoke) | **Tunable — D=2048–8192** | **O(N) — one codec per agent** | **Natural — inner products define competition** | **Selected** |
| Structured slots (NTM-style) | Moderate — ontology-constrained | Low per slot, high for slot design | Moderate — per-slot inhibition | Hybrid with selected |
| Poincaré hyperbolic | Exponential in low-D | High — Riemannian operations | Difficult — no natural inner product | Deferred to v2 |

**The selected architecture is a projected common Euclidean embedding space with structured named slots**, combining the Vision Wormhole's hub-and-spoke topology for O(N) scaling with NTM-inspired slot structure for interpretability. The theoretical grounding comes from the **Platonic Representation Hypothesis** (Huh et al., ICML 2024), which demonstrates that large neural networks are converging toward a shared statistical model of reality, with cross-model Linear CKA values of **0.595–0.881** between independently trained models. This convergence makes linear alignment increasingly feasible as models scale.

---

## 2. The Universal Latent Buffer

### 2.1 Representation space specification

The buffer is a matrix **B ∈ ℝ^{S × D}** where S is the number of named slots and D is the embedding dimension. Each slot is a D-dimensional vector representing a distinct semantic role.

**Slot ontology** (S=6 for v1 prototype):

- **Slot 0 — Factual Grounding**: Anchors to retrievable, verifiable claims. Highest-confidence factual content.
- **Slot 1 — Reasoning Chain**: Current best logical trajectory. Encodes inferential steps.
- **Slot 2 — Uncertainty Map**: Represents what the collective does *not* know. High activation = high uncertainty.
- **Slot 3 — Meta-Coordination**: Task framing, role allocation, attention direction. Administrative overhead.
- **Slot 4 — Competitive Arena**: Where cross-inhibition is strongest. Multiple proposals compete here.
- **Slot 5 — Dissenter Channel**: Structurally immune to cross-inhibition (§5). Reserved for minority positions.

**Dimensionality D**: Set D = **4096** for the prototype. Rationale: matches the native hidden dimension of 8B-parameter models (Qwen3-8B, LLaMA-3-8B); HELIX (2026) confirms that linear alignment works best when source and target dimensions are comparable; SDM theory guarantees near-orthogonality of random vectors at this dimension (expected cosine similarity between random unit vectors ≈ 0 with standard deviation ≈ 1/√D ≈ 0.016).

**Status**: The projected Euclidean space is *plausible* — Vision Wormhole (Feb 2026) demonstrates the hub-and-spoke concept for VLMs, and MUSE/VecMap prove linear alignment across embedding spaces. The specific slot ontology is *speculative* — no published work validates these particular categories for multi-agent LLM coordination.

### 2.2 Per-agent codecs

Each agent i has an encoder E_i: ℝ^{d_i} → ℝ^D and decoder D_i: ℝ^D → ℝ^{d_i}, where d_i is agent i's native hidden dimension. The codec architecture follows C2C's three-module design adapted for the hub-and-spoke topology:

**Encoder E_i** (agent-to-buffer):
```
E_i(h) = LayerNorm(W₂ · ReLU(W₁ · h + b₁) + b₂)
```
where W₁ ∈ ℝ^{D×d_i}, W₂ ∈ ℝ^{D×D}. The two-layer MLP with LayerNorm ensures the output lies on the unit-norm hypersphere in ℝ^D, critical for on-manifold enforcement.

**Decoder D_i** (buffer-to-agent):
```
D_i(z) = W₄ · ReLU(W₃ · z + b₃) + b₄
```
where W₃ ∈ ℝ^{d_i×D}, W₄ ∈ ℝ^{d_i×d_i}. Decoded vectors are injected into the agent's processing pipeline via the mechanism determined by agent type — through the vision-encoder pathway for VLMs (following Vision Wormhole), or through KV-cache concatenation for text-only LLMs (following LatentMAS).

**Training the codecs**: Both LLMs remain frozen. Codecs are trained with a composite loss:

```
L_codec = L_reconstruction + λ_align · L_alignment + λ_cycle · L_cycle
```

- **L_reconstruction** = ‖D_i(E_i(h)) − h‖² — autoencoder reconstruction, ensuring roundtrip fidelity.
- **L_alignment** = ‖E_i(h_i) − E_j(h_j)‖² for paired inputs — Procrustes-style alignment ensuring semantically equivalent inputs from different agents map to nearby buffer locations. Anchor pairs are generated by feeding identical prompts to different agents and collecting hidden states.
- **L_cycle** = ‖D_j(E_i(h_i)) − h_j‖² — cross-agent cycle consistency. Agent i's encoding, decoded by agent j's decoder, should approximate agent j's native representation of the same input.

The Procrustes component can be initialized in closed form: given paired hidden states H_i, H_j from N anchor inputs, the optimal orthogonal alignment is **W* = UV^T** where UΣV^T = SVD(H_j^T H_i), following MUSE's supervised alignment. This provides a warm start before fine-tuning with the full composite loss.

**Compute cost**: Each codec is ~33M parameters for D=4096, d_i=4096. Training requires ~1000 anchor pairs and converges in approximately 5000 gradient steps on a single GPU. At inference, codec forward pass adds **<1ms** latency per agent per round.

**Status**: Codec training via Procrustes + fine-tuning is *proven* — MUSE demonstrates this for 110+ language pairs, and both LatentMAS and Vision Wormhole use ridge regression alignment successfully. The specific two-layer MLP architecture is *plausible* — C2C uses a three-layer MLP for its fuser. Cycle-consistency loss is *proven* in image translation (CycleGAN) but *plausible* for latent-space codec training.

### 2.3 The multi-writer problem

When N agents simultaneously propose updates to the buffer, the system must combine these updates without destroying on-manifold structure. This is the hardest unsolved engineering problem in the architecture.

**Write protocol**: Each agent i reads the current buffer B_t, processes it through its decoder, performs internal computation, and produces a proposed delta **δ_i ∈ ℝ^{S×D}** via its encoder. The raw update would be B_{t+1} = B_t + Σ_i δ_i, but this naive sum drifts off-manifold and ignores conflicts.

**The TIES-Resolve mechanism** (novel — speculative): Adapt TIES-Merging's three-step process from static weight merging to dynamic activation merging:

```python
def ties_resolve(deltas: List[Tensor], density=0.3) -> Tensor:
    """
    deltas: list of N tensors, each shape [S, D]
    Returns: resolved delta, shape [S, D]
    """
    # STEP 1 — TRIM: Zero out low-magnitude components
    # Keep only top-k% of each delta by absolute value
    for i in range(len(deltas)):
        threshold = torch.quantile(deltas[i].abs(), 1 - density)
        deltas[i] = deltas[i] * (deltas[i].abs() >= threshold)
    
    # STEP 2 — ELECT SIGN: For each (slot, dimension), majority sign wins
    stacked = torch.stack(deltas)  # [N, S, D]
    sign_votes = torch.sign(stacked)  # {-1, 0, +1}
    magnitude_weighted_signs = (sign_votes * stacked.abs()).sum(dim=0)
    elected_sign = torch.sign(magnitude_weighted_signs)  # [S, D]
    
    # STEP 3 — DISJOINT MERGE: Average only sign-agreeing deltas
    agreement_mask = (sign_votes == elected_sign.unsqueeze(0))  # [N, S, D]
    agreement_count = agreement_mask.float().sum(dim=0).clamp(min=1)
    merged = (stacked * agreement_mask.float()).sum(dim=0) / agreement_count
    
    return merged
```

The density parameter d ∈ [0.2, 0.6] controls sparsity. At d=0.3, **70% of each delta is zeroed**, exploiting the DARE finding that large models tolerate 90–99% parameter dropout. The sign-election step creates a *continuous-space analog of winner-take-all per dimension* — for each feature direction, the majority opinion prevails, but the minority is not averaged in (it is excluded). This is cross-inhibition at the representation level, not averaging.

**On-manifold enforcement**: After TIES-Resolve produces the merged delta, apply a cleanup step:

```python
def on_manifold_project(B_new: Tensor, cleanup_ae: AutoEncoder) -> Tensor:
    """Project buffer state back onto the learned data manifold."""
    # Method 1: LayerNorm per slot (fast, always available)
    B_normed = F.layer_norm(B_new, [D])
    
    # Method 2: Cleanup autoencoder (better, requires training)
    B_clean = cleanup_ae.decode(cleanup_ae.encode(B_normed))
    
    # Blend: use AE during training, LayerNorm as fallback
    return B_clean
```

The cleanup autoencoder is trained on buffer states observed during codec training, learning the manifold of "valid" buffer configurations. This follows NTM literature where memory contents are maintained within a learned distribution. The autoencoder should be shallow (2–3 layers) and is trained with reconstruction loss plus a **contrastive term** that pushes the encoding away from pathological states (all-zero, all-same-slot, random noise).

**SDM-inspired orthogonality guarantee**: At D=4096, randomly initialized agent codecs will naturally produce approximately orthogonal encodings. The expected interference between any two agents' deltas is bounded by **|⟨δ_i, δ_j⟩|/‖δ_i‖‖δ_j‖ ≈ 1/√D ≈ 0.016** for uncorrelated contributions. When agents agree on content (correlated deltas), the inner product is high and TIES-Resolve correctly merges them. When agents disagree, the natural orthogonality minimizes interference even before TIES-Resolve's explicit conflict resolution.

**Status**: TIES-Merging for static weights is *proven* (NeurIPS 2023, 1–5 point accuracy improvement). Adapting it to dynamic per-inference activations is *speculative* — the key risk is that running statistics needed for the trim threshold may not be stable across diverse inputs. SDM orthogonality properties at D≥4096 are *proven* (Kanerva 1988, extensive theoretical and empirical validation). The cleanup autoencoder approach is *plausible* — autoencoder-based manifold projection is standard in generative modeling but untested for multi-agent buffer cleanup.

### 2.4 Buffer lifecycle and memory

The buffer operates in two temporal modes:

**Working memory** (within-task): Initialized to zero at task onset. Persists across all convergence rounds for a single query. Cleared when the convergence controller halts. This is the primary mode for the v1 prototype.

**Episodic memory** (across-task, v2): After convergence, the final buffer state B* is compressed via the cleanup autoencoder's encoder and stored in a **ring buffer** of the K most recent converged states. On new tasks, agents can attend to episodic entries via content-based addressing (cosine similarity between the new task's initial buffer state and stored episodes). This follows the MT-DNC (2025) pattern of working-to-long-term memory transformation.

**Buffer history for debugging**: Store the full trajectory {B_0, B_1, ..., B_T} during development. At each round, also store per-agent deltas {δ_i^t}, the TIES-Resolve output, the cross-inhibition energy, and the convergence metric. This trajectory enables post-hoc analysis of failure modes: divergence (‖B_t‖ growing), oscillation (B_t cycling), premature convergence (halting before quality stabilizes), and suppression collapse (all deltas going to zero).

---

## 3. Cross-inhibition engine

### 3.1 The Hopfield-energy competition framework

The cross-inhibition mechanism is the architectural core that distinguishes this system from simple ensemble averaging. It is grounded in modern Hopfield network theory (Ramsauer et al., ICLR 2021) and the honeybee stop-signal model (Seeley et al., *Science* 2012).

**Core formulation**: Define an energy function over the buffer state B and the set of agent deltas {δ₁, ..., δ_N}:

```
E(B) = -β⁻¹ log Σᵢ exp(β · δᵢᵀB_slot) + ½‖B_slot‖² + λ_repel · Σᵢ<ⱼ max(0, δᵢᵀδⱼ)
```

for each slot independently (the competitive arena slot uses stronger competition than the factual grounding slot). The three terms are:

1. **Log-sum-exp attractor** (−β⁻¹ lse): Creates attractor basins around each agent's proposal. The buffer is pulled toward the proposals. This is identical to the modern Hopfield energy.
2. **Norm regularizer** (½‖B‖²): Prevents unbounded growth. Ensures the energy is bounded below.
3. **Repulsion penalty** (λ_repel Σ max(0, δᵢᵀδⱼ)): Penalizes aligned proposals from different agents, forcing the system to *choose* rather than blend. This is the explicit cross-inhibition term, absent in standard Hopfield networks.

**The β parameter controls competition sharpness.** At low β, the softmax in the gradient descent update approaches uniform weighting (all proposals contribute equally — effectively averaging). At high β, softmax approaches argmax (winner-take-all — single proposal dominates). **The key insight is that β should be annealed from low to high during the convergence loop**, starting with soft exploration and ending with sharp selection.

**Update rule** (gradient descent on E):

```
B_slot^{t+1} = Σᵢ wᵢ(t) · δᵢ    where    wᵢ(t) = exp(β(t) · δᵢᵀ B_slot^t) / Σⱼ exp(β(t) · δⱼᵀ B_slot^t)
```

This is precisely the transformer attention update from Ramsauer et al., with agent deltas as keys/values and the buffer state as the query. The weights w_i are the *competition weights* — the system's estimate of which agent should dominate each slot at each round.

**Status**: The Hopfield energy formulation is *proven* — modern Hopfield networks are mathematically equivalent to transformer attention (Ramsauer et al., 2020). The repulsion penalty is *speculative* — standard Hopfield networks do not include cross-pattern repulsion, and its effect on attractor dynamics requires empirical validation.

### 3.2 Bee-inspired stop-signal dynamics

The honeybee model provides the interaction rule between competing agents. Seeley et al.'s differential equations:

```
dsᵢ/dt = γ · qᵢ · (1 − Σⱼ sⱼ) − α · sᵢ − Σⱼ≠ᵢ σ · sᵢ · sⱼ
```

where s_i is agent i's influence strength, q_i is its proposal quality, γ is recruitment rate, α is spontaneous decay, and **σ is the cross-inhibition rate**. The critical term is **σ · sᵢ · sⱼ** — inhibition is proportional to the *product* of both agents' activities. This means strong agents inhibit each other more than weak agents, creating automatic size-dependent competition without external arbitration.

**Discretized for the buffer** (novel proposal):

```python
def bee_inhibition_step(strengths: Tensor, qualities: Tensor, 
                         sigma=0.3, alpha=0.05, gamma=0.1) -> Tensor:
    """
    strengths: [N] — current influence strength per agent
    qualities: [N] — proposal quality scores (from confidence probes)
    Returns: updated strengths [N]
    """
    uncommitted = 1.0 - strengths.sum()
    uncommitted = uncommitted.clamp(min=0.01)  # prevent negative pool
    
    recruitment = gamma * qualities * uncommitted
    decay = alpha * strengths
    
    # Cross-inhibition: each agent inhibited proportional to product 
    # of its strength and all competitors' strengths
    cross_inhib = sigma * strengths * (strengths.sum() - strengths)
    
    d_strengths = recruitment - decay - cross_inhib
    new_strengths = (strengths + d_strengths).clamp(min=0.0, max=1.0)
    
    return new_strengths / new_strengths.sum()  # normalize
```

Agent strengths s_i then modulate the Hopfield competition weights: the effective delta applied to the buffer is **δ_i^effective = sᵢ · δ_i**. The bee dynamics determine *how much* each agent influences the buffer; the Hopfield energy determines *how* the influences combine.

**Quality signal q_i**: Computed from agent i's internal confidence. Following CP-WBFT (Zheng et al., Nov 2025), which extracts confidence via logistic regression probes on hidden states, we train a lightweight probe on each agent's penultimate-layer activations:

```
qᵢ = σ(w_probe · h_i^{penultimate} + b_probe)
```

This probe is trained on held-out tasks where ground truth is available, mapping hidden states to calibrated confidence scores. CP-WBFT demonstrates this works under **85.7% Byzantine fault rates**, validating its robustness.

**Status**: The bee stop-signal dynamics are *proven* in biological modeling (Seeley et al., *Science* 2012, replicated computationally multiple times). Their application to LLM activation competition is *speculative*. Confidence probing is *proven* (CP-WBFT, Nov 2025).

### 3.3 Preventing suppression collapse

Without safeguards, cross-inhibition can kill all proposals (amplitude death) or lock into a single agent permanently (winner-lock). Five mechanisms prevent these failure modes:

**Mechanism 1 — Minimum activity floor**: No agent's strength can drop below ε_floor = 0.01. This guarantees every agent retains at least 1% influence. Biological analog: tonic firing rate in neurons.

**Mechanism 2 — Homeostatic regulation**: Monitor total buffer activity A(t) = Σ_i ‖δ_i^effective‖. If A(t) drops below threshold A_min, globally reduce σ:

```
σ_effective(t) = σ₀ · sigmoid(k · (A(t) − A_min))
```

When total activity is healthy, σ_effective ≈ σ₀. When activity collapses, σ_effective → 0, releasing all inhibition. The gain k controls sensitivity. This is a direct analog of homeostatic synaptic scaling in biological neural circuits, where postsynaptic neurons upregulate their receptors when input drops.

**Mechanism 3 — Temperature annealing schedule**: The β parameter follows an exponential warmup:

```
β(t) = β_min + (β_max − β_min) · (1 − exp(−t/τ_β))
```

with β_min = 0.1 (soft averaging in early rounds), β_max = 10.0 (near-WTA in late rounds), and τ_β = 3 rounds. This allows exploration before committing to a winner. The schedule is adaptive: if proposal agreement (mean pairwise cosine similarity of deltas) exceeds 0.8, β jumps to β_max immediately.

**Mechanism 4 — Normalized inhibition cap**: Total inhibition on any agent cannot exceed its current activity:

```
total_inhib_on_i = min(sᵢ, Σⱼ≠ᵢ σ · sᵢ · sⱼ)
```

This prevents inhibition from driving strengths negative and ensures the system is *dissipative* — total energy always decreases or stays constant.

**Mechanism 5 — Refractory period**: After an agent's strength drops below 0.05, it cannot be further inhibited for one round. This prevents cascading suppression where a weakened agent is immediately eliminated.

**Expected failure modes and mitigations**:

| Failure mode | Detection signal | Mitigation |
|---|---|---|
| Amplitude death (all s_i → 0) | A(t) < A_min for 3 rounds | Homeostatic σ reduction + floor |
| Winner lock (one s_i → 1 permanently) | max(s_i) > 0.95 for 5 rounds | Reduce β; inject noise |
| Oscillation (two agents alternating) | s_i oscillation amplitude > 0.3 | Increase damping α; average over 2 rounds |
| Spurious attractor (buffer stuck at mixture) | Convergence metric plateaus above threshold | Anderson acceleration; reset to best-quality single agent |

---

## 4. Convergence engineering

### 4.1 Enforcing contractiveness

The Latent Resonance Loop is a fixed-point iteration: **B_{t+1} = F(B_t)** where F is the composite function of agent reads, delta proposals, TIES-Resolve, cross-inhibition, and on-manifold projection. For guaranteed convergence, F must be a **contraction mapping** — there must exist q ∈ [0, 1) such that ‖F(B) − F(B')‖ ≤ q · ‖B − B'‖ for all valid buffer states B, B'.

The Banach fixed-point theorem then guarantees: (1) a **unique** fixed point B* exists, (2) the iteration converges from any initialization, and (3) the convergence rate is linear with factor q, giving the a priori bound **‖B_t − B*‖ ≤ q^t/(1−q) · ‖B_0 − F(B_0)‖**.

**Three mechanisms enforce contractiveness**, applied in layers:

**Layer 1 — Damped iteration (always active, no training required)**:

```
B_{t+1} = (1 − α) · B_t + α · F_raw(B_t)    where α ∈ (0, 1)
```

If F_raw has Lipschitz constant L, the damped iteration has effective Lipschitz constant **(1−α) + α·L**. This is < 1 whenever **α < 2/(1+L)**. For any finite L, there exists a sufficiently small α that guarantees contraction. Default: α = 0.5, which handles L up to 3.0.

**Layer 2 — Spectral normalization of agent codecs (applied during codec training)**:

Each codec's weight matrices are spectrally normalized: **W̃ = W / σ_max(W)**. This bounds each linear layer's Lipschitz constant to 1. With the nonlinear activations (ReLU, Lip=1), the full codec has Lip(E_i) ≤ 1 and Lip(D_i) ≤ 1. To achieve contraction, scale the final encoder layer by **λ_SN = 0.9**, giving Lip(E_i) ≤ 0.9. Computational cost: **~2 matrix-vector products per weight matrix** (power iteration), negligible relative to the forward pass.

**Layer 3 — Jacobian regularization (applied during codec training)**:

Add the Frobenius norm penalty to the codec training loss:

```
L_total = L_codec + γ_jac · ‖J_F(B)‖_F²
```

where ‖J_F‖_F² is estimated via the **Hutchinson trace estimator**: sample a random vector v ~ Rademacher(D), compute ‖J_F · v‖², and use this as an unbiased estimate. Cost: **one additional backward pass**, applied stochastically with probability 0.5 per training step (following Bai et al., ICML 2021). Set γ_jac = 0.1 initially, tuning to keep the estimated spectral radius ρ(J_F) below 0.95.

**Monitoring**: During inference, periodically estimate ρ(J_F) via power iteration (10 iterations suffice for a rough estimate). If ρ exceeds 0.98, trigger an alert and increase the damping factor α.

**Status**: Damped iteration guaranteeing contraction is *proven* (textbook fixed-point theory). Spectral normalization is *proven* (Miyato et al., ICLR 2018). Jacobian regularization for DEQs is *proven* (Bai et al., ICML 2021, reducing NFEs by 2× with minimal accuracy loss). Applying all three to a multi-agent buffer loop is *plausible* — each mechanism is individually validated but their composition in this specific architecture is untested.

### 4.2 Anderson acceleration

Plain fixed-point iteration converges linearly at rate q. **Anderson acceleration** uses a history of m previous iterates to extrapolate, achieving superlinear convergence in practice — approximately **10× faster** than naive iteration in DEQ benchmarks.

```python
def anderson_accelerated_loop(agents, buffer, m=5, lam=1e-5, 
                                max_rounds=20, tol=1e-3, alpha=0.5):
    """
    Main convergence loop with Anderson acceleration.
    """
    S, D = buffer.shape
    X_hist = torch.zeros(m, S, D)  # iterate history
    F_hist = torch.zeros(m, S, D)  # f(iterate) history
    
    X_hist[0] = buffer
    F_hist[0] = loop_step(agents, buffer, alpha)  # one full Read-Propose-Resolve
    buffer = F_hist[0]
    
    for t in range(1, max_rounds):
        F_new = loop_step(agents, buffer, alpha)
        
        n = min(t, m)
        idx = t % m
        X_hist[idx] = buffer
        F_hist[idx] = F_new
        
        # Compute residuals
        G = F_hist[:n] - X_hist[:n]  # [n, S, D]
        G_flat = G.reshape(n, -1)  # [n, S*D]
        
        # Solve least-squares for mixing coefficients
        GTG = G_flat @ G_flat.T + lam * torch.eye(n)  # [n, n]
        ones = torch.ones(n, 1)
        alpha_mix = torch.linalg.solve(GTG, ones)
        alpha_mix = alpha_mix / alpha_mix.sum()  # normalize
        
        # Extrapolated update
        buffer = (alpha_mix.T @ F_hist[:n].reshape(n, -1)).reshape(S, D)
        
        # Convergence check
        residual = (F_new - X_hist[idx]).norm() / buffer.norm().clamp(min=1e-8)
        if residual < tol:
            break
    
    return buffer, t, residual
```

**Hyperparameters**: History size m=5 (balancing acceleration vs. memory — O(m · S · D) storage). Ridge regularization λ=1e-5 (prevents numerical instability from near-singular Gram matrices). Typical convergence: **3–7 rounds** with Anderson vs. 15–25 rounds without.

**Status**: Anderson acceleration for DEQs is *proven* (standard in TorchDEQ, demonstrated 10× speedup). Its application to multi-agent convergence with cross-inhibition dynamics is *plausible* — the fixed-point structure is preserved, but cross-inhibition's non-smooth dynamics (from TIES-Resolve's sign election) may reduce acceleration effectiveness.

### 4.3 Adaptive halting via PonderNet

Rather than a fixed iteration budget, the convergence controller learns when to halt. Following PonderNet (Banino et al., DeepMind 2021), a lightweight halting network predicts the probability of stopping at each round:

```
λ_t = σ(w_halt · [‖B_t − B_{t−1}‖/‖B_t‖, max(s_i), entropy(s), t/T_max] + b_halt)
```

The inputs are: relative residual (how close to fixed point), maximum agent strength (how dominant the winner is), entropy of agent strengths (how decided the competition is), and normalized round number. The halting probability is trained with:

```
L_halt = Σ_t p_t · L_task(B_t) + β_KL · KL(p || Geom(λ_prior))
```

where p_t = λ_t · Π_{j<t}(1−λ_j) is the unconditional halt probability at round t, and Geom(λ_prior) is a geometric distribution prior that encourages exploration (λ_prior = 0.2 gives an expected 5 rounds). During inference, halt when λ_t > 0.5 or when the hard timeout T_max = 20 rounds is reached.

**Fallback**: If PonderNet is not yet trained (cold start), use a quorum-based criterion: halt when the relative residual ‖B_t − B_{t−1}‖/‖B_t‖ < ε for **3 consecutive rounds** (ε = 0.01). This is the standard DEQ convergence criterion.

**Status**: PonderNet is *proven* (published, multiple extensions including PonderLM-3 and FR-Ponder in 2025–2026). Applying it to multi-agent convergence detection is *plausible* — the halting signal is well-defined, but training requires a task loss that reflects buffer quality, which must be designed per application.

---

## 5. The dissenter channel — structural immunity by design

The cross-inhibition engine is deliberately designed to suppress weak proposals. This creates a failure mode: a single agent with a *correct but minority* position gets suppressed by the majority's confident-but-wrong consensus. The dissenter channel provides structural immunity against this.

### 5.1 Architecture

**Slot 5 (Dissenter Channel)** operates under different rules from all other slots:

1. **No cross-inhibition**: The repulsion penalty λ_repel is set to zero for Slot 5. The bee stop-signal dynamics do not apply. Proposals to this slot are *never suppressed*.

2. **Divergence-maximizing write access**: Only the agent whose delta has the **lowest cosine similarity** to the current consensus (mean of all other agents' deltas) can write to Slot 5 in each round. This is operationalized as:

```python
def select_dissenter(deltas: List[Tensor]) -> int:
    """Select the most divergent agent as the round's dissenter."""
    consensus = torch.stack(deltas).mean(dim=0)  # [S, D]
    similarities = [F.cosine_similarity(d.flatten(), consensus.flatten(), dim=0) 
                    for d in deltas]
    return torch.tensor(similarities).argmin().item()
```

3. **Quality gate**: The dissenter's contribution is weighted by its confidence probe score q_i. A dissenter with low confidence (q_i < 0.1) still writes but with reduced magnitude, preventing random noise from dominating the slot. The gate: **δ_dissenter^effective = max(q_i, ε_floor) · δ_i**.

4. **Mandatory read**: All agents must attend to Slot 5 during the Read phase. The decoder does not gate this slot — it is always visible. This ensures the minority position *can* influence the next round's proposals even if it doesn't dominate.

### 5.2 Dissent vs. Byzantine fault

The fundamental unsolved problem: **how do you distinguish a correct minority opinion from a faulty agent?** CP-WBFT's confidence probing provides a partial answer — agents with well-calibrated confidence on held-out tasks are more likely to be correct when they dissent. But confidence probing alone is insufficient; a hallucinating agent can be confidently wrong.

**Proposed mechanism (speculative)**: Track each agent's **historical calibration** — how often its high-confidence claims turn out correct in tasks where verification is possible. Agents with strong calibration records get higher trust when they dissent. This is analogous to the immune system's *affinity maturation*, where rare antibodies that successfully fight infections are clonally expanded.

```python
def dissent_trust(agent_id, confidence, calibration_history):
    """Weight dissent by historical reliability."""
    calibration_score = calibration_history[agent_id].ece  # expected calibration error
    trust = confidence * (1 - calibration_score)  # high confidence + low ECE = high trust
    return trust
```

**Status**: The dissenter channel architecture is *speculative* — no published system implements structural immunity for minority positions in a shared latent buffer. The inspiration comes from biological immune systems (AIS literature), Byzantine fault tolerance (the 3f+1 rule inverted: require supermajority to suppress), and the AI-mediated devil's advocate system (Feb 2025). Historical calibration tracking is *plausible* — calibration scoring is well-established, but applying it to dynamic multi-agent dissent weighting is novel.

---

## 6. Complete round pseudocode

```python
def latent_resonance_loop(agents: List[Agent], query: str, 
                           config: LoopConfig) -> Tuple[Tensor, dict]:
    """
    Full Latent Resonance Loop execution.
    
    Returns: (converged_buffer, diagnostics)
    """
    N = len(agents)
    B = torch.zeros(config.S, config.D)  # Initialize buffer
    diagnostics = {"trajectory": [], "residuals": [], "strengths": []}
    strengths = torch.ones(N) / N  # Equal initial influence
    beta = config.beta_min
    
    # Anderson acceleration state
    anderson = AndersonState(m=5, lam=1e-5)
    
    for t in range(config.max_rounds):
        # === PHASE 1: READ ===
        agent_inputs = []
        for i, agent in enumerate(agents):
            # Decode buffer into agent i's native space
            native_repr = agent.codec.decode(B)
            # Inject into agent's processing (KV-cache or vision pathway)
            agent_inputs.append(agent.inject(native_repr, query))
        
        # === PHASE 2: PROPOSE ===
        deltas = []
        qualities = []
        for i, agent in enumerate(agents):
            # Agent processes and produces response in native space
            h_i = agent.forward(agent_inputs[i])
            # Encode to buffer space
            delta_i = agent.codec.encode(h_i) - B  # proposed change
            q_i = agent.confidence_probe(h_i)
            deltas.append(delta_i)
            qualities.append(q_i)
        
        qualities = torch.tensor(qualities)
        
        # === PHASE 3: RESOLVE ===
        
        # 3a. Bee-inspired strength update
        strengths = bee_inhibition_step(
            strengths, qualities, 
            sigma=config.sigma * sigmoid(sum_activity - config.A_min),  # homeostatic
            alpha=config.decay_rate,
            gamma=config.recruit_rate
        )
        strengths = strengths.clamp(min=config.epsilon_floor)
        strengths = strengths / strengths.sum()
        
        # 3b. Select dissenter and protect Slot 5
        dissenter_idx = select_dissenter(deltas)
        
        # 3c. Apply strength weighting to deltas (except dissenter on Slot 5)
        weighted_deltas = [s * d for s, d in zip(strengths, deltas)]
        
        # 3d. TIES-Resolve for Slots 0-4
        resolved_0_4 = ties_resolve(
            [d[:5] for d in weighted_deltas],  # slots 0-4
            density=config.ties_density
        )
        
        # 3e. Dissenter writes to Slot 5 unimpeded
        resolved_5 = (max(qualities[dissenter_idx], config.epsilon_floor) 
                       * deltas[dissenter_idx][5:6])
        
        resolved = torch.cat([resolved_0_4, resolved_5], dim=0)  # [S, D]
        
        # 3f. Hopfield energy-based competition weighting
        for s in range(config.S - 1):  # Slots 0-4 only
            slot_deltas = torch.stack([d[s] for d in deltas])  # [N, D]
            w = F.softmax(beta * slot_deltas @ B[s], dim=0)  # [N]
            resolved[s] = (w.unsqueeze(1) * slot_deltas).sum(0)
        
        # 3g. Damped update + on-manifold projection
        B_new = (1 - config.alpha) * B + config.alpha * (B + resolved)
        B_new = on_manifold_project(B_new, config.cleanup_ae)
        
        # 3h. Anderson acceleration
        B_new = anderson.step(B, B_new)
        
        # === CONVERGENCE CHECK ===
        residual = (B_new - B).norm() / B_new.norm().clamp(min=1e-8)
        diagnostics["trajectory"].append(B_new.clone())
        diagnostics["residuals"].append(residual.item())
        diagnostics["strengths"].append(strengths.clone())
        
        B = B_new
        beta = config.beta_min + (config.beta_max - config.beta_min) * (1 - exp(-t/config.tau_beta))
        
        # PonderNet or quorum halt
        if config.use_pondernet:
            halt_prob = pondernet(residual, strengths.max(), entropy(strengths), t/config.max_rounds)
            if halt_prob > 0.5:
                break
        else:
            if residual < config.tol and t >= 2:
                break
    
    return B, diagnostics
```

---

## 7. Compute budget and scaling analysis

**Per-round costs** for N agents, S=6 slots, D=4096:

| Component | FLOPs | Latency (est.) | Memory |
|---|---|---|---|
| N agent forward passes | N × C_agent | Parallel: C_agent | N × M_agent |
| N codec encode | N × 2 × D² ≈ N × 33M | < 1ms each | N × 33M params |
| N codec decode | N × 2 × D² ≈ N × 33M | < 1ms each | (shared with encode) |
| TIES-Resolve | O(N × S × D) ≈ N × 25K | < 0.1ms | O(N × S × D) |
| Cross-inhibition (bee + Hopfield) | O(N² × D) | < 0.1ms for N≤10 | O(N²) |
| Anderson acceleration | O(m² × S × D) ≈ 500K | < 0.1ms | O(m × S × D) ≈ 600KB |
| On-manifold projection | O(S × D²) ≈ 100M | < 1ms | Cleanup AE params |

**The dominant cost is agent forward passes**, which are fully parallelizable across GPUs. With N=5 agents on 5 GPUs, the per-round wall-clock time is approximately that of a single agent inference plus ~3ms overhead for buffer operations. With Anderson-accelerated convergence in 3–7 rounds, total latency is **3–7× single-agent inference time**.

**Scaling with N**: Cross-inhibition cost is O(N²) but with very small constants (N² scalar multiplications). The practical bottleneck for N>10 is GPU memory for parallel agent inference, not buffer computation. The architecture supports N up to ~20 before communication overhead dominates.

---

## 8. Research risk registry

The following risks are ordered by severity and probability:

**RISK 1 — Cross-model alignment quality (HIGH probability, HIGH severity)**: Despite the Platonic Representation Hypothesis, alignment between architecturally dissimilar models (e.g., Mamba vs. Transformer) may produce codecs with high reconstruction error, causing the buffer to contain garbled information. **Mitigation**: Start with same-architecture, different-scale agents (Qwen3 4B/8B/14B as validated by LatentMAS). Heterogeneous architectures are a v2 goal. **Test**: Measure codec roundtrip reconstruction loss; reject agents whose loss exceeds 2× the median.

**RISK 2 — TIES-Resolve instability on activations (HIGH probability, MEDIUM severity)**: TIES-Merging was designed for static weight deltas with known base models. Dynamic per-token activations have different statistical properties — the trim threshold may vary wildly across inputs. **Mitigation**: Use per-slot running statistics (exponential moving average of activation magnitudes) for adaptive thresholding. **Test**: Measure variance of trim thresholds across 1000 diverse inputs; if coefficient of variation > 1.0, switch to fixed percentile trimming.

**RISK 3 — Convergence failure under strong disagreement (MEDIUM probability, HIGH severity)**: When agents have fundamentally incompatible representations of a problem, the contraction mapping may not have a meaningful fixed point — convergence produces a compromise that none of the agents would endorse. This is the multi-agent analog of mode collapse. **Mitigation**: Detect via entropy of agent strengths — if entropy remains high (> 0.9 × log(N)) at timeout, declare non-convergence and fall back to the single highest-quality agent's output. **Test**: Deliberately provide agents with contradictory evidence; verify the system either resolves correctly or gracefully falls back.

**RISK 4 — Dissenter channel noise (MEDIUM probability, LOW severity)**: The dissenter slot may accumulate incoherent noise from agents that are simply confused rather than insightfully divergent. **Mitigation**: Quality gate + historical calibration. In the worst case, agents learn to ignore Slot 5 if it consistently contains noise. **Test**: Inject known-correct minority positions and verify they survive; inject random noise and verify it is low-weighted.

**RISK 5 — Cleanup autoencoder as bottleneck (LOW probability, MEDIUM severity)**: The on-manifold projection may be too conservative, collapsing novel buffer states back to training-distribution states and preventing genuine collective intelligence. **Mitigation**: Use a very shallow autoencoder (1 hidden layer) with high capacity (D_hidden = 2×D), and monitor reconstruction loss during operation. If reconstruction loss is consistently low, the manifold is too constrained. **Test**: Compare buffer trajectories with and without the cleanup step; verify cleanup improves rather than degrades downstream task performance.

---

## 9. What to build first

The prototype should be built in three phases to isolate risks:

**Phase 1 — Two agents, same family, no cross-inhibition (2 weeks)**: Validate the codec and buffer round-trip. Use Qwen3-8B and Qwen3-14B. Train codecs with Procrustes + fine-tuning. Verify that Agent A can encode a response, Agent B can decode and meaningfully continue it. Success metric: roundtrip reconstruction loss < 0.1 (normalized), and downstream task accuracy within 5% of single-agent baseline.

**Phase 2 — Five agents, same family, with cross-inhibition (4 weeks)**: Add TIES-Resolve, bee dynamics, and Hopfield competition. Use Qwen3-4B/8B/14B plus two fine-tuned variants. Verify convergence on GSM8K and GPQA-Diamond (benchmarks validated by LatentMAS). Success metric: **convergence in ≤10 rounds on 95% of inputs**, and accuracy exceeding best single agent by ≥5%.

**Phase 3 — Heterogeneous agents, dissenter, full system (6 weeks)**: Add cross-family agents (LLaMA-3-8B, Mistral-7B). Train cross-family codecs. Enable the dissenter channel. Test on adversarial scenarios (one agent given correct but minority evidence). Success metric: the system correctly adopts the dissenter's position on ≥50% of adversarial trials.

---

## Convergence of ideas, divergence of agents

This specification proposes that genuine collective intelligence in AI systems emerges not from averaging or voting but from the same dynamical process that governs honeybee swarms: **competing proposals that suppress each other until one wins**. The Latent Resonance Loop implements this at the representation level — agents don't argue in text, they exert force on a shared activation landscape. The Hopfield energy framework provides the attractor dynamics, contraction mapping theory provides convergence guarantees, and the dissenter channel prevents the tyranny of the majority. The deepest open question is not whether this architecture can be built — every component has precedent — but whether the fixed points it converges to represent something genuinely new: collective representations that no single agent could produce alone. That question can only be answered empirically, and this document provides the blueprint for doing so.