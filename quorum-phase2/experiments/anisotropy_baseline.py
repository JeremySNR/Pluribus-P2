"""Quick diagnostic: how anisotropic are GPT-2's hidden states?

If random word pairs already have >0.95 cosine similarity in the last layer,
then our codec's 0.998 cosine sim is preserving the cone direction (trivial)
while destroying the discriminative residuals.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

print("Loading GPT-2...")
model = AutoModel.from_pretrained("gpt2")
model.eval()
tok = AutoTokenizer.from_pretrained("gpt2")

RANDOM_WORDS = [
    "apple", "democracy", "purple", "running", "telescope",
    "angry", "molecule", "finance", "guitar", "ocean",
    "breakfast", "philosophy", "triangle", "whisper", "volcano",
    "algorithm", "curtain", "liberty", "penguin", "satellite",
]

print(f"\n--- Anisotropy baseline: {len(RANDOM_WORDS)} random words ---")

all_layer_hiddens = [[] for _ in range(13)]
with torch.no_grad():
    for w in RANDOM_WORDS:
        inp = tok(w, return_tensors="pt")
        out = model(**inp, output_hidden_states=True)
        for layer_idx in range(13):
            all_layer_hiddens[layer_idx].append(out.hidden_states[layer_idx][0, -1, :])

print("\nPairwise cosine similarity of UNRELATED words by layer:")
print(f"{'Layer':>8}  {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
print("-" * 48)

for layer_idx in range(13):
    hs = all_layer_hiddens[layer_idx]
    sims = []
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            sim = F.cosine_similarity(hs[i].unsqueeze(0), hs[j].unsqueeze(0)).item()
            sims.append(sim)
    mean_sim = sum(sims) / len(sims)
    print(f"{layer_idx:>8}  {mean_sim:>8.4f}  {min(sims):>8.4f}  {max(sims):>8.4f}  {torch.tensor(sims).std().item():>8.4f}")

last_layer_sims = []
hs = all_layer_hiddens[12]
for i in range(len(hs)):
    for j in range(i + 1, len(hs)):
        sim = F.cosine_similarity(hs[i].unsqueeze(0), hs[j].unsqueeze(0)).item()
        last_layer_sims.append(sim)
baseline = sum(last_layer_sims) / len(last_layer_sims)

print(f"\n--- What this means for our measurements ---")
print(f"  Random word baseline (last layer):   {baseline:.4f}")
print(f"  Our codec roundtrip cosine sim:      0.9980")
print(f"  Our cross-model alignment cosine:    0.9850")
print(f"")
usable = 1.0 - baseline
codec_pct = (0.998 - baseline) / usable * 100
align_pct = (0.985 - baseline) / usable * 100
print(f"  Meaningful similarity range:         {baseline:.4f} - 1.000 (width: {usable:.4f})")
print(f"  Codec roundtrip uses:                {codec_pct:.1f}% of the meaningful range")
print(f"  Cross-model alignment uses:          {align_pct:.1f}% of the meaningful range")
print(f"")
if baseline > 0.95:
    print(f"  CONFIRMED: Cosine similarity is near-meaningless in this space.")
    print(f"  The codec appears excellent (0.998) but is operating on a")
    print(f"  scale where random unrelated words are already at {baseline:.3f}.")
    print(f"  All discriminative information lives in the {usable:.3f} band above baseline.")
elif baseline > 0.80:
    print(f"  MODERATE anisotropy. Cosine sim has some discriminative value")
    print(f"  but the {baseline:.3f} floor compresses the meaningful range.")
else:
    print(f"  LOW anisotropy. Cosine similarity is a reasonable metric here.")
