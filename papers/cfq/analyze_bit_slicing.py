import torch
from safetensors import safe_open
import os
from collections import Counter

print("=" * 70)
print("ANALYZING GEMMA-2-27B FOR 1GB RAM COMPRESSION")
print("=" * 70)

snapshot = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-27b/snapshots/938270f5272feb02779b55c2bb2fffdd0f53ff0c")
shard_path = os.path.join(snapshot, "model-00002-of-00024.safetensors")

with safe_open(shard_path, framework="pt", device="cpu") as f:
    tensor_name = "model.layers.0.mlp.down_proj.weight"
    W = f.get_tensor(tensor_name).to(torch.bfloat16)

total_weights = W.numel()
print(f"Tensor: {tensor_name}")
print(f"Total weights: {total_weights:,}")

# 1. Find unique values and their frequencies
vals, counts = torch.unique(W, return_counts=True)
sorted_idx = torch.argsort(counts, descending=True)
vals_sorted = vals[sorted_idx]
counts_sorted = counts[sorted_idx]

print(f"\n1. Top 10 Most Common Values (Carriers):")
for i in range(10):
    val = vals_sorted[i].item()
    count = counts_sorted[i].item()
    print(f"   Value: {val:10.6f} | Frequency: {count:10,} ({count/total_weights:6.2%})")

# 2. Cumulative Coverage
cum_counts = torch.cumsum(counts_sorted, dim=0)
print(f"\n2. Coverage Analysis:")
for pct in [0.5, 0.8, 0.9, 0.95, 0.99]:
    idx = (cum_counts >= pct * total_weights).nonzero()[0].item()
    print(f"   {pct*100:2.0f}% of weights are covered by just {idx+1} unique values")

# 3. User's "Carrier + Fraction" idea
# Let's take the Top 1 value as the Carrier (usually 0 or near-0)
carrier = vals_sorted[0]
deltas = W - carrier
unique_deltas = torch.unique(deltas)
print(f"\n3. Unique Deltas (Fractional pieces):")
print(f"   Unique deltas from carrier {carrier.item():.6f}: {unique_deltas.numel():,}")

# 4. Bit-Slicing Simulation
# If we only store the index of the Top 16 values (4 bits) in RAM
top_16_indices = sorted_idx[:16]
top_16_vals = vals[top_16_indices]

# Measure "Accuracy" if we load only these 4 bits into RAM
# We map every weight to its nearest value among the top 16
# This is a VERY rough simulation of loading only the "Carrier Piece"
print(f"\n4. 4-bit 'Carrier' Simulation (Top 16 values):")
# For speed, we'll sample 1 million weights
sample_size = 1_000_000
W_sample = W.flatten()[:sample_size].float()
top_16_vals_f = top_16_vals.float()

# Find nearest top-16 value for each weight
dist = torch.abs(W_sample.unsqueeze(1) - top_16_vals_f.unsqueeze(0))
nearest_idx = torch.argmin(dist, dim=1)
W_reconstructed = top_16_vals_f[nearest_idx]

cos_sim = torch.nn.functional.cosine_similarity(W_sample.unsqueeze(0), W_reconstructed.unsqueeze(0)).item()
print(f"   Cosine similarity (4-bit Carrier only): {cos_sim:.4f}")

# 5. Memory Math for 27B
# 4 bits * 27B weights
size_4bit_gb = (4 * 27e9) / (8 * 1024**3)
print(f"\n5. Memory Projection (Gemma-27B):")
print(f"   If we load 4-bit Carrier per weight: {size_4bit_gb:.2f} GB")
print(f"   If we load 2-bit Carrier per weight: {size_4bit_gb/2:.2f} GB")
print(f"   If we load 1-bit Carrier per weight: {size_4bit_gb/4:.2f} GB")
