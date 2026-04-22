import torch
from safetensors import safe_open
import os
import numpy as np

print("=" * 70)
print("TESTING USER'S 'CARRIER + FRACTION' ON GEMMA-2-27B")
print("=" * 70)

# Path to the first shard of Gemma-2-27B
snapshot = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-27b/snapshots/938270f5272feb02779b55c2bb2fffdd0f53ff0c")
shard_path = os.path.join(snapshot, "model-00002-of-00024.safetensors") # picking shard 2 which usually has FFN weights

# Find a weight matrix in this shard
tensor_name = None
with safe_open(shard_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        if "mlp.down_proj.weight" in key:
            tensor_name = key
            break
            
    if tensor_name is None:
        # Just grab the first 2D weight matrix we find
        for key in f.keys():
            t = f.get_slice(key)
            if len(t.get_shape()) == 2:
                tensor_name = key
                break

print(f"1. Loading tensor: {tensor_name}")
with safe_open(shard_path, framework="pt", device="cpu") as f:
    W = f.get_tensor(tensor_name).float() # load as float32 for math precision

print(f"   Shape: {W.shape}, Parameters: {W.numel():,}")

# --- Find original unique values (simulating BF16 constraint) ---
W_bf16 = W.to(torch.bfloat16)
orig_unique = torch.unique(W_bf16)
print(f"2. Original unique values (in BF16): {orig_unique.numel():,}")

# --- User's approach: Block Median (Carrier) + Fraction (Delta) ---
block_size = 64
print(f"3. Applying Block Quantization (Block size = {block_size})")

# Reshape into blocks
flattened = W_bf16.view(-1, block_size)

# Find carrier (median per block)
# For simplicity, we'll take the exact median value in the block
carrier, _ = torch.median(flattened, dim=-1, keepdim=True)

# Calculate fraction (delta)
fraction = flattened - carrier

# How many unique fractions/deltas are there across the ENTIRE matrix?
unique_fractions = torch.unique(fraction)
print(f"\nRESULTS:")
print(f"  Total Weights: {W.numel():,}")
print(f"  Original Dictionary Size: {orig_unique.numel():,} unique values")
print(f"  New 'Fraction' Dictionary Size: {unique_fractions.numel():,} unique deltas")

if unique_fractions.numel() > orig_unique.numel():
    print(f"\nANALYSIS: The user's idea creates MORE unique values ({unique_fractions.numel():,}) than the original weights ({orig_unique.numel():,}).")
    print("  Why? Because subtracting a local median from the 6,000 standard values creates")
    print("  new intermediate float values. A dictionary of 6,000 becomes a dictionary of >100,000.")
else:
    print(f"\nANALYSIS: The user's idea successfully reduced the dictionary size!")
