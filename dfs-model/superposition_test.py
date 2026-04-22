"""
Superposition / Compressed Neuron Experiment

Can we pack N neurons into M < N super-neurons and still get correct output?

Tests:
1. Random Projection (JL): compress FFN activations, reconstruct, measure error
2. Product Quantization: codebook-based compression of activation vectors
3. Shared weights + ID: one shared weight vector + per-neuron identifier
4. End-to-end: generate with compressed FFN, compare to full model

Model: Qwen2-0.5B (small enough to iterate fast, same architecture as 7B)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import types
import time

print("=" * 70)
print("SUPERPOSITION / COMPRESSED NEURON EXPERIMENT")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model.eval()

n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
ffn_dim = model.config.intermediate_size
print(f"  Layers: {n_layers}, Hidden: {hidden_dim}, FFN: {ffn_dim}")

def generate(model, tokenizer, prompt, max_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id or 0)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ══════════════════════════════════════════════════════════════════════
# TEST 1: Random Projection of FFN activations
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[1/4] Random Projection — compress activations via JL")
print("=" * 70)

# Collect actual FFN intermediate activations for analysis
test_prompts = [
    "The capital of France is",
    "Shakespeare wrote Hamlet because",
    "Water is made of hydrogen and",
    "Neural networks learn by adjusting",
    "The speed of light in vacuum",
]

# Capture gate*up activations (the intermediate before down_proj)
captured = {}
def make_capture(layer_idx):
    def hook(module, input, output):
        with torch.no_grad():
            x = input[0].float()
            gate = F.silu(module.gate_proj(x))
            up = module.up_proj(x)
            captured[layer_idx] = (gate * up).detach()  # [batch, seq, ffn_dim]
    return hook

hooks = []
for i in range(n_layers):
    hooks.append(model.model.layers[i].mlp.register_forward_hook(make_capture(i)))

# Collect activations
all_acts = {i: [] for i in range(n_layers)}
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    for i in range(n_layers):
        all_acts[i].append(captured[i][0, -1, :].cpu())  # last token

for h in hooks:
    h.remove()

# Test random projection at different compression ratios
compression_ratios = [2, 4, 8, 16, 32]

print(f"\n  FFN dim: {ffn_dim}")
print(f"\n  Ratio | Super-neurons | Reconstruction cosine | Max error (relative)")
print("  " + "-" * 70)

torch.manual_seed(42)

for ratio in compression_ratios:
    m = ffn_dim // ratio  # number of super-neurons

    # Random projection matrix (Gaussian, scaled)
    R = torch.randn(m, ffn_dim) / np.sqrt(m)

    cosines = []
    rel_errors = []

    for i in [0, 11, 23]:  # sample layers
        for act in all_acts[i]:
            # Compress
            compressed = R @ act  # [m]

            # Reconstruct (pseudoinverse)
            reconstructed = R.T @ compressed  # [ffn_dim] (approximate)

            # Measure quality
            cos = F.cosine_similarity(act.unsqueeze(0), reconstructed.unsqueeze(0)).item()
            cosines.append(cos)

            rel_err = (act - reconstructed).norm() / act.norm()
            rel_errors.append(rel_err.item())

    print(f"  {ratio:5d}x | {m:13d} | {np.mean(cosines):21.4f} | {np.mean(rel_errors):.4f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 2: Random Projection of WEIGHT MATRICES
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[2/4] Random Projection of weight matrices")
print("  (Compress W_down from [hidden, ffn] to [hidden, M], reconstruct output)")
print("=" * 70)

# The key operation: output = down_proj(gate * up)
# down_proj: [hidden_dim, ffn_dim] × [ffn_dim] → [hidden_dim]
#
# With RP: compressed_act = R @ act  (R: [M, ffn_dim])
#           compressed_W  = W @ R^T   (W: [hidden, ffn_dim] → [hidden, M])
#           output_approx = compressed_W @ compressed_act
#
# This is: W @ R^T @ R @ act ≈ W @ act (by JL)

print(f"\n  Compressing down_proj weight matrix")
print(f"\n  Ratio | Super-dim | Output cosine | Output rel error | Memory saved")
print("  " + "-" * 75)

for ratio in compression_ratios:
    m = ffn_dim // ratio
    torch.manual_seed(42)
    R = torch.randn(m, ffn_dim) / np.sqrt(m)

    cosines = []
    rel_errors = []

    for layer_idx in [0, 11, 23]:
        W_down = model.model.layers[layer_idx].mlp.down_proj.weight.data.float()  # [hidden, ffn]

        # Compress weight matrix: W_compressed = W @ R^T  → [hidden, M]
        W_compressed = W_down @ R.T  # [hidden_dim, m]

        for act in all_acts[layer_idx]:
            # Full output
            full_output = W_down @ act  # [hidden_dim]

            # Compressed: compress activation, then use compressed weights
            compressed_act = R @ act  # [m]
            approx_output = W_compressed @ compressed_act  # [hidden_dim]

            cos = F.cosine_similarity(full_output.unsqueeze(0), approx_output.unsqueeze(0)).item()
            cosines.append(cos)

            rel_err = (full_output - approx_output).norm() / full_output.norm()
            rel_errors.append(rel_err.item())

    mem_original = hidden_dim * ffn_dim * 4  # bytes
    mem_compressed = (hidden_dim * m + m * ffn_dim) * 4  # W_compressed + R
    mem_saved = 1 - mem_compressed / mem_original

    print(f"  {ratio:5d}x | {m:9d} | {np.mean(cosines):13.4f} | {np.mean(rel_errors):16.4f} | {mem_saved:11.1%}")


# ══════════════════════════════════════════════════════════════════════
# TEST 3: Product Quantization of activations
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[3/4] Product Quantization — codebook compression")
print("  (Split FFN into blocks, learn codebook per block, lookup instead of compute)")
print("=" * 70)

# Collect more activations for codebook training
hooks = []
for i in range(n_layers):
    hooks.append(model.model.layers[i].mlp.register_forward_hook(make_capture(i)))

pq_prompts = test_prompts + [
    "The president of the United States",
    "Machine learning models are trained",
    "In quantum mechanics particles can",
    "The Amazon rainforest produces oxygen",
    "Bitcoin is a cryptocurrency that",
    "The derivative of sin x is",
    "Democracy requires participation from all",
    "The periodic table organizes chemical",
    "Photosynthesis converts sunlight into energy",
    "The theory of relativity states that",
]

pq_acts = {i: [] for i in range(n_layers)}
for prompt in pq_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        model(**inputs)
    for i in range(n_layers):
        # Grab ALL token positions for more training data
        for pos in range(seq_len):
            pq_acts[i].append(captured[i][0, pos, :].cpu())

for h in hooks:
    h.remove()

print(f"  Collected {len(pq_acts[0])} activation vectors per layer")

# Product Quantization: split vector into subvectors, cluster each
def train_pq(vectors, n_blocks, n_centroids=256):
    """Train a product quantizer."""
    dim = vectors.shape[1]
    block_size = dim // n_blocks
    codebooks = []
    codes = []

    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size if b < n_blocks - 1 else dim
        sub = vectors[:, start:end]  # [n_vectors, block_size]

        # K-means clustering (simple, 10 iterations)
        n_c = min(n_centroids, sub.shape[0])
        # Init: random subset
        perm = torch.randperm(sub.shape[0])[:n_c]
        centroids = sub[perm].clone()

        for _ in range(10):
            # Assign
            dists = torch.cdist(sub, centroids)  # [n, n_c]
            assignments = dists.argmin(dim=1)     # [n]
            # Update
            for c in range(n_c):
                mask = assignments == c
                if mask.sum() > 0:
                    centroids[c] = sub[mask].mean(dim=0)

        codebooks.append(centroids)
        codes.append(assignments)

    return codebooks, codes

def pq_reconstruct(codebooks, codes, n_blocks, dim):
    """Reconstruct vector from PQ codes."""
    block_size = dim // n_blocks
    parts = []
    for b in range(n_blocks):
        parts.append(codebooks[b][codes[b]])
    # Handle last block potentially being larger
    return torch.cat(parts, dim=-1)[:, :dim]

# Test PQ at different block counts
block_configs = [(8, 256), (16, 256), (32, 256), (64, 256)]

print(f"\n  Blocks | Centroids | Compression | Cosine sim | Rel error")
print("  " + "-" * 65)

for n_blocks, n_centroids in block_configs:
    test_layer = 11
    vectors = torch.stack(pq_acts[test_layer])

    codebooks, codes = train_pq(vectors, n_blocks, n_centroids)

    # Measure reconstruction quality
    reconstructed = pq_reconstruct(codebooks, codes, n_blocks, ffn_dim)

    cosines = F.cosine_similarity(vectors, reconstructed, dim=1)
    rel_errors = (vectors - reconstructed).norm(dim=1) / vectors.norm(dim=1).clamp(min=1e-8)

    # Compression: original = n_vectors × ffn_dim × 4 bytes
    # PQ = codebooks (n_blocks × n_centroids × block_size × 4) + codes (n_vectors × n_blocks × 1)
    block_size = ffn_dim // n_blocks
    codebook_bytes = n_blocks * n_centroids * block_size * 4
    code_bytes = vectors.shape[0] * n_blocks * 1  # 1 byte per code
    original_bytes = vectors.shape[0] * ffn_dim * 4
    compression = original_bytes / (codebook_bytes + code_bytes)

    print(f"  {n_blocks:6d} | {n_centroids:9d} | {compression:10.1f}x | {cosines.mean():.4f}    | {rel_errors.mean():.4f}")


# ══════════════════════════════════════════════════════════════════════
# TEST 4: End-to-end generation with compressed FFN
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[4/4] End-to-end: generate with compressed FFN")
print("=" * 70)

gen_prompts = [
    "The capital of France is",
    "Shakespeare wrote",
    "Water is made of",
]

for prompt in gen_prompts:
    full_out = generate(model, tokenizer, prompt)
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Full:   '{full_out}'")

# Test A: Random projection of down_proj (compress the WEIGHT, not activation)
# output = W_down @ (gate * up)
# ≈ (W_down @ R^T) @ (R @ (gate * up))
# = W_compressed @ compressed_activation

for ratio in [2, 4, 8]:
    m = ffn_dim // ratio
    torch.manual_seed(42)
    R = torch.randn(m, ffn_dim) / np.sqrt(m)

    # Pre-compute compressed weight matrices for all layers
    compressed_W_down = {}
    for i in range(n_layers):
        W = model.model.layers[i].mlp.down_proj.weight.data.float()
        compressed_W_down[i] = W @ R.T  # [hidden_dim, m]

    originals = {}
    for i in range(n_layers):
        originals[i] = model.model.layers[i].mlp.forward

        def make_rp_forward(layer_idx, R_mat, ratio_val):
            mlp = model.model.layers[layer_idx].mlp
            W_comp = compressed_W_down[layer_idx]

            def rp_forward(self_mlp, x):
                gate = F.silu(mlp.gate_proj(x))
                up = mlp.up_proj(x)
                combined = gate * up  # [batch, seq, ffn_dim]

                # Compress activation: [batch, seq, ffn_dim] → [batch, seq, m]
                compressed = F.linear(combined, R_mat)  # R is [m, ffn_dim]

                # Use compressed weight: W_comp is [hidden_dim, m]
                # F.linear(input, weight) computes input @ weight.T
                output = F.linear(compressed, W_comp)

                return output

            return rp_forward

        model.model.layers[i].mlp.forward = types.MethodType(
            make_rp_forward(i, R, ratio), model.model.layers[i].mlp
        )

    for prompt in gen_prompts:
        rp_out = generate(model, tokenizer, prompt)
        match = "MATCH" if rp_out == full_out else "diff"
        # Find full output for comparison
        full_ref = generate.__wrapped__(model, tokenizer, prompt) if hasattr(generate, '__wrapped__') else "?"
        print(f"  RP {ratio}x: '{rp_out[:70]}' [{match}]") if prompt == gen_prompts[0] else None

    # Print all prompts for this ratio
    for prompt in gen_prompts:
        rp_out = generate(model, tokenizer, prompt)
        print(f"  RP {ratio}x: '{prompt}' → '{rp_out[:65]}'")

    # Restore
    for i in originals:
        model.model.layers[i].mlp.forward = originals[i]

# Memory savings summary
print("\n  Memory comparison (per layer FFN):")
print(f"  Full:    gate({hidden_dim}×{ffn_dim}) + up({hidden_dim}×{ffn_dim}) + down({hidden_dim}×{ffn_dim})")
full_bytes = 3 * hidden_dim * ffn_dim * 4
print(f"           = {full_bytes / 1e6:.1f} MB")

for ratio in [2, 4, 8]:
    m = ffn_dim // ratio
    # gate and up stay full, only down is compressed
    rp_bytes = 2 * hidden_dim * ffn_dim * 4 + hidden_dim * m * 4 + m * ffn_dim * 4
    saving = 1 - rp_bytes / full_bytes
    print(f"  RP {ratio}x:  gate(full) + up(full) + down_compressed({hidden_dim}×{m}) + R({m}×{ffn_dim})")
    print(f"           = {rp_bytes / 1e6:.1f} MB ({saving:+.1%})")

    # If we also compress gate and up:
    rp_all = (hidden_dim * m + m * ffn_dim) * 4 * 3  # all three compressed
    saving_all = 1 - rp_all / full_bytes
    print(f"  RP {ratio}x (all 3): {rp_all / 1e6:.1f} MB ({saving_all:+.1%})")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
