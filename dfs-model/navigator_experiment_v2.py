"""
Navigator Experiment V2: Fix the input problem.

V1 failed because we compared low-rank activations computed from the WRONG input
(hidden_states[i] = output of layer i-1, not the actual MLP input which goes
through attention + layernorm first).

Fix: hook the MLP to capture its ACTUAL input, then compute low-rank from that.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

print("=" * 70)
print("NAVIGATOR EXPERIMENT V2: Correct MLP Input")
print("=" * 70)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

n_layers = model.config.n_layer
hidden_dim = model.config.n_embd
ffn_dim = model.config.n_inner or 4 * hidden_dim
print(f"  Layers: {n_layers}, Hidden: {hidden_dim}, FFN: {ffn_dim}")

test_prompts = [
    "The capital of France is",
    "In quantum mechanics, the wave function",
    "Shakespeare wrote Hamlet because",
    "The derivative of x squared is",
    "Machine learning models are trained by",
    "The cat sat on the mat and",
    "During World War II, the allies",
    "Python is a programming language that",
    "The speed of light in vacuum is",
    "To make a good pizza you need",
    "Neural networks learn by adjusting",
    "The president of the United States",
    "Water boils at 100 degrees because",
    "In the year 2024, artificial intelligence",
    "The mitochondria is the powerhouse of",
    "Bitcoin was invented by Satoshi",
    "Photosynthesis converts sunlight into",
    "The Pythagorean theorem states that",
    "Democracy is a system of government where",
    "Gravity is a fundamental force that",
]

# ── Step 1: SVD decompose all MLP weights ──
print("\n[1/4] SVD decomposition of MLP weights...")

ranks_to_test = [64, 128, 256, 384]

low_rank_by_rank = {}
for rank in ranks_to_test:
    low_rank_weights = {}
    for i in range(n_layers):
        mlp = model.transformer.h[i].mlp
        W_fc = mlp.c_fc.weight.data.float()     # [768, 3072]
        W_proj = mlp.c_proj.weight.data.float()  # [3072, 768]

        U_fc, S_fc, Vh_fc = torch.linalg.svd(W_fc, full_matrices=False)
        r = min(rank, S_fc.size(0))
        A_fc = U_fc[:, :r] * S_fc[:r].unsqueeze(0)  # [768, r]
        B_fc = Vh_fc[:r, :]                           # [r, 3072]

        U_proj, S_proj, Vh_proj = torch.linalg.svd(W_proj, full_matrices=False)
        r2 = min(rank, S_proj.size(0))
        A_proj = U_proj[:, :r2] * S_proj[:r2].unsqueeze(0)
        B_proj = Vh_proj[:r2, :]

        low_rank_weights[i] = {
            'A_fc': A_fc, 'B_fc': B_fc, 'bias_fc': mlp.c_fc.bias.data.float(),
        }
    low_rank_by_rank[rank] = low_rank_weights

print(f"  Computed SVD for ranks: {ranks_to_test}")

# ── Step 2: Capture ACTUAL MLP inputs and activations via hooks ──
print("\n[2/4] Capturing actual MLP inputs and activations...")

mlp_inputs = {}
mlp_activations = {}

def make_input_hook(layer_idx):
    def hook_fn(module, input, output):
        with torch.no_grad():
            h = input[0]  # actual MLP input (after attn + layernorm)
            mlp_inputs[layer_idx] = h[:, -1, :].detach().float()
            # Compute intermediate activation
            intermediate = module.c_fc(h)
            activated = module.act(intermediate)
            mlp_activations[layer_idx] = activated[:, -1, :].detach().float()
    return hook_fn

hooks = []
for i in range(n_layers):
    h = model.transformer.h[i].mlp.register_forward_hook(make_input_hook(i))
    hooks.append(h)

# Collect data
all_inputs = {i: [] for i in range(n_layers)}
all_activations = {i: [] for i in range(n_layers)}

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    for i in range(n_layers):
        all_inputs[i].append(mlp_inputs[i].squeeze(0))
        all_activations[i].append(mlp_activations[i].squeeze(0))

for h in hooks:
    h.remove()

# ── Step 3: Compare activation patterns at each rank ──
print("\n[3/4] Comparing activation patterns (correct input)...")

for rank in ranks_to_test:
    lr_weights = low_rank_by_rank[rank]
    cosine_sims = []
    top5_overlaps = []
    top10_overlaps = []

    for i in range(n_layers):
        lr = lr_weights[i]
        for p_idx in range(len(test_prompts)):
            h = all_inputs[i][p_idx]          # actual MLP input
            full_act = all_activations[i][p_idx]  # actual activation

            # Low-rank activation from SAME input
            lr_act = F.gelu(h @ lr['A_fc'] @ lr['B_fc'] + lr['bias_fc'])

            # Cosine similarity
            cos = F.cosine_similarity(full_act.unsqueeze(0), lr_act.unsqueeze(0)).item()
            cosine_sims.append(cos)

            # Top-5% overlap
            k5 = max(1, int(0.05 * ffn_dim))
            top_full = set(torch.topk(full_act.abs(), k5).indices.tolist())
            top_lr = set(torch.topk(lr_act.abs(), k5).indices.tolist())
            top5_overlaps.append(len(top_full & top_lr) / k5)

            # Top-10% overlap
            k10 = max(1, int(0.10 * ffn_dim))
            top_full_10 = set(torch.topk(full_act.abs(), k10).indices.tolist())
            top_lr_10 = set(torch.topk(lr_act.abs(), k10).indices.tolist())
            top10_overlaps.append(len(top_full_10 & top_lr_10) / k10)

    compression = (rank * hidden_dim + rank * ffn_dim) / (hidden_dim * ffn_dim)
    print(f"\n  Rank {rank:3d} ({compression:5.1%} of params):")
    print(f"    Cosine sim:     {np.mean(cosine_sims):.4f}")
    print(f"    Top-5% overlap: {np.mean(top5_overlaps):.1%}")
    print(f"    Top-10% overlap:{np.mean(top10_overlaps):.1%}")

# ── Step 4: Sparse execution with correct hooks ──
print("\n[4/4] End-to-end sparse execution test...")

best_rank = 384  # use highest rank for best chance
lr_weights = low_rank_by_rank[best_rank]

def generate_tokens(model, tokenizer, prompt, max_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

test_prompt = "The capital of France is"
full_output = generate_tokens(model, tokenizer, test_prompt)
print(f"\n  Prompt: '{test_prompt}'")
print(f"  Full model: '{full_output}'")

# Also test with multiple prompts
extra_prompts = [
    "Shakespeare wrote",
    "The meaning of life is",
    "Water is made of",
]

for keep_frac in [0.5, 0.3, 0.2, 0.1]:
    k = max(1, int(keep_frac * ffn_dim))

    originals = {}
    for i in range(n_layers):
        lr = lr_weights[i]
        orig_forward = model.transformer.h[i].mlp.forward.__func__ if hasattr(model.transformer.h[i].mlp.forward, '__func__') else None

        def make_masked_forward(layer_idx, k_val):
            lr_l = lr_weights[layer_idx]
            mlp_ref = model.transformer.h[layer_idx].mlp

            def masked_forward(self_mlp, x):
                with torch.no_grad():
                    lr_pred = F.gelu(x.float() @ lr_l['A_fc'] @ lr_l['B_fc'] + lr_l['bias_fc'])
                    # Per-token mask based on each token's predicted activation
                    top_k_idx = torch.topk(lr_pred.abs(), k_val, dim=-1).indices  # [batch, seq, k]

                    # Full computation
                    h = mlp_ref.c_fc(x)
                    h = mlp_ref.act(h)

                    # Create and apply mask
                    mask = torch.zeros_like(h)
                    mask.scatter_(-1, top_k_idx, 1.0)
                    h = h * mask

                    h = mlp_ref.c_proj(h)
                    h = mlp_ref.dropout(h)
                return h

            return masked_forward

        originals[i] = model.transformer.h[i].mlp.forward
        import types
        model.transformer.h[i].mlp.forward = types.MethodType(
            make_masked_forward(i, k), model.transformer.h[i].mlp
        )

    sparse_output = generate_tokens(model, tokenizer, test_prompt)

    # Restore
    for i in range(n_layers):
        model.transformer.h[i].mlp.forward = originals[i]

    match_str = "MATCH" if sparse_output == full_output else "DIFF"
    print(f"  Keep {keep_frac:4.0%} ({k:4d}/{ffn_dim}) rank={best_rank}: '{sparse_output[:80]}...' [{match_str}]")

# ── Sparsity analysis: how sparse are activations really? ──
print("\n" + "=" * 70)
print("SPARSITY DEEP DIVE")
print("=" * 70)

# What fraction of neurons are needed to capture X% of total activation energy?
for energy_target in [0.9, 0.95, 0.99]:
    neurons_needed = []
    for i in range(n_layers):
        for p_idx in range(len(test_prompts)):
            act = all_activations[i][p_idx]
            sorted_acts = torch.sort(act.abs(), descending=True).values
            cumsum = torch.cumsum(sorted_acts, dim=0)
            total = cumsum[-1]
            needed = (cumsum >= energy_target * total).nonzero(as_tuple=True)[0][0].item() + 1
            neurons_needed.append(needed / ffn_dim)
    print(f"  {energy_target:.0%} activation energy captured by {np.mean(neurons_needed):.1%} of neurons (avg)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
