"""
Navigator Experiment — SwiGLU Model (Qwen2-0.5B)

SwiGLU: output = (x @ W_gate * silu(x @ W_up)) @ W_down
The gate multiplication creates TRUE zeros — when gate ≈ 0, the neuron
contributes exactly 0 through W_down. This is the key difference from GELU.

Tests:
1. Measure true sparsity (exact zeros after gate)
2. Oracle test (perfect prediction — can we prune?)
3. SVD navigator test
4. Early-layer navigator test
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import types
import gc

print("=" * 70)
print("NAVIGATOR EXPERIMENT — SwiGLU (Qwen2-0.5B)")
print("=" * 70)

model_name = "Qwen/Qwen2-0.5B"
print(f"\nLoading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

config = model.config
n_layers = config.num_hidden_layers
hidden_dim = config.hidden_size
ffn_dim = config.intermediate_size
print(f"  Layers: {n_layers}, Hidden: {hidden_dim}, FFN: {ffn_dim}")
print(f"  Activation: {config.hidden_act}")
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

# Figure out MLP structure
mlp0 = model.model.layers[0].mlp
print(f"  MLP structure: {type(mlp0).__name__}")
print(f"    gate_proj: {mlp0.gate_proj.weight.shape}")
print(f"    up_proj:   {mlp0.up_proj.weight.shape}")
print(f"    down_proj: {mlp0.down_proj.weight.shape}")

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

# ── Step 1: Measure TRUE sparsity in SwiGLU ──
print("\n" + "=" * 70)
print("[1/5] Measuring SwiGLU sparsity...")
print("=" * 70)

# SwiGLU forward: output = silu(x @ gate_proj) * (x @ up_proj) → down_proj
# The key: silu(gate) can be very small, making the product ≈ 0

mlp_inputs_store = {}
gate_acts_store = {}
up_acts_store = {}
combined_acts_store = {}  # gate * up — this is what goes into down_proj

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        with torch.no_grad():
            x = input[0][:, -1, :].float()
            mlp_inputs_store[layer_idx] = x.detach()
            gate = F.silu(module.gate_proj(x))
            up = module.up_proj(x)
            combined = gate * up
            gate_acts_store[layer_idx] = gate.detach()
            up_acts_store[layer_idx] = up.detach()
            combined_acts_store[layer_idx] = combined.detach()
    return hook_fn

hooks = []
for i in range(n_layers):
    hooks.append(model.model.layers[i].mlp.register_forward_hook(make_hook(i)))

all_mlp_inputs = {i: [] for i in range(n_layers)}
all_gate_acts = {i: [] for i in range(n_layers)}
all_combined_acts = {i: [] for i in range(n_layers)}

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    for i in range(n_layers):
        all_mlp_inputs[i].append(mlp_inputs_store[i].squeeze(0))
        all_gate_acts[i].append(gate_acts_store[i].squeeze(0))
        all_combined_acts[i].append(combined_acts_store[i].squeeze(0))

for h in hooks:
    h.remove()

# Sparsity measurements
print("\n  Sparsity by threshold (fraction of neurons below X% of max):")
print(f"\n  Layer | Exact 0 | <1% max  | <5% max  | <10% max | <20% max")
print("  " + "-" * 65)

for i in range(n_layers):
    exact_zeros = []
    below_1 = []
    below_5 = []
    below_10 = []
    below_20 = []
    for p_idx in range(len(test_prompts)):
        act = all_combined_acts[i][p_idx]
        mx = act.abs().max().item()
        if mx == 0:
            exact_zeros.append(1.0)
            below_1.append(1.0)
            below_5.append(1.0)
            below_10.append(1.0)
            below_20.append(1.0)
        else:
            exact_zeros.append((act.abs() == 0).float().mean().item())
            below_1.append((act.abs() < 0.01 * mx).float().mean().item())
            below_5.append((act.abs() < 0.05 * mx).float().mean().item())
            below_10.append((act.abs() < 0.10 * mx).float().mean().item())
            below_20.append((act.abs() < 0.20 * mx).float().mean().item())

    print(f"  {i:5d} | {np.mean(exact_zeros):6.1%}  | {np.mean(below_1):6.1%}   | {np.mean(below_5):6.1%}   | {np.mean(below_10):6.1%}   | {np.mean(below_20):6.1%}")

# Energy analysis
print("\n  Neurons needed to capture X% of activation energy:")
for energy in [0.90, 0.95, 0.99]:
    needed = []
    for i in range(n_layers):
        for p_idx in range(len(test_prompts)):
            act = all_combined_acts[i][p_idx]
            sorted_acts = torch.sort(act.abs(), descending=True).values
            cumsum = torch.cumsum(sorted_acts, dim=0)
            total = cumsum[-1].item()
            if total == 0:
                needed.append(0)
            else:
                idx = (cumsum >= energy * total).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    needed.append((idx[0].item() + 1) / ffn_dim)
                else:
                    needed.append(1.0)
    print(f"  {energy:.0%} energy → {np.mean(needed):.1%} of neurons needed")

# ── Step 2: ORACLE TEST — the critical one ──
print("\n" + "=" * 70)
print("[2/5] Oracle test: perfect prediction, can we prune?")
print("=" * 70)

def generate(model, tokenizer, prompt, max_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

oracle_prompts = [
    "The capital of France is",
    "Shakespeare wrote",
    "Water is made of",
]

for prompt in oracle_prompts:
    full_out = generate(model, tokenizer, prompt)
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Full:   '{full_out}'")

    for keep_frac in [0.5, 0.3, 0.2, 0.1, 0.05]:
        k = max(1, int(keep_frac * ffn_dim))
        originals = {}

        for i in range(n_layers):
            originals[i] = model.model.layers[i].mlp.forward

            def make_oracle(layer_idx, k_val):
                mlp = model.model.layers[layer_idx].mlp

                def oracle_forward(self_mlp, x):
                    gate = F.silu(mlp.gate_proj(x))
                    up = mlp.up_proj(x)
                    combined = gate * up

                    # Oracle mask: keep top-k by actual magnitude
                    with torch.no_grad():
                        topk = torch.topk(combined.abs(), k_val, dim=-1)
                        mask = torch.zeros_like(combined)
                        mask.scatter_(-1, topk.indices, 1.0)

                    combined_masked = combined * mask
                    return mlp.down_proj(combined_masked)

                return oracle_forward

            model.model.layers[i].mlp.forward = types.MethodType(
                make_oracle(i, k), model.model.layers[i].mlp
            )

        oracle_out = generate(model, tokenizer, prompt)
        for i in originals:
            model.model.layers[i].mlp.forward = originals[i]

        match = "MATCH" if oracle_out == full_out else "diff"
        print(f"  Oracle keep {keep_frac:4.0%} ({k:4d}/{ffn_dim}): '{oracle_out[:70]}' [{match}]")

# ── Step 3: Navigator via SVD (same as v2 but for SwiGLU) ──
print("\n" + "=" * 70)
print("[3/5] SVD Navigator — activation pattern prediction")
print("=" * 70)

ranks_to_test = [64, 128, 256]
lr_by_rank = {}

for rank in ranks_to_test:
    lr_weights = {}
    for i in range(n_layers):
        mlp = model.model.layers[i].mlp
        # gate_proj and up_proj: [ffn_dim, hidden_dim] (standard Linear)
        W_gate = mlp.gate_proj.weight.data.float()  # [ffn_dim, hidden_dim]
        W_up = mlp.up_proj.weight.data.float()       # [ffn_dim, hidden_dim]

        for name, W in [('gate', W_gate), ('up', W_up)]:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(rank, S.size(0))
            # W ≈ U[:,:r] @ diag(S[:r]) @ Vh[:r,:]
            # For Linear: output = x @ W.T = x @ Vh.T @ diag(S) @ U.T
            lr_weights[(i, name)] = {
                'A': (U[:, :r] * S[:r].unsqueeze(0)),  # [ffn_dim, r]
                'Vh': Vh[:r, :],                         # [r, hidden_dim]
            }

        lr_weights[(i, 'gate_bias')] = getattr(mlp.gate_proj, 'bias', None)
        lr_weights[(i, 'up_bias')] = getattr(mlp.up_proj, 'bias', None)

    lr_by_rank[rank] = lr_weights

print("  SVD computed for gate_proj and up_proj at each rank")

for rank in ranks_to_test:
    lr_weights = lr_by_rank[rank]
    cosines = []
    top5 = []
    top10 = []

    for i in range(n_layers):
        g = lr_weights[(i, 'gate')]
        u = lr_weights[(i, 'up')]

        for p_idx in range(len(test_prompts)):
            h = all_mlp_inputs[i][p_idx].unsqueeze(0)  # [1, hidden_dim]
            actual = all_combined_acts[i][p_idx]

            # Low-rank: gate_lr = silu(x @ Vh.T @ A.T), up_lr = x @ Vh.T @ A.T
            gate_lr = F.silu(h @ g['Vh'].T @ g['A'].T)
            up_lr = h @ u['Vh'].T @ u['A'].T
            combined_lr = (gate_lr * up_lr).squeeze(0)

            cos = F.cosine_similarity(actual.unsqueeze(0), combined_lr.unsqueeze(0)).item()
            cosines.append(cos)

            k5 = max(1, int(0.05 * ffn_dim))
            t_act = set(torch.topk(actual.abs(), k5).indices.tolist())
            t_lr = set(torch.topk(combined_lr.abs(), k5).indices.tolist())
            top5.append(len(t_act & t_lr) / k5)

            k10 = max(1, int(0.10 * ffn_dim))
            t_act10 = set(torch.topk(actual.abs(), k10).indices.tolist())
            t_lr10 = set(torch.topk(combined_lr.abs(), k10).indices.tolist())
            top10.append(len(t_act10 & t_lr10) / k10)

    comp = 2 * (rank * hidden_dim + rank * ffn_dim) / (2 * hidden_dim * ffn_dim)
    print(f"\n  Rank {rank:3d} ({comp:5.1%} of gate+up params):")
    print(f"    Cosine sim:      {np.mean(cosines):.4f}")
    print(f"    Top-5% overlap:  {np.mean(top5):.1%}")
    print(f"    Top-10% overlap: {np.mean(top10):.1%}")

# ── Step 4: End-to-end sparse with navigator ──
print("\n" + "=" * 70)
print("[4/5] End-to-end: navigator-guided sparse execution")
print("=" * 70)

best_rank = 256
lr_weights = lr_by_rank[best_rank]

for prompt in oracle_prompts:
    full_out = generate(model, tokenizer, prompt)
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Full:   '{full_out}'")

    for keep_frac in [0.5, 0.3, 0.1]:
        k = max(1, int(keep_frac * ffn_dim))
        originals = {}

        for i in range(n_layers):
            originals[i] = model.model.layers[i].mlp.forward

            def make_nav_sparse(layer_idx, k_val):
                mlp = model.model.layers[layer_idx].mlp
                g = lr_by_rank[best_rank][(layer_idx, 'gate')]
                u = lr_by_rank[best_rank][(layer_idx, 'up')]

                def nav_forward(self_mlp, x):
                    # Navigator predicts which neurons fire
                    with torch.no_grad():
                        gate_lr = F.silu(x.float() @ g['Vh'].T @ g['A'].T)
                        up_lr = x.float() @ u['Vh'].T @ u['A'].T
                        combined_lr = gate_lr * up_lr
                        topk = torch.topk(combined_lr.abs(), k_val, dim=-1)
                        mask = torch.zeros_like(combined_lr)
                        mask.scatter_(-1, topk.indices, 1.0)

                    # Full computation, then mask
                    gate = F.silu(mlp.gate_proj(x))
                    up = mlp.up_proj(x)
                    combined = gate * up
                    combined_masked = combined * mask.to(combined.dtype)
                    return mlp.down_proj(combined_masked)

                return nav_forward

            model.model.layers[i].mlp.forward = types.MethodType(
                make_nav_sparse(i, k), model.model.layers[i].mlp
            )

        nav_out = generate(model, tokenizer, prompt)
        for i in originals:
            model.model.layers[i].mlp.forward = originals[i]

        match = "MATCH" if nav_out == full_out else "diff"
        print(f"  Nav(r={best_rank}) keep {keep_frac:.0%}: '{nav_out[:70]}' [{match}]")

# ── Step 5: Adjacent layer correlation in SwiGLU ──
print("\n" + "=" * 70)
print("[5/5] Adjacent layer correlation (SwiGLU)")
print("=" * 70)

print("\n  Layer pair | Top-5% Overlap | Cosine Sim")
print("  " + "-" * 45)
for i in range(n_layers - 1):
    overlaps = []
    cosines = []
    for p_idx in range(len(test_prompts)):
        a = all_combined_acts[i][p_idx]
        b = all_combined_acts[i+1][p_idx]
        k5 = max(1, int(0.05 * ffn_dim))
        ta = set(torch.topk(a.abs(), k5).indices.tolist())
        tb = set(torch.topk(b.abs(), k5).indices.tolist())
        overlaps.append(len(ta & tb) / k5)
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        cosines.append(cos)
    print(f"  {i:2d} → {i+1:2d}     | {np.mean(overlaps):13.1%} | {np.mean(cosines):.4f}")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
