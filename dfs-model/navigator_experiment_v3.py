"""
Navigator Experiment V3: Early layers as navigator.

Instead of SVD (static approximation), use the model's OWN first K layers
at full precision to predict activation patterns in later layers.

Hypothesis: layers 0-3 set up the representation. If we run those fully,
we can predict which neurons fire in layers 4-11 from the hidden state.

Also test: adjacent-layer prediction (layer N's activations predict layer N+1).
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import time

print("=" * 70)
print("NAVIGATOR V3: Early Layers as Predictor")
print("=" * 70)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

n_layers = model.config.n_layer
hidden_dim = model.config.n_embd
ffn_dim = model.config.n_inner or 4 * hidden_dim

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

# ── Capture all MLP inputs, activations, and hidden states ──
print("\n[1/5] Capturing full model activations...")

mlp_inputs_store = {}
mlp_acts_store = {}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        with torch.no_grad():
            h = input[0]
            mlp_inputs_store[layer_idx] = h[:, -1, :].detach().float()
            intermediate = module.c_fc(h)
            activated = module.act(intermediate)
            mlp_acts_store[layer_idx] = activated[:, -1, :].detach().float()
    return hook_fn

hooks = []
for i in range(n_layers):
    hooks.append(model.transformer.h[i].mlp.register_forward_hook(make_hook(i)))

all_mlp_inputs = {i: [] for i in range(n_layers)}
all_mlp_acts = {i: [] for i in range(n_layers)}
all_hidden_states = []

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = [h[:, -1, :].detach().float().squeeze(0) for h in out.hidden_states]
    all_hidden_states.append(hs)
    for i in range(n_layers):
        all_mlp_inputs[i].append(mlp_inputs_store[i].squeeze(0))
        all_mlp_acts[i].append(mlp_acts_store[i].squeeze(0))

for h in hooks:
    h.remove()

print(f"  Captured {len(test_prompts)} prompts × {n_layers} layers")

# ── Test 1: Adjacent layer correlation ──
print("\n[2/5] Adjacent layer activation correlation...")
print("  (Does layer N's active set predict layer N+1's active set?)")

print("\n  Layer pair | Top-5% Overlap | Top-10% Overlap | Cosine Sim")
print("  " + "-" * 60)

for i in range(n_layers - 1):
    overlaps_5 = []
    overlaps_10 = []
    cosines = []
    for p_idx in range(len(test_prompts)):
        act_a = all_mlp_acts[i][p_idx]
        act_b = all_mlp_acts[i+1][p_idx]

        k5 = max(1, int(0.05 * ffn_dim))
        top_a = set(torch.topk(act_a.abs(), k5).indices.tolist())
        top_b = set(torch.topk(act_b.abs(), k5).indices.tolist())
        overlaps_5.append(len(top_a & top_b) / k5)

        k10 = max(1, int(0.10 * ffn_dim))
        top_a10 = set(torch.topk(act_a.abs(), k10).indices.tolist())
        top_b10 = set(torch.topk(act_b.abs(), k10).indices.tolist())
        overlaps_10.append(len(top_a10 & top_b10) / k10)

        cos = F.cosine_similarity(act_a.unsqueeze(0), act_b.unsqueeze(0)).item()
        cosines.append(cos)

    print(f"  {i:2d} → {i+1:2d}     | {np.mean(overlaps_5):13.1%} | {np.mean(overlaps_10):14.1%} | {np.mean(cosines):.4f}")

# ── Test 2: Early layers predict later layers ──
print("\n[3/5] Early hidden state → later activation prediction...")
print("  (Can the hidden state after layer K predict activations in layer L?)")

# For each 'navigator depth' K, use hidden_states[K+1] (output of layer K)
# to predict which neurons fire in layers K+1 through 11.
# Prediction method: simple linear projection (h @ W_fc of target layer)

print("\n  Navigator  | Predicting | Top-5% Overlap | Top-10% Overlap | Cosine")
print("  (first K)  | Layer L    |                |                 |")
print("  " + "-" * 72)

for nav_depth in [2, 4, 6]:
    for target_layer in range(nav_depth, n_layers):
        overlaps_5 = []
        overlaps_10 = []
        cosines = []

        target_mlp = model.transformer.h[target_layer].mlp

        for p_idx in range(len(test_prompts)):
            # Use hidden state after nav_depth layers as input
            h_nav = all_hidden_states[p_idx][nav_depth]  # output of layer nav_depth-1

            # Predict: run target layer's weights on this hidden state
            # This is "what would this layer do if it saw the early representation?"
            with torch.no_grad():
                pred_act = F.gelu(target_mlp.c_fc(h_nav.unsqueeze(0))).squeeze(0)

            actual_act = all_mlp_acts[target_layer][p_idx]

            k5 = max(1, int(0.05 * ffn_dim))
            top_pred = set(torch.topk(pred_act.abs(), k5).indices.tolist())
            top_actual = set(torch.topk(actual_act.abs(), k5).indices.tolist())
            overlaps_5.append(len(top_pred & top_actual) / k5)

            k10 = max(1, int(0.10 * ffn_dim))
            top_pred10 = set(torch.topk(pred_act.abs(), k10).indices.tolist())
            top_actual10 = set(torch.topk(actual_act.abs(), k10).indices.tolist())
            overlaps_10.append(len(top_pred10 & top_actual10) / k10)

            cos = F.cosine_similarity(pred_act.unsqueeze(0), actual_act.unsqueeze(0)).item()
            cosines.append(cos)

        print(f"  First {nav_depth:2d}    | Layer {target_layer:2d}   | {np.mean(overlaps_5):13.1%} | {np.mean(overlaps_10):14.1%} | {np.mean(cosines):.4f}")
    print()

# ── Test 3: The real test — run first K layers fully, predict+sparse the rest ──
print("\n[4/5] End-to-end: first K layers full, rest sparse...")

def generate_full(model, tokenizer, prompt, max_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

import types

def run_sparse_generation(model, tokenizer, prompt, nav_depth, keep_frac, max_tokens=20):
    """Run first nav_depth layers at full precision, rest with sparse FFN."""
    k = max(1, int(keep_frac * ffn_dim))

    originals = {}
    for i in range(nav_depth, n_layers):
        originals[i] = model.transformer.h[i].mlp.forward
        target_mlp = model.transformer.h[i].mlp

        def make_sparse_forward(layer_idx, k_val):
            mlp = model.transformer.h[layer_idx].mlp

            def sparse_forward(self_mlp, x):
                # Full FFN computation
                h = mlp.c_fc(x)
                h_act = mlp.act(h)

                # But mask: keep only top-k by magnitude PER TOKEN
                with torch.no_grad():
                    topk = torch.topk(h_act.abs(), k_val, dim=-1)
                    mask = torch.zeros_like(h_act)
                    mask.scatter_(-1, topk.indices, 1.0)

                h_masked = h_act * mask
                out = mlp.c_proj(h_masked)
                out = mlp.dropout(out)
                return out

            return sparse_forward

        model.transformer.h[i].mlp.forward = types.MethodType(
            make_sparse_forward(i, k), model.transformer.h[i].mlp
        )

    result = generate_full(model, tokenizer, prompt, max_tokens)

    for i in originals:
        model.transformer.h[i].mlp.forward = originals[i]

    return result

test_cases = [
    "The capital of France is",
    "Shakespeare wrote",
    "Water is made of",
    "The meaning of life is",
]

for prompt in test_cases:
    full_out = generate_full(model, tokenizer, prompt)
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Full:  '{full_out}'")

    for nav_depth in [4, 6, 8]:
        for keep_frac in [0.5, 0.3, 0.1]:
            sparse_out = run_sparse_generation(model, tokenizer, prompt, nav_depth, keep_frac)
            match = "=" if sparse_out == full_out else "~" if sparse_out[:40] == full_out[:40] else "X"
            n_sparse = n_layers - nav_depth
            print(f"  Nav={nav_depth} Sparse={n_sparse}L Keep={keep_frac:.0%}: '{sparse_out[:70]}' [{match}]")

# ── Test 4: Oracle test — what if we had PERFECT prediction? ──
print("\n\n[5/5] Oracle test: using ACTUAL top-k (perfect navigator)...")
print("  (Upper bound: if we KNEW which neurons to keep, how much can we prune?)")

for prompt in test_cases[:2]:
    full_out = generate_full(model, tokenizer, prompt)
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Full:  '{full_out}'")

    # Oracle: mask using each token's OWN activation magnitudes (perfect prediction)
    for keep_frac in [0.5, 0.3, 0.2, 0.1, 0.05]:
        k = max(1, int(keep_frac * ffn_dim))
        originals = {}

        for i in range(n_layers):
            originals[i] = model.transformer.h[i].mlp.forward
            def make_oracle_forward(layer_idx, k_val):
                mlp = model.transformer.h[layer_idx].mlp
                def oracle_forward(self_mlp, x):
                    h = mlp.c_fc(x)
                    h_act = mlp.act(h)
                    # Oracle: keep top-k by actual magnitude
                    topk = torch.topk(h_act.abs(), k_val, dim=-1)
                    mask = torch.zeros_like(h_act)
                    mask.scatter_(-1, topk.indices, 1.0)
                    h_masked = h_act * mask
                    out = mlp.c_proj(h_masked)
                    out = mlp.dropout(out)
                    return out
                return oracle_forward

            model.transformer.h[i].mlp.forward = types.MethodType(
                make_oracle_forward(i, k), model.transformer.h[i].mlp
            )

        oracle_out = generate_full(model, tokenizer, prompt)
        for i in originals:
            model.transformer.h[i].mlp.forward = originals[i]

        match = "MATCH" if oracle_out == full_out else "diff"
        print(f"  Oracle keep {keep_frac:4.0%} ({k:4d}/{ffn_dim}): '{oracle_out[:70]}' [{match}]")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
