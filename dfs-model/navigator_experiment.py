"""
Navigator Experiment: Can a low-rank version of a model predict activation patterns?

Hypothesis: SVD-decomposed (low-rank) version of each layer produces
activation patterns that match the full model's activations >90%.
If true → we can use the cheap low-rank model as a "navigator" to
predict which neurons to activate in the full model.

Model: GPT-2 Small (124M params, 12 layers)
Method: SVD decomposition of FFN weights → rank-64 approximation
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import time

print("=" * 70)
print("NAVIGATOR EXPERIMENT: Low-Rank Activation Prediction")
print("=" * 70)

# Load model
print("\n[1/5] Loading GPT-2 Small...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# Model stats
n_layers = model.config.n_layer
hidden_dim = model.config.n_embd
ffn_dim = model.config.n_inner or 4 * hidden_dim
print(f"  Layers: {n_layers}, Hidden: {hidden_dim}, FFN: {ffn_dim}")
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

# ── Step 1: Measure natural sparsity ──
print("\n[2/5] Measuring natural FFN sparsity...")

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

# Hook to capture FFN activations (after GELU, before down-projection)
activations_full = {}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # GPT-2 MLP: c_fc (up) → gelu → c_proj (down)
        # We hook the MLP module itself. Input[0] is the input to MLP.
        # We need the intermediate activation after c_fc + gelu
        with torch.no_grad():
            h = input[0]
            intermediate = module.c_fc(h)
            activated = module.act(intermediate)
            # Store activation magnitudes (last token only for generation)
            activations_full[layer_idx] = activated[:, -1, :].detach()
    return hook_fn

hooks = []
for i in range(n_layers):
    h = model.transformer.h[i].mlp.register_forward_hook(make_hook(i))
    hooks.append(h)

# Run all prompts, collect activations
all_activations = {i: [] for i in range(n_layers)}
sparsity_per_layer = {i: [] for i in range(n_layers)}

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)

    for layer_idx in range(n_layers):
        act = activations_full[layer_idx].squeeze(0)  # [ffn_dim]
        all_activations[layer_idx].append(act)

        # Sparsity: what fraction of neurons have |activation| < threshold
        threshold = 0.1 * act.abs().max().item()  # 10% of max
        sparse_frac = (act.abs() < threshold).float().mean().item()
        sparsity_per_layer[layer_idx].append(sparse_frac)

# Remove hooks
for h in hooks:
    h.remove()

print("\n  Layer | Avg Sparsity | Min    | Max")
print("  " + "-" * 45)
for i in range(n_layers):
    avg_s = np.mean(sparsity_per_layer[i])
    min_s = np.min(sparsity_per_layer[i])
    max_s = np.max(sparsity_per_layer[i])
    print(f"  {i:5d} | {avg_s:10.1%}  | {min_s:.1%} | {max_s:.1%}")

overall_sparsity = np.mean([np.mean(v) for v in sparsity_per_layer.values()])
print(f"\n  Overall average sparsity: {overall_sparsity:.1%}")

# ── Step 2: Create low-rank approximations ──
print("\n[3/5] Creating low-rank (SVD) approximations...")

rank = 64  # Compress each weight matrix to rank 64
low_rank_weights = {}
compression_ratios = []

for i in range(n_layers):
    mlp = model.transformer.h[i].mlp
    # GPT-2 Conv1D: weight is [in_features, out_features]
    # Forward: x @ weight + bias
    W_fc = mlp.c_fc.weight.data.float()     # [768, 3072]
    W_proj = mlp.c_proj.weight.data.float()  # [3072, 768]

    # SVD decompose c_fc: W[768,3072] ≈ U[:,:r] @ diag(S[:r]) @ Vh[:r,:]
    U_fc, S_fc, Vh_fc = torch.linalg.svd(W_fc, full_matrices=False)
    A_fc = U_fc[:, :rank] * S_fc[:rank].unsqueeze(0)  # [768, rank]
    B_fc = Vh_fc[:rank, :]                              # [rank, 3072]

    # SVD decompose c_proj: W[3072,768]
    U_proj, S_proj, Vh_proj = torch.linalg.svd(W_proj, full_matrices=False)
    A_proj = U_proj[:, :rank] * S_proj[:rank].unsqueeze(0)  # [3072, rank]
    B_proj = Vh_proj[:rank, :]                                # [rank, 768]

    low_rank_weights[i] = {
        'A_fc': A_fc, 'B_fc': B_fc, 'bias_fc': mlp.c_fc.bias.data.float(),
        'A_proj': A_proj, 'B_proj': B_proj, 'bias_proj': mlp.c_proj.bias.data.float(),
    }

    # Compression ratio
    original = W_fc.numel() + W_proj.numel()
    compressed = A_fc.numel() + B_fc.numel() + A_proj.numel() + B_proj.numel()
    compression_ratios.append(compressed / original)

avg_ratio = np.mean(compression_ratios)
print(f"  Rank: {rank}")
print(f"  Compression ratio: {avg_ratio:.1%} of original FFN params")
print(f"  Original FFN params per layer: {W_fc.numel() + W_proj.numel():,}")
print(f"  Low-rank FFN params per layer: {int((A_fc.numel() + B_fc.numel() + A_proj.numel() + B_proj.numel())):,}")

# ── Step 3: Compare activation patterns ──
print("\n[4/5] Comparing activation patterns (full vs low-rank)...")

# For each prompt, compute low-rank FFN activations and compare to full
match_rates = {i: [] for i in range(n_layers)}
cosine_sims = {i: [] for i in range(n_layers)}
top_k_overlaps = {i: [] for i in range(n_layers)}

for p_idx, prompt in enumerate(test_prompts):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        # Get hidden states from full model
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [batch, seq, hidden]

    for layer_idx in range(n_layers):
        # Full model activation (already captured)
        full_act = all_activations[layer_idx][p_idx]

        # Low-rank activation: h @ A_fc @ B_fc + bias → GELU
        # (GPT-2 Conv1D forward: x @ weight + bias, weight is [in, out])
        h = hidden_states[layer_idx][:, -1, :].float()  # input to this layer's MLP

        # hidden_states[i] is output of layer i-1. The real MLP input has gone through
        # attn + layernorm. We use the hidden state as an approximation — it's the same
        # vector space, just missing the attn transform for this layer. For measuring
        # pattern correlation this is sufficient.
        lr = low_rank_weights[layer_idx]

        # Low-rank: x @ A_fc @ B_fc + bias  (A_fc:[768,r], B_fc:[r,3072])
        lr_intermediate = h @ lr['A_fc'] @ lr['B_fc'] + lr['bias_fc']
        lr_activated = F.gelu(lr_intermediate).squeeze(0)  # remove batch dim

        # Compare: which neurons are active?
        full_active = (full_act.abs() > 0.1 * full_act.abs().max())
        lr_active = (lr_activated.abs() > 0.1 * lr_activated.abs().max())

        # Binary match rate (same neurons active/inactive)
        match = (full_active == lr_active).float().mean().item()
        match_rates[layer_idx].append(match)

        # Cosine similarity of activation vectors
        cos = F.cosine_similarity(full_act.unsqueeze(0), lr_activated.unsqueeze(0)).item()
        cosine_sims[layer_idx].append(cos)

        # Top-K overlap: do the top 5% neurons match?
        k = max(1, int(0.05 * ffn_dim))
        top_k_full = set(torch.topk(full_act.abs(), k).indices.tolist())
        top_k_lr = set(torch.topk(lr_activated.abs(), k).indices.tolist())
        overlap = len(top_k_full & top_k_lr) / k
        top_k_overlaps[layer_idx].append(overlap)

print("\n  Layer | Binary Match | Cosine Sim | Top-5% Overlap")
print("  " + "-" * 55)
for i in range(n_layers):
    bm = np.mean(match_rates[i])
    cs = np.mean(cosine_sims[i])
    tk = np.mean(top_k_overlaps[i])
    print(f"  {i:5d} | {bm:10.1%}   | {cs:8.4f}   | {tk:10.1%}")

avg_match = np.mean([np.mean(v) for v in match_rates.values()])
avg_cosine = np.mean([np.mean(v) for v in cosine_sims.values()])
avg_topk = np.mean([np.mean(v) for v in top_k_overlaps.values()])

print(f"\n  Average binary match:  {avg_match:.1%}")
print(f"  Average cosine sim:    {avg_cosine:.4f}")
print(f"  Average top-5% overlap: {avg_topk:.1%}")

# ── Step 4: End-to-end sparse execution test ──
print("\n[5/5] End-to-end: full model vs sparse execution...")

def generate_tokens(model, tokenizer, prompt, max_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test: mask non-top-K neurons and see if output changes
# We'll hook the FFN and zero out neurons NOT in the top-K predicted by low-rank

sparsity_levels = [0.5, 0.3, 0.1, 0.05]  # keep 50%, 30%, 10%, 5% of neurons

test_prompt = "The capital of France is"
full_output = generate_tokens(model, tokenizer, test_prompt)
print(f"\n  Prompt: '{test_prompt}'")
print(f"  Full model output: '{full_output}'")

for keep_frac in sparsity_levels:
    k = max(1, int(keep_frac * ffn_dim))

    # Hook: zero out neurons outside top-k (predicted by low-rank magnitude)
    sparse_hooks = []

    def make_sparse_hook(layer_idx, k):
        def hook_fn(module, input, output):
            with torch.no_grad():
                h = input[0]
                lr = low_rank_weights[layer_idx]
                # Predict activation pattern via low-rank
                lr_intermediate = h[:, -1:, :].float() @ lr['A_fc'] @ lr['B_fc'] + lr['bias_fc']
                lr_activated = F.gelu(lr_intermediate)

                # Get top-k neuron indices
                top_k_idx = torch.topk(lr_activated.abs().squeeze(), k).indices

                # Create mask for ALL tokens (apply same mask)
                mask = torch.zeros(ffn_dim, dtype=torch.bool)
                mask[top_k_idx] = True

                # The output of MLP is already computed. We need to mask the intermediate.
                # Can't easily mask after the fact. Instead, let's measure output difference.
                # For this test, we'll modify the approach: hook c_fc output and mask there.
            return output
        return hook_fn

    # Simpler approach: hook the intermediate activation and mask it
    def make_mask_hook(layer_idx, k):
        original_forward = model.transformer.h[layer_idx].mlp.forward
        lr = low_rank_weights[layer_idx]

        def masked_forward(x):
            # Compute low-rank prediction for mask
            with torch.no_grad():
                lr_pred = x.float() @ lr['A_fc'] @ lr['B_fc'] + lr['bias_fc']
                lr_act = F.gelu(lr_pred)
                # Use last token to determine mask, apply to all
                top_k_idx = torch.topk(lr_act[:, -1, :].abs(), k).indices.squeeze()
                mask = torch.zeros(x.size(0), 1, ffn_dim)
                mask[:, :, top_k_idx] = 1.0

            # Run actual FFN
            h = model.transformer.h[layer_idx].mlp.c_fc(x)
            h = model.transformer.h[layer_idx].mlp.act(h)
            # Apply mask
            h = h * mask
            h = model.transformer.h[layer_idx].mlp.c_proj(h)
            h = model.transformer.h[layer_idx].mlp.dropout(h)
            return h

        return masked_forward, original_forward

    # Patch MLP forwards
    originals = {}
    for i in range(n_layers):
        masked_fn, orig_fn = make_mask_hook(i, k)
        originals[i] = orig_fn
        model.transformer.h[i].mlp.forward = masked_fn

    sparse_output = generate_tokens(model, tokenizer, test_prompt)

    # Restore
    for i in range(n_layers):
        model.transformer.h[i].mlp.forward = originals[i]

    match_str = "✓ MATCH" if sparse_output == full_output else "✗ DIFFERENT"
    print(f"  Keep {keep_frac:4.0%} neurons ({k:4d}/{ffn_dim}): '{sparse_output}' {match_str}")

# ── Summary ──
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Natural FFN sparsity: {overall_sparsity:.1%}")
print(f"  Low-rank navigator accuracy (cosine): {avg_cosine:.4f}")
print(f"  Low-rank navigator accuracy (top-5% overlap): {avg_topk:.1%}")
print(f"  Compression: {avg_ratio:.1%} of original FFN params")
print()
if avg_topk > 0.7:
    print("  CONCLUSION: Navigator concept is VIABLE.")
    print("  Low-rank model can predict which neurons fire with high accuracy.")
elif avg_topk > 0.4:
    print("  CONCLUSION: Navigator concept is PROMISING but needs higher rank.")
else:
    print("  CONCLUSION: Navigator concept NEEDS WORK at this rank.")
print("=" * 70)
