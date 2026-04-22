"""
Activation Index Experiment

Can we build a data structure that maps input → activation pattern?

Tests:
1. Per-token consistency: does the same word activate the same neurons across contexts?
2. Clustering: how many unique patterns exist?
3. Embedding-space index: can we look up nearest activation pattern?
4. Bloom filter: can a probabilistic structure predict activations?

Model: Qwen2-0.5B (SwiGLU)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict
import time
import types

print("=" * 70)
print("ACTIVATION INDEX EXPERIMENT")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model.eval()

n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
ffn_dim = model.config.intermediate_size
print(f"  Layers: {n_layers}, Hidden: {hidden_dim}, FFN: {ffn_dim}")

# ══════════════════════════════════════════════════════════════════════
# TEST 1: Per-token activation consistency across contexts
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[1/4] Per-token activation consistency")
print("  (Does 'France' activate the same neurons in different sentences?)")
print("=" * 70)

# Words appearing in multiple different contexts
test_contexts = {
    "France": [
        "The capital of France is",
        "France is a country in Europe",
        "I visited France last summer",
        "The president of France announced",
        "France won the world cup in",
        "The history of France is rich",
    ],
    "water": [
        "Water is made of hydrogen",
        "I need a glass of water",
        "The water temperature is rising",
        "Clean water is essential for",
        "Water flows downhill because of",
        "The water in the ocean is",
    ],
    "the": [
        "The capital of France is",
        "The cat sat on the mat",
        "The president of the United",
        "The derivative of x squared",
        "The speed of light in",
        "The meaning of life is",
    ],
    "is": [
        "The capital of France is",
        "Water is made of hydrogen",
        "Life is what you make",
        "Python is a programming language",
        "The sky is blue because",
        "This is a test of",
    ],
}

# For each target word, capture its activation when it appears in different contexts
# We need to find which token position corresponds to the target word

mlp_acts_store = {}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        with torch.no_grad():
            x = input[0].float()
            gate = F.silu(module.gate_proj(x))
            up = module.up_proj(x)
            combined = gate * up
            mlp_acts_store[layer_idx] = combined.detach()  # [batch, seq, ffn]
    return hook_fn

hooks = []
for i in range(n_layers):
    hooks.append(model.model.layers[i].mlp.register_forward_hook(make_hook(i)))

# Analyze three representative layers
test_layers = [0, 11, 23]  # first, middle, last

for word, contexts in test_contexts.items():
    print(f"\n  Word: '{word}'")

    # Collect activations for this word across contexts
    word_activations = {layer: [] for layer in test_layers}

    for ctx in contexts:
        tokens = tokenizer.encode(ctx)
        token_strs = [tokenizer.decode([t]) for t in tokens]

        # Find position of target word (case-insensitive, partial match ok)
        target_pos = None
        for pos, ts in enumerate(token_strs):
            if word.lower() in ts.lower():
                target_pos = pos
                break

        if target_pos is None:
            continue

        inputs = tokenizer(ctx, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)

        for layer in test_layers:
            act = mlp_acts_store[layer][0, target_pos, :].cpu()
            word_activations[layer].append(act)

    # Compare activations across contexts
    for layer in test_layers:
        acts = word_activations[layer]
        if len(acts) < 2:
            continue

        # Pairwise cosine similarity
        cosines = []
        top5_overlaps = []
        k5 = max(1, int(0.05 * ffn_dim))

        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                cos = F.cosine_similarity(acts[i].unsqueeze(0), acts[j].unsqueeze(0)).item()
                cosines.append(cos)

                ti = set(torch.topk(acts[i].abs(), k5).indices.tolist())
                tj = set(torch.topk(acts[j].abs(), k5).indices.tolist())
                top5_overlaps.append(len(ti & tj) / k5)

        print(f"    Layer {layer:2d}: cosine={np.mean(cosines):.4f}  top-5% overlap={np.mean(top5_overlaps):.1%}  (across {len(acts)} contexts)")

for h in hooks:
    h.remove()

# ══════════════════════════════════════════════════════════════════════
# TEST 2: How many unique activation patterns exist?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[2/4] Activation pattern clustering")
print("  (How many distinct patterns exist across many inputs?)")
print("=" * 70)

# Run many diverse prompts and collect activation patterns
diverse_prompts = [
    "The capital of France is",
    "How to cook pasta",
    "Einstein discovered relativity when",
    "The stock market crashed in",
    "Dogs are better than cats because",
    "The chemical formula for water",
    "In machine learning, overfitting occurs",
    "The tallest building in the world",
    "Mozart composed his first symphony",
    "Photosynthesis requires sunlight and",
    "The human brain has approximately",
    "JavaScript was created in ten",
    "The great wall of China was",
    "Quantum entanglement suggests that",
    "The French Revolution began in",
    "To solve a quadratic equation",
    "The Amazon rainforest produces about",
    "TCP IP protocol works by",
    "Shakespeare was born in Stratford",
    "The periodic table organizes elements",
    "Black holes form when massive",
    "The speed of sound in air",
    "Neural networks were inspired by",
    "The Treaty of Versailles was",
    "Fibonacci sequence starts with",
    "Climate change is caused primarily",
    "The Mona Lisa was painted by",
    "DNA contains genetic information encoded",
    "Supply and demand determines the",
    "The theory of evolution proposes",
]

hooks = []
for i in range(n_layers):
    hooks.append(model.model.layers[i].mlp.register_forward_hook(make_hook(i)))

# Collect patterns: for each prompt, get the LAST token's activation (most relevant for next-token prediction)
all_patterns = {layer: [] for layer in test_layers}
all_binary_patterns = {layer: [] for layer in test_layers}

for prompt in diverse_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    for layer in test_layers:
        act = mlp_acts_store[layer][0, -1, :].cpu()
        all_patterns[layer].append(act)
        # Binary pattern: which neurons are in top 10%
        k10 = max(1, int(0.1 * ffn_dim))
        binary = torch.zeros(ffn_dim)
        binary[torch.topk(act.abs(), k10).indices] = 1
        all_binary_patterns[layer].append(binary)

for h in hooks:
    h.remove()

# Analyze clustering potential
for layer in test_layers:
    patterns = torch.stack(all_patterns[layer])  # [n_prompts, ffn_dim]
    binary = torch.stack(all_binary_patterns[layer])

    # Pairwise cosine similarity matrix
    norms = patterns.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = patterns / norms
    cos_matrix = normed @ normed.T

    # Stats
    upper = cos_matrix[torch.triu(torch.ones_like(cos_matrix), diagonal=1).bool()]
    print(f"\n  Layer {layer:2d}:")
    print(f"    Pairwise cosine similarity: mean={upper.mean():.4f}, std={upper.std():.4f}")
    print(f"    Min={upper.min():.4f}, Max={upper.max():.4f}")

    # Binary pattern overlap (Jaccard similarity)
    intersections = binary @ binary.T  # [n, n]
    unions = k10 * 2 - intersections  # simplified
    jaccard = intersections / unions.clamp(min=1)
    upper_j = jaccard[torch.triu(torch.ones_like(jaccard), diagonal=1).bool()]
    print(f"    Binary top-10% Jaccard: mean={upper_j.mean():.4f}")

    # How many clusters would we need? Simple: count eigenvalues > threshold
    try:
        eigenvalues = torch.linalg.eigvalsh(cos_matrix)
        eigenvalues = eigenvalues.flip(0)  # descending
        total_energy = eigenvalues.sum()
        cumsum = torch.cumsum(eigenvalues, 0) / total_energy
        for target in [0.8, 0.9, 0.95]:
            n_clusters = (cumsum < target).sum().item() + 1
            print(f"    Clusters for {target:.0%} variance: {n_clusters}")
    except:
        print("    (eigenvalue computation failed)")

# ══════════════════════════════════════════════════════════════════════
# TEST 3: Embedding-based index (FAISS-style nearest neighbor)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[3/4] Embedding-based activation index")
print("  (Store input_embedding → activation_mask, nearest-neighbor lookup)")
print("=" * 70)

# Build index from diverse prompts, test on new prompts
# Index: map last-token hidden state → activation pattern
# At inference: compute hidden state up to some point → look up → use mask

hooks = []
hidden_states_store = {}

def make_hidden_hook(layer_idx):
    def hook_fn(module, input, output):
        # Capture input to MLP (post-attention, post-layernorm)
        hidden_states_store[layer_idx] = input[0][:, -1, :].detach().float()
    return hook_fn

for i in range(n_layers):
    hooks.append(model.model.layers[i].mlp.register_forward_hook(make_hidden_hook(i)))

# Build index from training prompts
index_prompts = diverse_prompts[:20]
test_index_prompts = diverse_prompts[20:]

index_hidden = {layer: [] for layer in test_layers}
index_acts = {layer: [] for layer in test_layers}

mlp_acts_store2 = {}
def make_act_hook(layer_idx):
    def hook_fn(module, input, output):
        with torch.no_grad():
            x = input[0].float()
            gate = F.silu(module.gate_proj(x))
            up = module.up_proj(x)
            mlp_acts_store2[layer_idx] = (gate * up)[:, -1, :].detach()
    return hook_fn

# Re-register activation hooks
for h in hooks:
    h.remove()
hooks = []
for i in range(n_layers):
    hooks.append(model.model.layers[i].mlp.register_forward_hook(make_act_hook(i)))
    h2 = model.model.layers[i].mlp.register_forward_hook(make_hidden_hook(i))
    hooks.append(h2)

for prompt in index_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    for layer in test_layers:
        index_hidden[layer].append(hidden_states_store[layer].squeeze(0))
        index_acts[layer].append(mlp_acts_store2[layer].squeeze(0))

# Now test: for each test prompt, find nearest hidden state in index, use its activation mask
for prompt in test_index_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)

    print(f"\n  Query: '{prompt[:50]}...'")
    for layer in test_layers:
        query_hidden = hidden_states_store[layer].squeeze(0)
        query_act = mlp_acts_store2[layer].squeeze(0)

        # Find nearest in index
        idx_h = torch.stack(index_hidden[layer])
        sims = F.cosine_similarity(query_hidden.unsqueeze(0), idx_h)
        best_idx = sims.argmax().item()
        best_sim = sims[best_idx].item()

        # Use best match's activation pattern as mask
        best_act = index_acts[layer][best_idx]

        k10 = max(1, int(0.1 * ffn_dim))
        top_pred = set(torch.topk(best_act.abs(), k10).indices.tolist())
        top_actual = set(torch.topk(query_act.abs(), k10).indices.tolist())
        overlap = len(top_pred & top_actual) / k10

        cos = F.cosine_similarity(query_act.unsqueeze(0), best_act.unsqueeze(0)).item()

        print(f"    Layer {layer:2d}: nearest='{index_prompts[best_idx][:30]}...' "
              f"(sim={best_sim:.3f}) → top-10% overlap={overlap:.1%}, cosine={cos:.4f}")

for h in hooks:
    h.remove()

# ══════════════════════════════════════════════════════════════════════
# TEST 4: Token-level activation index (the practical one)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[4/4] Token-level activation index + oracle pruning with it")
print("  (Build per-token average activation, use as mask)")
print("=" * 70)

# For each token that appears in our prompts, average its activation across all positions
# Then at inference, look up the current token's average activation mask

hooks = []
all_token_acts = {}  # {layer: {token_id: [list of activations]}}

def make_per_token_hook(layer_idx):
    def hook_fn(module, input, output):
        with torch.no_grad():
            x = input[0].float()
            gate = F.silu(module.gate_proj(x))
            up = module.up_proj(x)
            combined = gate * up
            all_token_acts[layer_idx] = combined.detach()  # [batch, seq, ffn]
    return hook_fn

for i in range(n_layers):
    hooks.append(model.model.layers[i].mlp.register_forward_hook(make_per_token_hook(i)))

# Collect all token activations from all prompts
token_act_sums = {}  # {(layer, token_id): [sum_vector, count]}

all_collect_prompts = diverse_prompts + [
    "Artificial intelligence will transform",
    "The temperature today is around",
    "Mathematics is the language of",
    "Cooking requires patience and good",
    "The universe began with a big",
    "Programming in Python is easy",
    "The Roman Empire fell because",
    "Electric cars are becoming more",
    "The human body contains about",
    "Music has the power to heal",
]

for prompt in all_collect_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs['input_ids'][0].tolist()
    with torch.no_grad():
        model(**inputs)
    for layer in test_layers:
        acts = all_token_acts[layer][0]  # [seq, ffn]
        for pos, tid in enumerate(token_ids):
            key = (layer, tid)
            if key not in token_act_sums:
                token_act_sums[key] = [torch.zeros(ffn_dim), 0]
            token_act_sums[key][0] += acts[pos].cpu().float()
            token_act_sums[key][1] += 1

for h in hooks:
    h.remove()

# Build token index: average activation per token
token_index = {}
for (layer, tid), (sum_vec, count) in token_act_sums.items():
    token_index[(layer, tid)] = sum_vec / count

n_tokens_indexed = len(set(tid for (_, tid) in token_index.keys()))
print(f"  Indexed {n_tokens_indexed} unique tokens across {len(all_collect_prompts)} prompts")

# Test: how well does the token index predict activations on NEW prompts?
test_token_prompts = [
    "The history of France includes many",
    "Water and oil do not mix because",
    "Shakespeare influenced modern literature through",
]

hooks = []
for i in range(n_layers):
    hooks.append(model.model.layers[i].mlp.register_forward_hook(make_per_token_hook(i)))

for prompt in test_token_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs['input_ids'][0].tolist()
    with torch.no_grad():
        model(**inputs)

    print(f"\n  Prompt: '{prompt}'")
    for layer in test_layers:
        actual_acts = all_token_acts[layer][0].cpu().float()  # [seq, ffn]

        overlaps = []
        cosines = []
        found = 0

        for pos, tid in enumerate(token_ids):
            key = (layer, tid)
            if key in token_index:
                found += 1
                pred = token_index[key]
                actual = actual_acts[pos]

                k10 = max(1, int(0.1 * ffn_dim))
                tp = set(torch.topk(pred.abs(), k10).indices.tolist())
                ta = set(torch.topk(actual.abs(), k10).indices.tolist())
                overlaps.append(len(tp & ta) / k10)

                cos = F.cosine_similarity(pred.unsqueeze(0), actual.unsqueeze(0)).item()
                cosines.append(cos)

        if overlaps:
            print(f"    Layer {layer:2d}: top-10% overlap={np.mean(overlaps):.1%}, "
                  f"cosine={np.mean(cosines):.4f} ({found}/{len(token_ids)} tokens in index)")
        else:
            print(f"    Layer {layer:2d}: no tokens found in index")

for h in hooks:
    h.remove()

# ── End-to-end: token-index-guided sparse generation ──
print("\n  --- End-to-end: token-index-guided sparse generation ---")

def generate(model, tokenizer, prompt, max_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id or 0)
    return tokenizer.decode(output[0], skip_special_tokens=True)

gen_prompts = ["The capital of France is", "Water is made of"]

for prompt in gen_prompts:
    full_out = generate(model, tokenizer, prompt)
    print(f"\n  Prompt: '{prompt}'")
    print(f"  Full:   '{full_out}'")

    for keep_frac in [0.5, 0.3]:
        k = max(1, int(keep_frac * ffn_dim))
        originals = {}

        for i in range(n_layers):
            originals[i] = model.model.layers[i].mlp.forward

            def make_token_index_forward(layer_idx, k_val):
                mlp = model.model.layers[layer_idx].mlp

                def indexed_forward(self_mlp, x):
                    gate = F.silu(mlp.gate_proj(x))
                    up = mlp.up_proj(x)
                    combined = gate * up

                    # Use actual top-k as mask (oracle — best case for any index)
                    with torch.no_grad():
                        topk = torch.topk(combined.abs(), k_val, dim=-1)
                        mask = torch.zeros_like(combined)
                        mask.scatter_(-1, topk.indices, 1.0)

                    return mlp.down_proj(combined * mask)

                return indexed_forward

            model.model.layers[i].mlp.forward = types.MethodType(
                make_token_index_forward(i, k), model.model.layers[i].mlp
            )

        idx_out = generate(model, tokenizer, prompt)
        for i in originals:
            model.model.layers[i].mlp.forward = originals[i]

        match = "MATCH" if idx_out == full_out else "diff"
        print(f"  Oracle keep {keep_frac:.0%}: '{idx_out[:70]}' [{match}]")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
