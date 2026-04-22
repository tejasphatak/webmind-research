"""
CPU DB Engine — Run a 7B model with disk-backed weights and LRU cache.

The model weights live on disk. An LRU cache holds only what fits in the
RAM budget. Layers are loaded on-demand, computed, then potentially evicted.

This simulates running a 70B model on a phone with 2-3 GB RAM.
We use Qwen2.5-7B (15 GB FP16) with configurable cache sizes.

Architecture:
  Disk (SSD) → LRU Cache (limited RAM) → CPU compute → output
"""

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from collections import OrderedDict
import os
import time
import json
import gc
import sys

MODEL_NAME = "Qwen/Qwen2.5-7B"
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/"
)

# Find the snapshot dir
snapshot_dirs = []
for d in os.listdir(MODEL_PATH):
    full = os.path.join(MODEL_PATH, d)
    if os.path.isdir(full):
        snapshot_dirs.append(full)
SNAPSHOT = snapshot_dirs[0]

print("=" * 70)
print("CPU DB ENGINE — 7B Model, Disk-Backed, LRU Cache")
print("=" * 70)

# ── Load config and tokenizer (tiny, always in RAM) ──
config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

n_layers = config.num_hidden_layers
hidden_dim = config.hidden_size
ffn_dim = config.intermediate_size
n_heads = config.num_attention_heads
n_kv_heads = config.num_key_value_heads
head_dim = hidden_dim // n_heads
vocab_size = config.vocab_size
rms_norm_eps = config.rms_norm_eps

print(f"  Model: {MODEL_NAME}")
print(f"  Layers: {n_layers}, Hidden: {hidden_dim}, FFN: {ffn_dim}")
print(f"  Heads: {n_heads}, KV heads: {n_kv_heads}, Head dim: {head_dim}")
print(f"  Vocab: {vocab_size}")

# ── Weight index: map tensor names → safetensor files ──
print("\n  Building weight index...")

weight_index_path = os.path.join(SNAPSHOT, "model.safetensors.index.json")
with open(weight_index_path) as f:
    weight_map = json.load(f)["weight_map"]

# Group by layer
layer_keys = {}
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    layer_keys[i] = [k for k in weight_map if k.startswith(prefix)]

embed_keys = [k for k in weight_map if "embed_tokens" in k]
norm_keys = [k for k in weight_map if k.startswith("model.norm.")]
lm_head_keys = [k for k in weight_map if k.startswith("lm_head.")]

print(f"  Weight keys per layer: {len(layer_keys[0])}")
print(f"  Embed keys: {len(embed_keys)}, Norm keys: {len(norm_keys)}, LM head keys: {len(lm_head_keys)}")


# ══════════════════════════════════════════════════════════════════════
# THE DB: Disk-backed weight store with LRU cache
# ══════════════════════════════════════════════════════════════════════

class ModelDB:
    """
    Treats the model as a database on disk.
    LRU cache holds recently used layers in RAM.
    Cache miss → load from disk → compute → potentially evict.
    """

    def __init__(self, snapshot_path, weight_map, cache_budget_bytes):
        self.snapshot_path = snapshot_path
        self.weight_map = weight_map
        self.cache_budget = cache_budget_bytes
        self.cache = OrderedDict()  # key → tensor dict
        self.cache_sizes = {}       # key → size in bytes
        self.current_size = 0
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'bytes_loaded': 0}
        self._file_handles = {}

    def _get_file(self, filename):
        """Get or open a safetensors file handle."""
        if filename not in self._file_handles:
            path = os.path.join(self.snapshot_path, filename)
            self._file_handles[filename] = safe_open(path, framework="pt", device="cpu")
        return self._file_handles[filename]

    def _load_tensor(self, key):
        """Load a single tensor from disk."""
        filename = self.weight_map[key]
        f = self._get_file(filename)
        return f.get_tensor(key)

    def _evict_until_fits(self, needed_bytes):
        """Evict LRU entries until we have room."""
        while self.current_size + needed_bytes > self.cache_budget and self.cache:
            evicted_key, evicted_data = self.cache.popitem(last=False)  # FIFO/LRU
            evicted_size = self.cache_sizes.pop(evicted_key)
            self.current_size -= evicted_size
            self.stats['evictions'] += 1
            del evicted_data
            gc.collect()

    def get_layer(self, layer_idx, keys):
        """
        Get all weights for a layer. Cache hit → instant. Miss → load from disk.
        Returns dict of {key: tensor}.
        """
        cache_key = f"layer_{layer_idx}"

        if cache_key in self.cache:
            self.stats['hits'] += 1
            self.cache.move_to_end(cache_key)  # mark as recently used
            return self.cache[cache_key]

        # Cache miss — load from disk
        self.stats['misses'] += 1
        t0 = time.time()

        layer_data = {}
        total_bytes = 0
        for key in keys:
            tensor = self._load_tensor(key)
            layer_data[key] = tensor
            total_bytes += tensor.numel() * tensor.element_size()

        load_time = time.time() - t0
        self.stats['bytes_loaded'] += total_bytes

        # Evict old entries to make room
        self._evict_until_fits(total_bytes)

        # Insert into cache
        self.cache[cache_key] = layer_data
        self.cache_sizes[cache_key] = total_bytes
        self.current_size += total_bytes

        return layer_data

    def get_tensor(self, key):
        """Get a single tensor (for embeddings, etc). Always cached."""
        cache_key = f"single_{key}"
        if cache_key in self.cache:
            self.stats['hits'] += 1
            return self.cache[cache_key]

        self.stats['misses'] += 1
        tensor = self._load_tensor(key)
        size = tensor.numel() * tensor.element_size()

        self._evict_until_fits(size)
        self.cache[cache_key] = tensor
        self.cache_sizes[cache_key] = size
        self.current_size += size
        return tensor

    def print_stats(self):
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        print(f"  DB Stats:")
        print(f"    Cache hits:    {self.stats['hits']}")
        print(f"    Cache misses:  {self.stats['misses']}")
        print(f"    Hit rate:      {hit_rate:.1%}")
        print(f"    Evictions:     {self.stats['evictions']}")
        print(f"    Bytes loaded:  {self.stats['bytes_loaded'] / 1e9:.2f} GB")
        print(f"    Cache usage:   {self.current_size / 1e9:.2f} / {self.cache_budget / 1e9:.2f} GB")


# ══════════════════════════════════════════════════════════════════════
# CPU-only transformer forward pass, layer by layer from DB
# ══════════════════════════════════════════════════════════════════════

def rms_norm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def build_rope(head_dim, seq_len, base=1000000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

def forward_layer(db, layer_idx, hidden_states, cos, sin, kv_cache=None):
    """Forward pass through one transformer layer, loading weights from DB."""
    prefix = f"model.layers.{layer_idx}."
    weights = db.get_layer(layer_idx, layer_keys[layer_idx])

    bsz, seq_len, _ = hidden_states.shape

    # Input layernorm
    ln_weight = weights[prefix + "input_layernorm.weight"]
    normed = rms_norm(hidden_states.float(), ln_weight.float(), rms_norm_eps)

    # Self-attention
    W_q = weights[prefix + "self_attn.q_proj.weight"].float()
    W_k = weights[prefix + "self_attn.k_proj.weight"].float()
    W_v = weights[prefix + "self_attn.v_proj.weight"].float()
    W_o = weights[prefix + "self_attn.o_proj.weight"].float()

    q_bias = weights.get(prefix + "self_attn.q_proj.bias")
    k_bias = weights.get(prefix + "self_attn.k_proj.bias")
    v_bias = weights.get(prefix + "self_attn.v_proj.bias")

    q = F.linear(normed, W_q, q_bias.float() if q_bias is not None else None)
    k = F.linear(normed, W_k, k_bias.float() if k_bias is not None else None)
    v = F.linear(normed, W_v, v_bias.float() if v_bias is not None else None)

    # Reshape for multi-head attention
    q = q.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    q, k = apply_rotary_pos_emb(q, k, cos[:, :, :seq_len, :], sin[:, :, :seq_len, :])

    # GQA: repeat KV heads
    if n_kv_heads < n_heads:
        repeat_factor = n_heads // n_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    # Attention
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

    # Causal mask
    if seq_len > 1:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    # Reshape and project
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, hidden_dim)
    attn_output = F.linear(attn_output, W_o)

    # Residual
    hidden_states = hidden_states.float() + attn_output

    # Post-attention layernorm
    ln2_weight = weights[prefix + "post_attention_layernorm.weight"]
    normed2 = rms_norm(hidden_states, ln2_weight.float(), rms_norm_eps)

    # MLP (SwiGLU)
    W_gate = weights[prefix + "mlp.gate_proj.weight"].float()
    W_up = weights[prefix + "mlp.up_proj.weight"].float()
    W_down = weights[prefix + "mlp.down_proj.weight"].float()

    gate = F.silu(F.linear(normed2, W_gate))
    up = F.linear(normed2, W_up)
    mlp_output = F.linear(gate * up, W_down)

    # Residual
    hidden_states = hidden_states + mlp_output

    return hidden_states


def forward_full(db, input_ids):
    """Full forward pass through the model, loading each layer from DB."""
    bsz, seq_len = input_ids.shape

    # Embedding (always loaded)
    embed_weight = db.get_tensor("model.embed_tokens.weight").float()
    hidden_states = F.embedding(input_ids, embed_weight)

    # Build RoPE
    cos, sin = build_rope(head_dim, seq_len, base=getattr(config, 'rope_theta', 1000000.0))

    # Forward through all layers
    for i in range(n_layers):
        hidden_states = forward_layer(db, i, hidden_states, cos, sin)

    # Final norm
    norm_weight = db.get_tensor("model.norm.weight").float()
    hidden_states = rms_norm(hidden_states, norm_weight, rms_norm_eps)

    # LM head
    lm_weight = db.get_tensor("lm_head.weight").float()
    logits = F.linear(hidden_states, lm_weight)

    return logits


def generate_token_by_token(db, prompt, max_tokens=20):
    """Generate tokens one at a time, loading layers from DB each time."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    generated = input_ids[0].tolist()
    total_time = 0
    prefill_time = 0

    for step in range(max_tokens):
        t0 = time.time()

        # Full forward pass (no KV cache for simplicity)
        ids = torch.tensor([generated], dtype=torch.long)
        logits = forward_full(db, ids)

        # Greedy: take the last token's prediction
        next_token = logits[0, -1, :].argmax().item()
        elapsed = time.time() - t0

        if step == 0:
            prefill_time = elapsed
        total_time += elapsed

        generated.append(next_token)

        # Stop on EOS
        if next_token == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    gen_tokens = len(generated) - input_ids.shape[1]

    return text, prefill_time, total_time, gen_tokens


# ══════════════════════════════════════════════════════════════════════
# BENCHMARK: Different cache sizes
# ══════════════════════════════════════════════════════════════════════

prompt = "The capital of France is"

# Test with different cache budgets
cache_sizes_gb = [1.0, 2.0, 4.0, 8.0, 14.0]

print("\n" + "=" * 70)
print("BENCHMARK: Cache size vs generation speed")
print(f"Prompt: '{prompt}'")
print("=" * 70)

for cache_gb in cache_sizes_gb:
    cache_bytes = int(cache_gb * 1e9)

    # Clear memory
    gc.collect()

    print(f"\n{'─' * 60}")
    print(f"  Cache budget: {cache_gb} GB")
    print(f"{'─' * 60}")

    db = ModelDB(SNAPSHOT, weight_map, cache_bytes)

    try:
        # Generate 5 tokens (quick test)
        text, prefill, total, n_tokens = generate_token_by_token(db, prompt, max_tokens=5)
        gen_time = total - prefill
        tok_per_sec = n_tokens / gen_time if gen_time > 0 and n_tokens > 0 else 0

        print(f"  Output: '{text}'")
        print(f"  Prefill (first token): {prefill:.2f}s")
        print(f"  Generation ({n_tokens} tokens): {gen_time:.2f}s")
        print(f"  Speed: {tok_per_sec:.2f} tokens/sec")
        db.print_stats()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        break

    del db
    gc.collect()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
