#!/usr/bin/env python3
"""
Tensor-Parallel Async Distributed Inference — v1
=================================================
Splits weight matrices within transformer layers across simulated devices
with variable shard sizes. Tests async stale partial sums with realistic
network conditions (latency, jitter, packet loss).

Target: 500MB GPU RAM per device. Model: Qwen 2.5 1.5B.
"""

import os, sys, json, time, math, gc, random, argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B")
TRAIN_STEPS = 800
EVAL_TOKENS = 50_000
LR = 1e-5
BATCH_SIZE = 4
SEQ_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Network simulation parameters
NETWORK_PROFILES = {
    "ideal":    {"latency_ms": 0,   "jitter_pct": 0,    "loss_pct": 0},
    "lan":      {"latency_ms": 2,   "jitter_pct": 10,   "loss_pct": 0.1},
    "wifi":     {"latency_ms": 10,  "jitter_pct": 20,   "loss_pct": 1},
    "internet": {"latency_ms": 50,  "jitter_pct": 30,   "loss_pct": 2},
    "mobile":   {"latency_ms": 100, "jitter_pct": 40,   "loss_pct": 5},
    "hostile":  {"latency_ms": 200, "jitter_pct": 50,   "loss_pct": 10},
}


@dataclass
class Result:
    name: str
    network: str
    ppl: float
    ppl_baseline: float
    gap_pct: float
    hellaswag_acc: Optional[float] = None
    train_loss: float = 0.0
    train_time_s: float = 0.0
    packets_dropped: int = 0
    packets_total: int = 0


# ---------------------------------------------------------------------------
# Network simulator
# ---------------------------------------------------------------------------
class NetworkSim:
    """Simulates realistic network conditions for partial sum delivery."""

    def __init__(self, latency_ms=0, jitter_pct=0, loss_pct=0):
        self.latency_ms = latency_ms
        self.jitter_pct = jitter_pct
        self.loss_pct = loss_pct
        self.dropped = 0
        self.total = 0

    def send(self, tensor):
        """Simulate sending a tensor. Returns (tensor_or_None, actual_latency_ms).
        None means packet was lost — caller should use stale value."""
        self.total += 1

        # Packet loss
        if random.random() * 100 < self.loss_pct:
            self.dropped += 1
            return None, 0

        # Latency + jitter (we don't actually sleep — just track for metrics)
        jitter = self.latency_ms * self.jitter_pct / 100 * random.uniform(-1, 1)
        actual_latency = max(0, self.latency_ms + jitter)

        return tensor.detach(), actual_latency


# ---------------------------------------------------------------------------
# Tensor-parallel layer: splits weight matrices across simulated devices
# ---------------------------------------------------------------------------
class TensorParallelLinear(nn.Module):
    """A linear layer split across N devices with async aggregation."""

    def __init__(self, in_features, out_features, n_shards=4,
                 shard_weights=None, network=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_shards = n_shards
        self.network = network or NetworkSim()

        # Variable shard sizes: split the INPUT dimension (columns of weight)
        # Each device holds W[:, col_start:col_end] and computes partial output
        # Results are SUMMED (not concatenated) — true tensor parallelism
        if shard_weights is None:
            shard_weights = [1] * n_shards
        total_weight = sum(shard_weights)

        self.shard_sizes = []  # sizes along input dimension
        assigned = 0
        for i, w in enumerate(shard_weights):
            if i == len(shard_weights) - 1:
                size = in_features - assigned
            else:
                size = (in_features * w) // total_weight
            self.shard_sizes.append(size)
            assigned += size

        # Create weight shards: each is [out_features, shard_in_size]
        self.shards = nn.ParameterList([
            nn.Parameter(torch.randn(out_features, self.shard_sizes[i]) * 0.01)
            for i in range(n_shards)
        ])

        # Stale partial results cache
        self.stale_partials = [None] * n_shards

    def forward(self, x, mode="sync"):
        """
        x: [B, S, in_features]
        Each shard computes x[:, :, col_start:col_end] @ shard.T → [B, S, out_features]
        Results are SUMMED.
        """
        # Split input along last dim for each shard
        x_splits = torch.split(x, self.shard_sizes, dim=-1)

        result = torch.zeros(x.size(0), x.size(1), self.shards[0].size(0),
                            device=x.device, dtype=x.dtype)

        for i, (shard, x_part) in enumerate(zip(self.shards, x_splits)):
            partial = F.linear(x_part, shard)  # [B, S, out_features]

            if mode == "sync":
                result = result + partial
            else:
                delivered, latency = self.network.send(partial)
                if delivered is not None:
                    self.stale_partials[i] = delivered
                    # Align shapes
                    if delivered.shape != result.shape:
                        delivered = delivered[:result.size(0), :result.size(1), :]
                    result = result + delivered
                elif self.stale_partials[i] is not None:
                    stale = self.stale_partials[i]
                    if stale.shape != result.shape:
                        stale = stale[:result.size(0), :result.size(1), :]
                    result = result + stale
                # else: skip (add zero — already initialized)

        return result

    def init_from_weight(self, weight):
        """Initialize shards from a full weight matrix [out, in].
        Split along input (column) dimension."""
        offset = 0
        for i, size in enumerate(self.shard_sizes):
            self.shards[i].data = weight[:, offset:offset+size].clone()
            offset += size


class TensorParallelTransformerBlock(nn.Module):
    """A single transformer block with tensor-parallel weight splitting."""

    def __init__(self, original_block, n_shards=4, shard_weights=None,
                 network=None):
        super().__init__()
        self.n_shards = n_shards
        self.network = network

        # Extract dimensions from original block
        attn = original_block.self_attn
        mlp = original_block.mlp
        hidden = attn.q_proj.in_features
        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim
        q_dim = num_heads * head_dim       # full Q dim
        kv_dim = num_kv_heads * head_dim   # KV dim (smaller with GQA)
        intermediate = mlp.gate_proj.out_features

        # Keep original norms (small, no need to shard)
        self.input_layernorm = original_block.input_layernorm
        self.post_attention_layernorm = original_block.post_attention_layernorm

        # Tensor-parallel attention projections (use actual output dims)
        self.q_proj = TensorParallelLinear(hidden, q_dim, n_shards,
                                           shard_weights, network)
        self.k_proj = TensorParallelLinear(hidden, kv_dim, n_shards,
                                           shard_weights, network)
        self.v_proj = TensorParallelLinear(hidden, kv_dim, n_shards,
                                           shard_weights, network)
        self.o_proj = TensorParallelLinear(hidden, hidden, n_shards,
                                           shard_weights, network)

        # Tensor-parallel MLP
        self.gate_proj = TensorParallelLinear(hidden, intermediate, n_shards,
                                              shard_weights, network)
        self.up_proj = TensorParallelLinear(hidden, intermediate, n_shards,
                                            shard_weights, network)
        self.down_proj = TensorParallelLinear(intermediate, hidden, n_shards,
                                              shard_weights, network)

        # Init from original weights
        self.q_proj.init_from_weight(attn.q_proj.weight.data)
        self.k_proj.init_from_weight(attn.k_proj.weight.data)
        self.v_proj.init_from_weight(attn.v_proj.weight.data)
        self.o_proj.init_from_weight(attn.o_proj.weight.data)
        self.gate_proj.init_from_weight(mlp.gate_proj.weight.data)
        self.up_proj.init_from_weight(mlp.up_proj.weight.data)
        self.down_proj.init_from_weight(mlp.down_proj.weight.data)

        # Attention params
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden = hidden
        self.num_kv_groups = num_heads // num_kv_heads

    def forward(self, hidden_states, mode="sync"):
        # Self-attention with tensor-parallel projections
        residual = hidden_states
        h = self.input_layernorm(hidden_states)

        # QKV projections (tensor-parallel)
        q = self.q_proj(h, mode)
        k = self.k_proj(h, mode)
        v = self.v_proj(h, mode)

        # Reshape for attention (handle GQA)
        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        # Standard attention (GPU-agnostic — no flash attention)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=h.device), diagonal=1).bool()
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.hidden)

        # Output projection (tensor-parallel)
        out = self.o_proj(out, mode)
        hidden_states = residual + out

        # MLP with tensor-parallel projections
        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)

        gate = self.gate_proj(h, mode)
        up = self.up_proj(h, mode)
        h = F.silu(gate) * up
        h = self.down_proj(h, mode)

        hidden_states = residual + h
        return hidden_states


class TensorParallelModel(nn.Module):
    """Full model with selected layers replaced by tensor-parallel versions."""

    def __init__(self, base_model, tp_layer_indices, n_shards=4,
                 shard_weights=None, network=None):
        super().__init__()
        self.base = base_model
        self.tp_indices = set(tp_layer_indices)

        # Freeze base model
        for p in base_model.parameters():
            p.requires_grad = False

        # Replace selected layers with tensor-parallel versions
        self.tp_blocks = nn.ModuleDict()
        for idx in tp_layer_indices:
            orig = base_model.model.layers[idx]
            tp = TensorParallelTransformerBlock(
                orig, n_shards, shard_weights, network)
            self.tp_blocks[str(idx)] = tp

        n_tp = len(tp_layer_indices)
        n_total = len(base_model.model.layers)
        print(f"  TP layers: {n_tp}/{n_total}, shards: {n_shards}, "
              f"shard_weights: {shard_weights}", flush=True)

    def forward(self, input_ids, mode="sync"):
        h = self.base.model.embed_tokens(input_ids)
        B, S = input_ids.shape
        pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.base.model.rotary_emb(h, pos_ids)

        for i, layer in enumerate(self.base.model.layers):
            if i in self.tp_indices:
                h = self.tp_blocks[str(i)](h, mode)
            else:
                out = layer(h, position_embeddings=pos_emb)
                h = out[0] if isinstance(out, tuple) else out

        if hasattr(self.base.model, 'norm'):
            h = self.base.model.norm(h)
        return self.base.lm_head(h)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def eval_ppl(model, tokenizer, mode="sync", n_tokens=EVAL_TOKENS):
    print(f"  eval ppl ({mode})...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    model.eval()
    total_loss, total_n = 0, 0
    with torch.no_grad():
        for s in ds:
            if total_n >= n_tokens:
                break
            if len(s["text"].strip()) < 20:
                continue
            inp = tokenizer(s["text"][:1000], return_tensors="pt",
                           truncation=True, max_length=SEQ_LEN).to(DEVICE)
            if inp["input_ids"].size(1) < 10:
                continue
            logits = model(inp["input_ids"], mode=mode)
            sl = logits[:, :-1, :].float()
            lab = inp["input_ids"][:, 1:]
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lab.reshape(-1),
                                   reduction="sum",
                                   ignore_index=tokenizer.pad_token_id or 0)
            total_loss += loss.item()
            total_n += lab.numel()
    avg = total_loss / max(total_n, 1)
    ppl = math.exp(min(avg, 100))
    print(f"    {mode}: ppl={ppl:.4f} ({total_n} tokens)", flush=True)
    return ppl


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, tokenizer, mode, network_name, steps=TRAIN_STEPS):
    print(f"\n  Training (mode={mode}, network={network_name}, "
          f"steps={steps})", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [s["text"] for s in ds if len(s["text"].strip()) > 50]

    n_train = sum(p.numel() for p in model.trainable_params())
    print(f"  Trainable params: {n_train:,}", flush=True)

    optimizer = torch.optim.AdamW(model.trainable_params(), lr=LR, eps=1e-4)
    model.train()
    total_loss, idx = 0, 0
    t0 = time.time()

    for step in range(steps):
        batch = [texts[idx % len(texts)][:1000] for _ in range(BATCH_SIZE)]
        idx += BATCH_SIZE
        inp = tokenizer(batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=SEQ_LEN).to(DEVICE)

        logits = model(inp["input_ids"], mode=mode)
        sl = logits[:, :-1, :].contiguous().float()
        lab = inp["input_ids"][:, 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1),
                              ignore_index=tokenizer.pad_token_id or 0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"    step {step+1}/{steps}  loss={total_loss/(step+1):.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    t = time.time() - t0
    fl = total_loss / steps
    print(f"  Done. loss={fl:.4f}, time={t:.1f}s", flush=True)
    return fl, t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Device capacity simulation: [500MB, 200MB, 1GB, 500MB]
    # Translates to relative shard weights: [2, 1, 4, 2]
    shard_weights = [2, 1, 4, 2]

    # Which layers to tensor-parallel: every 4th layer (7 out of 28)
    tp_layers = list(range(0, 28, 4))  # [0, 4, 8, 12, 16, 20, 24]

    # ---- SYNC BASELINE ----
    print("\n=== SYNC BASELINE ===", flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    net = NetworkSim(**NETWORK_PROFILES["ideal"])
    m = TensorParallelModel(base, tp_layers, 4, shard_weights, net).to(DEVICE)
    ppl_sync = eval_ppl(m, tokenizer, "sync")
    results.append(Result("sync_baseline", "ideal", ppl_sync, ppl_sync, 0.0))
    _save(results, out / "results.json")

    # Train sync baseline
    loss_sync, t_sync = train(m, tokenizer, "sync", "ideal")
    ppl_sync_trained = eval_ppl(m, tokenizer, "sync")
    results.append(Result("sync_trained", "ideal", ppl_sync_trained, ppl_sync,
                          ((ppl_sync_trained - ppl_sync) / ppl_sync) * 100,
                          train_loss=loss_sync, train_time_s=t_sync))
    _save(results, out / "results.json")
    del m, base; gc.collect(); torch.cuda.empty_cache()

    # ---- ASYNC with different network conditions ----
    for net_name in ["lan", "wifi", "internet", "mobile", "hostile"]:
        print(f"\n=== ASYNC ({net_name}) ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
        net = NetworkSim(**NETWORK_PROFILES[net_name])
        m = TensorParallelModel(base, tp_layers, 4, shard_weights, net).to(DEVICE)

        ppl_pre = eval_ppl(m, tokenizer, "async")
        loss, t = train(m, tokenizer, "async", net_name)
        ppl_post = eval_ppl(m, tokenizer, "async")

        # Collect network stats from all TP linear layers
        total_dropped, total_sent = 0, 0
        for block in m.tp_blocks.values():
            for proj in [block.q_proj, block.k_proj, block.v_proj, block.o_proj,
                        block.gate_proj, block.up_proj, block.down_proj]:
                total_dropped += proj.network.dropped
                total_sent += proj.network.total

        gap = ((ppl_post - ppl_sync) / ppl_sync) * 100
        print(f"\n  >>> {net_name}: ppl={ppl_post:.4f} (pre={ppl_pre:.4f}, "
              f"gap={gap:+.2f}%, dropped={total_dropped}/{total_sent})",
              flush=True)

        results.append(Result(f"async_{net_name}", net_name, ppl_post, ppl_sync,
                              gap, train_loss=loss, train_time_s=t,
                              packets_dropped=total_dropped,
                              packets_total=total_sent))
        _save(results, out / "results.json")
        del m, base; gc.collect(); torch.cuda.empty_cache()

    # ---- Final summary ----
    print("\n" + "=" * 70, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Name':<20s} {'Net':<10s} {'PPL':>8s} {'Gap%':>8s} "
          f"{'Loss':>7s} {'Drop%':>6s}", flush=True)
    print("-" * 65, flush=True)
    for r in results:
        lo = f"{r.train_loss:.4f}" if r.train_loss > 0 else "—"
        drop = f"{r.packets_dropped/max(r.packets_total,1)*100:.1f}" if r.packets_total > 0 else "—"
        print(f"{r.name:<20s} {r.network:<10s} {r.ppl:>8.2f} {r.gap_pct:>+7.2f}% "
              f"{lo:>7s} {drop:>6s}", flush=True)

    _save(results, out / "results.json")
    print(f"\nSaved to {out / 'results.json'}", flush=True)


def _save(results, path):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/workspace/results/")
    run(p.parse_args().output)
