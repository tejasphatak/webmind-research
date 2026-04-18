#!/usr/bin/env python3
"""
MoeMoe Gemma 4 E4B — Scale validation
======================================
Apply MoeMoe (erasure-coded tensor decomposition) to Gemma 4 E4B text backbone.
Phase 1: text-only. 42 layers, 8 heads, 2 KV heads, hidden=2560.
4 experts × 2 heads each.

Same protocol as GPT-2 proof:
1. Split pretrained weights by attention heads + MLP columns
2. Verify sync mode matches original (0.000% error)
3. Measure pre-training dropout degradation
4. Train with structured expert dropout (2000 steps)
5. Measure post-training resilience
"""

import os, json, time, math, gc, random, argparse, itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/gemma-4-E4B"
N_EXPERTS = 4
K_REQUIRED = 3
TRAIN_STEPS = 2000
LR = 5e-5
BATCH_SIZE = 2  # smaller batch for larger model
SEQ_LEN = 256
EVAL_TOKENS = 30_000  # reduced for speed on larger model


@dataclass
class Result:
    name: str
    ppl: float
    ppl_baseline: float
    gap_pct: float
    experts_active: int = 4
    dropped_expert: int = -1


class GemmaExpert(nn.Module):
    """One expert for Gemma 4 text layers."""

    def __init__(self, expert_id, n_experts, layer, config):
        super().__init__()
        self.id = expert_id
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        # Gemma 4: Q is [n_heads*head_dim, hidden], K/V are [n_kv_heads*head_dim, hidden]
        q_out = layer.self_attn.q_proj.weight.shape[0]
        k_out = layer.self_attn.k_proj.weight.shape[0]
        head_dim = q_out // n_heads
        hidden = config.hidden_size
        intermediate = config.intermediate_size

        heads_per_expert = n_heads // n_experts
        kv_heads_per_expert = max(1, n_kv_heads // n_experts)

        self.head_start = expert_id * heads_per_expert
        self.head_end = (expert_id + 1) * heads_per_expert
        self.kv_head_start = expert_id * kv_heads_per_expert
        self.kv_head_end = min((expert_id + 1) * kv_heads_per_expert, n_kv_heads)
        self.head_dim = head_dim
        self.n_heads = heads_per_expert
        self.n_kv_heads = kv_heads_per_expert
        self.n_kv_groups = heads_per_expert // kv_heads_per_expert
        self.hidden = hidden
        self.n_experts = n_experts

        attn = layer.self_attn

        # Q projection: split by heads
        q_dim = n_heads * head_dim
        hs = self.head_start * head_dim
        he = self.head_end * head_dim
        self.q_w = nn.Parameter(attn.q_proj.weight.data[hs:he, :].clone())

        # K, V projection: split by KV heads
        kv_hs = self.kv_head_start * head_dim
        kv_he = self.kv_head_end * head_dim
        self.k_w = nn.Parameter(attn.k_proj.weight.data[kv_hs:kv_he, :].clone())
        self.v_w = nn.Parameter(attn.v_proj.weight.data[kv_hs:kv_he, :].clone())

        # O projection: [hidden, q_total] — split input columns by heads
        o_total = attn.o_proj.weight.shape[1]  # q_total dim
        o_per_expert = o_total // n_experts
        o_start = expert_id * o_per_expert
        o_end = (expert_id + 1) * o_per_expert
        self.o_w = nn.Parameter(attn.o_proj.weight.data[:, o_start:o_end].clone())

        # MLP: split intermediate dimension
        int_per = intermediate // n_experts
        ms = expert_id * int_per
        me = (expert_id + 1) * int_per

        mlp = layer.mlp
        self.gate_w = nn.Parameter(mlp.gate_proj.weight.data[ms:me, :].clone())
        self.up_w = nn.Parameter(mlp.up_proj.weight.data[ms:me, :].clone())
        self.down_w = nn.Parameter(mlp.down_proj.weight.data[:, ms:me].clone())

    def compute_attention_partial(self, x, pos_emb=None):
        B, S, _ = x.shape

        q = F.linear(x, self.q_w)
        k = F.linear(x, self.k_w)
        v = F.linear(x, self.v_w)

        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if pos_emb is not None:
            cos, sin = pos_emb
            # Apply to Q
            q = (q * cos[:, :, :S, :]) + (self._rotate_half(q) * sin[:, :, :S, :])
            # Apply to K
            k = (k * cos[:, :, :S, :]) + (self._rotate_half(k) * sin[:, :, :S, :])

        # Expand KV for GQA
        if self.n_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
            k = k.reshape(B, self.n_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
            v = v.reshape(B, self.n_heads, S, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        partial = F.linear(out, self.o_w)
        return partial

    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def compute_mlp_partial(self, x):
        gate = F.linear(x, self.gate_w)
        up = F.linear(x, self.up_w)
        h = F.silu(gate) * up
        partial = F.linear(h, self.down_w)
        return partial


class MoeMoeGemma(nn.Module):
    """MoeMoe applied to Gemma 4 text backbone."""

    def __init__(self, base_model, n_experts=4):
        super().__init__()
        self.config = base_model.config
        self.n_experts = n_experts

        # Navigate to text/language model
        # Gemma 4 structure: model.language_model.{embed_tokens, layers, norm, rotary_emb}
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'language_model'):
            text_model = base_model.model.language_model
        elif hasattr(base_model, 'model'):
            text_model = base_model.model
        else:
            text_model = base_model

        self.embed_tokens = text_model.embed_tokens
        self.lm_head = base_model.lm_head if hasattr(base_model, 'lm_head') else None

        # Freeze shared components
        for p in self.embed_tokens.parameters():
            p.requires_grad = False
        if self.lm_head:
            for p in self.lm_head.parameters():
                p.requires_grad = False

        # Get text config
        text_config = base_model.config
        if hasattr(base_model.config, 'text_config'):
            text_config = base_model.config.text_config

        # Build experts per layer
        self.layer_experts = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()

        layers = text_model.layers
        self.n_layers = len(layers)

        for layer in layers:
            experts = nn.ModuleList([
                GemmaExpert(i, n_experts, layer, text_config)
                for i in range(n_experts)
            ])
            self.layer_experts.append(experts)
            self.norms_1.append(layer.input_layernorm)
            self.norms_2.append(layer.post_attention_layernorm)

            for p in layer.input_layernorm.parameters():
                p.requires_grad = False
            for p in layer.post_attention_layernorm.parameters():
                p.requires_grad = False

        # Final norm
        self.final_norm = text_model.norm if hasattr(text_model, 'norm') else None
        if self.final_norm:
            for p in self.final_norm.parameters():
                p.requires_grad = False

        # RoPE
        self.rotary_emb = text_model.rotary_emb if hasattr(text_model, 'rotary_emb') else None

        # PLE (Per-Layer Embeddings) — Gemma 4 specific, freeze
        if hasattr(text_model, 'embed_tokens_per_layer'):
            self.embed_tokens_per_layer = text_model.embed_tokens_per_layer
            for p in self.embed_tokens_per_layer.parameters():
                p.requires_grad = False
        else:
            self.embed_tokens_per_layer = None

        print(f"  MoeMoe Gemma: {self.n_layers} layers, {n_experts} experts, "
              f"{text_config.num_attention_heads} heads → "
              f"{text_config.num_attention_heads//n_experts} per expert", flush=True)

    def forward(self, input_ids, active_experts=None):
        if active_experts is None:
            active_experts = list(range(self.n_experts))

        B, S = input_ids.shape
        h = self.embed_tokens(input_ids)

        # Compute RoPE
        pos_emb = None
        if self.rotary_emb:
            pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.rotary_emb(h, pos_ids)

        sf = self.n_experts / len(active_experts) if len(active_experts) < self.n_experts else 1.0

        for layer_idx in range(self.n_layers):
            experts = self.layer_experts[layer_idx]
            ln1 = self.norms_1[layer_idx]
            ln2 = self.norms_2[layer_idx]

            normed = ln1(h)
            attn_partial = torch.zeros_like(h)
            for eid in active_experts:
                attn_partial = attn_partial + experts[eid].compute_attention_partial(normed, pos_emb)
            h = h + attn_partial * sf

            normed = ln2(h)
            mlp_partial = torch.zeros_like(h)
            for eid in active_experts:
                mlp_partial = mlp_partial + experts[eid].compute_mlp_partial(normed)
            h = h + mlp_partial * sf

        if self.final_norm:
            h = self.final_norm(h)
        if self.lm_head:
            return self.lm_head(h)
        return h


def eval_ppl(model, tokenizer, active_experts=None, n_tokens=EVAL_TOKENS):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    model.eval()
    total_loss, total_n = 0, 0
    with torch.no_grad():
        for s in ds:
            if total_n >= n_tokens:
                break
            if len(s["text"].strip()) < 20:
                continue
            inp = tokenizer(s["text"][:500], return_tensors="pt",
                           truncation=True, max_length=SEQ_LEN).to(DEVICE)
            if inp["input_ids"].size(1) < 10:
                continue
            logits = model(inp["input_ids"], active_experts=active_experts)
            sl = logits[:, :-1, :].float()
            lab = inp["input_ids"][:, 1:]
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)),
                                   lab.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_n += lab.numel()
    avg = total_loss / max(total_n, 1)
    ppl = math.exp(min(avg, 100))
    return ppl, total_n


def train_resilience(model, tokenizer, steps=TRAIN_STEPS):
    print(f"\n=== RESILIENCE TRAINING ({steps} steps) ===", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [s["text"] for s in ds if len(s["text"].strip()) > 50]

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_train:,}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, eps=1e-4)

    model.train()
    total_loss, idx = 0, 0
    t0 = time.time()

    for step in range(steps):
        dropped = random.randint(0, N_EXPERTS - 1)
        active = [i for i in range(N_EXPERTS) if i != dropped]

        batch = [texts[idx % len(texts)][:1000] for _ in range(BATCH_SIZE)]
        idx += BATCH_SIZE
        inp = tokenizer(batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=SEQ_LEN).to(DEVICE)

        logits = model(inp["input_ids"], active_experts=active)
        sl = logits[:, :-1, :].contiguous().float()
        lab = inp["input_ids"][:, 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1),
                              ignore_index=tokenizer.pad_token_id
                              if tokenizer.pad_token_id is not None else -100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % 200 == 0:
            print(f"    step {step+1}/{steps}  loss={total_loss/(step+1):.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    print(f"  Done. loss={total_loss/steps:.4f}, time={time.time()-t0:.1f}s",
          flush=True)
    return total_loss / steps, time.time() - t0


def run(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    base.eval()

    # ---- Original baseline ----
    print("\n=== ORIGINAL GEMMA 4 ===", flush=True)
    ppl_orig, n = eval_ppl_direct(base, tokenizer)
    print(f"  ppl={ppl_orig:.4f} ({n} tokens)", flush=True)
    results.append(Result("original", ppl_orig, ppl_orig, 0.0))
    _save(results, out / "results.json")

    # ---- Build MoeMoe ----
    print("\n=== Building MoeMoe ===", flush=True)
    moemoe = MoeMoeGemma(base, N_EXPERTS).to(DEVICE)
    del base; gc.collect(); torch.cuda.empty_cache()

    # ---- Sync: all 4 ----
    print("\n=== SYNC: All 4 experts ===", flush=True)
    ppl_sync, _ = eval_ppl(moemoe, tokenizer, list(range(4)))
    gap = ((ppl_sync - ppl_orig) / ppl_orig) * 100
    print(f"  ppl={ppl_sync:.4f} (gap={gap:+.4f}%)", flush=True)
    results.append(Result("sync_4of4", ppl_sync, ppl_orig, gap))
    _save(results, out / "results.json")

    # ---- Pre-training drops ----
    print("\n=== PRE-TRAINING: Drop each expert ===", flush=True)
    for dropped in range(4):
        active = [i for i in range(4) if i != dropped]
        ppl, _ = eval_ppl(moemoe, tokenizer, active)
        gap = ((ppl - ppl_orig) / ppl_orig) * 100
        print(f"  drop {dropped}: ppl={ppl:.2f} (gap={gap:+.1f}%)", flush=True)
        results.append(Result(f"pre_drop_{dropped}", ppl, ppl_orig, gap,
                              experts_active=3, dropped_expert=dropped))
    _save(results, out / "results.json")

    # ---- Resilience training ----
    train_resilience(moemoe, tokenizer)

    # ---- Post-training: all 4 ----
    print("\n=== POST-TRAINING: All 4 ===", flush=True)
    ppl_post, _ = eval_ppl(moemoe, tokenizer, list(range(4)))
    gap = ((ppl_post - ppl_orig) / ppl_orig) * 100
    print(f"  ppl={ppl_post:.4f} (gap={gap:+.2f}%)", flush=True)
    results.append(Result("post_4of4", ppl_post, ppl_orig, gap))

    # ---- Post-training: drops (THE KEY TEST) ----
    print("\n=== POST-TRAINING: Drop each expert (KEY TEST) ===", flush=True)
    for dropped in range(4):
        active = [i for i in range(4) if i != dropped]
        ppl, _ = eval_ppl(moemoe, tokenizer, active)
        gap = ((ppl - ppl_orig) / ppl_orig) * 100
        print(f"  drop {dropped}: ppl={ppl:.2f} (gap={gap:+.1f}%)", flush=True)
        results.append(Result(f"post_drop_{dropped}", ppl, ppl_orig, gap,
                              experts_active=3, dropped_expert=dropped))
    _save(results, out / "results.json")

    # ---- Summary ----
    print("\n" + "=" * 65, flush=True)
    print("MOEMOE GEMMA 4 RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Name':<25s} {'PPL':>8s} {'Gap%':>8s}", flush=True)
    print("-" * 45, flush=True)
    for r in results:
        print(f"{r.name:<25s} {r.ppl:>8.2f} {r.gap_pct:>+7.1f}%", flush=True)
    _save(results, out / "results.json")


def eval_ppl_direct(model, tokenizer, n_tokens=EVAL_TOKENS):
    """Eval original model directly."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    model.eval()
    total_loss, total_n = 0, 0
    with torch.no_grad():
        for s in ds:
            if total_n >= n_tokens:
                break
            if len(s["text"].strip()) < 20:
                continue
            inp = tokenizer(s["text"][:500], return_tensors="pt",
                           truncation=True, max_length=SEQ_LEN).to(DEVICE)
            if inp["input_ids"].size(1) < 10:
                continue
            logits = model(inp["input_ids"]).logits
            sl = logits[:, :-1, :].float()
            lab = inp["input_ids"][:, 1:]
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)),
                                   lab.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_n += lab.numel()
    avg = total_loss / max(total_n, 1)
    ppl = math.exp(min(avg, 100))
    return ppl, total_n


def _save(results, path):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/workspace/results/")
    run(p.parse_args().output)
