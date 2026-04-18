#!/usr/bin/env python3
"""
MoeMoe Resilience Experiment v1
================================
Train GPT-2 124M's 4 tensor-parallel experts with structured dropout
so they develop redundancy. Goal: dropping 1 of 4 experts causes
mild degradation (~20%) instead of collapse (10×).

Phase 1: Split pretrained GPT-2 into 4 experts (proven exact)
Phase 2: Train with structured dropout (randomly drop 1 expert per step)
Phase 3: Evaluate resilience (test all 4, test all combos of 3)
"""

import os, json, time, math, gc, random, argparse, itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EXPERTS = 4
K_REQUIRED = 3  # (4,3) erasure code — any 3 of 4 should work
TRAIN_STEPS = 2000
LR = 5e-5
BATCH_SIZE = 4
SEQ_LEN = 256
EVAL_TOKENS = 50_000


@dataclass
class Result:
    name: str
    ppl: float
    ppl_baseline: float
    gap_pct: float
    experts_active: int = 4
    dropped_expert: int = -1


class MoeMoeExpert(nn.Module):
    """One expert holding a subset of attention heads + MLP columns.
    Trainable — weights can be fine-tuned for redundancy."""

    def __init__(self, expert_id, n_experts, layer, config):
        super().__init__()
        self.id = expert_id
        n_heads = config.n_head
        head_dim = config.n_embd // n_heads
        heads_per_expert = n_heads // n_experts

        self.head_start = expert_id * heads_per_expert
        self.head_end = (expert_id + 1) * heads_per_expert
        self.head_dim = head_dim
        self.n_heads = heads_per_expert
        self.n_embd = config.n_embd
        self.n_experts = n_experts

        # Extract attention weights for this expert's heads
        c_attn_w = layer.attn.c_attn.weight.data
        c_attn_b = layer.attn.c_attn.bias.data
        qkv_dim = config.n_embd

        q_w = c_attn_w[:, :qkv_dim]
        k_w = c_attn_w[:, qkv_dim:2*qkv_dim]
        v_w = c_attn_w[:, 2*qkv_dim:]
        q_b = c_attn_b[:qkv_dim]
        k_b = c_attn_b[qkv_dim:2*qkv_dim]
        v_b = c_attn_b[2*qkv_dim:]

        hs = self.head_start * head_dim
        he = self.head_end * head_dim

        self.q_w = nn.Parameter(q_w[:, hs:he].clone())
        self.k_w = nn.Parameter(k_w[:, hs:he].clone())
        self.v_w = nn.Parameter(v_w[:, hs:he].clone())
        self.q_b = nn.Parameter(q_b[hs:he].clone())
        self.k_b = nn.Parameter(k_b[hs:he].clone())
        self.v_b = nn.Parameter(v_b[hs:he].clone())

        c_proj_w = layer.attn.c_proj.weight.data
        c_proj_b = layer.attn.c_proj.bias.data
        self.o_w = nn.Parameter(c_proj_w[hs:he, :].clone())
        self.o_b = nn.Parameter((c_proj_b / n_experts).clone())

        # MLP weights
        intermediate = config.n_embd * 4
        int_per = intermediate // n_experts
        ms = expert_id * int_per
        me = (expert_id + 1) * int_per

        self.mlp_fc_w = nn.Parameter(layer.mlp.c_fc.weight.data[:, ms:me].clone())
        self.mlp_fc_b = nn.Parameter(layer.mlp.c_fc.bias.data[ms:me].clone())
        self.mlp_proj_w = nn.Parameter(layer.mlp.c_proj.weight.data[ms:me, :].clone())
        self.mlp_proj_b = nn.Parameter((layer.mlp.c_proj.bias.data / n_experts).clone())

    def compute_attention_partial(self, x):
        B, S, _ = x.shape
        q = F.linear(x, self.q_w.T, self.q_b)
        k = F.linear(x, self.k_w.T, self.k_b)
        v = F.linear(x, self.v_w.T, self.v_b)

        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)

        # Scale output by N/(N-1) to compensate when one expert is dropped
        # This is learned during training — the model adapts
        partial = F.linear(out, self.o_w.T, self.o_b)
        return partial

    def compute_mlp_partial(self, x):
        h = F.linear(x, self.mlp_fc_w.T, self.mlp_fc_b)
        h = F.gelu(h, approximate='tanh')
        partial = F.linear(h, self.mlp_proj_w.T, self.mlp_proj_b)
        return partial


class MoeMoeModel(nn.Module):
    """MoeMoe: Multiple Overlapping Experts, Mutually Operating Ensemble."""

    def __init__(self, gpt2_model, n_experts=4):
        super().__init__()
        config = gpt2_model.config
        self.config = config
        self.n_experts = n_experts

        # Shared (frozen) components
        self.wte = gpt2_model.transformer.wte
        self.wpe = gpt2_model.transformer.wpe
        self.ln_f = gpt2_model.transformer.ln_f
        self.lm_head = gpt2_model.lm_head

        # Freeze shared components
        for p in self.wte.parameters():
            p.requires_grad = False
        for p in self.wpe.parameters():
            p.requires_grad = False
        for p in self.ln_f.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False

        # Build trainable experts for each layer
        self.layer_experts = nn.ModuleList()
        self.ln1_list = nn.ModuleList()
        self.ln2_list = nn.ModuleList()

        for layer in gpt2_model.transformer.h:
            experts = nn.ModuleList([
                MoeMoeExpert(i, n_experts, layer, config)
                for i in range(n_experts)
            ])
            self.layer_experts.append(experts)
            # Keep LayerNorms (frozen — they need full state)
            self.ln1_list.append(layer.ln_1)
            self.ln2_list.append(layer.ln_2)
            for p in layer.ln_1.parameters():
                p.requires_grad = False
            for p in layer.ln_2.parameters():
                p.requires_grad = False

    def forward(self, input_ids, active_experts=None, scale_factor=1.0):
        if active_experts is None:
            active_experts = list(range(self.n_experts))

        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        h = self.wte(input_ids) + self.wpe(pos)

        # Scale factor: when dropping experts, scale remaining contributions
        # to compensate for missing partial sums
        sf = self.n_experts / len(active_experts) if len(active_experts) < self.n_experts else 1.0

        for layer_idx in range(len(self.layer_experts)):
            experts = self.layer_experts[layer_idx]
            ln1 = self.ln1_list[layer_idx]
            ln2 = self.ln2_list[layer_idx]

            # Attention
            normed = ln1(h)
            attn_partial = torch.zeros_like(h)
            for eid in active_experts:
                attn_partial = attn_partial + experts[eid].compute_attention_partial(normed)
            h = h + attn_partial * sf

            # MLP
            normed = ln2(h)
            mlp_partial = torch.zeros_like(h)
            for eid in active_experts:
                mlp_partial = mlp_partial + experts[eid].compute_mlp_partial(normed)
            h = h + mlp_partial * sf

        h = self.ln_f(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# Training with structured dropout
# ---------------------------------------------------------------------------
def train_resilience(model, tokenizer, steps=TRAIN_STEPS):
    print(f"\n=== RESILIENCE TRAINING ({steps} steps) ===", flush=True)
    print(f"  Strategy: each step, randomly drop 1 of {N_EXPERTS} experts", flush=True)
    print(f"  Target: (N={N_EXPERTS}, K={K_REQUIRED}) erasure code", flush=True)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [s["text"] for s in ds if len(s["text"].strip()) > 50]

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_train:,}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, eps=1e-4)

    model.train()
    total_loss = 0
    idx = 0
    t0 = time.time()
    drop_counts = [0] * N_EXPERTS

    for step in range(steps):
        # Structured dropout: randomly drop 1 expert
        dropped = random.randint(0, N_EXPERTS - 1)
        active = [i for i in range(N_EXPERTS) if i != dropped]
        drop_counts[dropped] += 1

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
            avg = total_loss / (step + 1)
            print(f"    step {step+1}/{steps}  loss={avg:.4f}  "
                  f"elapsed={time.time()-t0:.1f}s  "
                  f"drops={drop_counts}", flush=True)

    t = time.time() - t0
    fl = total_loss / steps
    print(f"  Done. loss={fl:.4f}, time={t:.1f}s", flush=True)
    print(f"  Drop distribution: {drop_counts}", flush=True)
    return fl, t


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    print("Loading GPT-2 124M...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    gpt2.eval()

    # ---- Original baseline ----
    print("\n=== ORIGINAL GPT-2 ===", flush=True)
    ppl_orig, n = eval_ppl_orig(gpt2, tokenizer)
    print(f"  ppl={ppl_orig:.4f} ({n} tokens)", flush=True)
    results.append(Result("original", ppl_orig, ppl_orig, 0.0))
    _save(results, out / "results.json")

    # ---- Build MoeMoe model ----
    print("\n=== Building MoeMoe (4 experts) ===", flush=True)
    moemoe = MoeMoeModel(gpt2, N_EXPERTS).to(DEVICE)

    # ---- Pre-training eval: all 4 experts ----
    print("\n=== PRE-TRAINING: All 4 experts ===", flush=True)
    ppl_pre_4, _ = eval_ppl(moemoe, tokenizer, list(range(4)))
    gap = ((ppl_pre_4 - ppl_orig) / ppl_orig) * 100
    print(f"  ppl={ppl_pre_4:.4f} (gap={gap:+.4f}%)", flush=True)
    results.append(Result("pre_train_4of4", ppl_pre_4, ppl_orig, gap))

    # ---- Pre-training eval: each combo of 3 ----
    print("\n=== PRE-TRAINING: Drop each expert ===", flush=True)
    for dropped in range(4):
        active = [i for i in range(4) if i != dropped]
        ppl, _ = eval_ppl(moemoe, tokenizer, active)
        gap = ((ppl - ppl_orig) / ppl_orig) * 100
        print(f"  drop expert {dropped}: ppl={ppl:.2f} (gap={gap:+.1f}%)", flush=True)
        results.append(Result(f"pre_drop_{dropped}", ppl, ppl_orig, gap,
                              experts_active=3, dropped_expert=dropped))
    _save(results, out / "results.json")

    # ---- Train with structured dropout ----
    loss, train_time = train_resilience(moemoe, tokenizer)

    # ---- Post-training eval: all 4 experts ----
    print("\n=== POST-TRAINING: All 4 experts ===", flush=True)
    ppl_post_4, _ = eval_ppl(moemoe, tokenizer, list(range(4)))
    gap = ((ppl_post_4 - ppl_orig) / ppl_orig) * 100
    print(f"  ppl={ppl_post_4:.4f} (gap={gap:+.2f}%)", flush=True)
    results.append(Result("post_train_4of4", ppl_post_4, ppl_orig, gap,
                          experts_active=4))

    # ---- Post-training eval: each combo of 3 (THE KEY TEST) ----
    print("\n=== POST-TRAINING: Drop each expert (THE KEY TEST) ===", flush=True)
    for dropped in range(4):
        active = [i for i in range(4) if i != dropped]
        ppl, _ = eval_ppl(moemoe, tokenizer, active)
        gap = ((ppl - ppl_orig) / ppl_orig) * 100
        print(f"  drop expert {dropped}: ppl={ppl:.2f} (gap={gap:+.1f}%)", flush=True)
        results.append(Result(f"post_drop_{dropped}", ppl, ppl_orig, gap,
                              experts_active=3, dropped_expert=dropped))

    # ---- Post-training eval: each combo of 2 ----
    print("\n=== POST-TRAINING: Drop 2 experts ===", flush=True)
    for combo in itertools.combinations(range(4), 2):
        active = list(combo)
        ppl, _ = eval_ppl(moemoe, tokenizer, active)
        gap = ((ppl - ppl_orig) / ppl_orig) * 100
        dropped = [i for i in range(4) if i not in combo]
        print(f"  keep {combo}, drop {dropped}: ppl={ppl:.2f} (gap={gap:+.1f}%)",
              flush=True)
        results.append(Result(f"post_keep_{combo[0]}{combo[1]}", ppl, ppl_orig,
                              gap, experts_active=2))
    _save(results, out / "results.json")

    # ---- Final summary ----
    print("\n" + "=" * 70, flush=True)
    print("MOEMOE RESILIENCE RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Name':<25s} {'PPL':>8s} {'Gap%':>8s} {'Experts':>8s}", flush=True)
    print("-" * 55, flush=True)
    for r in results:
        print(f"{r.name:<25s} {r.ppl:>8.2f} {r.gap_pct:>+7.1f}% "
              f"{r.experts_active:>8d}", flush=True)

    _save(results, out / "results.json")
    print(f"\nSaved to {out / 'results.json'}", flush=True)


def eval_ppl_orig(model, tokenizer, n_tokens=EVAL_TOKENS):
    """Eval original GPT-2 model."""
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
            logits = model(**inp).logits
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
