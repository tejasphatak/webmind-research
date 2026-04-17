#!/usr/bin/env python3
"""
Decompose-Distribute-Collaborate: Architecture Proof
=====================================================
Split GPT-2 124M into 4 experts by attention heads + MLP columns.
NO training. Just split pretrained weights and sum partials.
Sync mode output MUST match original exactly.

Then test: expert dropout, compression, async staleness.
"""

import os, json, time, math, gc, argparse
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
EVAL_TOKENS = 50_000


@dataclass
class Result:
    name: str
    ppl: float
    ppl_baseline: float
    gap_pct: float
    bytes_per_token: float = 0.0
    experts_active: int = 4


class Expert:
    """One expert holding a subset of attention heads + MLP columns."""

    def __init__(self, expert_id, n_experts, layer, config):
        self.id = expert_id
        n_heads = config.n_head
        head_dim = config.n_embd // n_heads
        heads_per_expert = n_heads // n_experts

        self.head_start = expert_id * heads_per_expert
        self.head_end = (expert_id + 1) * heads_per_expert
        self.head_dim = head_dim
        self.n_heads = heads_per_expert
        self.n_embd = config.n_embd
        self.full_n_heads = n_heads

        # Extract this expert's attention head weights
        # GPT-2 stores Q,K,V concatenated in c_attn: [n_embd, 3*n_embd]
        c_attn_w = layer.attn.c_attn.weight.data  # [n_embd, 3*n_embd]
        c_attn_b = layer.attn.c_attn.bias.data     # [3*n_embd]

        # Split Q, K, V
        qkv_dim = config.n_embd
        q_w = c_attn_w[:, :qkv_dim]
        k_w = c_attn_w[:, qkv_dim:2*qkv_dim]
        v_w = c_attn_w[:, 2*qkv_dim:]
        q_b = c_attn_b[:qkv_dim]
        k_b = c_attn_b[qkv_dim:2*qkv_dim]
        v_b = c_attn_b[2*qkv_dim:]

        # Extract heads for this expert
        hs, he = self.head_start * head_dim, self.head_end * head_dim
        self.q_w = q_w[:, hs:he].clone()  # [n_embd, expert_head_dim]
        self.k_w = k_w[:, hs:he].clone()
        self.v_w = v_w[:, hs:he].clone()
        self.q_b = q_b[hs:he].clone()
        self.k_b = k_b[hs:he].clone()
        self.v_b = v_b[hs:he].clone()

        # Output projection: c_proj maps [n_embd] -> [n_embd]
        # Split rows (input side) by head
        c_proj_w = layer.attn.c_proj.weight.data  # [n_embd, n_embd]
        c_proj_b = layer.attn.c_proj.bias.data     # [n_embd]
        self.o_w = c_proj_w[hs:he, :].clone()      # [expert_head_dim, n_embd]
        # Bias is shared — only add 1/N of it per expert
        self.o_b = (c_proj_b / n_experts).clone()

        # MLP: split intermediate dimension
        # GPT-2 MLP: c_fc [n_embd, 4*n_embd], c_proj [4*n_embd, n_embd]
        intermediate = config.n_embd * 4
        int_per_expert = intermediate // n_experts
        ms = expert_id * int_per_expert
        me = (expert_id + 1) * int_per_expert

        self.mlp_fc_w = layer.mlp.c_fc.weight.data[:, ms:me].clone()  # [n_embd, int_per_expert]
        self.mlp_fc_b = layer.mlp.c_fc.bias.data[ms:me].clone()
        self.mlp_proj_w = layer.mlp.c_proj.weight.data[ms:me, :].clone()  # [int_per_expert, n_embd]
        self.mlp_proj_b = (layer.mlp.c_proj.bias.data / n_experts).clone()

    def compute_attention_partial(self, x):
        """Compute this expert's partial attention output.
        x: [B, S, n_embd] (full hidden state after LayerNorm)
        Returns: [B, S, n_embd] (partial contribution to residual)
        """
        B, S, _ = x.shape

        # Project to Q, K, V for our heads
        q = F.linear(x, self.q_w.T, self.q_b)  # [B, S, expert_head_dim]
        k = F.linear(x, self.k_w.T, self.k_b)
        v = F.linear(x, self.v_w.T, self.v_b)

        # Reshape to heads
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, n_heads, S, head_dim]

        # Concat heads and project
        out = out.transpose(1, 2).contiguous().view(B, S, -1)  # [B, S, expert_head_dim]
        partial = F.linear(out, self.o_w.T, self.o_b)  # [B, S, n_embd]
        return partial

    def compute_mlp_partial(self, x):
        """Compute this expert's partial MLP output.
        x: [B, S, n_embd] (full hidden state after LayerNorm)
        Returns: [B, S, n_embd] (partial contribution to residual)
        """
        h = F.linear(x, self.mlp_fc_w.T, self.mlp_fc_b)
        h = F.gelu(h, approximate='tanh')  # GPT-2 uses tanh approx GELU
        partial = F.linear(h, self.mlp_proj_w.T, self.mlp_proj_b)
        return partial


class DDCModel:
    """Decompose-Distribute-Collaborate model from pretrained GPT-2."""

    def __init__(self, model, n_experts=4):
        self.config = model.config
        self.n_experts = n_experts
        self.wte = model.transformer.wte
        self.wpe = model.transformer.wpe
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head

        # Build experts for each layer
        self.layer_experts = []
        self.layer_norms = []
        for layer in model.transformer.h:
            experts = [Expert(i, n_experts, layer, model.config)
                      for i in range(n_experts)]
            self.layer_experts.append(experts)
            self.layer_norms.append((layer.ln_1, layer.ln_2))

    def forward(self, input_ids, active_experts=None, compress_k=0):
        """
        active_experts: list of expert indices to use (default: all)
        compress_k: if > 0, PCA-compress partials to k dims
        """
        if active_experts is None:
            active_experts = list(range(self.n_experts))

        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        h = self.wte(input_ids) + self.wpe(pos)

        for layer_idx, (experts, (ln1, ln2)) in enumerate(
                zip(self.layer_experts, self.layer_norms)):

            # === Attention sub-layer ===
            normed = ln1(h)  # LayerNorm on FULL hidden state

            # Each active expert computes its partial attention
            attn_partial = torch.zeros_like(h)
            for eid in active_experts:
                partial = experts[eid].compute_attention_partial(normed)
                if compress_k > 0:
                    partial = self._compress_decompress(partial, compress_k)
                attn_partial = attn_partial + partial

            h = h + attn_partial  # Residual connection

            # === MLP sub-layer ===
            normed = ln2(h)  # LayerNorm on FULL hidden state

            # Each active expert computes its partial MLP
            mlp_partial = torch.zeros_like(h)
            for eid in active_experts:
                partial = experts[eid].compute_mlp_partial(normed)
                if compress_k > 0:
                    partial = self._compress_decompress(partial, compress_k)
                mlp_partial = mlp_partial + partial

            h = h + mlp_partial  # Residual connection

        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits

    def _compress_decompress(self, x, k):
        """PCA-style compress to k dims and reconstruct (lossy)."""
        B, S, D = x.shape
        flat = x.reshape(-1, D)  # [B*S, D]
        # SVD-based compression
        U, s, Vh = torch.linalg.svd(flat, full_matrices=False)
        # Keep top-k components
        compressed = U[:, :k] @ torch.diag(s[:k]) @ Vh[:k, :]
        return compressed.reshape(B, S, D)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def eval_ppl(model_fn, tokenizer, n_tokens=EVAL_TOKENS):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    total_loss, total_n = 0, 0
    with torch.no_grad():
        for s in ds:
            if total_n >= n_tokens:
                break
            if len(s["text"].strip()) < 20:
                continue
            inp = tokenizer(s["text"][:500], return_tensors="pt",
                           truncation=True, max_length=256)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            if inp["input_ids"].size(1) < 10:
                continue
            logits = model_fn(inp["input_ids"])
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
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    # ---- Original model baseline ----
    print("\n=== ORIGINAL GPT-2 (baseline) ===", flush=True)
    ppl_orig, n = eval_ppl(lambda ids: model(ids).logits, tokenizer)
    print(f"  ppl={ppl_orig:.4f} ({n} tokens)", flush=True)
    results.append(Result("original", ppl_orig, ppl_orig, 0.0))

    # ---- Build DDC model ----
    print("\n=== Building DDC (4 experts) ===", flush=True)
    ddc = DDCModel(model, N_EXPERTS)

    # Move expert weights to device
    for layer_experts in ddc.layer_experts:
        for exp in layer_experts:
            for attr in ['q_w','k_w','v_w','q_b','k_b','v_b','o_w','o_b',
                        'mlp_fc_w','mlp_fc_b','mlp_proj_w','mlp_proj_b']:
                setattr(exp, attr, getattr(exp, attr).to(DEVICE))

    # ---- Test 1: All 4 experts (should match original EXACTLY) ----
    print("\n=== SYNC: All 4 experts ===", flush=True)
    ppl_sync, n = eval_ppl(lambda ids: ddc.forward(ids), tokenizer)
    gap = ((ppl_sync - ppl_orig) / ppl_orig) * 100
    print(f"  ppl={ppl_sync:.4f} (gap={gap:+.4f}%)", flush=True)
    results.append(Result("sync_4_experts", ppl_sync, ppl_orig, gap,
                          experts_active=4))

    # ---- Test 2: Drop experts ----
    for n_active in [3, 2, 1]:
        active = list(range(n_active))
        print(f"\n=== DROP: {n_active} of 4 experts ===", flush=True)
        ppl, n = eval_ppl(lambda ids, a=active: ddc.forward(ids, a), tokenizer)
        gap = ((ppl - ppl_orig) / ppl_orig) * 100
        print(f"  ppl={ppl:.4f} (gap={gap:+.2f}%)", flush=True)
        results.append(Result(f"drop_to_{n_active}", ppl, ppl_orig, gap,
                              experts_active=n_active))

    # ---- Test 3: Compressed communication (PCA) ----
    for k in [64, 32, 16, 8]:
        print(f"\n=== COMPRESS: PCA k={k} ===", flush=True)
        ppl, n = eval_ppl(
            lambda ids, kk=k: ddc.forward(ids, compress_k=kk), tokenizer)
        gap = ((ppl - ppl_orig) / ppl_orig) * 100
        bytes_per_tok = k * 2 * N_EXPERTS * 2  # k fp16 values × experts × (attn+mlp)
        print(f"  ppl={ppl:.4f} (gap={gap:+.2f}%, {bytes_per_tok} bytes/tok)",
              flush=True)
        results.append(Result(f"compress_k{k}", ppl, ppl_orig, gap,
                              bytes_per_token=bytes_per_tok,
                              experts_active=4))

    # ---- Test 4: Combined: drop + compress ----
    print(f"\n=== DROP 1 + COMPRESS k=16 ===", flush=True)
    ppl, n = eval_ppl(
        lambda ids: ddc.forward(ids, active_experts=[0,1,2], compress_k=16),
        tokenizer)
    gap = ((ppl - ppl_orig) / ppl_orig) * 100
    print(f"  ppl={ppl:.4f} (gap={gap:+.2f}%)", flush=True)
    results.append(Result("drop1_compress16", ppl, ppl_orig, gap,
                          experts_active=3))

    # ---- Final summary ----
    print("\n" + "=" * 65, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Name':<25s} {'PPL':>8s} {'Gap%':>8s} {'Experts':>8s} "
          f"{'Bytes':>8s}", flush=True)
    print("-" * 65, flush=True)
    for r in results:
        b = f"{r.bytes_per_token:.0f}" if r.bytes_per_token > 0 else "—"
        print(f"{r.name:<25s} {r.ppl:>8.2f} {r.gap_pct:>+7.2f}% "
              f"{r.experts_active:>8d} {b:>8s}", flush=True)

    with open(out / "results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved to {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/workspace/results/")
    run(p.parse_args().output)
