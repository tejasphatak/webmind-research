#!/usr/bin/env python3
"""
Async Stale-Summary MVE v2 — Unfrozen layers + latency simulation
==================================================================
Fixes from v1: unfreeze last N layers so cross-device layers can
actually shift model output. Add latency simulation to measure tok/s.

Model: Qwen 2.5 1.5B (28 layers, 1536 hidden)
"""

import os, sys, json, time, math, gc, argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-1.5B"
N_DEVICES = 4
TRAIN_STEPS = 800
EVAL_TOKENS = 50_000
LR = 1e-5  # lower LR needed with 420M unfrozen params in fp16
BATCH_SIZE = 4
SEQ_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNFREEZE_LAST_N = 4  # unfreeze last 4 layers so model can adapt


@dataclass
class Result:
    name: str
    k: int
    ppl: float
    ppl_baseline: float
    gap_pct: float
    hellaswag_acc: Optional[float] = None
    train_loss: float = 0.0
    train_time_s: float = 0.0
    alphas: Optional[list] = None
    tok_per_sec: Optional[float] = None
    latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Cross-device layer (same as v1)
# ---------------------------------------------------------------------------
class AsyncCrossLayer(nn.Module):
    def __init__(self, hidden_dim, k=16, n_neighbors=2):
        super().__init__()
        self.compressor = nn.Linear(hidden_dim, k, bias=False)
        self.cross_proj = nn.Linear(k * n_neighbors, hidden_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.compressor.weight, std=0.01)
        nn.init.normal_(self.cross_proj.weight, std=0.01)

    def forward(self, h, stale_summaries=None):
        if stale_summaries is not None and len(stale_summaries) > 0:
            B, S, H = h.shape
            aligned = []
            for s in stale_summaries:
                if s.size(0) != B:
                    s = s[:B] if s.size(0) > B else s.repeat((B // s.size(0)) + 1, 1, 1)[:B]
                if s.size(1) != S:
                    s = s[:, :S, :] if s.size(1) > S else torch.cat([s, torch.zeros(s.size(0), S - s.size(1), s.size(2), device=s.device, dtype=s.dtype)], dim=1)
                aligned.append(s)
            cat = torch.cat(aligned, dim=-1)
            cross = self.cross_proj(cat.float()).to(h.dtype)
            h = h + self.alpha * cross
        summary = self.compressor(h.float()).to(h.dtype)
        return h, summary


class DistributedModel(nn.Module):
    def __init__(self, base, n_devices=4, k=16, unfreeze_last=4):
        super().__init__()
        self.base = base
        self.layers = base.model.layers
        n_layers = len(self.layers)
        self.lps = n_layers // n_devices
        self.boundaries = [self.lps * (i + 1) - 1 for i in range(n_devices - 1)]
        n_neighbors = max(len(self.boundaries) - 1, 1)

        # Freeze all, then unfreeze last N layers
        for p in base.parameters():
            p.requires_grad = False
        for i in range(max(0, n_layers - unfreeze_last), n_layers):
            for p in self.layers[i].parameters():
                p.requires_grad = True
        # Also unfreeze lm_head and norm
        for p in base.lm_head.parameters():
            p.requires_grad = True
        if hasattr(base.model, 'norm'):
            for p in base.model.norm.parameters():
                p.requires_grad = True

        self.cross_layers = nn.ModuleList([
            AsyncCrossLayer(base.config.hidden_size, k, n_neighbors)
            for _ in self.boundaries
        ])

    def forward(self, input_ids, mode="async", stale=None):
        h = self.base.model.embed_tokens(input_ids)
        B, S = input_ids.shape
        pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.base.model.rotary_emb(h, pos_ids)

        summaries = [None] * len(self.boundaries)
        for i, layer in enumerate(self.layers):
            out = layer(h, position_embeddings=pos_emb)
            h = out[0] if isinstance(out, tuple) else out
            if i in self.boundaries:
                bi = self.boundaries.index(i)
                if mode == "async":
                    neighbors = []
                    if stale:
                        for j, s in enumerate(stale):
                            if s is not None and j != bi:
                                neighbors.append(s)
                    h, sm = self.cross_layers[bi](h, neighbors if neighbors else None)
                    summaries[bi] = sm

        if hasattr(self.base.model, 'norm'):
            h = self.base.model.norm(h)
        return self.base.lm_head(h), summaries

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
            logits, _ = model(inp["input_ids"], mode=mode)
            sl = logits[:, :-1, :]
            lab = inp["input_ids"][:, 1:]
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lab.reshape(-1),
                                   reduction="sum", ignore_index=tokenizer.pad_token_id or 0)
            total_loss += loss.item()
            total_n += lab.numel()
    avg = total_loss / max(total_n, 1)
    ppl = math.exp(min(avg, 100))
    print(f"    {mode}: ppl={ppl:.4f} ({total_n} tokens)", flush=True)
    return ppl


def eval_hellaswag(model, tokenizer, n=500):
    print(f"  eval HellaSwag (n={n})...", flush=True)
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)
    except Exception as e:
        print(f"    SKIP: {e}", flush=True)
        return None
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, s in enumerate(ds):
            if i >= n:
                break
            ctx = s["ctx"]
            scores = []
            for ending in s["endings"]:
                inp = tokenizer(ctx + " " + ending, return_tensors="pt",
                               truncation=True, max_length=SEQ_LEN).to(DEVICE)
                logits, _ = model(inp["input_ids"], mode="async")
                sl = logits[:, :-1, :]
                lab = inp["input_ids"][:, 1:]
                loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lab.reshape(-1), reduction="mean")
                scores.append(-loss.item())
            if scores.index(max(scores)) == int(s["label"]):
                correct += 1
            total += 1
            if (i + 1) % 100 == 0:
                print(f"    ...{i+1}/{n} acc={correct/total:.3f}", flush=True)
    acc = correct / max(total, 1)
    print(f"    HellaSwag: {correct}/{total} = {acc:.3f}", flush=True)
    return acc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, tokenizer, k, steps=TRAIN_STEPS):
    print(f"\n  Training: K={k}, steps={steps}, unfreeze_last={UNFREEZE_LAST_N}", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [s["text"] for s in ds if len(s["text"].strip()) > 50]

    n_trainable = sum(p.numel() for p in model.trainable_params())
    print(f"  Trainable params: {n_trainable:,}", flush=True)

    # Keep model in fp16, use fp32 optimizer states for stability
    optimizer = torch.optim.AdamW(model.trainable_params(), lr=LR, eps=1e-4)
    model.train()
    total_loss, idx = 0, BATCH_SIZE  # skip warmup batch
    t0 = time.time()

    # Warmup: generate initial stale summaries
    warmup = tokenizer([texts[i % len(texts)][:1000] for i in range(BATCH_SIZE)],
                       return_tensors="pt", padding=True, truncation=True,
                       max_length=SEQ_LEN).to(DEVICE)
    with torch.no_grad():
        _, warmup_sums = model(warmup["input_ids"], mode="async", stale=None)
    stale = [s.detach() if s is not None else None for s in warmup_sums]

    for step in range(steps):
        batch = [texts[idx % len(texts)][:1000] for _ in range(BATCH_SIZE)]
        idx += BATCH_SIZE
        inp = tokenizer(batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=SEQ_LEN).to(DEVICE)

        logits, new_sums = model(inp["input_ids"], mode="async", stale=stale)
        stale = [s.detach() if s is not None else None for s in new_sums]

        # Compute loss in fp32 for numerical stability
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
            avg = total_loss / (step + 1)
            print(f"    step {step+1}/{steps}  loss={avg:.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    t = time.time() - t0
    fl = total_loss / steps
    alphas = [cl.alpha.item() for cl in model.cross_layers]
    print(f"  Done. loss={fl:.4f}, time={t:.1f}s, "
          f"alphas={[f'{a:.4f}' for a in alphas]}", flush=True)
    return fl, t, alphas


# ---------------------------------------------------------------------------
# Latency simulation
# ---------------------------------------------------------------------------
def measure_throughput(model, tokenizer, latency_ms=0, n_tokens=200):
    """Measure tok/s with simulated network latency at boundaries."""
    print(f"  Throughput test (latency={latency_ms}ms, {n_tokens} tokens)...", flush=True)
    model.eval()
    prompt = "The future of distributed computing is"
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inp["input_ids"]

    stale = None
    t0 = time.time()
    generated = 0

    with torch.no_grad():
        for _ in range(n_tokens):
            logits, new_sums = model(input_ids, mode="async", stale=stale)

            # Simulate network latency at each boundary
            if latency_ms > 0:
                time.sleep(latency_ms * len(model.boundaries) / 1000.0)

            stale = [s.detach() if s is not None else None for s in new_sums]

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # Keep context manageable
            if input_ids.size(1) > SEQ_LEN:
                input_ids = input_ids[:, -SEQ_LEN:]
            generated += 1

    elapsed = time.time() - t0
    tps = generated / elapsed
    print(f"    {generated} tokens in {elapsed:.2f}s = {tps:.2f} tok/s", flush=True)
    return tps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    # ---- Baseline ----
    print("\n=== BASELINE ===", flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    bm = DistributedModel(base, N_DEVICES, 16, 0).to(DEVICE)  # 0 unfrozen = fully frozen
    ppl_sync = eval_ppl(bm, tokenizer, "sync")
    hs_sync = eval_hellaswag(bm, tokenizer, 500)

    # Throughput baselines at different latencies
    for lat in [0, 10, 50, 100, 200]:
        tps = measure_throughput(bm, tokenizer, latency_ms=lat, n_tokens=100)
        results.append(Result(f"sync_lat{lat}ms", 0, ppl_sync, ppl_sync, 0.0,
                              tok_per_sec=tps, latency_ms=lat))

    results.append(Result("sync_baseline", 0, ppl_sync, ppl_sync, 0.0,
                          hellaswag_acc=hs_sync))
    _save(results, out / "results.json")
    del bm, base; gc.collect(); torch.cuda.empty_cache()

    # ---- Async experiments: K=16 (sweet spot from v1) ----
    for k in [4, 16, 64]:
        tag = f"async_k{k}"
        print(f"\n=== {tag} (unfreeze_last={UNFREEZE_LAST_N}) ===", flush=True)

        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
        m = DistributedModel(base, N_DEVICES, k, UNFREEZE_LAST_N).to(DEVICE)

        ppl_pre = eval_ppl(m, tokenizer, "async")
        loss, t, alphas = train(m, tokenizer, k)
        ppl_post = eval_ppl(m, tokenizer, "async")
        hs = eval_hellaswag(m, tokenizer, 500)

        gap = ((ppl_post - ppl_sync) / ppl_sync) * 100
        print(f"\n  >>> {tag}: ppl={ppl_post:.4f} (pre={ppl_pre:.4f}, "
              f"gap={gap:+.2f}%)", flush=True)

        results.append(Result(tag, k, ppl_post, ppl_sync, gap,
                              hellaswag_acc=hs, train_loss=loss,
                              train_time_s=t, alphas=alphas))

        # Throughput at different latencies
        for lat in [0, 10, 50, 100, 200]:
            tps = measure_throughput(m, tokenizer, latency_ms=lat, n_tokens=100)
            results.append(Result(f"{tag}_lat{lat}ms", k, ppl_post, ppl_sync, gap,
                                  tok_per_sec=tps, latency_ms=lat))

        _save(results, out / "results.json")
        del m, base; gc.collect(); torch.cuda.empty_cache()

    # ---- No-cross baseline (worst case: no communication at all) ----
    print("\n=== NO CROSS (fully isolated devices) ===", flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    nc = DistributedModel(base, N_DEVICES, 16, UNFREEZE_LAST_N).to(DEVICE)
    # Train with no cross-device info
    # Hackish: zero out all cross layers
    for cl in nc.cross_layers:
        cl.alpha.data.zero_()
        cl.alpha.requires_grad = False
    loss_nc, t_nc, _ = train(nc, tokenizer, 16)
    ppl_nc = eval_ppl(nc, tokenizer, "no_cross")
    gap_nc = ((ppl_nc - ppl_sync) / ppl_sync) * 100
    results.append(Result("no_cross_trained", 0, ppl_nc, ppl_sync, gap_nc,
                          train_loss=loss_nc))
    _save(results, out / "results.json")
    del nc, base; gc.collect(); torch.cuda.empty_cache()

    # ---- Final summary ----
    print("\n" + "=" * 70, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Name':<25s} {'K':>3s} {'PPL':>8s} {'Gap%':>7s} {'HS':>6s} "
          f"{'tok/s':>7s} {'lat_ms':>7s}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        hs = f"{r.hellaswag_acc:.3f}" if r.hellaswag_acc is not None else "—"
        tps = f"{r.tok_per_sec:.1f}" if r.tok_per_sec is not None else "—"
        lat = f"{r.latency_ms:.0f}" if r.latency_ms is not None else "—"
        print(f"{r.name:<25s} {r.k:>3d} {r.ppl:>8.2f} {r.gap_pct:>+6.2f}% "
              f"{hs:>6s} {tps:>7s} {lat:>7s}", flush=True)

    _save(results, out / "results.json")
    print(f"\nSaved to {out / 'results.json'}", flush=True)


def _save(results, path):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/workspace/results/")
    run(p.parse_args().output)
