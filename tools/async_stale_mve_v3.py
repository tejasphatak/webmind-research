#!/usr/bin/env python3
"""
Async Stale-Summary MVE v3 — Merged compressor + boundary-adjacent unfreezing
==============================================================================
Validates the distributed-native training protocol on Qwen3-1.7B.
8 partitions, ~3.5 layers/device, merged carrier-payload compressor.

Controls: sync, sync+compress, async+compress, no-cross
"""

import os, sys, json, time, math, gc, argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
N_DEVICES = 8
TRAIN_STEPS = 800
EVAL_TOKENS = 50_000
LR = 1e-5
BATCH_SIZE = 4
SEQ_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 16  # summary dimensions


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


class MergedCompressor(nn.Module):
    """Unified carrier-payload encoder/decoder at a partition boundary.
    Merges learned PCA-like projection with cross-device mixing."""

    def __init__(self, hidden_dim, k=16, n_neighbors=2):
        super().__init__()
        self.k = k
        self.n_neighbors = n_neighbors
        # Encoder: hidden → k (carrier-payload style learned projection)
        self.encoder = nn.Linear(hidden_dim, k, bias=False)
        # Decoder: k * n_neighbors → hidden (cross-device reconstruction)
        self.decoder = nn.Linear(k * n_neighbors, hidden_dim, bias=False)
        # Per-boundary mixing weight
        self.alpha = nn.Parameter(torch.tensor(0.05))
        nn.init.normal_(self.encoder.weight, std=0.01)
        nn.init.normal_(self.decoder.weight, std=0.01)

    def forward(self, h, neighbor_summaries=None):
        """
        h: [B, S, H] hidden states at this boundary
        neighbor_summaries: list of [B, S, K] from other boundaries (stale)
        Returns: (modified_h, summary_for_neighbors)
        """
        # Encode summary for neighbors
        summary = self.encoder(h.float()).to(h.dtype)

        # Decode neighbor summaries — always use fixed n_neighbors slots
        B, S, H = h.shape
        # Build fixed-size input: pad with zeros for missing neighbors
        slots = []
        for i in range(self.n_neighbors):
            if neighbor_summaries and i < len(neighbor_summaries):
                s = neighbor_summaries[i]
                if s.size(0) != B:
                    s = s[:B] if s.size(0) > B else s[:1].expand(B, -1, -1)
                if s.size(1) != S:
                    if s.size(1) > S:
                        s = s[:, :S, :]
                    else:
                        pad = torch.zeros(s.size(0), S - s.size(1), s.size(2),
                                         device=s.device, dtype=s.dtype)
                        s = torch.cat([s, pad], dim=1)
                slots.append(s)
            else:
                slots.append(torch.zeros(B, S, self.k, device=h.device, dtype=h.dtype))

        cat = torch.cat(slots, dim=-1)  # [B, S, K * n_neighbors] — always fixed size
        cross = self.decoder(cat.float()).to(h.dtype)
        h = h + self.alpha * cross

        return h, summary


class DistributedNativeModel(nn.Module):
    """Wraps a pretrained model with partition boundaries and merged compressors."""

    def __init__(self, base_model, n_devices=8, k=16):
        super().__init__()
        self.base = base_model
        self.layers = base_model.model.layers
        n_layers = len(self.layers)
        self.layers_per_device = n_layers // n_devices
        # Boundaries: last layer index of each device (except the last device)
        self.boundaries = [self.layers_per_device * (i + 1) - 1
                          for i in range(n_devices - 1)]
        n_neighbors = max(len(self.boundaries) - 1, 1)
        hidden_dim = base_model.config.hidden_size

        # Freeze everything first
        for p in base_model.parameters():
            p.requires_grad = False

        # Merged compressors at each boundary
        self.compressors = nn.ModuleList([
            MergedCompressor(hidden_dim, k, n_neighbors)
            for _ in self.boundaries
        ])

        # Unfreeze 1 transformer layer on each side of each boundary
        unfrozen_indices = set()
        for bi in self.boundaries:
            # Layer before boundary (last layer of sending device)
            unfrozen_indices.add(bi)
            # Layer after boundary (first layer of receiving device)
            if bi + 1 < n_layers:
                unfrozen_indices.add(bi + 1)

        for idx in unfrozen_indices:
            for p in self.layers[idx].parameters():
                p.requires_grad = True

        self.unfrozen_count = len(unfrozen_indices)
        print(f"  Architecture: {n_layers} layers, {n_devices} devices, "
              f"{self.layers_per_device} layers/device, "
              f"{len(self.boundaries)} boundaries, "
              f"{self.unfrozen_count} unfrozen transformer layers", flush=True)

    def forward(self, input_ids, mode="async", stale=None):
        """
        Modes:
          sync: no compression, no staleness (upper bound)
          sync_compress: compression at boundaries but fresh (not stale)
          async: compression + one-token-stale (the full system)
          no_cross: zero communication between devices (lower bound)
        """
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
                comp = self.compressors[bi]

                if mode == "sync":
                    # No compression, no staleness — just pass through
                    pass
                elif mode == "sync_compress":
                    # Compress and immediately reconstruct (no staleness)
                    # Use current summaries from OTHER boundaries as neighbors
                    neighbors = [s for j, s in enumerate(summaries)
                                if s is not None and j != bi]
                    h, sm = comp(h, neighbors if neighbors else None)
                    summaries[bi] = sm
                elif mode == "async":
                    # Use STALE summaries from previous token
                    neighbors = []
                    if stale:
                        for j, s in enumerate(stale):
                            if s is not None and j != bi:
                                neighbors.append(s)
                    h, sm = comp(h, neighbors if neighbors else None)
                    summaries[bi] = sm
                elif mode == "no_cross":
                    # Zero communication — encode summary but don't use neighbors
                    _, sm = comp(h, None)
                    summaries[bi] = sm

        if hasattr(self.base.model, 'norm'):
            h = self.base.model.norm(h)

        logits = self.base.lm_head(h)
        return logits, summaries

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_phase2(model, tokenizer, mode, steps=TRAIN_STEPS):
    print(f"\n  Phase 2 training (mode={mode}, steps={steps})", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [s["text"] for s in ds if len(s["text"].strip()) > 50]

    n_train = sum(p.numel() for p in model.trainable_params())
    print(f"  Trainable params: {n_train:,}", flush=True)

    optimizer = torch.optim.AdamW(model.trainable_params(), lr=LR, eps=1e-4)
    model.train()
    total_loss, idx = 0, BATCH_SIZE
    t0 = time.time()

    # Warmup: get initial summaries
    warmup = tokenizer([texts[i % len(texts)][:1000] for i in range(BATCH_SIZE)],
                       return_tensors="pt", padding=True, truncation=True,
                       max_length=SEQ_LEN).to(DEVICE)
    with torch.no_grad():
        _, warmup_sums = model(warmup["input_ids"], mode=mode, stale=None)
    stale = [s.detach() if s is not None else None for s in warmup_sums]

    for step in range(steps):
        batch = [texts[idx % len(texts)][:1000] for _ in range(BATCH_SIZE)]
        idx += BATCH_SIZE
        inp = tokenizer(batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=SEQ_LEN).to(DEVICE)

        logits, new_sums = model(inp["input_ids"], mode=mode, stale=stale)
        stale = [s.detach() if s is not None else None for s in new_sums]

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
    alphas = [c.alpha.item() for c in model.compressors]
    print(f"  Done. loss={fl:.4f}, time={t:.1f}s, "
          f"alphas={[f'{a:.4f}' for a in alphas]}", flush=True)
    return fl, t, alphas


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
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)).float(),
                                   lab.reshape(-1), reduction="sum",
                                   ignore_index=tokenizer.pad_token_id or 0)
            total_loss += loss.item()
            total_n += lab.numel()
    avg = total_loss / max(total_n, 1)
    ppl = math.exp(min(avg, 100))
    print(f"    {mode}: ppl={ppl:.4f} ({total_n} tokens)", flush=True)
    return ppl


def eval_hellaswag(model, tokenizer, n=500):
    print(f"  eval HellaSwag (n={n})...", flush=True)
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation",
                           trust_remote_code=False)
    except Exception as e:
        print(f"    SKIP: {e}", flush=True)
        return None
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, s in enumerate(ds):
            if i >= n:
                break
            scores = []
            for ending in s["endings"]:
                inp = tokenizer(s["ctx"] + " " + ending, return_tensors="pt",
                               truncation=True, max_length=SEQ_LEN).to(DEVICE)
                logits, _ = model(inp["input_ids"], mode="async")
                sl = logits[:, :-1, :].float()
                lab = inp["input_ids"][:, 1:]
                loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)),
                                       lab.reshape(-1), reduction="mean")
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

    # ---- SYNC BASELINE (no modification) ----
    print("\n=== SYNC BASELINE ===", flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    bm = DistributedNativeModel(base, N_DEVICES, K).to(DEVICE)
    ppl_sync = eval_ppl(bm, tokenizer, "sync")
    hs_sync = eval_hellaswag(bm, tokenizer, 500)
    results.append(Result("sync_baseline", 0, ppl_sync, ppl_sync, 0.0,
                          hellaswag_acc=hs_sync))
    _save(results, out / "results.json")
    del bm, base; gc.collect(); torch.cuda.empty_cache()

    # ---- For each mode: train + eval ----
    for mode_name, train_mode in [
        ("no_cross", "no_cross"),
        ("sync_compress", "sync_compress"),
        ("async_compress", "async"),
    ]:
        print(f"\n=== {mode_name.upper()} ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
        m = DistributedNativeModel(base, N_DEVICES, K).to(DEVICE)

        ppl_pre = eval_ppl(m, tokenizer, train_mode)
        loss, t, alphas = train_phase2(m, tokenizer, train_mode)
        ppl_post = eval_ppl(m, tokenizer, train_mode)
        hs = eval_hellaswag(m, tokenizer, 500)

        gap = ((ppl_post - ppl_sync) / ppl_sync) * 100
        print(f"\n  >>> {mode_name}: ppl={ppl_post:.4f} (pre={ppl_pre:.4f}, "
              f"gap={gap:+.2f}%)", flush=True)

        results.append(Result(mode_name, K, ppl_post, ppl_sync, gap,
                              hellaswag_acc=hs, train_loss=loss,
                              train_time_s=t, alphas=alphas))
        _save(results, out / "results.json")
        del m, base; gc.collect(); torch.cuda.empty_cache()

    # ---- Final summary ----
    print("\n" + "=" * 70, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Name':<20s} {'PPL':>8s} {'Gap%':>8s} {'HS':>6s} {'Loss':>7s}", flush=True)
    print("-" * 55, flush=True)
    for r in results:
        hs = f"{r.hellaswag_acc:.3f}" if r.hellaswag_acc is not None else "—"
        lo = f"{r.train_loss:.4f}" if r.train_loss > 0 else "—"
        print(f"{r.name:<20s} {r.ppl:>8.2f} {r.gap_pct:>+7.2f}% {hs:>6s} {lo:>7s}",
              flush=True)

    _save(results, out / "results.json")
    print(f"\nSaved to {out / 'results.json'}", flush=True)


def _save(results, path):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/workspace/results/")
    run(p.parse_args().output)
