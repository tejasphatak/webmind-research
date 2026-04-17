#!/usr/bin/env python3
"""
Async Stale-Summary Distributed Inference — Minimum Viable Experiment
=====================================================================
Simulates 4-device async inference on Gemma 3 1B (single GPU).

Axes (run in parallel where possible):
  1. detach vs no-detach (does upstream learn to compress?)
  2. K sweep: [4, 8, 16, 32, 64] summary dimensions
  3. Baselines: sync (full activations), async-stale, no-cross-device
  4. Task eval: perplexity + HellaSwag accuracy (downstream reasoning)

Usage:
  python3 async_stale_mve.py --output /workspace/results/
"""

import os
import sys
import json
import time
import argparse
import math
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-1.5B"
N_DEVICES = 4          # simulated device count
TRAIN_STEPS = 500      # phase-2 fine-tuning steps
EVAL_TOKENS = 50_000   # tokens for perplexity eval
LR = 1e-3
BATCH_SIZE = 4
SEQ_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_VALUES = [4, 8, 16, 32, 64]
DETACH_MODES = [True, False]  # True = original design, False = fix


@dataclass
class ExperimentResult:
    mode: str              # "sync", "async_detach", "async_nodetach", "no_cross"
    k: int
    detach: bool
    ppl_before: float      # perplexity before phase-2 training
    ppl_after: float       # perplexity after phase-2 training
    ppl_sync_baseline: float
    hellaswag_acc: Optional[float] = None
    train_loss_final: float = 0.0
    train_time_s: float = 0.0
    alpha_values: Optional[list] = None


# ---------------------------------------------------------------------------
# Async layer wrapper
# ---------------------------------------------------------------------------

class AsyncCrossDeviceLayer(nn.Module):
    """Wraps a transformer layer with async cross-device summary mechanism."""

    def __init__(self, hidden_dim: int, summary_dim: int = 16,
                 n_neighbors: int = 3, detach_summary: bool = True):
        super().__init__()
        self.summary_dim = summary_dim
        self.detach_summary = detach_summary

        # Learned compression: hidden_dim -> summary_dim
        self.compressor = nn.Linear(hidden_dim, summary_dim, bias=False)

        # Cross-device projection: summary from neighbors -> hidden_dim
        self.cross_proj = nn.Linear(summary_dim * n_neighbors, hidden_dim, bias=False)

        # Learnable mixing weight (per-layer)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Init small so cross-device starts as minor correction
        nn.init.normal_(self.compressor.weight, std=0.01)
        nn.init.normal_(self.cross_proj.weight, std=0.01)

    def forward(self, hidden_states: torch.Tensor,
                stale_summaries: Optional[list] = None):
        """
        Args:
            hidden_states: [batch, seq, hidden] — output of a transformer layer
            stale_summaries: list of [batch, seq, summary_dim] from neighbors (t-1)
        Returns:
            modified hidden_states, current summary for this device
        """
        # Cross-device contribution
        if stale_summaries is not None and len(stale_summaries) > 0:
            # Handle variable batch/sequence lengths
            batch_size = hidden_states.size(0)
            seq_len = hidden_states.size(1)
            aligned = []
            for s in stale_summaries:
                # Match batch size
                if s.size(0) != batch_size:
                    s = s[:batch_size] if s.size(0) > batch_size else s.repeat(
                        (batch_size // s.size(0)) + 1, 1, 1)[:batch_size]
                # Match seq length
                if s.size(1) > seq_len:
                    s = s[:, :seq_len, :]
                elif s.size(1) < seq_len:
                    pad = torch.zeros(s.size(0), seq_len - s.size(1), s.size(2),
                                     device=s.device, dtype=s.dtype)
                    s = torch.cat([s, pad], dim=1)
                aligned.append(s)
            cat_summaries = torch.cat(aligned, dim=-1)  # [B, S, K*N]
            cross = self.cross_proj(cat_summaries)        # [B, S, H]
            hidden_states = hidden_states + self.alpha * cross

        # Generate summary for neighbors
        if self.detach_summary:
            summary = self.compressor(hidden_states.detach())
        else:
            summary = self.compressor(hidden_states)

        return hidden_states, summary


# ---------------------------------------------------------------------------
# Simulated distributed model
# ---------------------------------------------------------------------------

class SimulatedDistributedModel(nn.Module):
    """
    Wraps a Gemma model, partitions layers across N simulated devices,
    and injects async cross-device layers at partition boundaries.
    """

    def __init__(self, base_model, n_devices: int = 4,
                 summary_dim: int = 16, detach: bool = True):
        super().__init__()
        self.base_model = base_model
        self.n_devices = n_devices
        self.summary_dim = summary_dim

        # Freeze the base model — only train cross-device layers
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Get transformer layers
        self.layers = base_model.model.layers
        n_layers = len(self.layers)
        self.layers_per_device = n_layers // n_devices

        # Boundaries: where one device ends and another begins
        self.boundaries = [self.layers_per_device * (i + 1) - 1
                          for i in range(n_devices - 1)]

        hidden_dim = base_model.config.hidden_size
        # Each boundary sees summaries from other boundaries (not itself)
        self.n_neighbors = len(self.boundaries) - 1 if len(self.boundaries) > 1 else 1

        # Cross-device layers at each boundary
        # Keep in fp32 for training stability; cast inputs/outputs as needed
        self.cross_layers = nn.ModuleList([
            AsyncCrossDeviceLayer(hidden_dim, summary_dim, self.n_neighbors, detach)
            for _ in self.boundaries
        ])

    def _run_layers(self, hidden, position_embeddings, mode, stale_summaries):
        """Run transformer layers with optional async cross-device injection.
        Returns (hidden, new_summaries)."""
        current_summaries = [None] * len(self.boundaries)

        for layer_idx, layer in enumerate(self.layers):
            if position_embeddings is not None:
                layer_out = layer(hidden, position_embeddings=position_embeddings)
            else:
                layer_out = layer(hidden)
            if isinstance(layer_out, tuple):
                hidden = layer_out[0]
            else:
                hidden = layer_out

            if layer_idx in self.boundaries:
                boundary_idx = self.boundaries.index(layer_idx)
                cross_layer = self.cross_layers[boundary_idx]

                if mode == "async":
                    stale = []
                    if stale_summaries is not None:
                        for i, s in enumerate(stale_summaries):
                            if s is not None and i != boundary_idx:
                                stale.append(s.float())
                    # Cross layers are fp32 for stability
                    hidden_f32 = hidden.float()
                    hidden_f32, summary = cross_layer(
                        hidden_f32, stale if stale else None)
                    hidden = hidden_f32.half()
                    current_summaries[boundary_idx] = summary

        if hasattr(self.base_model.model, 'norm'):
            hidden = self.base_model.model.norm(hidden)

        return hidden, current_summaries

    def forward_with_async(self, input_ids, mode="async",
                           stale_summaries=None):
        """
        Modes:
          - "sync": pass full hidden states at boundaries (upper bound)
          - "async": pass 1-token-stale compressed summaries
          - "no_cross": no cross-device communication (lower bound)
        Returns: (logits, current_summaries) so caller can feed staleness
        """
        hidden = self.base_model.model.embed_tokens(input_ids)

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(
            seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = None
        if hasattr(self.base_model.model, 'rotary_emb'):
            position_embeddings = self.base_model.model.rotary_emb(
                hidden, position_ids)

        hidden, current_summaries = self._run_layers(
            hidden, position_embeddings, mode, stale_summaries)

        logits = self.base_model.lm_head(hidden)
        return logits, current_summaries

    def trainable_parameters(self):
        return [p for p in self.cross_layers.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Training (Phase 2 — distributed fine-tuning)
# ---------------------------------------------------------------------------

def train_phase2(model, tokenizer, summary_dim, detach, steps=TRAIN_STEPS):
    """Fine-tune only the cross-device layers on C4 data."""
    print(f"\n{'='*60}")
    print(f"Phase 2 training: K={summary_dim}, detach={detach}, steps={steps}")
    print(f"{'='*60}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           split="train")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR)

    model.train()
    total_loss = 0
    t0 = time.time()

    # Filter empty samples and build text list
    all_texts = [s["text"] for s in dataset if len(s["text"].strip()) > 50]
    text_idx = 0

    # Warmup: generate initial stale summaries with a throwaway forward pass
    warmup_texts = [all_texts[i % len(all_texts)][:1000] for i in range(BATCH_SIZE)]
    warmup_inputs = tokenizer(warmup_texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=SEQ_LEN).to(DEVICE)
    with torch.no_grad():
        _, warmup_sums = model.forward_with_async(
            warmup_inputs["input_ids"], mode="async", stale_summaries=None)
    stale_summaries = [s.detach() if s is not None else None for s in warmup_sums]
    text_idx = BATCH_SIZE  # skip warmup samples

    for step in range(steps):
        texts = []
        for _ in range(BATCH_SIZE):
            texts.append(all_texts[text_idx % len(all_texts)][:1000])
            text_idx += 1

        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=SEQ_LEN).to(DEVICE)

        # Forward pass in async mode — feed stale summaries from prev step
        logits, new_summaries = model.forward_with_async(
            inputs["input_ids"], mode="async",
            stale_summaries=stale_summaries)

        # Update stale summaries for next step (simulates one-token delay)
        stale_summaries = [s.detach() if s is not None else None
                          for s in new_summaries]

        # Language modeling loss (shift by 1)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                              shift_labels.view(-1),
                              ignore_index=tokenizer.pad_token_id or 0)

        optimizer.zero_grad()
        if step == 0:
            # Debug: check which params have grad
            print(f"  [debug] loss requires_grad={loss.requires_grad}", flush=True)
            print(f"  [debug] trainable params: {len(model.trainable_parameters())}", flush=True)
            for i, p in enumerate(model.trainable_parameters()):
                print(f"    param[{i}] shape={p.shape} requires_grad={p.requires_grad}", flush=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if (step + 1) % 50 == 0:
            avg = total_loss / (step + 1)
            elapsed = time.time() - t0
            print(f"  step {step+1}/{steps}  loss={avg:.4f}  "
                  f"elapsed={elapsed:.1f}s", flush=True)

    train_time = time.time() - t0
    final_loss = total_loss / steps
    print(f"  Done. Final avg loss={final_loss:.4f}, time={train_time:.1f}s")

    # Collect alpha values
    alphas = [cl.alpha.item() for cl in model.cross_layers]
    print(f"  Learned alpha values: {[f'{a:.4f}' for a in alphas]}")

    return final_loss, train_time, alphas


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_perplexity(model, tokenizer, mode="async", n_tokens=EVAL_TOKENS):
    """Evaluate perplexity on C4 validation."""
    print(f"  Evaluating perplexity (mode={mode})...", flush=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           split="test")

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for sample in dataset:
            if total_tokens >= n_tokens:
                break

            inputs = tokenizer(sample["text"][:1000], return_tensors="pt",
                             truncation=True, max_length=SEQ_LEN).to(DEVICE)

            if inputs["input_ids"].size(1) < 10:
                continue

            logits, _ = model.forward_with_async(inputs["input_ids"], mode=mode)
            shift_logits = logits[:, :-1, :]
            shift_labels = inputs["input_ids"][:, 1:]

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="sum",
                ignore_index=tokenizer.pad_token_id or 0
            )

            n = shift_labels.numel()
            total_loss += loss.item()
            total_tokens += n

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))  # cap to avoid overflow
    print(f"    {mode}: loss={avg_loss:.4f}, ppl={ppl:.2f} ({total_tokens} tokens)")
    return ppl


def eval_hellaswag(model, tokenizer, n_samples=200):
    """Simple HellaSwag accuracy — pick the most likely continuation."""
    print(f"  Evaluating HellaSwag (n={n_samples})...")
    try:
        dataset = load_dataset("Rowan/hellaswag", split="validation",
                               trust_remote_code=False)
    except Exception as e:
        print(f"    HellaSwag load failed: {e}")
        return None

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= n_samples:
                break

            ctx = sample["ctx"]
            endings = sample["endings"]
            label = int(sample["label"])

            # Score each ending
            scores = []
            for ending in endings:
                text = ctx + " " + ending
                inputs = tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=SEQ_LEN).to(DEVICE)

                logits, _ = model.forward_with_async(inputs["input_ids"],
                                                     mode="async")
                shift_logits = logits[:, :-1, :]
                shift_labels = inputs["input_ids"][:, 1:]

                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="mean"
                )
                scores.append(-loss.item())  # higher = better

            pred = scores.index(max(scores))
            if pred == label:
                correct += 1
            total += 1

    acc = correct / max(total, 1)
    print(f"    HellaSwag accuracy: {correct}/{total} = {acc:.3f}")
    return acc


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    # --- Baseline: sync perplexity (no modification) ---
    print("\n" + "="*60)
    print("BASELINE: Original model (no async modification)")
    print("="*60)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(DEVICE)

    # Wrap as distributed model just to use same forward path
    sync_model = SimulatedDistributedModel(base_model, N_DEVICES, 16, True).to(DEVICE)
    ppl_sync = eval_perplexity(sync_model, tokenizer, mode="sync")
    ppl_no_cross = eval_perplexity(sync_model, tokenizer, mode="no_cross")

    # HellaSwag baseline
    hellaswag_sync = eval_hellaswag(sync_model, tokenizer)

    results.append(ExperimentResult(
        mode="sync_baseline", k=0, detach=False,
        ppl_before=ppl_sync, ppl_after=ppl_sync,
        ppl_sync_baseline=ppl_sync,
        hellaswag_acc=hellaswag_sync
    ))
    results.append(ExperimentResult(
        mode="no_cross_baseline", k=0, detach=False,
        ppl_before=ppl_no_cross, ppl_after=ppl_no_cross,
        ppl_sync_baseline=ppl_sync,
        hellaswag_acc=None
    ))

    # Save intermediate
    _save_results(results, output_path / "results.json")

    del sync_model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Sweep: K values × detach modes ---
    for k in K_VALUES:
        for detach in DETACH_MODES:
            tag = f"k{k}_{'detach' if detach else 'nodetach'}"
            print(f"\n{'='*60}")
            print(f"EXPERIMENT: {tag}")
            print(f"{'='*60}")

            # Fresh model each run
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16
            ).to(DEVICE)
            model = SimulatedDistributedModel(
                base, N_DEVICES, k, detach
            ).to(DEVICE)

            # Pre-training perplexity (async mode, untrained cross-layers)
            ppl_before = eval_perplexity(model, tokenizer, mode="async")

            # Phase 2 training
            final_loss, train_time, alphas = train_phase2(
                model, tokenizer, k, detach
            )

            # Post-training perplexity
            ppl_after = eval_perplexity(model, tokenizer, mode="async")

            # HellaSwag (only for K=16 to save time, plus best candidates)
            hellaswag = None
            if k == 16 or k == 32:
                hellaswag = eval_hellaswag(model, tokenizer)

            result = ExperimentResult(
                mode=f"async_{'detach' if detach else 'nodetach'}",
                k=k, detach=detach,
                ppl_before=ppl_before, ppl_after=ppl_after,
                ppl_sync_baseline=ppl_sync,
                hellaswag_acc=hellaswag,
                train_loss_final=final_loss,
                train_time_s=train_time,
                alpha_values=alphas
            )
            results.append(result)

            # Save after each run (crash safety)
            _save_results(results, output_path / "results.json")

            # Print summary
            gap = ((ppl_after - ppl_sync) / ppl_sync) * 100
            print(f"\n  >>> {tag}: ppl={ppl_after:.2f} "
                  f"(gap from sync: {gap:+.1f}%)")

            del model, base
            gc.collect()
            torch.cuda.empty_cache()

    # --- Final summary ---
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Mode':<30s} {'K':>3s} {'PPL':>8s} {'Gap%':>7s} {'HS-acc':>7s}")
    print("-" * 60)
    for r in results:
        gap = ((r.ppl_after - r.ppl_sync_baseline) / r.ppl_sync_baseline) * 100
        hs = f"{r.hellaswag_acc:.3f}" if r.hellaswag_acc is not None else "—"
        print(f"{r.mode:<30s} {r.k:>3d} {r.ppl_after:>8.2f} {gap:>+6.1f}% {hs:>7s}")

    _save_results(results, output_path / "results.json")
    print(f"\nResults saved to {output_path / 'results.json'}")


def _save_results(results, path):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/workspace/results/",
                        help="Output directory for results")
    args = parser.parse_args()
    run_experiment(args.output)
