"""
Paradigm 1: Activation Wire Compression for Decentralized Inference
"Carrier + Payload" experiment on real Gemma activations.

The make-or-break experiment:
1. Pass diverse prompts through the model.
2. Extract activations at each layer boundary via output_hidden_states.
3. Fit PCA on the activations (= "carrier" = shared basis between nodes).
4. Compute residual, sparsify (keep top-k% by magnitude = "payload").
5. Splice compressed activation back in via forward hook, run remainder.
6. Measure: KL divergence of final logits vs uncompressed baseline.
7. Sweep compression ratios to get a Pareto curve.

Author: Claude Opus 4.6 (Synapse Research)
Date: 2026-04-16
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-1b-it"


@torch.no_grad()
def get_baseline(model, input_ids: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Get all hidden states + final logits from an uncompressed forward pass."""
    out = model(input_ids, output_hidden_states=True)
    # hidden_states: tuple of (n_layers+1) tensors, [0]=embedding, [1..n]=post-block
    hidden_states = [h.detach().float().clone() for h in out.hidden_states]
    return hidden_states, out.logits.float().detach()


@torch.no_grad()
def forward_with_splice(
    model, input_ids: torch.Tensor,
    splice_layer: int, replacement: torch.Tensor
) -> torch.Tensor:
    """
    Run forward pass but replace the OUTPUT of transformer block `splice_layer`
    with `replacement`. Uses a forward hook on that block.

    splice_layer is 0-indexed into model.model.layers.
    hidden_states[splice_layer+1] corresponds to the output of layers[splice_layer].
    """
    replaced = [False]

    def replace_hook(module, input, output):
        if replaced[0]:
            return output
        replaced[0] = True
        # output is typically a tuple: (hidden_states, ...) or just hidden_states
        repl = replacement.to(output[0].dtype if isinstance(output, tuple) else output.dtype)
        if isinstance(output, tuple):
            return (repl,) + output[1:]
        return repl

    hook = model.model.layers[splice_layer].register_forward_hook(replace_hook)
    out = model(input_ids)
    hook.remove()
    return out.logits.float().detach()


def compress_carrier_payload(
    act: torch.Tensor,
    pca_rank: int,
    sparse_topk_frac: float,
) -> Tuple[torch.Tensor, Dict]:
    """
    Carrier + Payload compression.

    Carrier: PCA projection (rank-k approximation of the activation matrix).
    Payload: sparse residual (top-k% of residual entries by magnitude).

    Returns reconstructed activation + compression stats.
    """
    # act: (1, seq_len, hidden_dim) -> work with (seq_len, hidden_dim)
    A = act.squeeze(0).float()
    seq_len, hidden_dim = A.shape

    mean = A.mean(dim=0, keepdim=True)
    A_centered = A - mean
    U, S, Vt = torch.linalg.svd(A_centered, full_matrices=False)
    k = min(pca_rank, min(seq_len, hidden_dim))
    carrier = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :] + mean
    residual = A - carrier

    flat_res = residual.flatten()
    n_total = flat_res.numel()
    n_keep = max(1, int(sparse_topk_frac * n_total)) if sparse_topk_frac > 0 else 0

    if n_keep > 0:
        _, topk_idx = torch.topk(flat_res.abs(), n_keep)
        sparse_payload = torch.zeros_like(flat_res)
        sparse_payload[topk_idx] = flat_res[topk_idx]
        sparse_residual = sparse_payload.reshape_as(residual)
    else:
        sparse_residual = torch.zeros_like(residual)
        n_keep = 0

    reconstructed = carrier + sparse_residual

    # Wire cost calculation
    # Shared-basis model: PCA basis Vt[:k] pre-distributed to all nodes (amortized).
    # Per-activation wire: projections (seq_len × k) + mean (hidden_dim) + sparse (n_keep × 2)
    carrier_wire_floats = seq_len * k + hidden_dim
    payload_wire_floats = n_keep * 2
    baseline_floats = seq_len * hidden_dim
    total_wire = carrier_wire_floats + payload_wire_floats
    compression_ratio = baseline_floats / max(total_wire, 1)

    mse = F.mse_loss(reconstructed, A).item()
    cosine = F.cosine_similarity(
        reconstructed.flatten().unsqueeze(0),
        A.flatten().unsqueeze(0)
    ).item()

    stats = {
        "pca_rank": k,
        "sparse_topk_frac": sparse_topk_frac,
        "n_keep_sparse": n_keep,
        "baseline_floats": baseline_floats,
        "carrier_wire_floats": carrier_wire_floats,
        "payload_wire_floats": payload_wire_floats,
        "total_wire_floats": total_wire,
        "compression_ratio": compression_ratio,
        "reconstruction_mse": mse,
        "reconstruction_cosine": cosine,
        "variance_explained": float((S[:k] ** 2).sum() / (S ** 2).sum()),
    }
    return reconstructed.unsqueeze(0), stats


def run_experiment(
    model,
    tokenizer,
    prompts: List[str],
    splice_layers: List[int],
    pca_ranks: List[int],
    sparse_fracs: List[float],
) -> List[Dict]:
    results = []
    device = next(model.parameters()).device

    for p_idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True,
                              truncation=True, max_length=128).input_ids.to(device)

        hidden_states, baseline_logits = get_baseline(model, input_ids)
        baseline_top1 = baseline_logits[:, -1, :].argmax(dim=-1).item()
        baseline_top5 = set(torch.topk(baseline_logits[:, -1, :], 5).indices[0].tolist())

        for splice_layer in splice_layers:
            # hidden_states[splice_layer+1] = output of layers[splice_layer]
            act = hidden_states[splice_layer + 1]

            for rank in pca_ranks:
                for sfrac in sparse_fracs:
                    t0 = time.time()
                    reconstructed, comp_stats = compress_carrier_payload(act, rank, sfrac)

                    test_logits = forward_with_splice(
                        model, input_ids, splice_layer, reconstructed.to(device)
                    )
                    dt = time.time() - t0

                    ref_lp = F.log_softmax(baseline_logits[:, -1, :], dim=-1)
                    test_lp = F.log_softmax(test_logits[:, -1, :], dim=-1)
                    kl = F.kl_div(test_lp, ref_lp, reduction="batchmean", log_target=True).item()

                    test_top1 = test_logits[:, -1, :].argmax(dim=-1).item()
                    test_top5 = set(torch.topk(test_logits[:, -1, :], 5).indices[0].tolist())

                    results.append({
                        "prompt_idx": p_idx,
                        "splice_layer": splice_layer,
                        **comp_stats,
                        "kl_divergence": kl,
                        "top1_agree": int(test_top1 == baseline_top1),
                        "top5_overlap": len(test_top5 & baseline_top5),
                        "time_sec": dt,
                    })

        if p_idx % 4 == 0:
            last = results[-1]
            print(f"  [{time.strftime('%H:%M:%S')}] prompt {p_idx}/{len(prompts)} done "
                  f"({len(splice_layers) * len(pca_ranks) * len(sparse_fracs)} configs/prompt)")

    return results


PROMPTS = [
    "Explain the concept of Byzantine fault tolerance in distributed systems.",
    "Write a Python function that computes the nth Fibonacci number using matrix exponentiation.",
    "What are the three laws of thermodynamics and why do they matter for engineering?",
    "Translate the following English text to French: The quick brown fox jumps over the lazy dog.",
    "Describe the architecture of a transformer neural network in detail.",
    "If a train leaves Chicago at 60 mph and another leaves New York at 80 mph, when do they meet?",
    "What is the significance of Gödel's incompleteness theorems for artificial intelligence?",
    "Write a haiku about quantum entanglement.",
    "Explain how TCP congestion control works and why it matters for distributed AI inference.",
    "What is the manifold hypothesis and how does it relate to dimensionality reduction?",
    "Describe the key differences between fp16, bf16, and fp32 number representations.",
    "How does the Shor's algorithm threaten current cryptographic systems?",
    "Write pseudocode for a distributed consensus algorithm like Raft.",
    "Explain activation outliers in large language models and why they cause quantization problems.",
    "What is the holographic principle in physics and could it apply to neural networks?",
    "Describe the carrier-payload decomposition for signal compression.",
    "How do volunteer computing projects like BOINC ensure result integrity?",
    "What makes WebGPU different from WebGL for machine learning workloads?",
    "Explain the birthday paradox and its implications for hash collision probability.",
    "Write a brief proof that the square root of 2 is irrational.",
    "What is Kolmogorov complexity and why is it uncomputable?",
    "Describe how PCA works and when it fails as a dimensionality reduction technique.",
    "Explain the difference between lossless and lossy compression with examples.",
    "What is the curse of dimensionality and how does it affect nearest-neighbor search?",
    "How does error correction work in quantum computing?",
    "Describe the concept of information entropy in Shannon's framework.",
    "What are Kolmogorov-Arnold Networks and how do they differ from MLPs?",
    "Explain the SmoothQuant technique for LLM quantization.",
    "Write a function to compute the SVD of a matrix and explain each component.",
    "What is the Nyquist-Shannon sampling theorem and why does it matter for digital signals?",
    "Describe the concept of a Pareto frontier in multi-objective optimization.",
    "How does federated learning differ from traditional distributed training?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--n-prompts", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[{time.strftime('%H:%M:%S')}] Loading {args.model} on {device} ...")
    t0 = time.time()
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=hf_token, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    n_layers = len(model.model.layers)
    print(f"  loaded in {time.time()-t0:.1f}s; {n_layers} layers, dtype={dtype}")

    prompts = PROMPTS[:args.n_prompts]

    splice_layers = [4, n_layers // 2, n_layers - 4]
    pca_ranks = [2, 4, 8, 16, 32]
    sparse_fracs = [0.0, 0.005, 0.01, 0.05, 0.10]

    total = len(prompts) * len(splice_layers) * len(pca_ranks) * len(sparse_fracs)
    print(f"  Grid: {len(prompts)} prompts × {len(splice_layers)} layers × "
          f"{len(pca_ranks)} ranks × {len(sparse_fracs)} sparsities = {total} evals")

    results = run_experiment(model, tokenizer, prompts, splice_layers, pca_ranks, sparse_fracs)

    payload = {
        "meta": {
            "model_id": args.model,
            "n_layers": n_layers,
            "n_prompts": len(prompts),
            "splice_layers": splice_layers,
            "pca_ranks": pca_ranks,
            "sparse_fracs": sparse_fracs,
            "device": device,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    elapsed = time.time() - t0
    print(f"\nWrote {args.out} ({len(results)} records, {elapsed:.1f}s total)")


if __name__ == "__main__":
    main()
