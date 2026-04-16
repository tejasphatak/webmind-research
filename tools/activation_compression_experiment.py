"""
Paradigm 1: Activation Wire Compression for Decentralized Inference
"Carrier + Payload" experiment on real Gemma 3 1B IT activations.

The 2-hour make-or-break experiment (Gemini's spec):
1. Pass diverse prompts through Gemma 3 1B IT.
2. Extract fp16 activations at each layer boundary.
3. Fit PCA on the activations (= "carrier" = shared basis between nodes).
4. Compute residual, sparsify (keep top-k% by magnitude = "payload").
5. Reconstruct and continue forward pass from the splice point.
6. Measure: KL divergence of final logits vs uncompressed baseline.
7. Sweep compression ratios to get a Pareto curve.

Author: Claude Opus 4.6 (Synapse Research)
Date: 2026-04-16
Pre-reg: to be filed after exploratory run confirms feasibility.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-1b-it"


def extract_layer_activations(model, input_ids: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Run full forward pass, capture hidden state after each transformer block."""
    base = model.model
    hidden = base.embed_tokens(input_ids).float()
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    layer_acts = []
    for block in base.layers:
        out = block(hidden, position_ids=position_ids)
        if isinstance(out, tuple):
            out = out[0]
        hidden = out.float()
        layer_acts.append(hidden.detach().clone())

    final_hidden = base.norm(hidden)
    logits = model.lm_head(final_hidden)
    return layer_acts, logits.float()


def forward_from_layer(model, hidden: torch.Tensor, start_layer: int) -> torch.Tensor:
    """Continue forward pass from a given layer using a (possibly compressed) hidden state."""
    base = model.model
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    h = hidden.float()
    for block in list(base.layers)[start_layer:]:
        out = block(h, position_ids=position_ids)
        if isinstance(out, tuple):
            out = out[0]
        h = out.float()

    final_hidden = base.norm(h)
    logits = model.lm_head(final_hidden)
    return logits.float()


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

    # --- Carrier: PCA via SVD ---
    mean = A.mean(dim=0, keepdim=True)
    A_centered = A - mean
    U, S, Vt = torch.linalg.svd(A_centered, full_matrices=False)
    # Keep top-k components
    k = min(pca_rank, min(seq_len, hidden_dim))
    carrier = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :] + mean
    residual = A - carrier

    # --- Payload: sparse residual ---
    flat_res = residual.flatten()
    n_total = flat_res.numel()
    n_keep = max(1, int(sparse_topk_frac * n_total))
    topk_vals, topk_idx = torch.topk(flat_res.abs(), n_keep)
    sparse_payload = torch.zeros_like(flat_res)
    sparse_payload[topk_idx] = flat_res[topk_idx]
    sparse_residual = sparse_payload.reshape_as(residual)

    # Reconstruct
    reconstructed = carrier + sparse_residual

    # Compression stats
    # Wire cost (what crosses the network):
    # Carrier: PCA components = k * hidden_dim (Vt rows) + k * seq_len (U @ S cols) + hidden_dim (mean)
    # Actually for shared-basis scenario: Vt[:k] is the shared basis (pre-distributed).
    # Wire: U[:,:k] @ diag(S[:k]) = seq_len * k floats + mean (hidden_dim) + sparse indices + values
    carrier_wire_floats = seq_len * k + hidden_dim  # projections + mean
    payload_wire_floats = n_keep * 2  # value + index (index as float-equivalent)
    baseline_floats = seq_len * hidden_dim
    total_wire = carrier_wire_floats + payload_wire_floats
    compression_ratio = baseline_floats / total_wire

    # Reconstruction quality
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
    """
    For each prompt × splice_layer × pca_rank × sparse_frac:
    1. Run full uncompressed forward pass (baseline logits).
    2. Extract activation at splice_layer.
    3. Compress with carrier+payload.
    4. Splice compressed activation back in, run remainder of model.
    5. Measure KL divergence + top-1 agreement + perplexity shift.
    """
    results = []
    for p_idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True,
                              truncation=True, max_length=128).input_ids
        # Baseline
        with torch.no_grad():
            layer_acts, baseline_logits = extract_layer_activations(model, input_ids)
        baseline_top1 = baseline_logits[:, -1, :].argmax(dim=-1).item()
        baseline_top5 = set(torch.topk(baseline_logits[:, -1, :], 5).indices[0].tolist())

        for splice_layer in splice_layers:
            act = layer_acts[splice_layer]
            for rank in pca_ranks:
                for sfrac in sparse_fracs:
                    t0 = time.time()
                    reconstructed, comp_stats = compress_carrier_payload(act, rank, sfrac)
                    with torch.no_grad():
                        test_logits = forward_from_layer(model, reconstructed, splice_layer + 1)
                    dt = time.time() - t0

                    # KL divergence
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
                    if p_idx % 4 == 0 and splice_layer == splice_layers[0]:
                        print(f"  [{time.strftime('%H:%M:%S')}] p{p_idx:02d} L{splice_layer:02d} "
                              f"r={rank:2d} s={sfrac:.3f} "
                              f"CR={comp_stats['compression_ratio']:.1f}x "
                              f"cos={comp_stats['reconstruction_cosine']:.5f} "
                              f"KL={kl:.4e} top1={'Y' if test_top1 == baseline_top1 else 'N'}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-prompts", type=int, default=32)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    print(f"[{time.strftime('%H:%M:%S')}] Loading {MODEL_ID} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=hf_token, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    n_layers = len(model.model.layers)
    print(f"  loaded in {time.time()-t0:.1f}s; {n_layers} layers")

    # Diverse prompts — mix of reasoning, factual, code, creative
    seed_prompts = [
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
    prompts = seed_prompts[:args.n_prompts]

    # Experiment grid
    # Splice at early (layer 4), middle (n//2), and late (n-4) layers
    splice_layers = [4, n_layers // 2, n_layers - 4]
    # PCA ranks: low (aggressive compression) to high (mild compression)
    pca_ranks = [2, 4, 8, 16, 32]
    # Sparse residual fractions: 0 (carrier only), 0.01 (1%), 0.05 (5%), 0.10 (10%)
    sparse_fracs = [0.0, 0.005, 0.01, 0.05, 0.10]

    print(f"  Grid: {len(prompts)} prompts × {len(splice_layers)} layers × "
          f"{len(pca_ranks)} ranks × {len(sparse_fracs)} sparsities "
          f"= {len(prompts) * len(splice_layers) * len(pca_ranks) * len(sparse_fracs)} evals")

    results = run_experiment(model, tokenizer, prompts, splice_layers, pca_ranks, sparse_fracs)

    payload = {
        "meta": {
            "model_id": MODEL_ID,
            "n_layers": n_layers,
            "n_prompts": len(prompts),
            "splice_layers": splice_layers,
            "pca_ranks": pca_ranks,
            "sparse_fracs": sparse_fracs,
            "device": "cpu",
            "torch_version": torch.__version__,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"\nWrote {args.out} ({len(results)} records)")


if __name__ == "__main__":
    main()
