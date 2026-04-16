"""
Multi-model activation compression experiment.
Runs carrier-payload compression on any HuggingFace causal LM.

Usage:
    python multi_model_experiment.py --model google/gemma-3-1b-it --out results.json
    python multi_model_experiment.py --model meta-llama/Llama-3.1-8B-Instruct --out results.json
    python multi_model_experiment.py --model google/gemma-4-31b-it --out results.json
    python multi_model_experiment.py --model Qwen/Qwen2.5-32B-Instruct --out results.json
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
    "Explain the concept of Byzantine fault tolerance in distributed systems. Cover the Byzantine Generals Problem and practical BFT algorithms like PBFT.",
    "Write a Python function that computes the nth Fibonacci number using matrix exponentiation. Include type hints and a docstring.",
    "What are the three laws of thermodynamics and why do they matter for engineering? Give concrete examples for each law.",
    "Describe the architecture of a transformer neural network in detail. Cover multi-head attention, positional encoding, and layer normalization.",
    "What is the manifold hypothesis and how does it relate to dimensionality reduction in machine learning?",
    "Explain activation outliers in large language models and why they cause quantization problems. Reference LLM.int8 and SmoothQuant.",
    "How do volunteer computing projects like BOINC and Folding@Home ensure result integrity from untrusted participants?",
    "Describe the concept of information entropy in Shannon's framework and how it applies to data compression.",
    "Explain the difference between pipeline parallelism and tensor parallelism in distributed model serving. What are the communication patterns?",
    "What is Kolmogorov complexity and why is it uncomputable? How does this relate to practical compression algorithms?",
    "Describe the complete training pipeline for a large language model from data collection through RLHF alignment.",
    "Explain the mathematics behind PCA including the relationship to SVD and the optimality guarantees for variance preservation.",
    "What makes WebGPU different from WebGL for machine learning workloads? Discuss compute shaders and memory management.",
    "Describe federated learning and how it addresses privacy concerns. What are the key challenges with non-IID data distributions?",
    "Explain the holographic principle in physics. Could boundary-bulk duality have any analogy in neural network compression?",
    "Write a detailed comparison of lossy vs lossless compression. When is each appropriate? Give examples from both signal processing and ML.",
]

@torch.no_grad()
def get_baseline(model, input_ids):
    out = model(input_ids, output_hidden_states=True)
    hidden = [h.detach().float().cpu().clone() for h in out.hidden_states]
    return hidden, out.logits.float().cpu().detach()

@torch.no_grad()
def forward_with_splice(model, input_ids, splice_layer, replacement, device):
    replaced = [False]
    def hook(module, inp, output):
        if replaced[0]: return output
        replaced[0] = True
        repl = replacement.to(output[0].dtype if isinstance(output, tuple) else output.dtype).to(device)
        return (repl,) + output[1:] if isinstance(output, tuple) else repl
    h = model.model.layers[splice_layer].register_forward_hook(hook)
    out = model(input_ids)
    h.remove()
    return out.logits.float().cpu().detach()

def compress(act, rank, sparse_frac):
    A = act.squeeze(0).float()
    seq_len, hdim = A.shape
    mean = A.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(A - mean, full_matrices=False)
    k = min(rank, min(seq_len, hdim))
    carrier = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :] + mean
    res = A - carrier
    n_keep = max(1, int(sparse_frac * res.numel())) if sparse_frac > 0 else 0
    if n_keep > 0:
        _, idx = torch.topk(res.flatten().abs(), n_keep)
        sp = torch.zeros_like(res.flatten())
        sp[idx] = res.flatten()[idx]
        res_sparse = sp.reshape_as(res)
    else:
        res_sparse = torch.zeros_like(res)
    recon = carrier + res_sparse
    wire = seq_len * k + hdim + n_keep * 2
    baseline = seq_len * hdim
    return recon.unsqueeze(0), {
        "pca_rank": k, "sparse_frac": sparse_frac, "seq_len": seq_len, "hidden_dim": hdim,
        "compression_ratio": baseline / max(wire, 1),
        "reconstruction_cosine": F.cosine_similarity(recon.flatten().unsqueeze(0), A.flatten().unsqueeze(0)).item(),
        "variance_explained": float((S[:k]**2).sum() / (S**2).sum()),
    }

def get_model_layers(model):
    """Get the decoder layers from various model architectures."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError(f"Unknown model architecture: {type(model)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HF_TOKEN")

    print(f"[{time.strftime('%H:%M:%S')}] Loading {args.model} on {device}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=hf_token, dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    ).eval()

    layers = get_model_layers(model)
    n_layers = len(layers)
    print(f"  loaded in {time.time()-t0:.1f}s; {n_layers} layers, dtype={dtype}", flush=True)

    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM used: {mem:.1f} GB", flush=True)

    splice_layers = [n_layers // 6, n_layers // 2, n_layers - n_layers // 6]
    pca_ranks = [2, 4, 8, 16, 32, 64]
    sparse_fracs = [0.0, 0.005, 0.01, 0.05, 0.10]
    prompts = PROMPTS[:args.n_prompts]

    total = len(prompts) * len(splice_layers) * len(pca_ranks) * len(sparse_fracs)
    print(f"  Grid: {len(prompts)}p × {len(splice_layers)}L × {len(pca_ranks)}r × {len(sparse_fracs)}s = {total}", flush=True)

    results = []
    for p_idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=args.max_length, add_special_tokens=True).input_ids
        if device == "cuda":
            input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]

        hidden, base_logits = get_baseline(model, input_ids)
        base_top1 = base_logits[:, -1, :].argmax(-1).item()
        base_top5 = set(torch.topk(base_logits[:, -1, :], 5).indices[0].tolist())

        for sl in splice_layers:
            act = hidden[sl + 1]
            for rank in pca_ranks:
                if rank >= seq_len:
                    continue
                for sf in sparse_fracs:
                    recon, stats = compress(act, rank, sf)
                    test_logits = forward_with_splice(model, input_ids, sl, recon, device)
                    ref_lp = F.log_softmax(base_logits[:, -1, :], dim=-1)
                    test_lp = F.log_softmax(test_logits[:, -1, :], dim=-1)
                    kl = F.kl_div(test_lp, ref_lp, reduction="batchmean", log_target=True).item()
                    t1 = test_logits[:, -1, :].argmax(-1).item()
                    t5 = set(torch.topk(test_logits[:, -1, :], 5).indices[0].tolist())
                    results.append({
                        "prompt_idx": p_idx, "actual_seq_len": seq_len,
                        "splice_layer": sl, **stats,
                        "kl_divergence": kl,
                        "top1_agree": int(t1 == base_top1),
                        "top5_overlap": len(t5 & base_top5),
                    })
        print(f"  [{time.strftime('%H:%M:%S')}] prompt {p_idx}/{len(prompts)} ({seq_len} tok)", flush=True)

    elapsed = time.time() - t0
    payload = {
        "meta": {"model_id": args.model, "n_layers": n_layers, "device": device,
                 "dtype": str(dtype), "n_prompts": len(prompts), "elapsed_sec": elapsed,
                 "splice_layers": splice_layers, "pca_ranks": pca_ranks, "sparse_fracs": sparse_fracs},
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"\nWrote {args.out} ({len(results)} records, {elapsed:.1f}s)", flush=True)

if __name__ == "__main__":
    main()
