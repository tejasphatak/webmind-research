"""
Long-context validation adapted for vision-language (multimodal) models.

Differs from long_context_validation.py:
- Uses Qwen2_5_VLForConditionalGeneration (or AutoModel for VL) explicitly
- Navigates model.model.language_model.layers for VL models
- Skips vision encoder entirely (pure text prompts)

Author: Claude Opus 4.6
Date: 2026-04-16
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig

LONG_CORPUS = """
Distributed computing has evolved dramatically since the era of mainframes. The fundamental insight
that computation can be split across multiple independent machines opened pathways to systems like
MapReduce, Apache Spark, and modern serverless architectures. In these systems, the central challenge
is coordination: how do you ensure that computations distributed across hundreds or thousands of
machines produce consistent results despite failures, network partitions, and clock skew? The
answer, developed over decades, involves consensus protocols like Paxos and Raft, eventually
consistent replication, and careful reasoning about the CAP theorem's tradeoffs.

Large language models represent an interesting case study in distributed computing because they are
simultaneously embarrassingly parallel (each token position's computation can proceed in parallel
within a layer) and painfully sequential (each layer depends on the previous layer's output). The
attention mechanism introduces quadratic complexity in sequence length, which becomes the dominant
cost at long contexts. Researchers have proposed many solutions: sparse attention patterns, linear
attention approximations, flash attention kernels, sliding window attention, and most recently,
state space models like Mamba which replace attention entirely with a structured state space.

When we shard a transformer across multiple devices, we face a fundamental design choice: do we
split it along the model axis (tensor parallelism, each device holds a slice of each layer), the
layer axis (pipeline parallelism, each device holds a slice of layers), or the data axis (data
parallelism, each device processes different examples)? The optimal choice depends on hardware
characteristics, network bandwidth, and the model's architectural details. Modern frameworks like
Megatron-LM, DeepSpeed, and FairScale combine these strategies adaptively.
""" * 10  # repeat for longer corpus


def get_model_layers(model):
    # Multimodal: Gemma 4, Qwen-VL
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        return model.model.language_model.layers
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        return model.language_model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    raise ValueError(f"Unknown model: {type(model)}")


def load_model(model_id, hf_token):
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(model_id, token=hf_token)
    mt = cfg.model_type
    print(f"  model_type: {mt}", flush=True)
    # Try specific VL class first
    if mt == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    elif mt == "gemma4":
        from transformers import AutoModelForCausalLM as ModelCls
    elif mt == "mllama":
        from transformers import MllamaForConditionalGeneration as ModelCls
    else:
        # Try AutoModel (loads anything)
        ModelCls = AutoModel
    return ModelCls.from_pretrained(model_id, token=hf_token, dtype=torch.float16,
                                     device_map="auto", low_cpu_mem_usage=True)


def build_long_prompts(tokenizer, n_prompts, seq_lens, seed=0xC0FFEE):
    rng = np.random.default_rng(seed)
    all_tokens = tokenizer(LONG_CORPUS, return_tensors="pt", add_special_tokens=False).input_ids[0]
    total = all_tokens.numel()
    print(f"  corpus tokens available: {total}", flush=True)
    by_len = {}
    for L in seq_lens:
        if L >= total:
            L = total - 1
        starts = rng.integers(0, max(1, total - L - 1), size=n_prompts)
        by_len[L] = torch.stack([all_tokens[s:s+L].clone() for s in starts], dim=0)
    return by_len


@torch.no_grad()
def measure_effective_rank(model, input_ids, splice_layer, device):
    # Use forward hook to capture the output of the splice_layer transformer block
    layers = get_model_layers(model)
    captured = []
    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured.append(out.detach().float().cpu().clone())
    h = layers[splice_layer].register_forward_hook(hook_fn)
    try:
        _ = model(input_ids=input_ids)
    finally:
        h.remove()
    if not captured:
        raise RuntimeError("Hook didn't capture")
    act = captured[0].squeeze(0)  # (seq, hidden)
    seq_len, hidden_dim = act.shape
    mean = act.mean(dim=0, keepdim=True)
    A = act - mean
    _, S, _ = torch.linalg.svd(A, full_matrices=False)
    var = S ** 2
    cum = torch.cumsum(var, 0) / var.sum()
    ranks = {}
    for t in [0.9, 0.95, 0.99, 0.999]:
        ranks[str(t)] = int((cum < t).sum().item() + 1)
    return {
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "total_rank_bound": min(seq_len, hidden_dim),
        "ranks_for_variance": ranks,
        "singular_values_top32": S[:32].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HF_TOKEN")

    print(f"[{time.strftime('%H:%M:%S')}] Loading {args.model}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model, hf_token).eval()
    layers = get_model_layers(model)
    print(f"  loaded in {time.time()-t0:.1f}s; {len(layers)} layers", flush=True)

    prompts_by_len = build_long_prompts(tokenizer, args.n_prompts, args.seq_lens)
    n_layers = len(layers)
    splice_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

    results = []
    for sl, prompts in prompts_by_len.items():
        print(f"\n--- seq_len = {sl} ({prompts.shape[0]} prompts) ---", flush=True)
        for p_idx in range(prompts.shape[0]):
            input_ids = prompts[p_idx:p_idx+1].to(device)
            for layer_i in splice_layers:
                t_s = time.time()
                m = measure_effective_rank(model, input_ids, layer_i, device)
                results.append({"prompt_idx": p_idx, "splice_layer": layer_i, **m})
                if p_idx == 0:
                    r = m['ranks_for_variance']
                    print(f"  splice_layer={layer_i}  seq={m['seq_len']}  "
                          f"r90={r['0.9']} r95={r['0.95']} r99={r['0.99']} r999={r['0.999']}  "
                          f"bound={m['total_rank_bound']}  [{time.time()-t_s:.1f}s]", flush=True)

    payload = {
        "meta": {"model_id": args.model, "n_layers": n_layers, "n_prompts": args.n_prompts,
                 "seq_lens": args.seq_lens, "splice_layers": splice_layers, "device": device,
                 "dtype": "torch.float16", "elapsed_sec": time.time()-t0},
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f: json.dump(payload, f)
    print(f"\nWrote {args.out} ({len(results)} records, {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
