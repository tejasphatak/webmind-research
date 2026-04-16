"""
Long-context validation for Carrier-Payload.

Addresses Gemini's cross-verification critique: short prompts (18-30 tokens)
make seq_len the bottleneck on matrix rank, so "rank 16 captures 99%" is a
tautology. This experiment tests at realistic context lengths (512, 1024, 2048)
to find the true intrinsic dimensionality.

Key test: does the PCA rank required to capture 99% variance scale with
seq_len, or saturate at some small number regardless of seq_len?

- If rank saturates (e.g., 64 suffices at 2048 tokens) → genuine low-rank structure
- If rank scales with seq_len (e.g., needs 512 at 2048 tokens) → no low-rank structure

Usage:
    python long_context_validation.py --model <id> --out <path> [--seq-lens 512 1024 2048]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        return model.model.language_model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError(f"Unknown model architecture: {type(model)}")


# A long corpus of diverse text — scientific, narrative, code-like, conversational
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

Pipeline parallelism is particularly interesting for decentralized inference because it requires
transmitting only the hidden states between devices, rather than the weights themselves. This means
the bandwidth requirement scales with the hidden dimension and sequence length, not the parameter
count. For a model with 5000-dimensional hidden states and 2000-token sequences in float16, each
pipeline stage transmits approximately 20 megabytes of activations. This might seem small, but at
typical inference throughputs of thousands of tokens per second, the aggregate bandwidth can saturate
gigabit network links.

The story of quantization in neural networks begins with an observation from the early 2010s: neural
network weights are typically over-parameterized. This means that the same function can be expressed
with lower-precision weights. Initial attempts at 8-bit quantization showed promise for some tasks
but struggled with large language models due to the presence of activation outliers. SmoothQuant and
LLM.int8() developed techniques to handle these outliers, either by pre-processing activations to
smooth their distributions or by using mixed precision where outliers are kept at higher precision.

The activation outlier phenomenon deserves closer examination. In large language models, a small
fraction of hidden dimensions carry disproportionately large values. These outliers are not noise,
but meaningful signal: they often correspond to specific linguistic features like punctuation,
special tokens, or positional information. Quantizing these dimensions with the same scale as the
rest of the activations destroys their information content. The SmoothQuant paper by Xiao et al.
2023 showed that you can migrate the challenge from activations to weights by applying a scaling
transformation, making both tractable to quantize.

Consider the physical implementation of a modern GPU. The Nvidia H100 has 80 billion transistors
organized into streaming multiprocessors, each containing CUDA cores, tensor cores, and register
files. The memory hierarchy spans multiple levels: register files at approximately 1 cycle latency,
shared memory at 10-20 cycles, L2 cache at hundreds of cycles, and HBM3 memory at over 1000 cycles.
For neural network inference, the bottleneck is typically HBM bandwidth, which has grown much more
slowly than compute throughput over the past decade. This phenomenon, known as the memory wall,
explains why sparsity and quantization have become essential techniques.

Let me describe how the attention mechanism works in technical detail. Given an input sequence of
tokens embedded as vectors, attention computes three projections: queries Q, keys K, and values V.
The attention output is a weighted combination of the values, where the weights are determined by
the similarity between queries and keys. Specifically, we compute Q times K transpose, divide by
the square root of the key dimension for numerical stability, apply a softmax to normalize into a
probability distribution, and then multiply by V to get the output. Multi-head attention performs
this computation in parallel across several subspaces, providing the model with multiple ways to
relate different positions.

The softmax operation is particularly important to understand because it has some unusual properties.
It is shift-invariant: adding a constant to all logits doesn't change the output probabilities. It
saturates: very large logits produce outputs close to one, very small logits produce outputs close
to zero, and this saturation can cause gradient flow problems during training. The temperature of
the softmax can be controlled by a scalar multiplier: higher temperatures produce more uniform
distributions, lower temperatures produce sharper peaks. Inference sampling typically uses
temperature around 0.7 to 1.0 for a balance of coherence and diversity.

Now consider the challenge of distributed inference on volunteer devices. Unlike a datacenter where
all machines are identical, fast, and connected by high-speed interconnects, volunteer devices are
heterogeneous in every dimension: different GPU architectures, different memory capacities,
different operating systems, different network conditions, different reliability. A phone might
disconnect when the user answers a call. A laptop might throttle when the battery is low. A desktop
might have consistent uptime but slow upload bandwidth. Designing a system that handles this
gracefully is the central engineering challenge of projects like Petals and Synapse.

Historical attempts at volunteer computing include SETI@home for radio signal analysis, Folding@home
for protein structure prediction, and World Community Grid for various scientific applications.
These projects demonstrated that millions of volunteers could contribute compute to scientific
problems, but they also revealed important challenges: cheating detection, work unit design,
progressive backup, and the social engineering of maintaining user engagement over years or
decades. Distributed inference adds new complications because the computation must complete in
real-time to be useful for interactive applications like chatbots.

The question of how much information is actually contained in neural network activations has
received surprisingly little attention in the systems literature. From a representation learning
perspective, the intrinsic dimensionality of activations has been studied (Aghajanyan et al. 2020
found that fine-tuning objectives can be solved in very low-dimensional subspaces), but the
implications for activation transport in distributed inference have not been systematically
explored. This is the gap that Carrier-Payload compression aims to address: if activations occupy
a low-dimensional manifold, we can transmit only the manifold coordinates plus a sparse correction.

Analog computing for neural networks has a long history, dating back to the Connection Machine of
the 1980s and earlier neuromorphic chips like Carver Mead's silicon retinas. The modern revival of
analog AI is driven by the energy efficiency of compute-in-memory architectures. In a conventional
digital processor, data must be moved from memory to the arithmetic units for each operation,
which dominates the energy cost. Analog crossbar arrays allow the memory and computation to happen
in the same location: input voltages applied to rows, weights stored as conductances in the cells,
output currents summing automatically via Kirchhoff's current law. The theoretical energy
advantage can be 100x or more, though practical implementations have struggled with device
variability and the analog-digital conversion overhead at boundaries.

"""  # The corpus is long; tokenized to 2000+ tokens


def build_long_prompts(tokenizer, n_prompts: int, seq_lens: list[int], seed: int = 0xC0FFEE) -> dict[int, torch.Tensor]:
    """Returns dict of seq_len -> (n_prompts, seq_len) prompt tensors, deterministic."""
    rng = np.random.default_rng(seed)
    all_tokens = tokenizer(LONG_CORPUS, return_tensors="pt", add_special_tokens=False).input_ids[0]
    total = all_tokens.numel()
    prompts_by_len = {}
    for L in seq_lens:
        if L >= total:
            print(f"  WARN: seq_len {L} > corpus {total}, truncating")
            L = total - 1
        prompts = []
        for _ in range(n_prompts):
            start = int(rng.integers(0, max(1, total - L - 1)))
            prompts.append(all_tokens[start:start + L].clone())
        prompts_by_len[L] = torch.stack(prompts, dim=0)
    return prompts_by_len


@torch.no_grad()
def measure_effective_rank(model, input_ids: torch.Tensor, splice_layer: int,
                           device: str, variance_thresholds: list[float] = None):
    """
    Measure TRUE intrinsic dimensionality at a splice layer for long sequences.
    Returns: how many PCA components needed to capture each variance threshold.
    """
    if variance_thresholds is None:
        variance_thresholds = [0.90, 0.95, 0.99, 0.999]
    out = model(input_ids, output_hidden_states=True)
    act = out.hidden_states[splice_layer + 1].float().squeeze(0).cpu()  # (seq, hidden)
    seq_len, hidden_dim = act.shape

    mean = act.mean(dim=0, keepdim=True)
    A_centered = act - mean
    _, S, _ = torch.linalg.svd(A_centered, full_matrices=False)

    # Cumulative variance explained
    var = S ** 2
    cum_var = torch.cumsum(var, dim=0) / var.sum()
    ranks_needed = {}
    for t in variance_thresholds:
        rank = int((cum_var < t).sum().item() + 1)
        ranks_needed[t] = rank

    return {
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "total_rank_bound": min(seq_len, hidden_dim),
        "ranks_for_variance": ranks_needed,
        "singular_values_top32": S[:32].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HF_TOKEN")

    print(f"[{time.strftime('%H:%M:%S')}] Loading {args.model}...", flush=True)
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
    n_layers = len(get_model_layers(model))
    print(f"  loaded in {time.time()-t0:.1f}s; {n_layers} layers, dtype={dtype}", flush=True)

    prompts_by_len = build_long_prompts(tokenizer, args.n_prompts, args.seq_lens)
    splice_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

    results = []
    for seq_len, prompts in prompts_by_len.items():
        print(f"\n--- seq_len = {seq_len} ({prompts.shape[0]} prompts) ---", flush=True)
        for p_idx in range(prompts.shape[0]):
            input_ids = prompts[p_idx:p_idx+1].to(device)
            for sl in splice_layers:
                t_s = time.time()
                metrics = measure_effective_rank(model, input_ids, sl, device)
                results.append({
                    "prompt_idx": p_idx,
                    "splice_layer": sl,
                    **metrics,
                })
                if p_idx == 0:
                    print(f"  splice_layer={sl}  seq_len={metrics['seq_len']}  "
                          f"ranks for var: "
                          f"90%={metrics['ranks_for_variance'][0.90]}  "
                          f"95%={metrics['ranks_for_variance'][0.95]}  "
                          f"99%={metrics['ranks_for_variance'][0.99]}  "
                          f"99.9%={metrics['ranks_for_variance'][0.999]}  "
                          f"bound={metrics['total_rank_bound']}  "
                          f"[{time.time()-t_s:.1f}s]", flush=True)

    payload = {
        "meta": {
            "model_id": args.model, "n_layers": n_layers,
            "n_prompts": args.n_prompts, "seq_lens": args.seq_lens,
            "splice_layers": splice_layers, "device": device,
            "dtype": str(dtype), "elapsed_sec": time.time() - t0,
        },
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f: json.dump(payload, f)
    print(f"\nWrote {args.out} ({len(results)} records, {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
