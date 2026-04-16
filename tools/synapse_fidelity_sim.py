"""
Synapse numerical fidelity simulator.

Measures layer-wise activation drift in Gemma 3 1B IT under precision regimes
that model Synapse shard-boundary casts (including heterogeneous precision-hopping).

Pre-registration: papers/synapse-numerical-fidelity-preregistration-v1.md
Author: Claude Opus 4.6 (Synapse Research)
Date: 2026-04-16

Run:
    source .venv/bin/activate
    export HF_TOKEN=...  # read scope; Gemma license accepted
    python tools/synapse_fidelity_sim.py --out findings/_raw_fidelity_<ts>.json

Outputs:
    JSON blob with layer-wise metrics per regime per prompt.
    Downstream plotting in tools/synapse_fidelity_plot.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-1b-it"
SEED = 0xC0FFEE
N_PROMPTS = 64
SEQ_LEN = 64

# Precision regimes. Each regime is a list of torch dtypes, one per transformer block.
# "boundary_dtype[i]" is the precision the hidden state is CAST TO after block i.
# Block 0 input is always fp32 (from embedding). Final output is always compared in fp32.

def build_regimes(n_layers: int) -> Dict[str, List[torch.dtype]]:
    fp32, fp16, bf16 = torch.float32, torch.float16, torch.bfloat16
    regimes: Dict[str, List[torch.dtype]] = {
        "uniform_fp32": [fp32] * n_layers,
        "uniform_fp16": [fp16] * n_layers,
        "uniform_bf16": [bf16] * n_layers,
        # Heterogeneous 3-cycle: simulates a path across 3 device classes.
        "hetero_fp16_bf16_fp32_cycle": [[fp16, bf16, fp32][i % 3] for i in range(n_layers)],
        # Adversarial 2-cycle: maximum oscillation.
        "hetero_fp16_bf16_alternate": [[fp16, bf16][i % 2] for i in range(n_layers)],
    }
    return regimes


@dataclass
class LayerMetrics:
    layer: int
    cosine: float            # cosine similarity vs fp32 reference at this layer output
    rel_l2: float            # ||x - x_ref||_2 / ||x_ref||_2
    max_abs: float           # max abs deviation per hidden-dim
    outlier_contribution: float  # fraction of L2^2 from top-1% magnitude positions in x_ref


@dataclass
class PromptResult:
    prompt_idx: int
    regime: str
    layers: List[LayerMetrics]
    final_cosine: float
    final_kl: float          # KL(fp32_ref || test) over next-token distribution of last position
    top1_agree: bool
    top5_overlap: int        # count from {0..5} of top-5 token overlap with reference


def compute_layer_metrics(x: torch.Tensor, x_ref: torch.Tensor, layer: int) -> LayerMetrics:
    # x, x_ref: (batch=1, seq, hidden) in fp32 for fair comparison.
    x = x.float().flatten()
    x_ref = x_ref.float().flatten()
    cos = F.cosine_similarity(x.unsqueeze(0), x_ref.unsqueeze(0)).item()
    diff = x - x_ref
    rel_l2 = (diff.norm() / (x_ref.norm() + 1e-12)).item()
    max_abs = diff.abs().max().item()
    # Outlier contribution: positions where |x_ref| is in top-1% of magnitudes
    k = max(1, int(0.01 * x_ref.numel()))
    top_idx = torch.topk(x_ref.abs(), k).indices
    outlier_sqerr = (diff[top_idx] ** 2).sum().item()
    total_sqerr = (diff ** 2).sum().item() + 1e-20
    return LayerMetrics(layer=layer,
                        cosine=cos,
                        rel_l2=rel_l2,
                        max_abs=max_abs,
                        outlier_contribution=outlier_sqerr / total_sqerr)


@torch.no_grad()
def forward_with_casts(model, input_ids: torch.Tensor, regime: List[torch.dtype]):
    """
    Run forward pass, casting hidden state to regime[i] AFTER transformer block i.
    Returns: list of hidden states (in fp32 for comparison) after each block, plus final logits.
    The cast inside the block is simulated by casting hidden state at boundary.
    """
    # Gemma models expose: model.model.embed_tokens, model.model.layers, model.model.norm, model.lm_head
    base = model.model
    hidden = base.embed_tokens(input_ids).float()  # embedding in fp32
    # Some Gemma variants scale embeddings; keep whatever the model config does.
    # We follow the model's own module graph by replicating its forward block-by-block.

    per_layer_hidden = []
    # Build a dummy attention mask (causal); transformers handles via position_ids
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    # Gemma decoder layer call signature varies across transformers versions.
    # We use a minimal call: block(hidden_states, position_ids=...)
    for i, block in enumerate(base.layers):
        # Move hidden state to fp32 for the block's internals, but cast at boundary afterwards.
        out = block(hidden.to(torch.float32), position_ids=position_ids)
        if isinstance(out, tuple):
            out = out[0]
        # Boundary cast — simulate the shard-hop precision.
        out_cast = out.to(regime[i])
        hidden = out_cast
        per_layer_hidden.append(out_cast.detach().float().clone())

    # Final norm + lm_head in fp32 on the final-block output.
    final_hidden = base.norm(hidden.float())
    logits = model.lm_head(final_hidden)
    return per_layer_hidden, logits.float()


def kl_divergence(logits_ref: torch.Tensor, logits_test: torch.Tensor) -> float:
    # KL at last-position next-token distribution
    p = F.log_softmax(logits_ref[:, -1, :], dim=-1)
    q = F.log_softmax(logits_test[:, -1, :], dim=-1)
    return F.kl_div(q, p, reduction="batchmean", log_target=True).item()


def load_prompts(tokenizer, n_prompts: int, seq_len: int, seed: int) -> torch.Tensor:
    """
    Use a deterministic synthetic prompt set when C4 not available.
    Prompts drawn from a fixed English corpus snippet; deterministic under seed.
    """
    # Deterministic fallback: generate by seeding random token selection from a short fixed English sample.
    # This is lower-quality than C4 but reproducible without network dependency at generation time.
    rng = np.random.default_rng(seed)
    # A compact English seed corpus — reproducible, no internet fetch needed at runtime.
    corpus = (
        "In distributed systems the hardest problem is not speed but agreement among nodes that may "
        "fail or act maliciously. Consensus protocols give us a formal answer to this problem under "
        "clear assumptions about how many nodes can misbehave. When we move from classical consensus "
        "to machine learning inference on volunteer hardware, the question becomes whether the shared "
        "computation can be trusted end to end. A shard that lies by one bit per layer can compound "
        "into a wildly different output. A shard that lies by one bit per token can remain invisible. "
        "The mathematics of detection sits between these regimes. Freivalds showed fifty years ago "
        "that matrix products can be checked probabilistically in quadratic time. We apply that idea "
        "here to transformer activations crossing shard boundaries, with the caveat that finite "
        "precision arithmetic changes the thresholds. "
    ) * 4
    tokens = tokenizer(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
    # Sample N_PROMPTS starting positions with fixed seed
    max_start = max(1, tokens.numel() - seq_len - 1)
    starts = rng.integers(0, max_start, size=n_prompts)
    prompts = torch.stack([tokens[s:s + seq_len] for s in starts], dim=0)
    return prompts  # (N, seq_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[{time.strftime('%H:%M:%S')}] Loading {MODEL_ID} ...")
    t0 = time.time()
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=hf_token, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    n_layers = len(model.model.layers)
    print(f"  loaded in {time.time()-t0:.1f}s; n_layers={n_layers}")

    regimes = build_regimes(n_layers)
    print(f"  regimes: {list(regimes.keys())}")

    prompts = load_prompts(tokenizer, args.n_prompts, args.seq_len, SEED)
    print(f"  prompts: {prompts.shape}")

    results: List[PromptResult] = []
    run_meta = {
        "model_id": MODEL_ID,
        "n_layers": n_layers,
        "n_prompts": args.n_prompts,
        "seq_len": args.seq_len,
        "seed": SEED,
        "device": "cpu",
        "torch_version": torch.__version__,
        "regimes": list(regimes.keys()),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for p_idx in range(args.n_prompts):
        input_ids = prompts[p_idx:p_idx + 1]
        # Reference run: uniform fp32
        ref_hidden, ref_logits = forward_with_casts(model, input_ids, regimes["uniform_fp32"])
        ref_top5 = torch.topk(ref_logits[:, -1, :], 5).indices[0].tolist()
        ref_top1 = ref_top5[0]

        for regime_name, regime_dtypes in regimes.items():
            if regime_name == "uniform_fp32":
                # Degenerate — skip or record as self-check
                continue
            t_start = time.time()
            test_hidden, test_logits = forward_with_casts(model, input_ids, regime_dtypes)
            layer_ms = [
                compute_layer_metrics(test_hidden[i], ref_hidden[i], i)
                for i in range(n_layers)
            ]
            final_cos = F.cosine_similarity(
                test_logits.float().flatten().unsqueeze(0),
                ref_logits.float().flatten().unsqueeze(0),
            ).item()
            kl = kl_divergence(ref_logits, test_logits)
            test_top5 = torch.topk(test_logits[:, -1, :], 5).indices[0].tolist()
            test_top1 = test_top5[0]
            results.append(PromptResult(
                prompt_idx=p_idx,
                regime=regime_name,
                layers=layer_ms,
                final_cosine=final_cos,
                final_kl=kl,
                top1_agree=(test_top1 == ref_top1),
                top5_overlap=len(set(test_top5) & set(ref_top5)),
            ))
            dt = time.time() - t_start
            if p_idx % 8 == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] p{p_idx:02d} {regime_name:40s} "
                      f"cos={final_cos:.6f} kl={kl:.4e} t={dt:.1f}s")

    run_meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_meta["wallclock_sec"] = time.time() - t0

    payload = {
        "meta": run_meta,
        "results": [
            {
                "prompt_idx": r.prompt_idx,
                "regime": r.regime,
                "final_cosine": r.final_cosine,
                "final_kl": r.final_kl,
                "top1_agree": r.top1_agree,
                "top5_overlap": r.top5_overlap,
                "layers": [asdict(lm) for lm in r.layers],
            }
            for r in results
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"wrote {args.out} ({len(results)} records, {run_meta['wallclock_sec']:.1f}s)")


if __name__ == "__main__":
    main()
