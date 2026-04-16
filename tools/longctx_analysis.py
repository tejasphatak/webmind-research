"""
Analyze long-context validation results.

The critical question from Gemini's cross-verification:
Does the PCA rank required to capture 99% variance of transformer activations
scale with sequence length, or saturate at a bounded value?

If saturates → P1's "effective dim ~16-64" claim is REAL, not a seq_len artifact.
If scales linearly → P1's claim must be rewritten; compression ratios interpretable
  only per-prompt-length, not as a universal property.

Usage: python longctx_analysis.py --files findings/longctx_*.json
"""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def analyze(fp: str):
    """Returns per-seq_len summary: rank for 99% variance, etc."""
    with open(fp) as f: data = json.load(f)
    meta = data['meta']
    results = data['results']
    model = meta['model_id']

    # Group by seq_len × splice_layer
    by_len_layer = defaultdict(list)
    for r in results:
        by_len_layer[(r['seq_len'], r['splice_layer'])].append(r)

    # For each seq_len, compute the rank needed to hit each variance threshold
    summary = []
    for (sl, layer), rs in sorted(by_len_layer.items()):
        ranks_99 = [r['ranks_for_variance']['0.99'] for r in rs]
        ranks_999 = [r['ranks_for_variance']['0.999'] for r in rs]
        ranks_95 = [r['ranks_for_variance']['0.95'] for r in rs]
        ranks_90 = [r['ranks_for_variance']['0.90'] for r in rs]
        bounds = [r['total_rank_bound'] for r in rs]
        summary.append({
            "seq_len": sl, "splice_layer": layer, "n_prompts": len(rs),
            "rank_90_mean": float(np.mean(ranks_90)),
            "rank_95_mean": float(np.mean(ranks_95)),
            "rank_99_mean": float(np.mean(ranks_99)),
            "rank_999_mean": float(np.mean(ranks_999)),
            "bound_mean": float(np.mean(bounds)),
            "rank_99_frac_of_bound": float(np.mean(ranks_99) / np.mean(bounds)),
        })
    return model, summary


def plot_scaling(summaries: dict[str, list[dict]], out_path: str):
    """Plot rank-for-99% vs seq_len for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, summary in summaries.items():
        # Average across splice layers per seq_len
        by_sl = defaultdict(list)
        for s in summary:
            by_sl[s['seq_len']].append(s['rank_99_mean'])
        seq_lens = sorted(by_sl)
        means = [np.mean(by_sl[sl]) for sl in seq_lens]
        ax.plot(seq_lens, means, 'o-', label=f"{model.split('/')[-1]} (rank for 99% var)", lw=2, markersize=8)

    # Reference lines
    ax.plot([256, 512, 1024, 2048], [256, 512, 1024, 2048], 'k--', lw=1, alpha=0.5, label="y = seq_len (linear scaling = NO low-rank)")
    ax.axhline(32, color='green', ls=':', alpha=0.5, label="rank=32 (P1's implied bound)")
    ax.axhline(64, color='orange', ls=':', alpha=0.5, label="rank=64")
    ax.axhline(128, color='red', ls=':', alpha=0.5, label="rank=128")

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("PCA rank for 99% variance")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Long-context validation: does effective rank saturate or scale?", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def verdict(summaries: dict[str, list[dict]]) -> str:
    """Generate the honest verdict: saturates or scales?"""
    lines = []
    lines.append("=" * 70)
    lines.append("LONG-CONTEXT VALIDATION VERDICT")
    lines.append("=" * 70)
    for model, summary in summaries.items():
        lines.append(f"\n### {model}")
        lines.append(f"{'seq_len':>10} {'rank 99%':>12} {'bound':>10} {'rank/bound':>12}")
        by_sl = defaultdict(list)
        for s in summary:
            by_sl[s['seq_len']].append(s['rank_99_mean'])
        for sl in sorted(by_sl):
            r99 = np.mean(by_sl[sl])
            lines.append(f"{sl:>10} {r99:>12.1f} {sl:>10d} {r99/sl*100:>11.1f}%")

        seq_lens = sorted(by_sl)
        means = [np.mean(by_sl[sl]) for sl in seq_lens]
        if len(seq_lens) >= 2:
            # Log-log slope
            log_sl = np.log(seq_lens)
            log_r = np.log(means)
            slope = np.polyfit(log_sl, log_r, 1)[0]
            lines.append(f"\n  log-log slope = {slope:.2f}")
            if slope < 0.3:
                lines.append(f"  ✓ VERDICT: rank SATURATES (slope < 0.3). P1 low-rank claim SURVIVES.")
            elif slope < 0.7:
                lines.append(f"  ~ VERDICT: rank grows sub-linearly. P1 claim partially holds (CR still grows with seq_len).")
            else:
                lines.append(f"  ✗ VERDICT: rank scales linearly-ish with seq_len. P1 low-rank claim DOES NOT HOLD universally.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out-plot", default="plots/longctx_scaling.png")
    ap.add_argument("--out-verdict", default="findings/longctx_verdict.md")
    args = ap.parse_args()

    summaries = {}
    for fp in args.files:
        model, summary = analyze(fp)
        summaries[model] = summary

    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    plot_scaling(summaries, args.out_plot)

    v = verdict(summaries)
    print(v)
    with open(args.out_verdict, "w") as f:
        f.write("# Long-Context Validation Verdict\n\n")
        f.write("Generated from: " + ", ".join(args.files) + "\n\n")
        f.write("```\n" + v + "\n```\n")
    print(f"\nSaved verdict to {args.out_verdict}")


if __name__ == "__main__":
    main()
