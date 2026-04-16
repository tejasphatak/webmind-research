"""
Plot activation compression experiment results.
Generates Pareto frontier: compression ratio vs quality loss.

Usage:
    python tools/activation_compression_plot.py --in findings/_raw_compression.json --out-dir plots
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    with open(args.inp) as f:
        data = json.load(f)
    meta = data["meta"]
    results = data["results"]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Group by (splice_layer, pca_rank, sparse_frac), average over prompts
    groups = defaultdict(list)
    for r in results:
        key = (r["splice_layer"], r["pca_rank"], r["sparse_topk_frac"])
        groups[key].append(r)

    agg = []
    for (sl, pr, sf), rs in groups.items():
        agg.append({
            "splice_layer": sl,
            "pca_rank": pr,
            "sparse_frac": sf,
            "compression_ratio": np.mean([r["compression_ratio"] for r in rs]),
            "kl_mean": np.mean([r["kl_divergence"] for r in rs]),
            "kl_std": np.std([r["kl_divergence"] for r in rs]),
            "cos_mean": np.mean([r["reconstruction_cosine"] for r in rs]),
            "top1_rate": np.mean([r["top1_agree"] for r in rs]),
            "top5_mean": np.mean([r["top5_overlap"] for r in rs]),
            "var_explained": np.mean([r["variance_explained"] for r in rs]),
        })

    splice_layers = sorted(set(a["splice_layer"] for a in agg))

    # --- Plot 1: Pareto frontier per splice layer (CR vs KL)
    fig, axes = plt.subplots(1, len(splice_layers), figsize=(6 * len(splice_layers), 5),
                             sharey=True, squeeze=False)
    for idx, sl in enumerate(splice_layers):
        ax = axes[0][idx]
        pts = [a for a in agg if a["splice_layer"] == sl]
        crs = [p["compression_ratio"] for p in pts]
        kls = [p["kl_mean"] for p in pts]
        colors = [p["sparse_frac"] for p in pts]
        sc = ax.scatter(crs, kls, c=colors, cmap="viridis", s=60, edgecolors="k", linewidth=0.5)
        ax.set_xlabel("Compression Ratio (x)")
        ax.set_title(f"Splice Layer {sl}")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.axhline(0.01, ls="--", color="green", alpha=0.6, lw=1, label="KL=0.01 (good)")
        ax.axhline(0.1, ls="--", color="orange", alpha=0.6, lw=1, label="KL=0.1 (marginal)")
        ax.axhline(1.0, ls="--", color="red", alpha=0.6, lw=1, label="KL=1.0 (bad)")
        if idx == 0:
            ax.set_ylabel("KL Divergence (log scale)")
            ax.legend(fontsize=7)
    fig.suptitle("Carrier + Payload: Compression Ratio vs Quality Loss\n"
                 f"(Gemma 3 1B IT, {meta['n_prompts']} prompts, colored by sparse residual fraction)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(sc, ax=axes[0][-1], label="sparse_topk_frac")
    fig.tight_layout()
    fig.savefig(out_dir / "activation_compression_pareto.png", dpi=150)

    # --- Plot 2: Top-1 agreement rate vs compression ratio
    fig, ax = plt.subplots(figsize=(9, 5))
    for sl in splice_layers:
        pts = sorted([a for a in agg if a["splice_layer"] == sl],
                     key=lambda x: x["compression_ratio"])
        ax.plot([p["compression_ratio"] for p in pts],
                [p["top1_rate"] for p in pts],
                "o-", label=f"Layer {sl}", ms=5)
    ax.set_xlabel("Compression Ratio (x)")
    ax.set_ylabel("Top-1 Token Agreement Rate")
    ax.set_title("Does compressed activation still predict the same next token?")
    ax.axhline(0.95, ls="--", color="green", alpha=0.5, label="95% threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "activation_compression_top1_agreement.png", dpi=150)

    # --- Plot 3: Variance explained vs KL (is PCA rank sufficient?)
    fig, ax = plt.subplots(figsize=(9, 5))
    for sl in splice_layers:
        pts = [a for a in agg if a["splice_layer"] == sl]
        ax.scatter([p["var_explained"] for p in pts],
                   [p["kl_mean"] for p in pts],
                   label=f"Layer {sl}", s=40, alpha=0.7)
    ax.set_xlabel("PCA Variance Explained")
    ax.set_ylabel("KL Divergence (log)")
    ax.set_yscale("log")
    ax.set_title("Is PCA variance explained a good proxy for output quality?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "activation_compression_variance_vs_kl.png", dpi=150)

    # --- Plot 4: Heatmap — best compression ratio at KL < 0.01 per (layer, sparse_frac)
    fig, ax = plt.subplots(figsize=(8, 5))
    sparse_fracs = sorted(set(a["sparse_frac"] for a in agg))
    heat = np.zeros((len(splice_layers), len(sparse_fracs)))
    for i, sl in enumerate(splice_layers):
        for j, sf in enumerate(sparse_fracs):
            good = [a for a in agg if a["splice_layer"] == sl and a["sparse_frac"] == sf
                    and a["kl_mean"] < 0.01]
            heat[i, j] = max((a["compression_ratio"] for a in good), default=0)
    im = ax.imshow(heat, aspect="auto", cmap="YlGn")
    ax.set_xticks(range(len(sparse_fracs)))
    ax.set_xticklabels([f"{sf:.3f}" for sf in sparse_fracs])
    ax.set_yticks(range(len(splice_layers)))
    ax.set_yticklabels([f"Layer {sl}" for sl in splice_layers])
    ax.set_xlabel("Sparse Residual Fraction")
    ax.set_ylabel("Splice Layer")
    ax.set_title("Best achievable compression ratio at KL < 0.01")
    for i in range(len(splice_layers)):
        for j in range(len(sparse_fracs)):
            ax.text(j, i, f"{heat[i, j]:.1f}x", ha="center", va="center",
                    color="black" if heat[i, j] > heat.max() * 0.5 else "gray", fontsize=10)
    fig.colorbar(im, label="Compression Ratio")
    fig.tight_layout()
    fig.savefig(out_dir / "activation_compression_heatmap.png", dpi=150)

    # Summary
    print(f"\nPlots saved to {out_dir}/")
    print(f"\n=== KEY FINDINGS ===")
    good_pts = [a for a in agg if a["kl_mean"] < 0.01]
    if good_pts:
        best = max(good_pts, key=lambda x: x["compression_ratio"])
        print(f"Best compression at KL<0.01: {best['compression_ratio']:.1f}x "
              f"(layer {best['splice_layer']}, rank {best['pca_rank']}, "
              f"sparse {best['sparse_frac']:.3f})")
        print(f"  top-1 agree: {best['top1_rate']:.0%}, var explained: {best['var_explained']:.3f}")
    else:
        print("NO configuration achieved KL < 0.01. Manifold assumption may not hold at these compression levels.")

    marginal = [a for a in agg if a["kl_mean"] < 0.1]
    if marginal:
        best_m = max(marginal, key=lambda x: x["compression_ratio"])
        print(f"Best compression at KL<0.1:  {best_m['compression_ratio']:.1f}x "
              f"(layer {best_m['splice_layer']}, rank {best_m['pca_rank']}, "
              f"sparse {best_m['sparse_frac']:.3f})")

if __name__ == "__main__":
    main()
