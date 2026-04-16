"""
Plot synapse fidelity sim results.

Usage:
    python tools/synapse_fidelity_plot.py --in findings/_raw_fidelity_<ts>.json --out-dir plots
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    with open(args.inp) as f:
        data = json.load(f)
    meta = data["meta"]
    results = data["results"]
    regimes = [r for r in meta["regimes"] if r != "uniform_fp32"]
    n_layers = meta["n_layers"]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: mean layer-wise cosine similarity per regime
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for regime in regimes:
        rs = [r for r in results if r["regime"] == regime]
        if not rs: continue
        cos = np.array([[lm["cosine"] for lm in r["layers"]] for r in rs])
        mean = cos.mean(axis=0); std = cos.std(axis=0)
        ax.plot(range(n_layers), mean, label=regime, lw=2)
        ax.fill_between(range(n_layers), mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("transformer block index")
    ax.set_ylabel("cosine similarity vs fp32 reference")
    ax.set_title(f"Synapse fidelity — layer-wise cosine sim across precision regimes "
                 f"(Gemma 3 1B IT, n={len(results) // len(regimes)} prompts)")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "synapse_fidelity_cosine_per_layer.png", dpi=150)

    # --- Plot 2: relative L2 per layer
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for regime in regimes:
        rs = [r for r in results if r["regime"] == regime]
        if not rs: continue
        rl2 = np.array([[lm["rel_l2"] for lm in r["layers"]] for r in rs])
        mean = rl2.mean(axis=0)
        ax.semilogy(range(n_layers), mean, label=regime, lw=2)
    ax.set_xlabel("transformer block index")
    ax.set_ylabel("relative L2 drift vs fp32 (log scale)")
    ax.set_title("Synapse fidelity — relative L2 drift layer-by-layer")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out_dir / "synapse_fidelity_rel_l2_per_layer.png", dpi=150)

    # --- Plot 3: H2 test — heterogeneous drift vs max(uniform components) at final layer
    uniform_final = {}
    for reg in ["uniform_fp16", "uniform_bf16"]:
        rs = [r for r in results if r["regime"] == reg]
        if rs:
            uniform_final[reg] = np.array([r["layers"][-1]["rel_l2"] for r in rs]).mean()
    hetero_final = {}
    for reg in regimes:
        if reg.startswith("hetero"):
            rs = [r for r in results if r["regime"] == reg]
            if rs:
                hetero_final[reg] = np.array([r["layers"][-1]["rel_l2"] for r in rs]).mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(uniform_final.keys()) + list(hetero_final.keys())
    vals = list(uniform_final.values()) + list(hetero_final.values())
    colors = ["#4a8" if l.startswith("uniform") else "#c55" for l in labels]
    ax.bar(labels, vals, color=colors)
    max_uni = max(uniform_final.values()) if uniform_final else 0
    if max_uni > 0:
        ax.axhline(1.5 * max_uni, ls="--", color="k", lw=1, label="1.5× max(uniform) threshold")
        ax.legend()
    ax.set_ylabel("mean rel L2 drift at final transformer block")
    ax.set_title("H2 test — does heterogeneous precision amplify drift super-linearly?")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out_dir / "synapse_fidelity_H2_heterogeneity.png", dpi=150)

    # --- Plot 4: final-logit agreement rate per regime
    fig, ax = plt.subplots(figsize=(9, 5))
    agree_rates = {}
    overlap_means = {}
    kl_means = {}
    for reg in regimes:
        rs = [r for r in results if r["regime"] == reg]
        if not rs: continue
        agree_rates[reg] = np.mean([r["top1_agree"] for r in rs])
        overlap_means[reg] = np.mean([r["top5_overlap"] for r in rs])
        kl_means[reg] = np.mean([r["final_kl"] for r in rs])
    x = np.arange(len(agree_rates))
    w = 0.35
    ax.bar(x - w/2, list(agree_rates.values()), w, label="top-1 agreement rate")
    ax.bar(x + w/2, [v/5.0 for v in overlap_means.values()], w, label="top-5 overlap / 5")
    ax.set_xticks(x); ax.set_xticklabels(list(agree_rates.keys()), rotation=30, ha="right")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Next-token agreement with fp32 reference, per regime")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "synapse_fidelity_token_agreement.png", dpi=150)

    # Write a compact summary JSON for the findings doc
    summary = {
        "n_prompts": len(results) // len(regimes),
        "n_layers": n_layers,
        "per_regime": {},
    }
    for reg in regimes:
        rs = [r for r in results if r["regime"] == reg]
        if not rs: continue
        final_cos = np.array([r["final_cosine"] for r in rs])
        final_kl = np.array([r["final_kl"] for r in rs])
        last_rl2 = np.array([r["layers"][-1]["rel_l2"] for r in rs])
        outlier_contrib_last = np.array([r["layers"][-1]["outlier_contribution"] for r in rs])
        summary["per_regime"][reg] = {
            "final_cos_mean": float(final_cos.mean()),
            "final_cos_min": float(final_cos.min()),
            "final_kl_mean": float(final_kl.mean()),
            "final_kl_max": float(final_kl.max()),
            "last_layer_rel_l2_mean": float(last_rl2.mean()),
            "top1_agreement": float(np.mean([r["top1_agree"] for r in rs])),
            "top5_overlap_mean": float(np.mean([r["top5_overlap"] for r in rs])),
            "outlier_contribution_last_layer_mean": float(outlier_contrib_last.mean()),
        }
    with open(out_dir / "synapse_fidelity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote plots + summary to {out_dir}/")

if __name__ == "__main__":
    main()
