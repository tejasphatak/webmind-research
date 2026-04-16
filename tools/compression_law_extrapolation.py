"""
Compression Law Discovery + Long-Context Extrapolation

Follow-up to compression_pattern_ml.py. Goal:
1. Discover a physical equation for compression quality (not just a black-box model)
2. Use it to predict long-context behavior at seq_len=2048 BEFORE empirical data arrives
3. Report uncertainty honestly; we'll validate when long-context completes

Key insight from prior run: rank/seq_len dominates (0.44 importance). So the
natural normalized variable is u = rank / seq_len. At u >= 1, we hit the
SVD identity (trivial compression). At u << 1, real compression; KL should
depend on how close u is to the intrinsic rank.

Hypothesis: log(KL) ≈ A - B · log(u) for u in a meaningful range, where B
captures how fast KL decays as we add PCA components.

Author: Claude Opus 4.6
Date: 2026-04-16
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


MODELS = [
    ('gemma3_1b',   'Gemma 3 1B',   1536, 26),
    ('llama_8b',    'Llama 3.1 8B', 4096, 32),
    ('qwen_32b',    'Qwen 2.5 32B', 5120, 64),
    ('gemma4_31b',  'Gemma 4 31B',  5376, 60),
]


def load_all():
    records = []
    for key, name, hidden_dim, n_layers in MODELS:
        fp = Path(f'findings/multimodel_{key}.json')
        if not fp.exists(): continue
        with open(fp) as f: data = json.load(f)
        for r in data['results']:
            r['model_key'] = key
            r['model_name'] = name
            r['hidden_dim'] = hidden_dim
            r['n_layers'] = n_layers
            r['u'] = r['pca_rank'] / r['actual_seq_len']
            r['v'] = r['pca_rank'] / hidden_dim
            r['log_kl'] = math.log(max(r['kl_divergence'], 1e-12))
            records.append(r)
    return records


def fit_physical_law(records, sparse_frac=0.0):
    """
    Hypothesis: log(KL) = A + B · log(1 - u)  where u = rank/seq_len
    (as u → 1, KL → 0 because we're doing full-rank SVD)

    Also try: log(KL) = A + B · log(u) + C · log(hidden_dim)
    """
    rs = [r for r in records if r['sparse_frac'] == sparse_frac
          and r['kl_divergence'] >= 0
          and r['u'] < 0.99]  # avoid SVD-identity tail

    if len(rs) < 20:
        return None

    u = np.array([r['u'] for r in rs])
    hd = np.array([r['hidden_dim'] for r in rs])
    log_kl = np.array([r['log_kl'] for r in rs])

    # Model: log_kl = a + b * log(1 - u) + c * log(hd)
    X = np.column_stack([np.log(1 - u + 1e-6), np.log(hd)])
    X = np.column_stack([np.ones(len(X)), X])  # intercept
    coef, resid, rank, sv = np.linalg.lstsq(X, log_kl, rcond=None)
    a, b, c = coef
    pred = X @ coef
    r2 = r2_score(log_kl, pred)
    return {
        "a": float(a), "b": float(b), "c": float(c),
        "r2": float(r2), "n": len(rs),
        "equation": f"log(KL) = {a:.3f} + {b:.3f}·log(1-u) + {c:.3f}·log(hd)",
        "interpretation": (
            f"At u=rank/seq_len fixed, doubling hidden_dim changes KL by factor 2^{c:.2f}. "
            f"At fixed hd, KL scales as (1-u)^{b:.2f} — so as u → 1 (rank → seq_len), KL → 0."
        ),
    }


def gbr_predict_longctx(records, target_seq_lens=(256, 512, 1024, 2048), target_ranks=(8, 16, 32, 64, 128)):
    """Train GBR on short-context data and predict at long context. Unreliable but informative."""
    from sklearn.ensemble import GradientBoostingRegressor
    feats = ['pca_rank', 'sparse_frac', 'actual_seq_len', 'hidden_dim', 'splice_layer']
    valid = [r for r in records if r['kl_divergence'] >= 0]
    X = np.array([[r[f] for f in feats] for r in valid])
    y = np.array([r['log_kl'] for r in valid])
    gbr = GradientBoostingRegressor(n_estimators=500, max_depth=4, random_state=42)
    gbr.fit(X, y)

    predictions = {}
    for key, name, hd, nl in MODELS:
        for sl in target_seq_lens:
            for rk in target_ranks:
                if rk >= sl: continue
                x = np.array([[rk, 0.0, sl, hd, nl // 2]])
                log_kl_pred = gbr.predict(x)[0]
                predictions[f"{name}_seq{sl}_rank{rk}"] = {
                    "predicted_kl": float(math.exp(log_kl_pred)),
                    "predicted_log_kl": float(log_kl_pred),
                    "caveat": f"EXTRAPOLATION: trained on seq_len 17-30, predicting seq_len={sl}",
                }
    return predictions


def main():
    records = load_all()
    print(f"Loaded {len(records)} records\n")

    # Find the best physical-law fit
    print("=== PHYSICAL LAW FIT: log(KL) = a + b·log(1 - rank/seq_len) + c·log(hidden_dim) ===\n")
    for sf in [0.0, 0.005, 0.01, 0.05, 0.10]:
        fit = fit_physical_law(records, sparse_frac=sf)
        if fit:
            print(f"sparse_frac={sf}:")
            print(f"  {fit['equation']}")
            print(f"  R² = {fit['r2']:.4f}  (n={fit['n']})")
            print(f"  {fit['interpretation']}")
            print()

    # Sanity check: KL at common configurations
    print("=== PREDICTED KL AT LONG CONTEXT (GBR extrapolation) ===")
    print("CAVEAT: GBR trained on seq_len 17-30, extrapolating is UNRELIABLE.")
    print("Report bounds, not point estimates.\n")
    preds = gbr_predict_longctx(records)
    # Show a few representative ones
    for key, name, hd, nl in MODELS:
        for sl in [256, 512, 1024, 2048]:
            k = f"{name}_seq{sl}_rank16"
            if k in preds:
                print(f"  {name:<18} seq={sl:<5} rank=16: predicted KL = {preds[k]['predicted_kl']:.2e}")
        print()

    # Generate plot: empirical vs model prediction
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: the physical law with empirical data overlaid
    ax = axes[0]
    fit = fit_physical_law(records, sparse_frac=0.0)
    if fit:
        u_grid = np.linspace(0.01, 0.95, 100)
        for key, name, hd, nl in MODELS:
            rs = [r for r in records if r['model_key'] == key and r['sparse_frac'] == 0.0
                  and r['kl_divergence'] > 0 and r['u'] < 0.99]
            u_emp = [r['u'] for r in rs]
            kl_emp = [r['kl_divergence'] for r in rs]
            ax.scatter(u_emp, kl_emp, s=20, alpha=0.6, label=f"{name} empirical")
            # Law prediction
            log_kl_pred = fit['a'] + fit['b']*np.log(1 - u_grid) + fit['c']*np.log(hd)
            ax.plot(u_grid, np.exp(log_kl_pred), '--', alpha=0.5)
    ax.set_xlabel("u = rank / seq_len")
    ax.set_ylabel("KL divergence")
    ax.set_yscale('log')
    ax.set_title(f"Physical law fit: R²={fit['r2']:.3f}\n{fit['equation']}" if fit else "fit failed", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which='both')

    # Plot 2: predicted KL vs seq_len at fixed rank=16
    ax = axes[1]
    for key, name, hd, nl in MODELS:
        seqs = [256, 512, 1024, 2048]
        ranks = [16]
        kls = []
        for sl in seqs:
            k = f"{name}_seq{sl}_rank16"
            if k in preds:
                kls.append(preds[k]['predicted_kl'])
            else:
                kls.append(np.nan)
        ax.plot(seqs, kls, 'o-', label=name, lw=2, markersize=8)
    ax.set_xlabel("seq_len")
    ax.set_ylabel("Predicted KL (GBR, rank=16, sparse=0)")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title("GBR extrapolation to long context\n(unreliable outside training range)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig("plots/compression_law.png", dpi=150)
    print("\nSaved plots/compression_law.png")

    # Write predictions to JSON for later comparison
    with open("findings/compression_law_predictions.json", "w") as f:
        json.dump({
            "physical_law_fit_sparse0": fit_physical_law(records, 0.0),
            "gbr_predictions": preds,
            "caveats": [
                "GBR is trained on seq_len 17-30. Predictions at seq_len >= 256 are EXTRAPOLATION.",
                "Physical law fit has R² around 0.6 on training range — predictive but not exact.",
                "Validate against long-context empirical data (Qwen + Gemma4 running now)."
            ]
        }, f, indent=2)
    print("Saved findings/compression_law_predictions.json")


if __name__ == "__main__":
    main()
