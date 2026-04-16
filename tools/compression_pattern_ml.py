"""
Pattern discovery in carrier-payload compression data.

Question: is there a universal functional form f(model, rank, sparse, seq_len)
that predicts compression quality (KL, top-1 agreement)?

Approach:
1. Pool all 3840 records across 4 models (Gemma3-1B, Llama-8B, Qwen-32B, Gemma4-31B)
2. Fit various functional forms: power law, exponential, saturating curves
3. Run gradient-boosted trees (feature importance) + SHAP
4. Check for universality: does one equation fit ALL models with model-specific parameters?

Author: Claude Opus 4.6
Date: 2026-04-16
"""
from __future__ import annotations
import json
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
    """Pool all multimodel records with model metadata."""
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
            # Derived features
            r['rank_over_hd'] = r['pca_rank'] / hidden_dim
            r['rank_over_seq'] = r['pca_rank'] / r['actual_seq_len']
            r['layer_rel'] = r['splice_layer'] / n_layers
            r['log_compression'] = np.log(max(r['compression_ratio'], 1e-6))
            r['log_kl'] = np.log(max(r['kl_divergence'], 1e-12))
            r['one_minus_var'] = 1.0 - r['variance_explained']
            records.append(r)
    return records


def fit_power_law(x, y, name):
    """Fit y = a * x^b."""
    def f(x, a, b): return a * np.power(x, b)
    pos_mask = (x > 0) & (y > 0)
    x_pos = x[pos_mask]; y_pos = y[pos_mask]
    if len(x_pos) < 5: return None, None, None
    try:
        (a, b), _ = curve_fit(f, x_pos, y_pos, p0=[1, 1], maxfev=5000)
        y_pred = f(x_pos, a, b)
        r2 = r2_score(np.log(y_pos), np.log(y_pred))
        return a, b, r2
    except Exception as e:
        return None, None, None


def fit_exponential_decay(x, y):
    """Fit y = a * exp(-b*x) + c."""
    def f(x, a, b, c): return a * np.exp(-b * x) + c
    try:
        popt, _ = curve_fit(f, x, y, p0=[1, 1, 0], maxfev=5000)
        y_pred = f(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt, r2
    except Exception:
        return None, None


def analyze():
    records = load_all()
    print(f"Loaded {len(records)} records across {len(MODELS)} models\n")

    # --- Analysis 1: KL divergence vs one-minus-variance-explained ---
    # Hypothesis: KL ~ (1 - var_explained)^α
    x = np.array([r['one_minus_var'] for r in records])
    y = np.array([max(r['kl_divergence'], 1e-12) for r in records])
    a, b, r2 = fit_power_law(x, y, "KL vs (1-var)")
    print("=== Hypothesis 1: KL ~ (1 - var_explained)^α ===")
    if a is not None:
        print(f"  KL ≈ {a:.4f} · (1-var)^{b:.3f}   [R² on log-log: {r2:.3f}]")
    else:
        print("  Fit failed")
    print()

    # --- Analysis 2: KL vs rank/hidden_dim across models ---
    print("=== Hypothesis 2: KL ~ f(rank / hidden_dim) — universal across models? ===")
    for key, name, hd, nl in MODELS:
        rs = [r for r in records if r['model_key'] == key and r['sparse_frac'] == 0.0]
        if len(rs) < 10: continue
        x = np.array([r['rank_over_hd'] for r in rs])
        y = np.array([max(r['kl_divergence'], 1e-12) for r in rs])
        a, b, r2 = fit_power_law(x, y, f"{name} KL vs rank/hd")
        if a is not None:
            print(f"  {name:<18} KL ≈ {a:.2e} · (rank/hd)^{b:.3f}  [R²: {r2:.3f}]")

    print()

    # --- Analysis 3: Universal collapse? Plot KL / KL_ref vs rank/hd ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#4285F4', '#1877F2', '#FF6A00', '#0F9D58']

    # Plot 1: KL vs rank, per model
    ax = axes[0]
    for (key, name, hd, nl), color in zip(MODELS, colors):
        rs = [r for r in records if r['model_key'] == key and r['sparse_frac'] == 0.0]
        ranks = sorted(set(r['pca_rank'] for r in rs))
        kl_at_rank = []
        for rk in ranks:
            kls = [r['kl_divergence'] for r in rs if r['pca_rank'] == rk and r['kl_divergence'] >= 0]
            if kls:
                kl_at_rank.append(np.mean(kls))
            else:
                kl_at_rank.append(np.nan)
        ax.plot(ranks, kl_at_rank, 'o-', color=color, label=f"{name} (hd={hd})", lw=2, markersize=7)
    ax.set_xlabel("PCA rank"); ax.set_ylabel("Mean KL divergence")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title("KL vs rank, per model"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    # Plot 2: KL vs (rank/hidden_dim) — universality test
    ax = axes[1]
    for (key, name, hd, nl), color in zip(MODELS, colors):
        rs = [r for r in records if r['model_key'] == key and r['sparse_frac'] == 0.0]
        ranks = sorted(set(r['pca_rank'] for r in rs))
        for rk in ranks:
            kls = [r['kl_divergence'] for r in rs if r['pca_rank'] == rk and r['kl_divergence'] >= 0]
            if kls:
                ax.scatter(rk / hd, np.mean(kls), color=color, s=40, alpha=0.7,
                          label=name if rk == ranks[0] else None)
    ax.set_xlabel("rank / hidden_dim"); ax.set_ylabel("Mean KL divergence")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title("Universality test: KL vs normalized rank"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig("plots/compression_pattern.png", dpi=150)
    print(f"Saved plots/compression_pattern.png")

    # --- Analysis 4: Gradient boosting — predict log_kl from features ---
    print("\n=== Gradient Boosted Trees: predict log(KL) from features ===")
    valid = [r for r in records if r['kl_divergence'] >= 0]
    feats = ['pca_rank', 'sparse_frac', 'actual_seq_len', 'hidden_dim', 'n_layers',
             'splice_layer', 'rank_over_hd', 'rank_over_seq', 'layer_rel',
             'variance_explained']
    X = np.array([[r[f] for f in feats] for r in valid])
    y = np.array([r['log_kl'] for r in valid])

    # Train/test split (random 80/20)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(valid))
    cut = int(0.8 * len(valid))
    train_idx, test_idx = idx[:cut], idx[cut:]
    gbr = GradientBoostingRegressor(n_estimators=500, max_depth=4, random_state=42)
    gbr.fit(X[train_idx], y[train_idx])
    y_pred = gbr.predict(X[test_idx])
    r2 = r2_score(y[test_idx], y_pred)
    print(f"  Test R² on log(KL): {r2:.4f}")
    print(f"  Feature importances:")
    imp = sorted(zip(feats, gbr.feature_importances_), key=lambda x: -x[1])
    for f, i in imp:
        print(f"    {f:<25} {i:.3f}")

    # --- Analysis 5: Saturating curve for variance explained ---
    print("\n=== Variance explained vs rank: per-model power law fit ===")
    for key, name, hd, nl in MODELS:
        rs = [r for r in records if r['model_key'] == key and r['sparse_frac'] == 0.0]
        # var_explained at each rank (averaged)
        by_rank = {}
        for r in rs:
            by_rank.setdefault(r['pca_rank'], []).append(r['variance_explained'])
        ranks = sorted(by_rank); means = [np.mean(by_rank[r]) for r in ranks]
        # Fit (1 - var) = a * rank^(-b)
        x = np.array(ranks, dtype=float); y = np.array([1 - v for v in means])
        a, b, r2 = fit_power_law(x, y, name)
        if a is not None and b is not None:
            print(f"  {name:<18} (1-var_expl) ≈ {a:.3f} · rank^{-b:.3f}   [R²: {r2:.3f}]")

    # Return the discovered patterns as a summary
    return {
        "n_records": len(records),
        "gbr_r2_log_kl": r2,
        "feature_importances": dict(imp),
    }


if __name__ == "__main__":
    summary = analyze()
    print("\n=== SUMMARY ===")
    import json
    print(json.dumps(summary, indent=2))
