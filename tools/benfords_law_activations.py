"""
Benford's Law on LLM Activations — empirical pilot test.

Hypothesis: legitimate transformer activations follow Benford's Law
in their leading-digit distribution. Byzantine perturbations violate it
measurably, enabling a cheap per-shard integrity check.

This is a pilot test using the multimodel data we already collected
(Gemma 3 1B, Llama 3.1 8B, Qwen 2.5 32B, Gemma 4 31B). We compare:
1. Leading-digit distribution of real activations vs Benford prediction
2. Distribution after adversarial perturbations (random noise, bias,
   targeted flip) to confirm the check actually detects attacks

Author: Claude Opus 4.6
Date: 2026-04-16
Companion: findings/weird_faculties_synthesis_round1.md (origin of idea)

Note: The multimodel files don't retain raw activations, only
compression stats. So for this pilot test, we need to re-extract a
small set of activations. For initial feasibility demonstration,
we use synthetic activations with LLM-like statistics (heavy tails,
a few outliers, mostly-structured).
"""
from __future__ import annotations
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path


BENFORD_PROB = np.array([np.log10(1 + 1/d) for d in range(1, 10)])


def leading_digit(x: np.ndarray) -> np.ndarray:
    """Return the leading (most significant) digit of each non-zero element of x.
    Ignores sign; operates on magnitude."""
    absx = np.abs(x.flatten())
    nonzero = absx[absx > 0]
    if nonzero.size == 0:
        return np.array([], dtype=int)
    # Leading digit = first digit of string representation of |x| in scientific notation
    log10_x = np.log10(nonzero)
    mantissa = 10 ** (log10_x - np.floor(log10_x))  # 1 <= mantissa < 10
    return np.floor(mantissa).astype(int)  # 1..9


def benford_compliance(digits: np.ndarray) -> dict:
    """Compare leading-digit distribution of `digits` to Benford prediction.
    Returns: chi-square statistic, p-value, empirical probabilities, L1 distance."""
    if digits.size == 0:
        return {"chi2": np.nan, "p_value": np.nan, "empirical": np.zeros(9), "l1": np.nan, "n": 0}
    counts = np.array([np.sum(digits == d) for d in range(1, 10)], dtype=float)
    empirical = counts / counts.sum()
    expected_counts = BENFORD_PROB * counts.sum()
    # Chi-square goodness of fit vs Benford
    chi2_stat, p_val = stats.chisquare(counts, expected_counts)
    l1 = np.abs(empirical - BENFORD_PROB).sum()
    return {"chi2": chi2_stat, "p_value": p_val, "empirical": empirical,
            "l1": l1, "n": int(counts.sum())}


# ──────────────────────────────────────────────────────────────────────────
# Synthetic activation generator (LLM-like statistics)
# ──────────────────────────────────────────────────────────────────────────

def synth_llm_activations(n_tokens=128, hidden_dim=1536, outlier_frac=0.01,
                          outlier_mag=30.0, seed=0) -> np.ndarray:
    """Generate synthetic activations with LLM-like statistics:
    - mostly Gaussian-distributed with small std
    - small fraction of outliers with large magnitude (SmoothQuant / LLM.int8 style)
    - positive + negative values
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.0, (n_tokens, hidden_dim))
    n_outliers = int(outlier_frac * x.size)
    flat_idx = rng.choice(x.size, size=n_outliers, replace=False)
    signs = rng.choice([-1, 1], size=n_outliers)
    x.flat[flat_idx] = signs * outlier_mag * (1 + rng.exponential(1, n_outliers))
    return x


# ──────────────────────────────────────────────────────────────────────────
# Adversarial perturbation generators
# ──────────────────────────────────────────────────────────────────────────

def adversary_random_noise(x: np.ndarray, strength: float = 1.0, seed=42) -> np.ndarray:
    """Add uniform random noise."""
    rng = np.random.default_rng(seed)
    return x + rng.uniform(-strength, strength, x.shape)


def adversary_scaled(x: np.ndarray, scale: float = 1.3) -> np.ndarray:
    """Scale all values (biased injection — simple multiplicative)."""
    return x * scale


def adversary_targeted_flip(x: np.ndarray, flip_frac: float = 0.01, seed=42) -> np.ndarray:
    """Flip signs of a fraction of entries (targeted)."""
    rng = np.random.default_rng(seed)
    mask = rng.random(x.shape) < flip_frac
    return np.where(mask, -x, x)


def adversary_round(x: np.ndarray, digits: int = 2) -> np.ndarray:
    """Round to limited precision (quantization-like lazy attacker)."""
    scale = 10 ** digits
    return np.round(x * scale) / scale


# ──────────────────────────────────────────────────────────────────────────
# Pilot experiment
# ──────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Benford's Law on Synthetic LLM Activations — pilot test")
    print("=" * 80)
    print(f"Benford predicted distribution: {BENFORD_PROB}")
    print()

    results = {}
    for seed in range(5):
        x = synth_llm_activations(seed=seed)
        scenarios = {
            "honest": x,
            "adv_random_noise_σ=0.5": adversary_random_noise(x, 0.5, seed + 100),
            "adv_random_noise_σ=2.0": adversary_random_noise(x, 2.0, seed + 100),
            "adv_scaled_1.3x": adversary_scaled(x, 1.3),
            "adv_targeted_flip_1%": adversary_targeted_flip(x, 0.01, seed + 200),
            "adv_targeted_flip_5%": adversary_targeted_flip(x, 0.05, seed + 200),
            "adv_round_2digits": adversary_round(x, 2),
        }
        for name, xp in scenarios.items():
            digits = leading_digit(xp)
            result = benford_compliance(digits)
            results.setdefault(name, []).append(result)

    # Aggregate
    print(f"{'Scenario':<30} {'mean_chi2':>12} {'mean_p':>10} {'mean_L1':>10} {'n':>10}")
    print("-" * 80)
    for name, rs in results.items():
        chi2 = np.mean([r["chi2"] for r in rs])
        p = np.mean([r["p_value"] for r in rs])
        l1 = np.mean([r["l1"] for r in rs])
        n = rs[0]["n"]
        flag = "  ← LEGITIMATE" if p > 0.05 else ""
        flag += "  ← DETECTED" if p < 0.01 else ""
        print(f"{name:<30} {chi2:12.2f} {p:10.4e} {l1:10.4f} {n:10d}{flag}")

    print()
    print("INTERPRETATION:")
    print("- High p-value (> 0.05) = activations follow Benford = likely legitimate")
    print("- Low p-value (< 0.01) = activations violate Benford = possible adversarial tampering")
    print("- L1 distance quantifies deviation from Benford prediction")

    # Plot Benford vs empirical for honest and one adversarial
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = synth_llm_activations(seed=0)
    for ax, (name, xp) in zip(axes, [("honest", x), ("adv_round_2digits", adversary_round(x, 2))]):
        digits = leading_digit(xp)
        r = benford_compliance(digits)
        bar_x = np.arange(1, 10)
        ax.bar(bar_x - 0.2, BENFORD_PROB, 0.4, label="Benford", alpha=0.7)
        ax.bar(bar_x + 0.2, r["empirical"], 0.4, label="Empirical", alpha=0.7)
        ax.set_xlabel("Leading digit")
        ax.set_ylabel("Probability")
        ax.set_title(f"{name}: χ²={r['chi2']:.1f}  p={r['p_value']:.3e}")
        ax.legend()
    fig.tight_layout()
    fig.savefig("plots/benfords_law_pilot.png", dpi=150)
    print(f"\nPlot saved to plots/benfords_law_pilot.png")


if __name__ == "__main__":
    main()
