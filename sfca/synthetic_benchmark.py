"""
SFCA Synthetic Benchmark
========================

Ground-truth benchmark for Shapley Faculty Credit Assignment (SFCA).

Creates a synthetic multi-agent environment with KNOWN ground-truth faculty
contributions, then compares credit-assignment algorithms on how well they
recover the ground truth.

This is the rigorous experimental backbone for the SFCA paper. Moves the
claim from "n=1 live A/B on personal system" to "reproducible benchmark
with ground truth".

Author: Tejas Phatak (with Claude Opus 4.6 / Anthropic)
Date: 2026-04-16
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import spearmanr

# Make sfca.py importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from sfca import monte_carlo_shapley, BeatRecord


# ── Ground-truth environment ──────────────────────────────────────────────

@dataclass
class SyntheticEnv:
    """
    Multi-agent environment with known ground-truth faculty weights.

    Each beat:
      1. Sample coalition S from beats distribution (Bernoulli per faculty)
      2. Compute latent signal: z = Σ_{i∈S} w_i* + Σ_{(i,j)∈S²} J_ij* + noise
      3. Outcome: r = +1 if z > θ_up, -1 if z < θ_down, else 0
    """
    n_faculties: int
    true_weights: np.ndarray              # w_i* ∈ [0,1], shape (N,)
    true_synergies: np.ndarray            # J_ij* symmetric, shape (N,N), diag=0
    coalition_prob: float = 0.5           # prob each faculty fires
    noise_std: float = 0.5
    theta_up: float = 0.3
    theta_down: float = -0.3
    rng: np.random.Generator = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(0)

    def sample_beat(self) -> tuple[np.ndarray, int, float]:
        """Returns (coalition_mask, outcome, latent_signal)."""
        S = self.rng.binomial(1, self.coalition_prob, self.n_faculties).astype(bool)
        if not S.any():
            # Force at least one faculty
            S[self.rng.integers(self.n_faculties)] = True
        linear = self.true_weights[S].sum()
        synergy = 0.0
        idx = np.where(S)[0]
        for i in idx:
            for j in idx:
                if j > i:
                    synergy += self.true_synergies[i, j]
        z = linear + synergy + self.rng.normal(0, self.noise_std)
        if z > self.theta_up:
            r = 1
        elif z < self.theta_down:
            r = -1
        else:
            r = 0
        return S, r, z


# ── Credit-assignment algorithms ──────────────────────────────────────────

def ema_credit(beats: list[tuple[np.ndarray, int]], n: int, alpha: float = 0.1) -> np.ndarray:
    """Baseline: flat EMA credit. Each firing faculty gets full outcome."""
    w = np.zeros(n)
    for S, r in beats:
        w = (1 - alpha) * w + alpha * r * S.astype(float)
    return w


def counterfactual_credit(
    beats: list[tuple[np.ndarray, int]],
    n: int,
    value_fn_history: Callable,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Leave-one-out credit (related to COMA).
    Each faculty gets credit = r * (v(S) - v(S \\ {i})).
    Uses an empirical value estimator that updates with each beat.
    """
    # Build history incrementally — v(T) estimated from running statistics
    # Shortcut: use per-faculty + per-pair running averages of r
    w = np.zeros(n)
    singleton_mean = np.zeros(n)
    singleton_count = np.zeros(n)
    for S, r in beats:
        idx = np.where(S)[0]
        if len(idx) == 0:
            continue
        # Estimate v(S) via Σ_i singleton_mean[i] × S[i] / max(1, |S|)
        # Estimate v(S \\ {i}) similarly without i
        if singleton_count.sum() > 0:
            v_S = 0.0
            for i in idx:
                if singleton_count[i] > 0:
                    v_S += singleton_mean[i]
            v_S /= max(1, len(idx))
            for i in idx:
                # v without i
                if len(idx) > 1:
                    v_Si = 0.0
                    for j in idx:
                        if j != i and singleton_count[j] > 0:
                            v_Si += singleton_mean[j]
                    v_Si /= (len(idx) - 1)
                    marginal = v_S - v_Si
                else:
                    marginal = r
                w[i] = (1 - alpha) * w[i] + alpha * marginal * r
        else:
            # Bootstrap: first few beats, flat credit
            for i in idx:
                w[i] = (1 - alpha) * w[i] + alpha * r
        # Update singleton stats
        for i in idx:
            singleton_count[i] += 1
            singleton_mean[i] += (r - singleton_mean[i]) / singleton_count[i]
    return w


def sfca_credit(
    beats: list[tuple[np.ndarray, int]],
    n: int,
    ledger_warmup: int = 50,
    shapley_samples: int = 500,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Shapley-based credit assignment.
    After `ledger_warmup` beats, use Monte Carlo Shapley with a historical
    value function computed from the ledger.
    """
    # Build a BeatRecord ledger as we go; fit value function on ledger
    ledger: list[BeatRecord] = []
    faculty_names = [f"f{i}" for i in range(n)]
    w = np.zeros(n)

    # Pre-store all idx lists
    for t, (S, r) in enumerate(beats):
        idx = np.where(S)[0]
        fset = frozenset(faculty_names[i] for i in idx)
        ledger.append(BeatRecord(faculty_set=fset, outcome=r))

        if t < ledger_warmup:
            # Bootstrap with EMA
            for i in idx:
                w[i] = (1 - alpha) * w[i] + alpha * r
            continue

        # Build value function from current ledger (rolling window)
        window = ledger[-min(len(ledger), 500):]
        # For speed, estimate v(T) as mean outcome of beats whose set is a superset of T
        # Precompute per-faculty conditional means
        per_fac_mean = np.zeros(n)
        per_fac_count = np.zeros(n)
        for b in window:
            for f in b.faculty_set:
                i = int(f[1:])
                per_fac_count[i] += 1
                per_fac_mean[i] += b.outcome

        def value_fn(T: frozenset) -> float:
            if not T:
                return 0.0
            s = 0.0
            c = 0
            for f in T:
                i = int(f[1:])
                if per_fac_count[i] > 0:
                    s += per_fac_mean[i] / per_fac_count[i]
                    c += 1
            return s / max(1, c)

        credits = monte_carlo_shapley(
            fset, r, value_fn, num_samples=shapley_samples,
        )
        for f, credit in credits.items():
            i = int(f[1:])
            w[i] = (1 - alpha) * w[i] + alpha * credit

    return w


# ── Experiment runner ──────────────────────────────────────────────────────

def run_single_config(
    n_faculties: int,
    n_beats: int,
    synergy_strength: float,
    noise_std: float,
    seed: int,
) -> dict:
    """Run one config: generate env, sample beats, run 3 algorithms, return metrics."""
    rng = np.random.default_rng(seed)

    # Ground-truth weights: mix of high, medium, low contributors
    true_w = rng.uniform(-0.5, 1.0, n_faculties)

    # Synergy matrix (symmetric, diag=0)
    J = rng.uniform(-0.3, 0.3, (n_faculties, n_faculties)) * synergy_strength
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    env = SyntheticEnv(
        n_faculties=n_faculties,
        true_weights=true_w,
        true_synergies=J,
        noise_std=noise_std,
        rng=rng,
    )

    beats = []
    for _ in range(n_beats):
        S, r, _ = env.sample_beat()
        beats.append((S, r))

    # Run algorithms
    t0 = time.time()
    w_ema = ema_credit(beats, n_faculties)
    t_ema = time.time() - t0

    t0 = time.time()
    w_cf = counterfactual_credit(beats, n_faculties, None)
    t_cf = time.time() - t0

    t0 = time.time()
    w_sfca = sfca_credit(beats, n_faculties, shapley_samples=300)
    t_sfca = time.time() - t0

    # Metrics: correlation with true weights (ranking quality)
    def correlate(w_est, w_true):
        if np.std(w_est) < 1e-12:
            return 0.0
        r, _ = spearmanr(w_est, w_true)
        return 0.0 if np.isnan(r) else r

    def normalized_l2(w_est, w_true):
        # Rescale w_est to have same scale as w_true for L2 comparison
        if np.std(w_est) < 1e-12:
            return float(np.linalg.norm(w_true))
        w_est_n = (w_est - w_est.mean()) / (w_est.std() + 1e-12)
        w_true_n = (w_true - w_true.mean()) / (w_true.std() + 1e-12)
        return float(np.linalg.norm(w_est_n - w_true_n) / np.sqrt(n_faculties))

    results = {
        "n_faculties": n_faculties,
        "n_beats": n_beats,
        "synergy_strength": synergy_strength,
        "noise_std": noise_std,
        "seed": seed,
        "true_weights": true_w.tolist(),
        "ema": {
            "weights": w_ema.tolist(),
            "spearman_r": float(correlate(w_ema, true_w)),
            "l2_norm": float(normalized_l2(w_ema, true_w)),
            "time_sec": t_ema,
        },
        "counterfactual": {
            "weights": w_cf.tolist(),
            "spearman_r": float(correlate(w_cf, true_w)),
            "l2_norm": float(normalized_l2(w_cf, true_w)),
            "time_sec": t_cf,
        },
        "sfca": {
            "weights": w_sfca.tolist(),
            "spearman_r": float(correlate(w_sfca, true_w)),
            "l2_norm": float(normalized_l2(w_sfca, true_w)),
            "time_sec": t_sfca,
        },
    }
    return results


def run_benchmark():
    """Run the full sweep: sizes × synergy levels × seeds."""
    configs = []
    # Main sweep: how does synergy strength affect relative performance?
    for n in [5, 10, 19]:
        for synergy in [0.0, 0.5, 1.0, 2.0]:
            for seed in range(5):  # 5 seeds per config
                configs.append((n, 500, synergy, 0.3, seed))

    print(f"Running {len(configs)} configurations...")
    all_results = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        r = run_single_config(*cfg)
        dt = time.time() - t0
        all_results.append(r)
        if i % 5 == 0:
            print(f"  [{i+1}/{len(configs)}] n={cfg[0]} synergy={cfg[2]} seed={cfg[4]}  "
                  f"EMA r={r['ema']['spearman_r']:.3f}  "
                  f"SFCA r={r['sfca']['spearman_r']:.3f}  "
                  f"CF r={r['counterfactual']['spearman_r']:.3f}  "
                  f"[{dt:.1f}s]", flush=True)

    out_path = Path("findings/sfca_synthetic_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": all_results}, f)
    print(f"\nWrote {out_path}")
    return all_results


if __name__ == "__main__":
    results = run_benchmark()

    # Quick summary
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r["n_faculties"], r["synergy_strength"])].append(r)

    print("\n=== SUMMARY (Spearman rank correlation with ground truth, mean across seeds) ===")
    print(f"{'n':>3} {'synergy':>8} {'EMA':>8} {'CF':>8} {'SFCA':>8}  {'SFCA-EMA':>9}")
    for (n, syn), rs in sorted(groups.keys()):
        group = groups[(n, syn)]
        ema = np.mean([r["ema"]["spearman_r"] for r in group])
        cf = np.mean([r["counterfactual"]["spearman_r"] for r in group])
        sfca = np.mean([r["sfca"]["spearman_r"] for r in group])
        print(f"{n:3d} {syn:8.1f} {ema:8.3f} {cf:8.3f} {sfca:8.3f}  {sfca - ema:+9.3f}")
