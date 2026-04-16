"""
Keystone Shard Metric for Synapse Volunteer Networks.

Origin: Round 1 weird-faculty exercise (ecologist lens — keystone species).
Hypothesis: not all shards in a distributed inference pipeline are equally
critical. "Keystone" shards are those whose failure disproportionately
affects query latency, correctness, or fleet stability.

Formalization:
Let G = (V, E) be the pipeline graph. V = volunteers, E = activation flows.
Let Q = distribution of queries and L(q, v) = latency of serving query q
when volunteer v is operational, L(q, v_) = when v is down.

Keystone score of volunteer v:
    K(v) = E_{q ~ Q} [ L(q, v_) - L(q, v) ] / (E_{q ~ Q} [ L(q, v) ])

Higher K(v) = more critical shard. Allocate more redundancy / priority
to high-K volunteers.

This module provides:
- Keystone score computation from measured latency distributions
- Monte Carlo estimation when exhaustive computation is expensive
- Redundancy-allocation policy given a keystone score vector

Author: Claude Opus 4.6
Date: 2026-04-16
"""
from __future__ import annotations
import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class NetworkSample:
    """A single observation: query id, assigned shards, measured latency, outcome."""
    query_id: str
    shards_used: list[str]
    latency_ms: float
    success: bool


def keystone_score_from_ablation(
    samples: list[NetworkSample],
    shard_id: str,
    impact_metric: str = "latency"
) -> float:
    """
    Compute keystone score for a single shard by comparing queries that did
    vs did not include it.

    Args:
        samples: historical query observations
        shard_id: the shard to score
        impact_metric: "latency" or "failure_rate"

    Returns:
        keystone score; higher = more critical
    """
    with_shard = [s for s in samples if shard_id in s.shards_used]
    without_shard = [s for s in samples if shard_id not in s.shards_used]

    if not with_shard or not without_shard:
        return 0.0

    if impact_metric == "latency":
        # Latency bump when this shard is bypassed (e.g., replaced with a slower alternative or timeout)
        mean_with = np.mean([s.latency_ms for s in with_shard])
        mean_without = np.mean([s.latency_ms for s in without_shard])
        return (mean_without - mean_with) / max(mean_with, 1e-6)
    elif impact_metric == "failure_rate":
        fail_with = 1 - np.mean([s.success for s in with_shard])
        fail_without = 1 - np.mean([s.success for s in without_shard])
        return fail_without - fail_with
    else:
        raise ValueError(f"unknown metric {impact_metric}")


def keystone_scores_monte_carlo(
    simulate_query: Callable,
    active_shards: list[str],
    n_samples: int = 100,
    impact_metric: str = "latency",
) -> dict[str, float]:
    """
    Estimate keystone scores via ablation Monte Carlo.

    Args:
        simulate_query: callable that takes a set of active shards and returns NetworkSample
        active_shards: list of shard IDs currently in fleet
        n_samples: queries to simulate per condition

    Returns:
        dict mapping shard_id -> keystone score
    """
    scores = {}
    # Baseline: all shards active
    baseline = [simulate_query(active_shards) for _ in range(n_samples)]
    base_latency = np.mean([s.latency_ms for s in baseline])
    base_fail = 1 - np.mean([s.success for s in baseline])

    for shard in active_shards:
        # Ablate: remove this shard, simulate
        ablated_shards = [s for s in active_shards if s != shard]
        ablated = [simulate_query(ablated_shards) for _ in range(n_samples)]
        abl_latency = np.mean([s.latency_ms for s in ablated])
        abl_fail = 1 - np.mean([s.success for s in ablated])

        if impact_metric == "latency":
            scores[shard] = (abl_latency - base_latency) / max(base_latency, 1e-6)
        else:
            scores[shard] = abl_fail - base_fail

    return scores


def redundancy_allocation(
    keystone_scores: dict[str, float],
    total_redundancy_budget: int,
    min_per_shard: int = 1,
) -> dict[str, int]:
    """
    Given keystone scores and a total budget of "redundancy slots" (extra replicas
    beyond the primary), allocate proportionally to keystone score.

    Args:
        keystone_scores: dict from shard_id -> score
        total_redundancy_budget: total number of extra replicas across fleet
        min_per_shard: minimum extras each shard gets (floor)

    Returns:
        dict mapping shard_id -> number of replicas
    """
    shards = list(keystone_scores.keys())
    scores = np.array([max(keystone_scores[s], 0) for s in shards])

    # Start with min
    allocation = {s: min_per_shard for s in shards}
    remaining = total_redundancy_budget - len(shards) * min_per_shard
    if remaining <= 0:
        return allocation

    # Proportional top-up
    if scores.sum() > 0:
        weights = scores / scores.sum()
        extras = np.floor(weights * remaining).astype(int)
        # Distribute any leftover to top shards
        leftover = remaining - extras.sum()
        top_idx = np.argsort(-scores)
        for i in range(leftover):
            extras[top_idx[i % len(shards)]] += 1

        for s, extra in zip(shards, extras):
            allocation[s] += int(extra)
    else:
        # No signal — spread evenly
        for s in shards:
            allocation[s] += remaining // len(shards)

    return allocation


# ──────────────────────────────────────────────────────────────────────────
# Demo with synthetic Synapse-like data
# ──────────────────────────────────────────────────────────────────────────

def demo():
    """Demo showing keystone identification on synthetic mixed-criticality fleet."""
    rng = np.random.default_rng(42)

    # Simulate a fleet of 10 volunteer shards. Some are critical (many queries
    # route through them), some are redundant.
    shard_criticality = {
        "s0": 0.9,  # keystone
        "s1": 0.8,  # keystone
        "s2": 0.3,  # middle
        "s3": 0.1,  # redundant
        "s4": 0.2,
        "s5": 0.85, # keystone
        "s6": 0.05,
        "s7": 0.4,
        "s8": 0.1,
        "s9": 0.3,
    }

    def simulate_query(active):
        base_latency = 100.0
        # Missing any critical shard → latency penalty
        penalty = sum(10.0 * shard_criticality[s] for s in shard_criticality
                     if s not in active) * (1 + 0.1 * rng.standard_normal())
        fail_prob = 0.0
        for s in shard_criticality:
            if s not in active:
                fail_prob = max(fail_prob, shard_criticality[s] * 0.3)  # top missing shard dominates
        success = rng.random() > fail_prob
        used = active.copy()
        return NetworkSample(
            query_id=f"q{rng.integers(0, 100000)}",
            shards_used=used,
            latency_ms=base_latency + penalty,
            success=success,
        )

    active = list(shard_criticality.keys())
    scores = keystone_scores_monte_carlo(
        simulate_query, active, n_samples=200, impact_metric="latency"
    )

    print("=" * 60)
    print("KEYSTONE SHARD DEMO")
    print("=" * 60)
    print(f"{'Shard':<8} {'true criticality':>18} {'measured score':>18}")
    for s in sorted(active, key=lambda x: -scores[x]):
        print(f"{s:<8} {shard_criticality[s]:>18.2f} {scores[s]:>18.4f}")

    # Correlation between true and measured
    true_v = np.array([shard_criticality[s] for s in active])
    meas_v = np.array([scores[s] for s in active])
    rho, _ = __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(true_v, meas_v)
    print(f"\nSpearman rank correlation (true vs measured): {rho:.3f}")

    # Redundancy allocation with budget 20 extra replicas
    alloc = redundancy_allocation(scores, total_redundancy_budget=20, min_per_shard=1)
    print("\n=== REDUNDANCY ALLOCATION (budget=20 total extras) ===")
    print(f"{'Shard':<8} {'replicas':>10} {'criticality':>14}")
    for s in sorted(active, key=lambda x: -scores[x]):
        print(f"{s:<8} {alloc[s]:>10d} {shard_criticality[s]:>14.2f}")


if __name__ == "__main__":
    demo()
