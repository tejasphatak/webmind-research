"""
SFCA — Shapley Faculty Credit Assignment.

Given beat outcomes + faculty sets used, compute each faculty's Shapley-value
credit via Monte Carlo permutation sampling. Used to fairly attribute
contribution across cooperating faculties in a multi-persona LLM agent system.

v0 — dependencies: numpy (stdlib fallback for tiny cases).
Licensed MIT under Webmind.

Theory:
  Each beat is a coalition game. Faculties in the activated set S collaborate
  to produce an outcome r ∈ {-1, 0, +1}. We want φ_i, the marginal
  contribution of faculty i, averaged over all orderings:

    φ_i = (1/|S|!) Σ_π∈Π(S) [ v(π_≤i) - v(π_<i) ]

  Exact Shapley is NP-hard (|S|!); MC approximation uses random permutations:

    φ_i ≈ (1/K) Σ_k=1..K [ v(S_≤i in π_k) - v(S_<i in π_k) ]

  K=1000 samples is plenty for |S| ≤ 19; std-error scales as O(1/√K).

Value function v(T):
  v(T) = estimated probability that a beat activating exactly T yields ACTIVE.
  Initialized from a prior; updated online from ledger data.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class BeatRecord:
    """A single observed beat: which faculties fired, what outcome."""
    faculty_set: frozenset[str]
    outcome: int  # -1 (BLOCKED), 0 (QUIET), +1 (ACTIVE)


def monte_carlo_shapley(
    faculty_set: Iterable[str],
    outcome: int,
    value_fn: Callable[[frozenset[str]], float],
    num_samples: int = 1000,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """
    Compute Monte Carlo Shapley credits for one beat observation.

    Args:
      faculty_set: faculties that participated in this beat
      outcome: the observed outcome signal (e.g. -1 / 0 / +1)
      value_fn: v(T) returning expected outcome if only T had activated
      num_samples: number of random permutations
      rng: optional random.Random for reproducibility

    Returns:
      { faculty_name: marginal credit } summing approximately to v(full) * outcome
    """
    rng = rng or random.Random()
    faculties = list(faculty_set)
    n = len(faculties)
    if n == 0:
        return {}
    if n == 1:
        # Single faculty gets all credit
        return {faculties[0]: float(outcome)}

    # Normalize so v(∅) = 0 (standard Shapley convention). This ensures
    # efficiency axiom: Σφᵢ = v(full). Without it, Σφᵢ = v(full) − v(∅)
    # which is mathematically valid but confuses the efficiency check.
    v0 = value_fn(frozenset())
    def _v(T: frozenset) -> float:
        return value_fn(T) - v0

    credits = {f: 0.0 for f in faculties}
    for _ in range(num_samples):
        perm = faculties[:]
        rng.shuffle(perm)
        prev = frozenset()
        prev_v = 0.0  # _v(∅) = 0 by construction
        for f in perm:
            cur = prev | {f}
            cur_v = _v(cur)
            credits[f] += (cur_v - prev_v) * outcome
            prev = cur
            prev_v = cur_v
    for f in credits:
        credits[f] /= num_samples
    return credits


def exact_shapley(
    faculty_set: Iterable[str],
    outcome: int,
    value_fn: Callable[[frozenset[str]], float],
) -> dict[str, float]:
    """Exact Shapley, O(n!). Only use for small n (≤ 8) validation."""
    faculties = list(faculty_set)
    n = len(faculties)
    if n == 0:
        return {}
    # Normalize v(∅) = 0
    v0 = value_fn(frozenset())
    def _v(T: frozenset) -> float:
        return value_fn(T) - v0
    credits = {f: 0.0 for f in faculties}
    n_fact = math.factorial(n)
    for perm in itertools.permutations(faculties):
        prev = frozenset()
        prev_v = 0.0
        for f in perm:
            cur = prev | {f}
            cur_v = _v(cur)
            credits[f] += (cur_v - prev_v) * outcome
            prev = cur
            prev_v = cur_v
    for f in credits:
        credits[f] /= n_fact
    return credits


def efficiency_check(credits: dict[str, float], value_fn, faculty_set, outcome, tol=1e-6) -> bool:
    """Shapley efficiency axiom: sum of credits = (v(full) − v(∅)) × outcome."""
    S = frozenset(faculty_set)
    expected = (value_fn(S) - value_fn(frozenset())) * outcome
    return abs(sum(credits.values()) - expected) < tol


# ── Value function estimators ──────────────────────────────────────────

class HistoricalValueFn:
    """
    v(T) estimated from historical beat ledger.
    Returns empirical ACTIVE-rate of beats whose faculty_set is a superset of T.
    Falls back to prior when insufficient data.
    """
    def __init__(self, ledger: list[BeatRecord], prior_mean: float = 0.5, min_samples: int = 3):
        self.ledger = ledger
        self.prior = prior_mean
        self.min_samples = min_samples

    def __call__(self, T: frozenset[str]) -> float:
        if not T:
            return self.prior
        matches = [b.outcome == 1 for b in self.ledger if T.issubset(b.faculty_set)]
        if len(matches) < self.min_samples:
            return self.prior
        return sum(matches) / len(matches)


class SimpleLinearValueFn:
    """
    v(T) = σ( w₀ + Σ_{f∈T} w_f ), where w_f are learned faculty weights.
    Useful as a toy/testing value function with known structure.
    """
    def __init__(self, weights: dict[str, float], bias: float = 0.0):
        self.w = weights
        self.b = bias

    def __call__(self, T: frozenset[str]) -> float:
        x = self.b + sum(self.w.get(f, 0.0) for f in T)
        return 1.0 / (1.0 + math.exp(-x))
