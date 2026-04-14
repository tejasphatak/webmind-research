"""
Unit tests for SFCA — validates MC approximation against exact Shapley on
toy cases, verifies mathematical axioms (efficiency, symmetry, null-player).

Run: python3 -m unittest test_sfca -v
"""

import random
import unittest

from sfca import (
    monte_carlo_shapley,
    exact_shapley,
    efficiency_check,
    HistoricalValueFn,
    SimpleLinearValueFn,
    BeatRecord,
)


class SFCATest(unittest.TestCase):

    # ── Axiom tests (the famous 4 Shapley axioms) ──────────────────────

    def test_efficiency_exact_small_game(self):
        """Sum of Shapley values equals (v(full) − v(∅)) · outcome (standard Shapley efficiency)."""
        faculties = ["A", "B", "C"]
        vfn = SimpleLinearValueFn({"A": 1.0, "B": 0.5, "C": -0.2})
        credits = exact_shapley(faculties, outcome=1, value_fn=vfn)
        total = sum(credits.values())
        expected = (vfn(frozenset(faculties)) - vfn(frozenset())) * 1
        self.assertAlmostEqual(total, expected, places=6)

    def test_symmetry_exact(self):
        """If two faculties have identical marginal contributions, they get equal credit."""
        faculties = ["A", "B", "C"]
        # A and B are symmetric (same weight); C differs
        vfn = SimpleLinearValueFn({"A": 1.0, "B": 1.0, "C": 0.3})
        credits = exact_shapley(faculties, outcome=1, value_fn=vfn)
        self.assertAlmostEqual(credits["A"], credits["B"], places=6)
        self.assertNotAlmostEqual(credits["A"], credits["C"], places=3)

    def test_null_player_exact(self):
        """A faculty that never changes v() gets zero credit."""
        faculties = ["A", "B", "NULL"]
        vfn = SimpleLinearValueFn({"A": 1.0, "B": 0.5, "NULL": 0.0})
        credits = exact_shapley(faculties, outcome=1, value_fn=vfn)
        # NULL contributes 0 to the linear combination, so its marginal is 0
        # Well, NULL still moves sigmoid because of saturation, but only slightly.
        # Use a truly null function:
        def null_vfn(T):
            T_clean = frozenset(f for f in T if f != "NULL")
            return SimpleLinearValueFn({"A": 1.0, "B": 0.5})(T_clean)
        credits = exact_shapley(faculties, outcome=1, value_fn=null_vfn)
        self.assertAlmostEqual(credits["NULL"], 0.0, places=6)

    def test_linearity_exact(self):
        """φ(v1 + v2) = φ(v1) + φ(v2). Holds for any value functions."""
        faculties = ["A", "B"]
        v1 = SimpleLinearValueFn({"A": 0.8, "B": 0.4})
        v2 = SimpleLinearValueFn({"A": 0.2, "B": 0.6})
        v_sum = lambda T: v1(T) + v2(T)
        c1 = exact_shapley(faculties, outcome=1, value_fn=v1)
        c2 = exact_shapley(faculties, outcome=1, value_fn=v2)
        c_sum = exact_shapley(faculties, outcome=1, value_fn=v_sum)
        for f in faculties:
            self.assertAlmostEqual(c1[f] + c2[f], c_sum[f], places=6)

    # ── MC convergence tests ───────────────────────────────────────────

    def test_mc_approximates_exact_small_n(self):
        """Monte Carlo with many samples approaches exact Shapley."""
        faculties = ["A", "B", "C", "D", "E"]
        vfn = SimpleLinearValueFn({"A": 1.0, "B": 0.5, "C": -0.2, "D": 0.8, "E": 0.1})
        exact = exact_shapley(faculties, outcome=1, value_fn=vfn)
        rng = random.Random(42)
        mc = monte_carlo_shapley(faculties, outcome=1, value_fn=vfn, num_samples=5000, rng=rng)
        for f in faculties:
            # 5000 samples: stderr < 2%
            self.assertAlmostEqual(exact[f], mc[f], places=2,
                                   msg=f"MC diverges on faculty {f}: exact={exact[f]:.4f} mc={mc[f]:.4f}")

    def test_mc_reproducible_with_seed(self):
        faculties = ["A", "B", "C"]
        vfn = SimpleLinearValueFn({"A": 1.0, "B": 0.5, "C": -0.2})
        m1 = monte_carlo_shapley(faculties, 1, vfn, 500, random.Random(7))
        m2 = monte_carlo_shapley(faculties, 1, vfn, 500, random.Random(7))
        self.assertEqual(m1, m2)

    # ── Edge cases ─────────────────────────────────────────────────────

    def test_empty_set_returns_empty(self):
        vfn = SimpleLinearValueFn({})
        self.assertEqual(monte_carlo_shapley([], 1, vfn, 100), {})

    def test_single_faculty_gets_full_credit(self):
        vfn = SimpleLinearValueFn({"A": 1.0})
        credits = monte_carlo_shapley(["A"], 1, vfn, 100)
        self.assertEqual(credits, {"A": 1.0})

    def test_negative_outcome_flips_sign(self):
        faculties = ["A", "B"]
        vfn = SimpleLinearValueFn({"A": 1.0, "B": 0.5})
        pos = exact_shapley(faculties, outcome=1, value_fn=vfn)
        neg = exact_shapley(faculties, outcome=-1, value_fn=vfn)
        for f in faculties:
            self.assertAlmostEqual(pos[f], -neg[f], places=6)

    def test_zero_outcome_zero_credits(self):
        faculties = ["A", "B", "C"]
        vfn = SimpleLinearValueFn({"A": 1.0, "B": 0.5, "C": -0.3})
        credits = monte_carlo_shapley(faculties, outcome=0, value_fn=vfn, num_samples=500)
        for c in credits.values():
            self.assertAlmostEqual(c, 0.0, places=6)

    # ── HistoricalValueFn tests ────────────────────────────────────────

    def test_historical_vfn_falls_back_to_prior(self):
        vfn = HistoricalValueFn(ledger=[], prior_mean=0.5, min_samples=3)
        self.assertEqual(vfn(frozenset(["X"])), 0.5)

    def test_historical_vfn_empirical_rate(self):
        ledger = [
            BeatRecord(frozenset(["A", "B"]), 1),
            BeatRecord(frozenset(["A", "B"]), 1),
            BeatRecord(frozenset(["A", "B"]), 0),
            BeatRecord(frozenset(["A", "B", "C"]), 1),
            BeatRecord(frozenset(["A", "B", "C"]), 1),
        ]
        vfn = HistoricalValueFn(ledger, prior_mean=0.5, min_samples=2)
        # {A,B} subset of all 5 records → ACTIVE in 4/5 = 0.8
        self.assertAlmostEqual(vfn(frozenset(["A", "B"])), 4/5)
        # {A,B,C} subset of last 2 → ACTIVE in 2/2 = 1.0
        self.assertAlmostEqual(vfn(frozenset(["A", "B", "C"])), 1.0)

    # ── Performance sanity ─────────────────────────────────────────────

    def test_n19_completes_fast(self):
        """1000 MC samples on N=19 must finish in under 2s (SRE target: <1s on e2-standard-4)."""
        import time
        faculties = [f"F{i}" for i in range(19)]
        weights = {f: (i - 9) / 10 for i, f in enumerate(faculties)}
        vfn = SimpleLinearValueFn(weights)
        t0 = time.time()
        credits = monte_carlo_shapley(faculties, 1, vfn, num_samples=1000, rng=random.Random(0))
        elapsed = time.time() - t0
        self.assertLess(elapsed, 2.0, f"Too slow: {elapsed:.2f}s")
        self.assertEqual(len(credits), 19)


if __name__ == "__main__":
    unittest.main()
