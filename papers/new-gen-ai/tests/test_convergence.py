"""
Tests for the Convergence Loop.

Verifies against HLD spec:
- Converged = answer found, concepts returned
- Not converged = "I don't know" (invariant #4: honest about failure)
- Query anchor prevents drift (residual connection)
- Each hop is inspectable (invariant #2: every answer has a source)
- Low-confidence neurons filtered out
- Empty DB → honest abort
- Confidence-weighted blending (replaces softmax)
- Movement decreases toward convergence
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from neuron import NeuronDB, VECTOR_DIM
from convergence import ConvergenceLoop, ConvergenceResult, MultiHopConvergence, MultiHopResult

DIM = 300


def make_db(dim=DIM):
    return NeuronDB(dim=dim)


def unit_vector(idx: int, dim=DIM) -> np.ndarray:
    """Unit vector along a single dimension."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx % dim] = 1.0
    return v


def random_vector(seed: int, dim=DIM) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestConvergenceBasic:

    def test_converges_on_nearby_neuron(self):
        """Query near a neuron should converge to it."""
        db = make_db()
        # Insert a neuron
        target_vec = unit_vector(0)
        db.insert(target_vec, confidence=0.7)

        # Query very close to it
        query = unit_vector(0)
        query[1] = 0.01  # tiny perturbation
        query = query / np.linalg.norm(query)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        result = loop.converge(query)

        assert result.converged is True
        assert len(result.concepts) > 0
        assert result.confidence > 0

    def test_converges_with_multiple_neurons(self):
        """Multiple related neurons should reinforce convergence."""
        db = make_db()
        # Cluster of nearby neurons
        base = random_vector(42)
        for i in range(5):
            v = base + np.random.RandomState(i).randn(DIM).astype(np.float32) * 0.05
            v = v / np.linalg.norm(v)
            db.insert(v, confidence=0.6)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        result = loop.converge(base)

        assert result.converged is True
        assert len(result.concepts) >= 1

    def test_empty_db_does_not_converge(self):
        """Empty DB → no answer. Honest about failure."""
        db = make_db()
        loop = ConvergenceLoop(db, max_hops=5, k=5)
        result = loop.converge(random_vector(0))

        assert result.converged is False
        assert result.concepts == []
        assert result.confidence == 0.0

    def test_zero_query_does_not_converge(self):
        """Zero vector query → honest abort."""
        db = make_db()
        db.insert(random_vector(0), confidence=0.5)

        loop = ConvergenceLoop(db, max_hops=5, k=5)
        result = loop.converge(np.zeros(DIM, dtype=np.float32))

        assert result.converged is False


class TestConvergenceAnchor:

    def test_anchor_prevents_drift(self):
        """
        Query anchor should keep the search near the original query.
        Without anchor, the loop would drift to wherever the densest
        cluster of neurons is. With anchor, it stays near the query.
        """
        db = make_db()

        # Two clusters: one at dim 0, one at dim 1
        for i in range(5):
            v = unit_vector(0)
            v[2 + i] = 0.1 * (i + 1)
            v = v / np.linalg.norm(v)
            db.insert(v, confidence=0.6)

        for i in range(10):  # bigger cluster at dim 1
            v = unit_vector(1)
            v[2 + i] = 0.1 * (i + 1)
            v = v / np.linalg.norm(v)
            db.insert(v, confidence=0.8)

        # Query near cluster 0
        query = unit_vector(0)
        loop = ConvergenceLoop(db, max_hops=10, k=5)
        result = loop.converge(query)

        # Result should still be closer to dim 0 than dim 1
        # because the anchor keeps pulling back
        if result.converged:
            assert result.vector[0] > result.vector[1]

    def test_alpha_increases_with_hops(self):
        """Later hops should weight the query anchor more heavily."""
        db = make_db()
        # Spread neurons around so convergence takes multiple hops
        for i in range(20):
            db.insert(random_vector(i), confidence=0.5)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        result = loop.converge(random_vector(99))

        if len(result.hops) >= 2:
            # Later hops should show less movement (anchor stabilizes)
            first_movement = result.hops[0].movement
            last_movement = result.hops[-1].movement
            # Last movement should be smaller or equal
            # (convergence = decreasing movement)
            assert last_movement <= first_movement + 0.01  # small tolerance


class TestConvergenceHonesty:

    def test_low_confidence_neurons_filtered(self):
        """Neurons below min_confidence should not participate."""
        db = make_db()
        # All neurons have very low confidence
        for i in range(5):
            db.insert(random_vector(i), confidence=0.01)

        loop = ConvergenceLoop(db, max_hops=5, k=5, min_confidence=0.1)
        result = loop.converge(random_vector(0))

        assert result.converged is False
        assert result.confidence == 0.0

    def test_non_convergence_penalizes_confidence(self):
        """Non-convergence should reduce the reported confidence."""
        db = make_db()
        # Sparse, scattered neurons — won't converge easily
        for i in range(5):
            db.insert(unit_vector(i * 50), confidence=0.6)

        loop = ConvergenceLoop(db, max_hops=3, k=5)
        result = loop.converge(random_vector(99))

        if not result.converged:
            # Confidence should be penalized (halved per spec)
            assert result.confidence < 0.6


class TestConvergenceTrace:

    def test_trace_has_hops(self):
        """Each hop should be recorded for inspectability."""
        db = make_db()
        for i in range(5):
            db.insert(random_vector(i), confidence=0.5)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        result = loop.converge(random_vector(0))

        assert len(result.hops) > 0
        for hop in result.hops:
            assert hop.hop_number >= 0
            assert len(hop.neighbors) > 0
            assert hop.current.shape == (DIM,)

    def test_trace_string_readable(self):
        """Invariant #2: trace should be human-readable."""
        db = make_db()
        for i in range(3):
            db.insert(random_vector(i), confidence=0.5)

        loop = ConvergenceLoop(db, max_hops=5, k=3)
        result = loop.converge(random_vector(0))

        trace = result.trace()
        assert "Convergence:" in trace
        assert "Hop 0:" in trace

    def test_trace_shows_neuron_ids(self):
        """Trace should show which neurons participated."""
        db = make_db()
        n0 = db.insert(random_vector(0), confidence=0.7)
        n1 = db.insert(random_vector(1), confidence=0.6)

        loop = ConvergenceLoop(db, max_hops=5, k=5)
        result = loop.converge(random_vector(0))

        trace = result.trace()
        # Should mention at least one neuron ID
        assert "n0" in trace or "n1" in trace

    def test_converged_result_has_concepts(self):
        """Converged result should list the participating neurons."""
        db = make_db()
        target = random_vector(42)
        db.insert(target, confidence=0.8)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        result = loop.converge(target)

        assert result.converged is True
        assert len(result.concepts) > 0
        assert all(hasattr(c, 'id') for c in result.concepts)


class TestConvergenceBlending:

    def test_high_confidence_dominates_blend(self):
        """Higher confidence neurons should have more influence."""
        db = make_db()
        v_high = unit_vector(0)
        v_low = unit_vector(1)

        db.insert(v_high, confidence=0.8)
        db.insert(v_low, confidence=0.1)

        # Query between them
        query = np.zeros(DIM, dtype=np.float32)
        query[0] = 0.5
        query[1] = 0.5
        query = query / np.linalg.norm(query)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        result = loop.converge(query)

        # Result should lean toward the high-confidence neuron (dim 0)
        if result.converged:
            assert result.vector[0] > result.vector[1]

    def test_equal_confidence_blends_evenly(self):
        """Equal confidence neurons should blend roughly equally."""
        db = make_db()
        v1 = unit_vector(0)
        v2 = unit_vector(1)

        db.insert(v1, confidence=0.5)
        db.insert(v2, confidence=0.5)

        query = np.zeros(DIM, dtype=np.float32)
        query[0] = 0.5
        query[1] = 0.5
        query = query / np.linalg.norm(query)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        result = loop.converge(query)

        if result.converged:
            # Both dimensions should have similar magnitude
            ratio = abs(result.vector[0]) / (abs(result.vector[1]) + 1e-10)
            assert 0.5 < ratio < 2.0


class TestConvergenceEdgeCases:

    def test_single_neuron_converges_immediately(self):
        """One neuron in DB → should converge in 1-2 hops."""
        db = make_db()
        v = random_vector(42)
        db.insert(v, confidence=0.7)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        result = loop.converge(v)

        assert result.converged is True
        assert len(result.hops) <= 3

    def test_max_hops_respected(self):
        """Should not exceed max_hops."""
        db = make_db()
        for i in range(20):
            db.insert(random_vector(i), confidence=0.5)

        max_h = 3
        loop = ConvergenceLoop(db, max_hops=max_h, k=5)
        result = loop.converge(random_vector(99))

        assert len(result.hops) <= max_h

    def test_convergence_threshold_configurable(self):
        """Stricter threshold should require more hops or fail."""
        db = make_db()
        for i in range(10):
            db.insert(random_vector(i), confidence=0.5)

        query = random_vector(0)

        # Lenient threshold
        loop_easy = ConvergenceLoop(db, max_hops=10, k=5,
                                     convergence_threshold=0.90)
        result_easy = loop_easy.converge(query)

        # Strict threshold
        loop_hard = ConvergenceLoop(db, max_hops=10, k=5,
                                     convergence_threshold=0.999)
        result_hard = loop_hard.converge(query)

        # Strict should take at least as many hops (or fail)
        if result_easy.converged and result_hard.converged:
            assert len(result_hard.hops) >= len(result_easy.hops)


class TestMultiHopConvergence:
    """
    Tests for multi-hop reasoning across convergence rounds.

    The key insight: single convergence finds one neighborhood.
    Multi-hop chains rounds so concepts from round N shift the
    query for round N+1, reaching new regions of concept space.
    """

    def _make_cluster(self, db, center_dim: int, count: int = 3,
                      confidence: float = 0.6, spread: float = 0.05):
        """Create a cluster of neurons near a unit vector dimension."""
        neurons = []
        for i in range(count):
            v = unit_vector(center_dim)
            # Add small perturbation so they're not identical
            v[center_dim + 1 if center_dim + 1 < DIM else 0] = spread * (i + 1)
            v = v / np.linalg.norm(v)
            n = db.insert(v, confidence=confidence)
            neurons.append(n)
        return neurons

    def test_single_hop_still_works(self):
        """Single-hop queries should work unchanged through multi-hop."""
        db = make_db()
        target = random_vector(42)
        db.insert(target, confidence=0.7)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        mh = MultiHopConvergence(loop, max_rounds=3)
        result = mh.reason(target)

        assert result.converged is True
        assert len(result.concepts) > 0
        assert len(result.rounds) >= 1

    def test_two_hop_finds_distant_concept(self):
        """
        Two clusters in different regions, connected by a bridge neuron.
        Query near cluster A should find cluster B via the bridge.

        Layout:
          Cluster A (dim 0 region) -- Bridge (between 0 and 50) -- Cluster B (dim 50 region)
        """
        db = make_db()

        # Cluster A: centered on dim 0
        cluster_a = self._make_cluster(db, center_dim=0, count=3, confidence=0.7)

        # Bridge neuron: between dim 0 and dim 50
        bridge_vec = np.zeros(DIM, dtype=np.float32)
        bridge_vec[0] = 0.5
        bridge_vec[50] = 0.5
        bridge_vec = bridge_vec / np.linalg.norm(bridge_vec)
        bridge = db.insert(bridge_vec, confidence=0.6)

        # Cluster B: centered on dim 50
        cluster_b = self._make_cluster(db, center_dim=50, count=3, confidence=0.7)

        # Query near cluster A
        query = unit_vector(0)

        loop = ConvergenceLoop(db, max_hops=10, k=5, min_relevance=0.1)
        mh = MultiHopConvergence(loop, max_rounds=3, concept_blend_weight=0.5)
        result = mh.reason(query)

        assert result.converged is True
        # Should have found concepts from both clusters
        found_ids = {c.id for c in result.concepts}
        cluster_a_ids = {n.id for n in cluster_a}
        cluster_b_ids = {n.id for n in cluster_b}

        # Must find cluster A (directly near query)
        assert found_ids & cluster_a_ids, "Should find cluster A (near query)"
        # Should find bridge or cluster B (via multi-hop)
        assert bridge.id in found_ids or (found_ids & cluster_b_ids), \
            "Multi-hop should discover bridge or cluster B"

    def test_no_new_concepts_stops_early(self):
        """If round 2 finds the same concepts as round 1, stop."""
        db = make_db()
        # Single tight cluster — round 2 can't find anything new
        for i in range(5):
            v = random_vector(i)
            db.insert(v, confidence=0.6)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        mh = MultiHopConvergence(loop, max_rounds=5)
        result = mh.reason(random_vector(0))

        # Should stop early — not use all 5 rounds
        assert len(result.rounds) <= 3

    def test_empty_db_multi_hop(self):
        """Empty DB → honest abort, even with multi-hop."""
        db = make_db()
        loop = ConvergenceLoop(db, max_hops=5, k=5)
        mh = MultiHopConvergence(loop, max_rounds=3)
        result = mh.reason(random_vector(0))

        assert result.converged is False
        assert result.concepts == []
        assert len(result.rounds) == 1  # tries once, fails, stops

    def test_zero_vector_multi_hop(self):
        """Zero vector → immediate abort."""
        db = make_db()
        db.insert(random_vector(0), confidence=0.5)

        loop = ConvergenceLoop(db, max_hops=5, k=5)
        mh = MultiHopConvergence(loop, max_rounds=3)
        result = mh.reason(np.zeros(DIM, dtype=np.float32))

        assert result.converged is False

    def test_multi_hop_trace_shows_rounds(self):
        """Trace should show each round for inspectability."""
        db = make_db()
        for i in range(10):
            db.insert(random_vector(i), confidence=0.5)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        mh = MultiHopConvergence(loop, max_rounds=3)
        result = mh.reason(random_vector(0))

        trace = result.trace()
        assert "Multi-hop:" in trace
        assert "Round 1" in trace

    def test_concepts_not_duplicated(self):
        """Same neuron found in multiple rounds should appear once."""
        db = make_db()
        v = random_vector(42)
        db.insert(v, confidence=0.7)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        mh = MultiHopConvergence(loop, max_rounds=3)
        result = mh.reason(v)

        ids = [c.id for c in result.concepts]
        assert len(ids) == len(set(ids)), "Concepts should not be duplicated"

    def test_max_rounds_respected(self):
        """Should not exceed max_rounds."""
        db = make_db()
        # Many spread-out neurons to encourage multiple rounds
        for i in range(50):
            db.insert(unit_vector(i * 5), confidence=0.5)

        loop = ConvergenceLoop(db, max_hops=5, k=5, min_relevance=0.05)
        mh = MultiHopConvergence(loop, max_rounds=2)
        result = mh.reason(random_vector(0))

        assert len(result.rounds) <= 2

    def test_confidence_from_all_rounds(self):
        """Confidence should reflect concepts from all rounds."""
        db = make_db()
        # High confidence cluster
        for i in range(3):
            v = random_vector(i)
            db.insert(v, confidence=0.8)

        loop = ConvergenceLoop(db, max_hops=10, k=5)
        mh = MultiHopConvergence(loop, max_rounds=3)
        result = mh.reason(random_vector(0))

        if result.converged:
            assert result.confidence > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
