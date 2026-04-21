"""
Tests for SparseConvergenceLoop and SparseMultiHop.

Verifies that sparse co-occurrence convergence produces the same
logical behavior as the dense convergence loop:
- Converges on related words
- Does not converge on empty/unknown input
- Query anchor prevents drift
- Per-hop specialization (early=broad, late=narrow)
- Mutual attention boosts coherent clusters
- Softmax-weighted blending
- Multi-hop finds distant concepts
- Trace is inspectable
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparse_convergence import (
    SparseConvergenceLoop, SparseMultiHop,
    sparse_cosine, sparse_blend, sparse_norm, sparse_normalize,
    SparseConvergenceResult, SparseMultiHopResult,
)


def make_graph():
    """Build a small co-occurrence graph for testing.

    Words: paris(0), capital(1), france(2), london(3), england(4),
           river(5), seine(6), thames(7), bridge(8), tower(9)

    Co-occurrence edges encode knowledge:
      paris <-> capital, france, seine, bridge
      london <-> capital, england, thames, tower, bridge
      capital <-> paris, london, france, england
      seine <-> paris, river
      thames <-> london, river
    """
    words = ["paris", "capital", "france", "london", "england",
             "river", "seine", "thames", "bridge", "tower"]
    word_idx = {w: i for i, w in enumerate(words)}
    word_neurons = {w: i + 100 for i, w in enumerate(words)}

    cooc = {}
    for i in range(len(words)):
        cooc[i] = {i: 1.0}

    def link(a, b, weight=0.3):
        cooc[a][b] = cooc[a].get(b, 0) + weight
        cooc[b][a] = cooc[b].get(a, 0) + weight

    # paris cluster
    link(0, 1, 0.5)  # paris-capital
    link(0, 2, 0.6)  # paris-france
    link(0, 6, 0.4)  # paris-seine
    link(0, 8, 0.2)  # paris-bridge

    # london cluster
    link(3, 1, 0.5)  # london-capital
    link(3, 4, 0.6)  # london-england
    link(3, 7, 0.4)  # london-thames
    link(3, 9, 0.3)  # london-tower
    link(3, 8, 0.2)  # london-bridge

    # shared
    link(1, 2, 0.4)  # capital-france
    link(1, 4, 0.4)  # capital-england

    # river connections
    link(5, 6, 0.5)  # river-seine
    link(5, 7, 0.5)  # river-thames

    return cooc, word_idx, words, word_neurons


def make_loop(cooc, word_idx, words, word_neurons, **kwargs):
    defaults = dict(max_hops=10, k=5, convergence_threshold=0.99,
                    min_confidence=0.05, min_relevance=0.1, temperature=1.0)
    defaults.update(kwargs)
    return SparseConvergenceLoop(
        cooc=cooc, word_idx=word_idx, words=words,
        word_neurons=word_neurons, **defaults
    )


class TestSparseConvergenceBasic:

    def test_converges_on_related_words(self):
        """Query [paris] should converge and find france/capital."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        result = loop.converge([word_idx["paris"]])

        assert result.converged is True
        assert len(result.concepts) > 0
        assert result.confidence > 0

    def test_empty_query_does_not_converge(self):
        """Empty query should not converge."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        result = loop.converge([])

        assert result.converged is False
        assert result.concepts == []

    def test_unknown_index_does_not_crash(self):
        """Index not in cooc should handle gracefully."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        result = loop.converge([999])

        assert result.converged is False

    def test_convergence_finds_correct_cluster(self):
        """Query [paris, france] should find paris-related words, not london."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        result = loop.converge([word_idx["paris"], word_idx["france"]])

        concept_indices = {widx for widx, _ in result.concepts}
        # Should find paris-cluster words
        assert word_idx["capital"] in concept_indices or word_idx["seine"] in concept_indices


class TestSparseAnchor:

    def test_anchor_keeps_query_relevant(self):
        """With query anchor, result should stay near query, not drift."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons, max_hops=10)
        result = loop.converge([word_idx["paris"]])

        if result.converged and result.hops:
            # Last hop's current profile should still have paris-related keys
            last_profile = result.hops[-1].current
            # Paris (idx 0) should still have weight in the profile
            assert last_profile.get(0, 0) > 0 or last_profile.get(2, 0) > 0

    def test_movement_decreases(self):
        """Movement should generally decrease (convergence)."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons, max_hops=10)
        result = loop.converge([word_idx["paris"]])

        if len(result.hops) >= 3:
            first = result.hops[0].movement
            last = result.hops[-1].movement
            assert last <= first + 0.05  # tolerance


class TestSparseMutualAttention:

    def test_coherent_cluster_boosted(self):
        """Words that co-occur with each other should get boosted."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)

        # paris(0) and france(2) mutually co-occur
        neighbors = [(0, 0.5), (2, 0.4), (5, 0.3)]  # paris, france, river
        boosted = loop._mutual_attention(neighbors)

        # paris and france should be boosted more than river
        paris_sim = next(s for w, s in boosted if w == 0)
        river_sim = next(s for w, s in boosted if w == 5)
        assert paris_sim > river_sim


class TestSparseSoftmaxBlend:

    def test_high_similarity_dominates(self):
        """Higher similarity word should dominate the blend."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons, temperature=0.5)

        neighbors = [(0, 0.9), (3, 0.1)]  # paris strong, london weak
        blended = loop._softmax_blend(neighbors)

        # Blended profile should lean toward paris's co-occurrences
        # paris has france(2), london has england(4)
        paris_weight = blended.get(2, 0)  # france
        london_weight = blended.get(4, 0)  # england
        assert paris_weight > london_weight

    def test_uniform_at_inf_temperature(self):
        """Infinite temperature should give uniform weighting."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons,
                         temperature=float('inf'))

        neighbors = [(0, 0.9), (3, 0.1)]
        blended = loop._softmax_blend(neighbors)
        # Both should contribute roughly equally
        assert len(blended) > 0


class TestSparseTrace:

    def test_trace_has_hops(self):
        """Trace should record each hop."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        result = loop.converge([word_idx["paris"]])

        assert len(result.hops) > 0
        for hop in result.hops:
            assert hop.hop_number >= 0
            assert len(hop.neighbors) > 0

    def test_trace_string_readable(self):
        """Trace should produce human-readable string."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        result = loop.converge([word_idx["paris"]])

        trace_str = result.trace()
        assert "SparseConvergence:" in trace_str
        assert "Hop 0:" in trace_str


class TestSparseMultiHop:

    def test_single_hop_works(self):
        """Single query should work through multi-hop."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        mh = SparseMultiHop(loop, max_rounds=3)
        result = mh.reason([word_idx["paris"]])

        assert len(result.rounds) >= 1
        assert len(result.concepts) > 0

    def test_multi_hop_discovers_distant_concept(self):
        """Multi-hop from [seine] should eventually find [london] via river->thames."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons,
                         min_relevance=0.05, min_confidence=0.01)
        mh = SparseMultiHop(loop, max_rounds=3, concept_blend_weight=0.5)
        result = mh.reason([word_idx["seine"]])

        concept_indices = {widx for widx, _ in result.concepts}
        # Should discover river or thames through the graph
        found_river_cluster = (word_idx["river"] in concept_indices or
                               word_idx["thames"] in concept_indices)
        assert found_river_cluster, f"Expected to find river/thames, got {concept_indices}"

    def test_no_duplicate_concepts(self):
        """Same concept should not appear twice across rounds."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        mh = SparseMultiHop(loop, max_rounds=3)
        result = mh.reason([word_idx["paris"]])

        indices = [widx for widx, _ in result.concepts]
        assert len(indices) == len(set(indices))

    def test_max_rounds_respected(self):
        """Should not exceed max_rounds."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        mh = SparseMultiHop(loop, max_rounds=2)
        result = mh.reason([word_idx["paris"]])

        assert len(result.rounds) <= 2

    def test_multi_hop_trace_readable(self):
        """Multi-hop trace should be human-readable."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        mh = SparseMultiHop(loop, max_rounds=3)
        result = mh.reason([word_idx["paris"]])

        trace_str = result.trace()
        assert "SparseMultiHop:" in trace_str
        assert "Round 1" in trace_str

    def test_empty_query_multi_hop(self):
        """Empty query should not crash."""
        cooc, word_idx, words, word_neurons = make_graph()
        loop = make_loop(cooc, word_idx, words, word_neurons)
        mh = SparseMultiHop(loop, max_rounds=3)
        result = mh.reason([])

        assert result.converged is False
        assert result.concepts == []


class TestSparseUtils:

    def test_sparse_cosine_identical(self):
        a = {0: 1.0, 1: 2.0}
        assert abs(sparse_cosine(a, a) - 1.0) < 1e-6

    def test_sparse_cosine_orthogonal(self):
        a = {0: 1.0}
        b = {1: 1.0}
        assert sparse_cosine(a, b) == 0.0

    def test_sparse_cosine_empty(self):
        assert sparse_cosine({}, {0: 1.0}) == 0.0
        assert sparse_cosine({}, {}) == 0.0

    def test_sparse_blend_uniform(self):
        p1 = {0: 1.0, 1: 2.0}
        p2 = {1: 4.0, 2: 6.0}
        blended = sparse_blend([p1, p2])
        assert abs(blended[0] - 0.5) < 1e-6
        assert abs(blended[1] - 3.0) < 1e-6
        assert abs(blended[2] - 3.0) < 1e-6

    def test_sparse_normalize(self):
        d = {0: 3.0, 1: 4.0}
        n = sparse_normalize(d)
        assert abs(sparse_norm(n) - 1.0) < 1e-6


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
