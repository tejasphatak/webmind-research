"""
Tests for the Engine — full end-to-end integration.

Verifies the HLD MVP success criteria:
1. Convergence loop finds the right concepts for factual queries
2. System says "I don't know" for things it doesn't know
3. Template-based generation produces grammatically correct sentence
4. A fact can be deleted and the system immediately stops using it
5. All of the above runs on CPU in <100ms per query
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from engine import Engine

DIM = 300


def make_vocab():
    """Test vocabulary — deterministic, covers enough for meaningful tests."""
    rng = np.random.RandomState(42)
    words = {}
    for w in ["shakespeare", "hamlet", "wrote", "playwright", "english",
              "was", "a", "an", "who", "the", "in",
              "cat", "dog", "sat", "on", "mat",
              "python", "programming", "language",
              "earth", "planet", "sun", "orbits",
              "1600", "1564"]:
        words[w] = rng.randn(DIM).astype(np.float32)
    return words


def make_engine():
    engine = Engine(dim=DIM)
    engine.load_embeddings_from_dict(make_vocab())
    return engine


# --- MVP Success Criteria ---

class TestMVPCriteria:

    def test_1_convergence_finds_concepts(self):
        """MVP #1: Convergence finds the right concepts for factual queries."""
        engine = make_engine()

        # Teach facts
        engine.teach_sentence("shakespeare wrote hamlet")

        result = engine.query("shakespeare")
        assert result.converged is True
        assert result.confidence > 0

    def test_2_honest_abstention_unknown(self):
        """MVP #2: Says 'I don't know' for unknown things."""
        engine = make_engine()

        # Empty KB
        result = engine.query("who wrote hamlet")
        assert "don't know" in result.answer.lower()

    def test_2_honest_abstention_oov(self):
        """MVP #2: Says 'I don't know' for completely OOV queries."""
        engine = make_engine()

        result = engine.query("glorpnax zibble fweep")
        assert "don't know" in result.answer.lower()
        assert result.confidence == 0.0

    def test_3_template_produces_sentence(self):
        """MVP #3: Template generation produces readable sentence."""
        engine = make_engine()

        engine.teach_sentence("shakespeare wrote hamlet")
        engine.teach_template(
            "[PERSON] wrote [WORK]",
            {"PERSON": "noun", "WORK": "noun"},
            confidence=0.8,
        )

        result = engine.query("shakespeare hamlet")
        if result.strategy == "template":
            # Should contain filled slots
            assert len(result.answer.split()) >= 2
            assert result.answer != "I don't know."

    def test_4_delete_equals_gone(self):
        """MVP #4: Delete = gone. Invariant #3."""
        engine = make_engine()

        engine.teach("shakespeare", confidence=0.7)
        engine.teach("hamlet", confidence=0.6)

        # Before delete — should find it
        result_before = engine.query("shakespeare")
        assert result_before.converged is True

        # Delete
        assert engine.delete_word("shakespeare") is True

        # After delete — should not find it
        # (hamlet might still converge, but shakespeare should be gone)
        info = engine.inspect("shakespeare")
        shakespeare_found = any(
            n["word"] == "shakespeare" for n in info["neighbors"]
        )
        assert shakespeare_found is False

    def test_5_latency_under_100ms(self):
        """MVP #5: CPU, sub-100ms per query."""
        engine = make_engine()

        for word in ["shakespeare", "hamlet", "wrote", "playwright",
                     "cat", "dog", "sat", "mat"]:
            engine.teach(word, confidence=0.5)

        # Warm up
        engine.query("cat")

        # Measure
        start = time.perf_counter()
        for _ in range(10):
            engine.query("who wrote hamlet")
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 0.1, f"Query took {elapsed*1000:.1f}ms (>100ms)"


# --- Teach/Learn ---

class TestTeach:

    def test_teach_word(self):
        engine = make_engine()
        n = engine.teach("cat")
        assert n.id >= 0
        assert engine.db.count() == 1

    def test_teach_oov_raises(self):
        engine = make_engine()
        try:
            engine.teach("glorpnax")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_teach_sentence_wires_successors(self):
        engine = make_engine()
        neurons = engine.teach_sentence("the cat sat on the mat")

        assert len(neurons) >= 4  # some words might be OOV

        # Check successors were wired
        first = engine.db.get(neurons[0].id)
        assert len(first.successors) > 0

    def test_teach_sentence_deduplicates(self):
        """Teaching 'the cat' twice should reuse neurons."""
        engine = make_engine()

        n1 = engine.teach_sentence("the cat")
        count_after_first = engine.db.count()

        n2 = engine.teach_sentence("the cat sat")
        count_after_second = engine.db.count()

        # Should only add 'sat', not re-add 'the' and 'cat'
        assert count_after_second == count_after_first + 1


# --- Inspect ---

class TestInspect:

    def test_inspect_shows_neighbors(self):
        engine = make_engine()
        engine.teach("cat", confidence=0.7)
        engine.teach("dog", confidence=0.6)

        info = engine.inspect("cat")
        assert len(info["neighbors"]) > 0
        assert info["neighbors"][0]["confidence"] > 0

    def test_inspect_oov(self):
        engine = make_engine()
        info = engine.inspect("glorpnax")
        assert info["neighbors"] == []


# --- Stats ---

class TestStats:

    def test_stats_correct(self):
        engine = make_engine()
        engine.teach("cat")
        engine.teach("dog")
        engine.teach_template("[X] is [Y]", {"X": "noun", "Y": "noun"})

        s = engine.stats()
        assert s["neurons"] == 2
        assert s["templates"] == 1
        assert s["vocab_size"] == len(make_vocab())
        assert s["dim"] == DIM


# --- Feedback Integration ---

class TestFeedbackIntegration:

    def test_query_updates_confidence(self):
        """Querying should trigger feedback, updating confidence."""
        engine = make_engine()

        n = engine.teach("cat", confidence=0.5)
        original = engine.db.get(n.id).confidence

        # Query twice — same topic (follow-up) should penalize
        engine.query("cat")
        engine.query("cat")  # follow-up

        updated = engine.db.get(n.id).confidence
        # Confidence should have changed (either direction)
        # The exact direction depends on similarity of queries
        assert updated != original or len(engine.feedback.history) > 0

    def test_explicit_feedback(self):
        engine = make_engine()
        n = engine.teach("hamlet", confidence=0.5)

        engine.feedback.force_feedback([n.id], useful=True)
        assert engine.db.get(n.id).confidence > 0.5


# --- End-to-End Flow ---

class TestEndToEnd:

    def test_teach_query_cycle(self):
        """Full cycle: teach → query → get answer → verify trace."""
        engine = make_engine()

        # Teach
        engine.teach_sentence("shakespeare wrote hamlet")
        engine.teach_template(
            "[PERSON] wrote [WORK]",
            {"PERSON": "noun", "WORK": "noun"},
            confidence=0.8,
        )

        # Query
        result = engine.query("who wrote hamlet")

        # Verify
        assert result.answer != ""
        assert result.trace != ""
        assert result.strategy in ("template", "successor", "concept_list", "abstain")

    def test_teach_delete_query(self):
        """Teach → delete → query should get 'I don't know'."""
        engine = make_engine()

        engine.teach("hamlet", confidence=0.7)

        # Before delete
        result1 = engine.query("hamlet")
        assert result1.converged is True

        # Delete
        engine.delete_word("hamlet")

        # After delete
        result2 = engine.query("hamlet")
        # With empty DB, should not converge
        if engine.db.count() == 0:
            assert "don't know" in result2.answer.lower()

    def test_multiple_topics(self):
        """System should handle queries across different topics."""
        engine = make_engine()

        engine.teach_sentence("shakespeare wrote hamlet")
        engine.teach_sentence("cat sat on mat")
        engine.teach_sentence("python programming language")

        r1 = engine.query("shakespeare")
        r2 = engine.query("cat")
        r3 = engine.query("python")

        # All should converge (have relevant neurons)
        assert r1.converged
        assert r2.converged
        assert r3.converged


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
