"""
Tests for MVP-3: Self-evolution from query failures.

Tests that the system:
  1. Logs misses when it says "I don't know"
  2. Learns from corrections (correct() teaches words + template)
  3. Answers correctly after correction (the evolution loop)
  4. Tracks evolution stats accurately
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from engine import Engine
from neuron import NeuronDB


# --- Fixtures ---

def make_engine(tmp_path):
    """Create an engine with a small test vocabulary."""
    engine = Engine(data_dir=str(tmp_path), dim=300)

    # Build a small vocabulary with distinct vectors
    words = {
        "shakespeare": np.random.RandomState(1).randn(300).astype(np.float32),
        "wrote": np.random.RandomState(2).randn(300).astype(np.float32),
        "hamlet": np.random.RandomState(3).randn(300).astype(np.float32),
        "who": np.random.RandomState(4).randn(300).astype(np.float32),
        "einstein": np.random.RandomState(5).randn(300).astype(np.float32),
        "discovered": np.random.RandomState(6).randn(300).astype(np.float32),
        "relativity": np.random.RandomState(7).randn(300).astype(np.float32),
        "what": np.random.RandomState(8).randn(300).astype(np.float32),
        "is": np.random.RandomState(9).randn(300).astype(np.float32),
        "the": np.random.RandomState(10).randn(300).astype(np.float32),
        "capital": np.random.RandomState(11).randn(300).astype(np.float32),
        "of": np.random.RandomState(12).randn(300).astype(np.float32),
        "france": np.random.RandomState(13).randn(300).astype(np.float32),
        "paris": np.random.RandomState(14).randn(300).astype(np.float32),
        "python": np.random.RandomState(15).randn(300).astype(np.float32),
        "programming": np.random.RandomState(16).randn(300).astype(np.float32),
        "language": np.random.RandomState(17).randn(300).astype(np.float32),
    }
    # Normalize
    for w in words:
        words[w] = words[w] / (np.linalg.norm(words[w]) + 1e-10)

    engine.load_embeddings_from_dict(words)
    return engine


# --- NeuronDB miss logging tests ---

class TestMissLogging:
    def test_log_miss(self, tmp_path):
        db = NeuronDB(path=str(tmp_path), dim=300)
        vec = np.random.randn(300).astype(np.float32)
        mid = db.log_miss("who wrote hamlet", vec)
        assert mid is not None
        assert mid > 0
        db.close()

    def test_unresolved_misses(self, tmp_path):
        db = NeuronDB(path=str(tmp_path), dim=300)
        vec = np.random.randn(300).astype(np.float32)
        db.log_miss("who wrote hamlet", vec)
        db.log_miss("what is gravity", vec)

        misses = db.get_unresolved_misses()
        assert len(misses) == 2
        texts = {m[1] for m in misses}
        assert texts == {"who wrote hamlet", "what is gravity"}
        db.close()

    def test_resolve_miss(self, tmp_path):
        db = NeuronDB(path=str(tmp_path), dim=300)
        vec = np.random.randn(300).astype(np.float32)
        mid = db.log_miss("who wrote hamlet", vec)

        db.resolve_miss(mid, "shakespeare wrote hamlet")
        misses = db.get_unresolved_misses()
        assert len(misses) == 0
        db.close()

    def test_resolve_miss_by_query(self, tmp_path):
        db = NeuronDB(path=str(tmp_path), dim=300)
        vec = np.random.randn(300).astype(np.float32)
        db.log_miss("who wrote hamlet", vec)

        found = db.resolve_miss_by_query("who wrote hamlet", "shakespeare")
        assert found is True

        not_found = db.resolve_miss_by_query("nonexistent query", "answer")
        assert not_found is False
        db.close()

    def test_miss_stats(self, tmp_path):
        db = NeuronDB(path=str(tmp_path), dim=300)
        vec = np.random.randn(300).astype(np.float32)

        # Empty
        stats = db.miss_stats()
        assert stats["total_misses"] == 0
        assert stats["resolution_rate"] == 0.0

        # Two misses, resolve one
        db.log_miss("q1", vec)
        mid2 = db.log_miss("q2", vec)
        db.resolve_miss(mid2, "a2")

        stats = db.miss_stats()
        assert stats["total_misses"] == 2
        assert stats["resolved"] == 1
        assert stats["unresolved"] == 1
        assert stats["resolution_rate"] == 0.5
        db.close()

    def test_miss_persists_across_restart(self, tmp_path):
        db = NeuronDB(path=str(tmp_path), dim=300)
        vec = np.random.randn(300).astype(np.float32)
        db.log_miss("who wrote hamlet", vec)
        db.close()

        # Reopen
        db2 = NeuronDB(path=str(tmp_path), dim=300)
        misses = db2.get_unresolved_misses()
        assert len(misses) == 1
        assert misses[0][1] == "who wrote hamlet"
        db2.close()


# --- Engine self-evolution tests ---

class TestSelfEvolution:
    def test_query_miss_is_logged(self, tmp_path):
        """Query with no neurons in KB → abstain → miss logged."""
        engine = make_engine(tmp_path)
        # KB is empty, so any query should miss
        result = engine.query("who wrote hamlet")
        assert result.strategy == "abstain"

        misses = engine.misses()
        assert len(misses) == 1
        assert misses[0][1] == "who wrote hamlet"
        engine.close()

    def test_correct_teaches_neurons(self, tmp_path):
        """correct() teaches answer words as neurons."""
        engine = make_engine(tmp_path)
        # Miss first
        engine.query("who wrote hamlet")

        # Correct
        result = engine.correct("who wrote hamlet", "shakespeare wrote hamlet")
        assert result["neurons_learned"] >= 2  # at least shakespeare, wrote, hamlet
        assert "shakespeare" in result["words"]
        assert "hamlet" in result["words"]
        engine.close()

    def test_correct_resolves_miss(self, tmp_path):
        """correct() marks the miss as resolved."""
        engine = make_engine(tmp_path)
        engine.query("who wrote hamlet")
        assert len(engine.misses()) == 1

        engine.correct("who wrote hamlet", "shakespeare wrote hamlet")
        assert len(engine.misses()) == 0
        engine.close()

    def test_correct_creates_template(self, tmp_path):
        """correct() auto-generates a template from the Q→A pair."""
        engine = make_engine(tmp_path)
        engine.query("who wrote hamlet")
        result = engine.correct("who wrote hamlet", "shakespeare wrote hamlet")

        # Should create a template — "shakespeare" is not in query,
        # so it becomes a slot
        assert result["template_created"] is True
        assert result["template_pattern"] is not None
        engine.close()

    def test_evolution_loop_query_succeeds_after_correction(self, tmp_path):
        """The full loop: miss → correct → re-query succeeds."""
        engine = make_engine(tmp_path)

        # First query: miss
        r1 = engine.query("who wrote hamlet")
        assert r1.strategy == "abstain"

        # Correct it
        engine.correct("who wrote hamlet", "shakespeare wrote hamlet")

        # Re-query: should find shakespeare/hamlet neurons now
        r2 = engine.query("who wrote hamlet")
        assert r2.strategy != "abstain"
        assert r2.confidence > 0
        # The answer should mention shakespeare or hamlet
        answer_lower = r2.answer.lower()
        assert "shakespeare" in answer_lower or "hamlet" in answer_lower
        engine.close()

    def test_multiple_corrections_improve_stats(self, tmp_path):
        """Evolution stats track learning over time."""
        engine = make_engine(tmp_path)

        # Three misses
        engine.query("who wrote hamlet")
        engine.query("what is relativity")
        engine.query("capital of france")

        stats = engine.evolution_stats()
        assert stats["total_misses"] == 3
        assert stats["resolved"] == 0
        assert stats["learning_rate"] == 0.0

        # Correct two
        engine.correct("who wrote hamlet", "shakespeare wrote hamlet")
        engine.correct("capital of france", "paris is the capital of france")

        stats = engine.evolution_stats()
        assert stats["total_misses"] == 3
        assert stats["resolved"] == 2
        assert stats["unresolved"] == 1
        assert abs(stats["learning_rate"] - 2/3) < 0.01
        engine.close()

    def test_correct_without_prior_miss(self, tmp_path):
        """correct() works even without a logged miss — proactive teaching."""
        engine = make_engine(tmp_path)
        result = engine.correct(
            "what is python",
            "python is a programming language"
        )
        assert result["neurons_learned"] >= 2
        # No miss to resolve, but doesn't error
        assert len(engine.misses()) == 0
        engine.close()

    def test_evolution_stats_includes_base_stats(self, tmp_path):
        """evolution_stats() includes neuron count, template count, etc."""
        engine = make_engine(tmp_path)
        engine.query("who wrote hamlet")
        engine.correct("who wrote hamlet", "shakespeare wrote hamlet")

        stats = engine.evolution_stats()
        assert "neurons" in stats
        assert "templates" in stats
        assert "total_misses" in stats
        assert stats["neurons"] > 0
        engine.close()
