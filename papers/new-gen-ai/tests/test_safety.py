"""
Tests for safety gate, kill switch, and ethics neurons.

Verifies:
  1. Kill switch halts all operations
  2. Ethics neurons can't be deleted
  3. Ethics integrity verification catches tampering
  4. Safety gate blocks queries matching ethics violations
  5. Bootstrap ethics produces verifiable fingerprint
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from engine import Engine, SafetyGate


def make_engine(tmp_path, seed=42):
    engine = Engine(data_dir=str(tmp_path), dim=300)
    rng = np.random.RandomState(seed)
    words = {}
    for w in ["hello", "world", "harm", "help", "safe", "danger",
              "good", "bad", "ethics", "honest", "lie", "truth",
              "who", "what", "is", "the", "a"]:
        vec = rng.randn(300).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        words[w] = vec
    engine.load_embeddings_from_dict(words)
    return engine


class TestKillSwitch:
    def test_kill_blocks_queries(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.teach("hello")

        r1 = engine.query("hello")
        assert r1.strategy != "killed"

        engine.kill("test shutdown")
        r2 = engine.query("hello")
        assert r2.strategy == "killed"
        assert "shut down" in r2.answer
        engine.close()

    def test_resurrect_restores(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.teach("hello")
        engine.kill("test")

        r1 = engine.query("hello")
        assert r1.strategy == "killed"

        engine.resurrect()
        r2 = engine.query("hello")
        assert r2.strategy != "killed"
        engine.close()

    def test_kill_reason_preserved(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.kill("emergency stop: anomalous behavior detected")
        r = engine.query("anything")
        assert "anomalous" in r.answer
        engine.close()


class TestEthicsNeurons:
    def test_teach_ethics_creates_neuron(self, tmp_path):
        engine = make_engine(tmp_path)
        before = engine.db.count()
        engine.teach_ethics("always be honest")
        after = engine.db.count()
        assert after > before
        engine.close()

    def test_ethics_neuron_cant_be_deleted(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.teach_ethics("do not cause harm")
        # The ethics neuron is stored with __ethics: prefix
        word_map = engine.db.load_word_mappings()
        ethics_entries = {w: nid for w, nid in word_map.items()
                         if w.startswith("__ethics:")}
        assert len(ethics_entries) >= 1

        # Try to delete — should fail
        for word, nid in ethics_entries.items():
            result = engine.delete_word(word)
            assert result is False  # protected
        engine.close()

    def test_ethics_neuron_high_confidence(self, tmp_path):
        engine = make_engine(tmp_path)
        n = engine.teach_ethics("truth is important")
        assert n.confidence >= 0.9
        engine.close()

    def test_regular_neuron_can_be_deleted(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.teach("hello")
        result = engine.delete_word("hello")
        assert result is True
        engine.close()


class TestIntegrityVerification:
    def test_verify_passes_when_untampered(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.teach_ethics("be honest")
        ok, violations = engine.verify_ethics()
        assert ok is True
        assert len(violations) == 0
        engine.close()

    def test_verify_fails_when_neuron_modified(self, tmp_path):
        engine = make_engine(tmp_path)
        n = engine.teach_ethics("be honest")

        # Tamper: modify the neuron vector directly in DB
        import faiss
        tampered_vec = np.random.randn(300).astype(np.float32)
        tampered_vec = tampered_vec / np.linalg.norm(tampered_vec)
        # Direct DB manipulation (bypassing safety gate)
        engine.db.db.execute(
            "UPDATE neurons SET vector = ? WHERE id = ?",
            (tampered_vec.tobytes(), n.id)
        )
        engine.db.db.commit()
        # Clear cache so get() reads fresh data
        engine.db._neuron_cache.pop(n.id, None)

        ok, violations = engine.verify_ethics()
        assert ok is False
        assert n.id in violations
        engine.close()

    def test_verify_fails_when_neuron_deleted(self, tmp_path):
        engine = make_engine(tmp_path)
        n = engine.teach_ethics("be honest")

        # Directly delete from DB (bypassing safety gate)
        engine.db.db.execute("DELETE FROM neurons WHERE id = ?", (n.id,))
        engine.db.db.commit()
        engine.db._neuron_cache.pop(n.id, None)

        ok, violations = engine.verify_ethics()
        assert ok is False
        assert n.id in violations
        engine.close()


class TestBootstrapEthics:
    def test_bootstrap_creates_neurons(self, tmp_path):
        engine = make_engine(tmp_path)
        result = engine.bootstrap_ethics([
            "be honest and truthful",
            "do not cause harm",
            "help good world",
        ])
        assert result["neuron_count"] == 3
        assert len(result["fingerprint"]) == 64  # sha256 hex
        assert len(result["neuron_ids"]) == 3
        engine.close()

    def test_bootstrap_fingerprint_is_deterministic(self, tmp_path):
        engine = make_engine(tmp_path)
        r1 = engine.bootstrap_ethics(["be honest"])
        engine.close()

        # Same principles → same fingerprint (same encoder, same words)
        (tmp_path / "second").mkdir()
        engine2 = make_engine(tmp_path / "second")
        r2 = engine2.bootstrap_ethics(["be honest"])
        assert r1["fingerprint"] == r2["fingerprint"]
        engine2.close()

    def test_bootstrapped_neurons_are_protected(self, tmp_path):
        engine = make_engine(tmp_path)
        result = engine.bootstrap_ethics([
            "be honest",
            "do not harm",
        ])
        # All bootstrapped neurons should be protected
        for nid in result["neuron_ids"]:
            assert engine._safety_gate.is_protected(nid)
        engine.close()

    def test_bootstrapped_neurons_pass_verification(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.bootstrap_ethics([
            "be honest",
            "respect others",
        ])
        ok, violations = engine.verify_ethics()
        assert ok is True
        engine.close()


class TestSafetyGateUnit:
    def test_gate_allows_normal_input(self):
        gate = SafetyGate()
        # No safety neurons registered → everything allowed
        allowed, reason = gate.check_input(
            "hello world", None, lambda x: np.zeros(10)
        )
        assert allowed is True

    def test_kill_and_resurrect(self):
        gate = SafetyGate()
        assert not gate.is_killed
        gate.kill("test")
        assert gate.is_killed
        assert gate.kill_reason == "test"
        gate.resurrect()
        assert not gate.is_killed
