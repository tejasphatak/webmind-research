"""
Tests for persistence: neurons, templates, word mappings, FAISS index.

Verifies:
- Create engine, teach facts, close → reopen → everything survives
- Templates persist and reload correctly
- Word→neuron mappings persist
- FAISS index saves/loads (fast boot)
- Delete persists across restarts
- In-memory mode still works (no path = no persistence)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from neuron import NeuronDB, VECTOR_DIM
from generator import TemplateStore
from encoder import Encoder

DIM = 300


def random_vector(seed: int, dim=DIM) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestNeuronPersistence:

    def test_neurons_survive_restart(self):
        """Insert neurons, close DB, reopen → neurons still there."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: insert
            db = NeuronDB(path=tmpdir, dim=DIM)
            v1 = random_vector(1)
            v2 = random_vector(2)
            n1 = db.insert(v1, confidence=0.7)
            n2 = db.insert(v2, confidence=0.3)
            db.close()

            # Session 2: reopen
            db2 = NeuronDB(path=tmpdir, dim=DIM)
            assert db2.count() == 2

            reloaded1 = db2.get(n1.id)
            assert reloaded1 is not None
            assert abs(reloaded1.confidence - 0.7) < 0.001

            reloaded2 = db2.get(n2.id)
            assert reloaded2 is not None
            assert abs(reloaded2.confidence - 0.3) < 0.001
            db2.close()

    def test_successors_persist(self):
        """Successor relationships survive restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = NeuronDB(path=tmpdir, dim=DIM)
            n1 = db.insert(random_vector(1), confidence=0.5)
            n2 = db.insert(random_vector(2), confidence=0.5)
            db.update_successors(n1.id, n2.id, 0.8)
            db.update_predecessors(n2.id, n1.id)
            db.close()

            db2 = NeuronDB(path=tmpdir, dim=DIM)
            r1 = db2.get(n1.id)
            r2 = db2.get(n2.id)
            assert len(r1.successors) == 1
            assert r1.successors[0][0] == n2.id
            assert abs(r1.successors[0][1] - 0.8) < 0.001  # float32 precision
            assert n1.id in r2.predecessors
            db2.close()

    def test_confidence_updates_persist(self):
        """Confidence changes survive restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = NeuronDB(path=tmpdir, dim=DIM)
            n = db.insert(random_vector(1), confidence=0.5)
            db.update_confidence(n.id, useful=True)
            db.close()

            db2 = NeuronDB(path=tmpdir, dim=DIM)
            r = db2.get(n.id)
            assert r.confidence > 0.5  # was reinforced
            db2.close()

    def test_delete_persists(self):
        """Deleted neurons stay deleted after restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = NeuronDB(path=tmpdir, dim=DIM)
            n1 = db.insert(random_vector(1), confidence=0.5)
            n2 = db.insert(random_vector(2), confidence=0.5)
            db.delete(n1.id)
            assert db.count() == 1
            db.close()

            db2 = NeuronDB(path=tmpdir, dim=DIM)
            assert db2.count() == 1
            assert db2.get(n1.id) is None
            assert db2.get(n2.id) is not None
            db2.close()


class TestFAISSPersistence:

    def test_faiss_index_saves_and_loads(self):
        """FAISS index file should be created on close and loaded on open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = NeuronDB(path=tmpdir, dim=DIM)
            v1 = random_vector(1)
            v2 = random_vector(2)
            db.insert(v1, confidence=0.7)
            db.insert(v2, confidence=0.3)
            db.close()

            # Check that the index file exists
            index_path = Path(tmpdir) / "neurons.faiss"
            assert index_path.exists(), "FAISS index file should be saved on close"

            # Reopen — should load from file, not rebuild
            db2 = NeuronDB(path=tmpdir, dim=DIM)
            assert db2.count() == 2

            # Search should work
            results = db2.search(v1, k=2)
            assert len(results) == 2
            db2.close()

    def test_search_works_after_reload(self):
        """Spatial search should return correct results after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = NeuronDB(path=tmpdir, dim=DIM)
            v_target = random_vector(42)
            v_far = random_vector(99)
            n_target = db.insert(v_target, confidence=0.7)
            db.insert(v_far, confidence=0.5)
            db.close()

            db2 = NeuronDB(path=tmpdir, dim=DIM)
            results = db2.search(v_target, k=1)
            assert len(results) == 1
            assert results[0].id == n_target.id
            db2.close()


class TestTemplatePersistence:

    def _make_encoder(self, dim=DIM):
        """Create an encoder with test embeddings."""
        enc = Encoder(data_dir="/tmp/test-enc", dim=dim)
        enc.load_from_dict({
            "wrote": random_vector(10, dim),
            "discovered": random_vector(11, dim),
            "is": random_vector(12, dim),
        })
        return enc

    def test_templates_survive_restart(self):
        """Templates saved via TemplateStore should reload from SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enc = self._make_encoder()

            # Session 1: add templates
            db = NeuronDB(path=tmpdir, dim=DIM)
            store = TemplateStore(enc, db=db)
            store.add(
                "[PERSON] wrote [WORK]",
                {"PERSON": "noun", "WORK": "noun"},
                confidence=0.8,
            )
            store.add(
                "[PERSON] discovered [CONCEPT]",
                {"PERSON": "noun", "CONCEPT": "noun"},
                confidence=0.7,
            )
            assert store.count() == 2
            db.close()

            # Session 2: reopen
            db2 = NeuronDB(path=tmpdir, dim=DIM)
            store2 = TemplateStore(enc, db=db2)
            assert store2.count() == 2

            t0 = store2.templates[0]
            assert t0.pattern == "[PERSON] wrote [WORK]"
            assert t0.slots == {"PERSON": "noun", "WORK": "noun"}
            assert abs(t0.confidence - 0.8) < 0.001

            t1 = store2.templates[1]
            assert t1.pattern == "[PERSON] discovered [CONCEPT]"
            db2.close()

    def test_template_delete_persists(self):
        """Deleted templates stay deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enc = self._make_encoder()

            db = NeuronDB(path=tmpdir, dim=DIM)
            store = TemplateStore(enc, db=db)
            t = store.add("[X] is [Y]", {"X": "noun", "Y": "noun"}, 0.5)
            assert store.count() == 1
            store.delete(t.id)
            assert store.count() == 0
            db.close()

            db2 = NeuronDB(path=tmpdir, dim=DIM)
            store2 = TemplateStore(enc, db=db2)
            assert store2.count() == 0
            db2.close()

    def test_in_memory_templates_no_crash(self):
        """TemplateStore without DB should work (in-memory only)."""
        enc = self._make_encoder()
        store = TemplateStore(enc, db=None)
        store.add("[A] is [B]", {"A": "noun", "B": "noun"}, 0.5)
        assert store.count() == 1

    def test_template_search_after_reload(self):
        """Template search should work correctly after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enc = self._make_encoder()

            db = NeuronDB(path=tmpdir, dim=DIM)
            store = TemplateStore(enc, db=db)
            store.add(
                "[PERSON] wrote [WORK]",
                {"PERSON": "noun", "WORK": "noun"}, 0.8,
            )
            db.close()

            db2 = NeuronDB(path=tmpdir, dim=DIM)
            store2 = TemplateStore(enc, db=db2)
            # Search with the "wrote" vector
            wrote_vec = enc.encode_word("wrote")
            results = store2.search(wrote_vec, k=1)
            assert len(results) == 1
            assert results[0].pattern == "[PERSON] wrote [WORK]"
            db2.close()


class TestWordMappingPersistence:

    def test_word_mappings_survive_restart(self):
        """Word→neuron mappings persist across restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = NeuronDB(path=tmpdir, dim=DIM)
            n1 = db.insert(random_vector(1), confidence=0.5)
            n2 = db.insert(random_vector(2), confidence=0.5)
            db.save_word_mapping("shakespeare", n1.id)
            db.save_word_mapping("hamlet", n2.id)
            db.close()

            db2 = NeuronDB(path=tmpdir, dim=DIM)
            mappings = db2.load_word_mappings()
            assert mappings["shakespeare"] == n1.id
            assert mappings["hamlet"] == n2.id
            db2.close()

    def test_word_mapping_delete_persists(self):
        """Deleted word mappings stay deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = NeuronDB(path=tmpdir, dim=DIM)
            n = db.insert(random_vector(1), confidence=0.5)
            db.save_word_mapping("test", n.id)
            db.delete_word_mapping("test")
            db.close()

            db2 = NeuronDB(path=tmpdir, dim=DIM)
            mappings = db2.load_word_mappings()
            assert "test" not in mappings
            db2.close()

    def test_word_mapping_update(self):
        """Updating a word mapping replaces the old one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = NeuronDB(path=tmpdir, dim=DIM)
            n1 = db.insert(random_vector(1), confidence=0.5)
            n2 = db.insert(random_vector(2), confidence=0.5)
            db.save_word_mapping("word", n1.id)
            db.save_word_mapping("word", n2.id)  # overwrite
            db.close()

            db2 = NeuronDB(path=tmpdir, dim=DIM)
            mappings = db2.load_word_mappings()
            assert mappings["word"] == n2.id
            db2.close()


class TestEndToEndPersistence:

    def test_full_engine_persistence(self):
        """
        Complete test: create engine with data_dir, teach facts and templates,
        close, reopen → everything is still there.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            enc_vectors = {
                "shakespeare": random_vector(1),
                "wrote": random_vector(2),
                "hamlet": random_vector(3),
                "einstein": random_vector(4),
                "discovered": random_vector(5),
                "relativity": random_vector(6),
            }

            # Session 1: teach
            from engine import Engine
            engine = Engine(data_dir=tmpdir, dim=DIM)
            engine.load_embeddings_from_dict(enc_vectors)
            engine.teach_sentence("shakespeare wrote hamlet")
            engine.teach_template(
                "[PERSON] wrote [WORK]",
                {"PERSON": "noun", "WORK": "noun"}, 0.8,
            )
            stats1 = engine.stats()
            engine.close()

            # Session 2: reopen
            engine2 = Engine(data_dir=tmpdir, dim=DIM)
            engine2.load_embeddings_from_dict(enc_vectors)
            stats2 = engine2.stats()

            assert stats2["neurons"] == stats1["neurons"], \
                f"Neurons: {stats2['neurons']} != {stats1['neurons']}"
            assert stats2["templates"] == stats1["templates"], \
                f"Templates: {stats2['templates']} != {stats1['templates']}"

            # Word mappings should be loaded
            assert "shakespeare" in engine2._word_neurons
            assert "hamlet" in engine2._word_neurons

            # Search should still work
            results = engine2.db.search(enc_vectors["shakespeare"], k=1)
            assert len(results) >= 1

            engine2.close()

    def test_incremental_teaching(self):
        """Multiple sessions of teaching should accumulate knowledge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enc_vectors = {
                "cat": random_vector(10),
                "sat": random_vector(11),
                "mat": random_vector(12),
                "dog": random_vector(13),
                "ran": random_vector(14),
            }

            # Session 1
            from engine import Engine
            e1 = Engine(data_dir=tmpdir, dim=DIM)
            e1.load_embeddings_from_dict(enc_vectors)
            e1.teach_sentence("cat sat mat")
            count1 = e1.db.count()
            e1.close()

            # Session 2: teach more
            e2 = Engine(data_dir=tmpdir, dim=DIM)
            e2.load_embeddings_from_dict(enc_vectors)
            assert e2.db.count() == count1  # previous neurons still there
            e2.teach_sentence("dog ran")
            assert e2.db.count() > count1  # new neurons added
            e2.close()

            # Session 3: verify all
            e3 = Engine(data_dir=tmpdir, dim=DIM)
            e3.load_embeddings_from_dict(enc_vectors)
            assert "cat" in e3._word_neurons
            assert "dog" in e3._word_neurons
            e3.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
