"""
Tests for the Neuron and NeuronDB.

Verifies against HLD spec:
- Neuron: vector(384) + confidence + successors(10) + predecessors(3) + timestamp + temporal + level
- Confidence: ×1.1 on useful, ×0.9 on useless, capped at ±0.8
- Successors: top-K=10 eviction (new competes for slot, replaces lowest)
- Predecessors: top-3 most recent
- Delete = gone (invariant #3)
- Proximity = connection (spatial search finds nearest)
- Temporal neurons decay with age
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from neuron import (
    CONFIDENCE_BOOST, CONFIDENCE_CAP, CONFIDENCE_DECAY, CONFIDENCE_FLOOR,
    DEFAULT_CONFIDENCE, MAX_PREDECESSORS, MAX_SUCCESSORS, VECTOR_DIM,
    Level, Neuron, NeuronDB,
)


def make_vector(seed: int = 0) -> np.ndarray:
    """Generate a deterministic 384-dim vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(VECTOR_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# --- Neuron dataclass tests ---

class TestNeuronUnit:

    def test_default_fields(self):
        v = make_vector(0)
        n = Neuron(id=0, vector=v)
        assert n.confidence == DEFAULT_CONFIDENCE
        assert n.successors == []
        assert n.predecessors == []
        assert n.temporal is False
        assert n.level == Level.WORD

    def test_reinforce_increases_confidence(self):
        n = Neuron(id=0, vector=make_vector(0), confidence=0.5)
        n.reinforce()
        assert abs(n.confidence - 0.5 * CONFIDENCE_BOOST) < 1e-6

    def test_reinforce_capped_at_max(self):
        n = Neuron(id=0, vector=make_vector(0), confidence=0.79)
        n.reinforce()  # 0.79 * 1.1 = 0.869 > 0.8
        assert n.confidence == CONFIDENCE_CAP

    def test_weaken_decreases_confidence(self):
        n = Neuron(id=0, vector=make_vector(0), confidence=0.5)
        n.weaken()
        assert abs(n.confidence - 0.5 * CONFIDENCE_DECAY) < 1e-6

    def test_weaken_floored(self):
        n = Neuron(id=0, vector=make_vector(0), confidence=-0.75)
        n.weaken()  # -0.75 * 0.9 = -0.675, but floor check
        # Actually -0.75 * 0.9 = -0.675 which is > -0.8, so no floor hit
        assert n.confidence > CONFIDENCE_FLOOR

    def test_weaken_at_floor(self):
        n = Neuron(id=0, vector=make_vector(0), confidence=CONFIDENCE_FLOOR)
        n.weaken()  # -0.8 * 0.9 = -0.72, which is > -0.8
        assert n.confidence >= CONFIDENCE_FLOOR

    def test_successor_add(self):
        n = Neuron(id=0, vector=make_vector(0))
        n.add_successor(1, 0.9)
        n.add_successor(2, 0.8)
        assert len(n.successors) == 2
        assert (1, 0.9) in n.successors

    def test_successor_update_existing(self):
        n = Neuron(id=0, vector=make_vector(0))
        n.add_successor(1, 0.5)
        n.add_successor(1, 0.9)  # update, not duplicate
        assert len(n.successors) == 1
        assert n.successors[0] == (1, 0.9)

    def test_successor_eviction_at_max(self):
        n = Neuron(id=0, vector=make_vector(0))
        # Fill to MAX_SUCCESSORS
        for i in range(MAX_SUCCESSORS):
            n.add_successor(i, 0.1 * (i + 1))

        assert len(n.successors) == MAX_SUCCESSORS

        # Add one with higher confidence than the lowest
        n.add_successor(99, 0.95)
        assert len(n.successors) == MAX_SUCCESSORS
        # Lowest (id=0, conf=0.1) should be evicted
        ids = [s[0] for s in n.successors]
        assert 99 in ids
        assert 0 not in ids  # evicted

    def test_successor_no_eviction_if_weaker(self):
        n = Neuron(id=0, vector=make_vector(0))
        for i in range(MAX_SUCCESSORS):
            n.add_successor(i, 0.5)
        n.add_successor(99, 0.1)  # weaker than all existing
        ids = [s[0] for s in n.successors]
        assert 99 not in ids

    def test_predecessor_add(self):
        n = Neuron(id=0, vector=make_vector(0))
        n.add_predecessor(1)
        n.add_predecessor(2)
        assert n.predecessors == [1, 2]

    def test_predecessor_max_keeps_recent(self):
        n = Neuron(id=0, vector=make_vector(0))
        n.add_predecessor(1)
        n.add_predecessor(2)
        n.add_predecessor(3)
        assert len(n.predecessors) == MAX_PREDECESSORS
        n.add_predecessor(4)
        assert len(n.predecessors) == MAX_PREDECESSORS
        assert 1 not in n.predecessors  # oldest evicted
        assert 4 in n.predecessors

    def test_predecessor_no_duplicates(self):
        n = Neuron(id=0, vector=make_vector(0))
        n.add_predecessor(1)
        n.add_predecessor(1)
        assert n.predecessors == [1]

    def test_effective_confidence_non_temporal(self):
        n = Neuron(id=0, vector=make_vector(0), confidence=0.7, temporal=False)
        assert n.effective_confidence(current_time=999999) == 0.7

    def test_effective_confidence_temporal_decays(self):
        now = int(time.time())
        n = Neuron(id=0, vector=make_vector(0), confidence=0.7,
                   temporal=True, timestamp=now - 86400)  # 24 hours ago
        eff = n.effective_confidence(current_time=now)
        assert eff < 0.7  # should decay
        assert eff > 0.0  # shouldn't be zero

    def test_effective_confidence_temporal_fresh(self):
        now = int(time.time())
        n = Neuron(id=0, vector=make_vector(0), confidence=0.7,
                   temporal=True, timestamp=now)
        eff = n.effective_confidence(current_time=now)
        assert abs(eff - 0.7) < 0.01  # fresh = no decay

    def test_level_enum(self):
        assert Level.CHARACTER == 0
        assert Level.WORD == 1
        assert Level.CONCEPT == 2


# --- NeuronDB tests ---

class TestNeuronDB:

    def setup_method(self):
        self.db = NeuronDB()  # in-memory

    def teardown_method(self):
        self.db.close()

    def test_insert_and_get(self):
        v = make_vector(42)
        neuron = self.db.insert(v, confidence=0.7)
        assert neuron.id == 0
        assert neuron.confidence == 0.7

        retrieved = self.db.get(neuron.id)
        assert retrieved is not None
        assert retrieved.id == neuron.id
        assert abs(retrieved.confidence - 0.7) < 1e-6

    def test_insert_auto_increments_id(self):
        n0 = self.db.insert(make_vector(0))
        n1 = self.db.insert(make_vector(1))
        n2 = self.db.insert(make_vector(2))
        assert n0.id == 0
        assert n1.id == 1
        assert n2.id == 2

    def test_count(self):
        assert self.db.count() == 0
        self.db.insert(make_vector(0))
        self.db.insert(make_vector(1))
        assert self.db.count() == 2

    def test_get_nonexistent(self):
        assert self.db.get(999) is None

    def test_search_finds_nearest(self):
        # Insert two distant vectors
        v1 = np.zeros(VECTOR_DIM, dtype=np.float32)
        v1[0] = 1.0  # unit vector along dim 0

        v2 = np.zeros(VECTOR_DIM, dtype=np.float32)
        v2[1] = 1.0  # unit vector along dim 1

        self.db.insert(v1, confidence=0.5)  # id=0
        self.db.insert(v2, confidence=0.5)  # id=1

        # Query near v1
        query = np.zeros(VECTOR_DIM, dtype=np.float32)
        query[0] = 0.9
        query[2] = 0.1

        results = self.db.search(query, k=1)
        assert len(results) == 1
        assert results[0].id == 0  # closer to v1

    def test_search_returns_k_results(self):
        for i in range(20):
            self.db.insert(make_vector(i))
        results = self.db.search(make_vector(5), k=5)
        assert len(results) == 5

    def test_search_k_larger_than_db(self):
        self.db.insert(make_vector(0))
        self.db.insert(make_vector(1))
        results = self.db.search(make_vector(0), k=10)
        assert len(results) == 2  # only 2 exist

    def test_search_empty_db(self):
        results = self.db.search(make_vector(0), k=5)
        assert results == []

    def test_delete_removes_neuron(self):
        """Invariant #3: Delete = gone. No retraining."""
        n = self.db.insert(make_vector(42))
        assert self.db.get(n.id) is not None
        assert self.db.count() == 1

        deleted = self.db.delete(n.id)
        assert deleted is True
        assert self.db.get(n.id) is None
        assert self.db.count() == 0

    def test_delete_removes_from_search(self):
        """After delete, search should not find the neuron."""
        v = np.zeros(VECTOR_DIM, dtype=np.float32)
        v[0] = 1.0
        n = self.db.insert(v)

        # Searchable before delete
        results = self.db.search(v, k=1)
        assert len(results) == 1

        self.db.delete(n.id)

        # Gone from search after delete
        results = self.db.search(v, k=1)
        assert len(results) == 0

    def test_delete_nonexistent(self):
        assert self.db.delete(999) is False

    def test_update_confidence_useful(self):
        n = self.db.insert(make_vector(0), confidence=0.5)
        self.db.update_confidence(n.id, useful=True)
        updated = self.db.get(n.id)
        assert abs(updated.confidence - 0.5 * CONFIDENCE_BOOST) < 1e-6

    def test_update_confidence_not_useful(self):
        n = self.db.insert(make_vector(0), confidence=0.5)
        self.db.update_confidence(n.id, useful=False)
        updated = self.db.get(n.id)
        assert abs(updated.confidence - 0.5 * CONFIDENCE_DECAY) < 1e-6

    def test_update_confidence_respects_cap(self):
        n = self.db.insert(make_vector(0), confidence=0.79)
        self.db.update_confidence(n.id, useful=True)
        updated = self.db.get(n.id)
        assert updated.confidence == CONFIDENCE_CAP

    def test_update_successors(self):
        n = self.db.insert(make_vector(0))
        other = self.db.insert(make_vector(1))
        self.db.update_successors(n.id, other.id, 0.85)

        updated = self.db.get(n.id)
        assert len(updated.successors) == 1
        assert updated.successors[0][0] == other.id
        assert abs(updated.successors[0][1] - 0.85) < 1e-5

    def test_update_predecessors(self):
        n = self.db.insert(make_vector(0))
        pred = self.db.insert(make_vector(1))
        self.db.update_predecessors(n.id, pred.id)

        updated = self.db.get(n.id)
        assert pred.id in updated.predecessors

    def test_successor_persistence_through_get(self):
        """Successors survive serialize→deserialize round-trip."""
        n = self.db.insert(make_vector(0))
        self.db.update_successors(n.id, 10, 0.9)
        self.db.update_successors(n.id, 20, 0.8)
        self.db.update_successors(n.id, 30, 0.7)

        retrieved = self.db.get(n.id)
        assert len(retrieved.successors) == 3
        ids = [s[0] for s in retrieved.successors]
        assert 10 in ids
        assert 20 in ids
        assert 30 in ids

    def test_vector_normalized_on_insert(self):
        """Vectors should be L2-normalized for cosine similarity."""
        v = np.ones(VECTOR_DIM, dtype=np.float32) * 5.0  # not normalized
        n = self.db.insert(v)
        retrieved = self.db.get(n.id)
        norm = np.linalg.norm(retrieved.vector)
        assert abs(norm - 1.0) < 1e-5

    def test_multiple_deletes_and_inserts(self):
        """DB stays consistent through insert-delete-insert cycles."""
        n0 = self.db.insert(make_vector(0))
        n1 = self.db.insert(make_vector(1))
        n2 = self.db.insert(make_vector(2))
        assert self.db.count() == 3

        self.db.delete(n1.id)
        assert self.db.count() == 2

        n3 = self.db.insert(make_vector(3))
        assert self.db.count() == 3

        # n1 gone, others present
        assert self.db.get(n0.id) is not None
        assert self.db.get(n1.id) is None
        assert self.db.get(n2.id) is not None
        assert self.db.get(n3.id) is not None

    def test_level_persists(self):
        n = self.db.insert(make_vector(0), level=Level.CONCEPT)
        retrieved = self.db.get(n.id)
        assert retrieved.level == Level.CONCEPT

    def test_temporal_persists(self):
        n = self.db.insert(make_vector(0), temporal=True)
        retrieved = self.db.get(n.id)
        assert retrieved.temporal is True


# --- Run ---

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
