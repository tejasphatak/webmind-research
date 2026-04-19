"""
Neuron: the atomic unit of the reasoning engine.

A neuron is a point in N-dimensional concept space with a trust score.
No text. No labels. Just a vector, a confidence, and connections to neighbors.

Storage: FAISS index for spatial search + SQLite for metadata.
"""

import sqlite3
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

VECTOR_DIM = 384
MAX_SUCCESSORS = 10
MAX_PREDECESSORS = 3
CONFIDENCE_CAP = 0.8
CONFIDENCE_FLOOR = -0.8
CONFIDENCE_BOOST = 1.1
CONFIDENCE_DECAY = 0.9
DEFAULT_CONFIDENCE = 0.5


class Level(IntEnum):
    CHARACTER = 0
    WORD = 1
    CONCEPT = 2


@dataclass
class Neuron:
    """A point in concept space with a trust score."""

    id: int
    vector: np.ndarray                          # float32[384]
    confidence: float = DEFAULT_CONFIDENCE
    successors: list = field(default_factory=list)   # [(neuron_id, confidence)]
    predecessors: list = field(default_factory=list)  # [neuron_id, ...]
    timestamp: int = 0
    temporal: bool = False
    level: Level = Level.WORD

    def reinforce(self):
        """Fired and was useful. Strengthen."""
        self.confidence = min(self.confidence * CONFIDENCE_BOOST, CONFIDENCE_CAP)

    def weaken(self):
        """Fired and was not useful. Weaken."""
        self.confidence = max(self.confidence * CONFIDENCE_DECAY, CONFIDENCE_FLOOR)

    def add_successor(self, neuron_id: int, conf: float):
        """Add or update a successor. Evict lowest if full."""
        # Update existing
        for i, (sid, _) in enumerate(self.successors):
            if sid == neuron_id:
                self.successors[i] = (neuron_id, conf)
                return

        if len(self.successors) < MAX_SUCCESSORS:
            self.successors.append((neuron_id, conf))
        else:
            # Evict lowest confidence
            min_idx = min(range(len(self.successors)), key=lambda i: self.successors[i][1])
            if conf > self.successors[min_idx][1]:
                self.successors[min_idx] = (neuron_id, conf)

    def add_predecessor(self, neuron_id: int):
        """Track a predecessor. Keep top-3 most recent."""
        if neuron_id in self.predecessors:
            return
        if len(self.predecessors) < MAX_PREDECESSORS:
            self.predecessors.append(neuron_id)
        else:
            self.predecessors.pop(0)
            self.predecessors.append(neuron_id)

    def effective_confidence(self, current_time: Optional[int] = None):
        """Confidence adjusted for recency if temporal."""
        if not self.temporal or current_time is None:
            return self.confidence
        age_hours = (current_time - self.timestamp) / 3600
        decay = max(0.1, 1.0 / (1.0 + age_hours / 24.0))
        return self.confidence * decay


# --- Serialization helpers for successor/predecessor lists ---

def _encode_successors(successors: list) -> bytes:
    """Pack [(id, conf), ...] into bytes."""
    parts = []
    for sid, conf in successors:
        parts.append(struct.pack('<if', sid, conf))
    return b''.join(parts)


def _decode_successors(data: bytes) -> list:
    """Unpack bytes into [(id, conf), ...]."""
    size = struct.calcsize('<if')
    result = []
    for i in range(0, len(data), size):
        sid, conf = struct.unpack('<if', data[i:i + size])
        result.append((sid, conf))
    return result


def _encode_predecessors(predecessors: list) -> bytes:
    parts = [struct.pack('<i', pid) for pid in predecessors]
    return b''.join(parts)


def _decode_predecessors(data: bytes) -> list:
    size = struct.calcsize('<i')
    return [struct.unpack('<i', data[i:i + size])[0] for i in range(0, len(data), size)]


class NeuronDB:
    """
    Neuron storage: FAISS for spatial search, SQLite for metadata.

    Proximity IS connection. No explicit wiring graph.
    Two neurons are "connected" if they're close in vector space.
    """

    def __init__(self, path: Optional[str] = None, dim: int = VECTOR_DIM):
        self.dim = dim
        self.path = path
        self._next_id = 0

        # FAISS index: flat L2 for MVP (swap to IVF at scale)
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine sim on normalized vectors

        # SQLite for metadata
        db_path = str(Path(path) / "neurons.db") if path else ":memory:"
        self.db = sqlite3.connect(db_path)
        self._init_schema()

        # In-memory id→faiss_position map (FAISS uses sequential positions)
        self._id_to_pos = {}
        self._pos_to_id = {}

        if path:
            if not self._load_index_from_file():
                self._load_index()

    def _init_schema(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS neurons (
                id INTEGER PRIMARY KEY,
                confidence REAL NOT NULL DEFAULT 0.5,
                successors BLOB,
                predecessors BLOB,
                timestamp INTEGER NOT NULL,
                temporal INTEGER NOT NULL DEFAULT 0,
                level INTEGER NOT NULL DEFAULT 1,
                vector BLOB NOT NULL
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS templates (
                id INTEGER PRIMARY KEY,
                pattern TEXT NOT NULL,
                slots TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                vector BLOB NOT NULL
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS word_neurons (
                word TEXT PRIMARY KEY,
                neuron_id INTEGER NOT NULL,
                FOREIGN KEY (neuron_id) REFERENCES neurons(id)
            )
        """)
        self.db.commit()

    def _load_index(self):
        """Rebuild FAISS index from SQLite on startup."""
        rows = self.db.execute(
            "SELECT id, vector FROM neurons ORDER BY id"
        ).fetchall()
        if not rows:
            return

        for row in rows:
            nid = row[0]
            vec = np.frombuffer(row[1], dtype=np.float32).copy()
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            pos = self.index.ntotal
            self.index.add(vec.reshape(1, -1))
            self._id_to_pos[nid] = pos
            self._pos_to_id[pos] = nid

        self._next_id = max(r[0] for r in rows) + 1

    def insert(self, vector: np.ndarray, confidence: float = DEFAULT_CONFIDENCE,
               level: Level = Level.WORD, temporal: bool = False) -> Neuron:
        """Insert a new neuron. Returns the created Neuron."""
        nid = self._next_id
        self._next_id += 1

        vec = np.array(vector, dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)  # normalize for cosine sim

        now = int(time.time())
        neuron = Neuron(
            id=nid, vector=vec, confidence=confidence,
            timestamp=now, temporal=temporal, level=level,
        )

        # FAISS
        pos = self.index.ntotal
        self.index.add(vec.reshape(1, -1))
        self._id_to_pos[nid] = pos
        self._pos_to_id[pos] = nid

        # SQLite
        self.db.execute(
            "INSERT INTO neurons (id, confidence, successors, predecessors, timestamp, temporal, level, vector) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (nid, confidence, b'', b'', now, int(temporal), int(level), vec.tobytes())
        )
        self.db.commit()

        return neuron

    def get(self, neuron_id: int) -> Optional[Neuron]:
        """Retrieve a neuron by ID."""
        row = self.db.execute(
            "SELECT id, confidence, successors, predecessors, timestamp, temporal, level, vector "
            "FROM neurons WHERE id = ?", (neuron_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_neuron(row)

    def _row_to_neuron(self, row) -> Neuron:
        nid, conf, succ_bytes, pred_bytes, ts, temporal, level, vec_bytes = row
        return Neuron(
            id=nid,
            vector=np.frombuffer(vec_bytes, dtype=np.float32).copy(),
            confidence=conf,
            successors=_decode_successors(succ_bytes) if succ_bytes else [],
            predecessors=_decode_predecessors(pred_bytes) if pred_bytes else [],
            timestamp=ts,
            temporal=bool(temporal),
            level=Level(level),
        )

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[Neuron]:
        """Find k nearest neurons by cosine similarity."""
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        vec = vec / (np.linalg.norm(vec) + 1e-10)

        scores, positions = self.index.search(vec, k)
        results = []
        for pos, score in zip(positions[0], scores[0]):
            if pos < 0:
                continue
            nid = self._pos_to_id.get(int(pos))
            if nid is not None:
                neuron = self.get(nid)
                if neuron is not None:
                    results.append(neuron)
        return results

    def delete(self, neuron_id: int) -> bool:
        """
        Delete = gone. Immediately. No retraining. Invariant #3.

        Note: FAISS IndexFlatIP doesn't support removal. We mark as deleted
        in SQLite and rebuild the index. For MVP this is fine.
        """
        neuron = self.get(neuron_id)
        if neuron is None:
            return False

        self.db.execute("DELETE FROM neurons WHERE id = ?", (neuron_id,))
        self.db.commit()

        # Rebuild FAISS index without the deleted neuron
        self._rebuild_index()
        return True

    def update_confidence(self, neuron_id: int, useful: bool):
        """Update confidence based on whether the neuron was useful."""
        neuron = self.get(neuron_id)
        if neuron is None:
            return

        if useful:
            neuron.reinforce()
        else:
            neuron.weaken()

        self.db.execute(
            "UPDATE neurons SET confidence = ? WHERE id = ?",
            (neuron.confidence, neuron_id)
        )
        self.db.commit()

    def update_successors(self, neuron_id: int, successor_id: int, conf: float):
        """Add a successor relationship."""
        neuron = self.get(neuron_id)
        if neuron is None:
            return
        neuron.add_successor(successor_id, conf)
        self.db.execute(
            "UPDATE neurons SET successors = ? WHERE id = ?",
            (_encode_successors(neuron.successors), neuron_id)
        )
        self.db.commit()

    def update_predecessors(self, neuron_id: int, predecessor_id: int):
        """Add a predecessor relationship."""
        neuron = self.get(neuron_id)
        if neuron is None:
            return
        neuron.add_predecessor(predecessor_id)
        self.db.execute(
            "UPDATE neurons SET predecessors = ? WHERE id = ?",
            (_encode_predecessors(neuron.predecessors), neuron_id)
        )
        self.db.commit()

    def count(self) -> int:
        """Total neurons in the DB."""
        row = self.db.execute("SELECT COUNT(*) FROM neurons").fetchone()
        return row[0]

    def _rebuild_index(self):
        """Rebuild FAISS index from SQLite. Used after deletes."""
        self.index = faiss.IndexFlatIP(self.dim)
        self._id_to_pos = {}
        self._pos_to_id = {}
        self._load_index()

    # --- FAISS index persistence ---

    def save_index(self):
        """Save FAISS index to disk for fast reload."""
        if self.path:
            index_path = str(Path(self.path) / "neurons.faiss")
            faiss.write_index(self.index, index_path)

    def _load_index_from_file(self) -> bool:
        """Try to load FAISS index from disk. Returns True if successful."""
        if not self.path:
            return False
        index_path = Path(self.path) / "neurons.faiss"
        if not index_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))
        # Rebuild id↔pos maps from SQLite
        rows = self.db.execute(
            "SELECT id FROM neurons ORDER BY id"
        ).fetchall()
        for pos, (nid,) in enumerate(rows):
            self._id_to_pos[nid] = pos
            self._pos_to_id[pos] = nid
        if rows:
            self._next_id = max(r[0] for r in rows) + 1
        return True

    # --- Template persistence ---

    def save_template(self, template_id: int, pattern: str, slots_json: str,
                      confidence: float, vector: np.ndarray):
        """Save a template to SQLite."""
        vec_bytes = np.array(vector, dtype=np.float32).tobytes()
        self.db.execute(
            "INSERT OR REPLACE INTO templates (id, pattern, slots, confidence, vector) "
            "VALUES (?, ?, ?, ?, ?)",
            (template_id, pattern, slots_json, confidence, vec_bytes)
        )
        self.db.commit()

    def load_templates(self) -> list:
        """Load all templates from SQLite. Returns list of (id, pattern, slots_json, confidence, vector)."""
        rows = self.db.execute(
            "SELECT id, pattern, slots, confidence, vector FROM templates ORDER BY id"
        ).fetchall()
        result = []
        for tid, pattern, slots_json, conf, vec_bytes in rows:
            vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
            result.append((tid, pattern, slots_json, conf, vec))
        return result

    def delete_template(self, template_id: int) -> bool:
        """Delete a template. Invariant #3."""
        cursor = self.db.execute(
            "DELETE FROM templates WHERE id = ?", (template_id,)
        )
        self.db.commit()
        return cursor.rowcount > 0

    # --- Word→neuron mapping persistence ---

    def save_word_mapping(self, word: str, neuron_id: int):
        """Save a word→neuron mapping."""
        self.db.execute(
            "INSERT OR REPLACE INTO word_neurons (word, neuron_id) VALUES (?, ?)",
            (word, neuron_id)
        )
        self.db.commit()

    def load_word_mappings(self) -> dict:
        """Load all word→neuron mappings. Returns {word: neuron_id}."""
        rows = self.db.execute(
            "SELECT word, neuron_id FROM word_neurons"
        ).fetchall()
        return {word: nid for word, nid in rows}

    def delete_word_mapping(self, word: str) -> bool:
        """Delete a word mapping."""
        cursor = self.db.execute(
            "DELETE FROM word_neurons WHERE word = ?", (word,)
        )
        self.db.commit()
        return cursor.rowcount > 0

    def close(self):
        self.save_index()
        self.db.close()
