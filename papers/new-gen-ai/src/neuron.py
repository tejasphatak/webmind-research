"""
Neuron: the atomic unit of the reasoning engine.

A neuron is a point in concept space with a trust score.
No text. No labels. Just a vector, a confidence, and connections.

Storage: numpy matrix for search, SQLite for metadata.
No FAISS. Custom brute-force cosine similarity.
Supports growing dimensions — vectors expand as the system learns.
"""

import sqlite3
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

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
    vector: np.ndarray                          # float32[dim]
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
        for i, (sid, _) in enumerate(self.successors):
            if sid == neuron_id:
                self.successors[i] = (neuron_id, conf)
                return

        if len(self.successors) < MAX_SUCCESSORS:
            self.successors.append((neuron_id, conf))
        else:
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


# --- Serialization helpers ---

def _encode_successors(successors: list) -> bytes:
    parts = []
    for sid, conf in successors:
        parts.append(struct.pack('<if', sid, conf))
    return b''.join(parts)


def _decode_successors(data: bytes) -> list:
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
    Neuron storage: numpy for search, SQLite for metadata.

    Search is brute-force cosine similarity over a numpy matrix.
    At our scale (< 100K neurons), this is sub-millisecond.
    No external dependencies for search — pure numpy.

    Supports dynamic dimensions: vectors can change size as the
    encoder learns new concepts. Existing vectors are zero-padded
    when dimensions grow.
    """

    def __init__(self, path: Optional[str] = None, dim: int = VECTOR_DIM):
        self.dim = dim
        self.path = path
        self._next_id = 0

        # In-memory vector matrix for search
        self._vectors = None    # shape (n_neurons, dim) or None
        self._id_to_row = {}    # neuron_id → row index in matrix
        self._row_to_id = {}    # row index → neuron_id

        # SQLite for metadata
        db_path = str(Path(path) / "neurons.db") if path else ":memory:"
        self.db = sqlite3.connect(db_path)
        self._init_schema()

        # Word mapping cache
        self._word_map_cache = None

        if path:
            self._load_from_sqlite()

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
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS sentence_neurons (
                sentence_id INTEGER NOT NULL,
                neuron_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY (sentence_id, neuron_id)
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentence_neurons_nid
            ON sentence_neurons(neuron_id)
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS misses (
                id INTEGER PRIMARY KEY,
                query_text TEXT NOT NULL,
                query_vector BLOB NOT NULL,
                timestamp INTEGER NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0,
                resolved_timestamp INTEGER,
                answer_text TEXT
            )
        """)
        self.db.commit()

    def _load_from_sqlite(self):
        """Rebuild search matrix from SQLite on startup."""
        rows = self.db.execute(
            "SELECT id, vector FROM neurons ORDER BY id"
        ).fetchall()
        if not rows:
            return

        for row_idx, (nid, vec_bytes) in enumerate(rows):
            vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            if self._vectors is None:
                self._vectors = vec.reshape(1, -1)
            else:
                # Handle dimension mismatch (growing dims)
                if vec.shape[0] < self._vectors.shape[1]:
                    vec = np.pad(vec, (0, self._vectors.shape[1] - vec.shape[0]))
                elif vec.shape[0] > self._vectors.shape[1]:
                    pad_width = vec.shape[0] - self._vectors.shape[1]
                    self._vectors = np.pad(self._vectors, ((0, 0), (0, pad_width)))
                self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])

            self._id_to_row[nid] = row_idx
            self._row_to_id[row_idx] = nid

        self._next_id = max(r[0] for r in rows) + 1

    # --- Core operations ---

    def insert(self, vector: np.ndarray, confidence: float = DEFAULT_CONFIDENCE,
               level: Level = Level.WORD, temporal: bool = False) -> Neuron:
        """Insert a new neuron. Returns the created Neuron."""
        nid = self._next_id
        self._next_id += 1

        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        now = int(time.time())
        neuron = Neuron(
            id=nid, vector=vec, confidence=confidence,
            timestamp=now, temporal=temporal, level=level,
        )

        # Add to search matrix
        row_idx = len(self._id_to_row)
        if self._vectors is None:
            self._vectors = vec.reshape(1, -1)
        else:
            # Handle dimension mismatch
            if vec.shape[0] < self._vectors.shape[1]:
                vec = np.pad(vec, (0, self._vectors.shape[1] - vec.shape[0]))
            elif vec.shape[0] > self._vectors.shape[1]:
                pad_width = vec.shape[0] - self._vectors.shape[1]
                self._vectors = np.pad(self._vectors, ((0, 0), (0, pad_width)))
            self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])

        self._id_to_row[nid] = row_idx
        self._row_to_id[row_idx] = nid

        # SQLite
        self.db.execute(
            "INSERT INTO neurons (id, confidence, successors, predecessors, "
            "timestamp, temporal, level, vector) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (nid, confidence, b'', b'', now, int(temporal), int(level), vec.tobytes())
        )
        self.db.commit()

        # Invalidate word map cache
        self._word_map_cache = None

        return neuron

    def get(self, neuron_id: int) -> Optional[Neuron]:
        """Retrieve a neuron by ID."""
        row = self.db.execute(
            "SELECT id, confidence, successors, predecessors, "
            "timestamp, temporal, level, vector "
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

    def search(self, query_vector: np.ndarray, k: int = 5) -> list:
        """
        Find k nearest neurons by cosine similarity.
        Pure numpy — brute-force matrix multiply.
        Sub-millisecond for < 100K neurons.
        """
        if self._vectors is None or len(self._id_to_row) == 0:
            return []

        vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        vec = vec / norm

        # Handle dimension mismatch
        mat = self._vectors
        if vec.shape[0] < mat.shape[1]:
            vec = np.pad(vec, (0, mat.shape[1] - vec.shape[0]))
        elif vec.shape[0] > mat.shape[1]:
            vec = vec[:mat.shape[1]]

        # Cosine similarity: matrix @ vector
        sims = mat @ vec

        k = min(k, len(sims))
        if k <= 0:
            return []

        top_k = np.argpartition(-sims, k - 1)[:k] if len(sims) > k else np.arange(len(sims))
        top_k = top_k[np.argsort(-sims[top_k])]

        results = []
        for row_idx in top_k:
            nid = self._row_to_id.get(int(row_idx))
            if nid is not None:
                neuron = self.get(nid)
                if neuron is not None:
                    results.append(neuron)
        return results

    def delete(self, neuron_id: int) -> bool:
        """Delete = gone. Immediately. Invariant #3."""
        neuron = self.get(neuron_id)
        if neuron is None:
            return False

        self.db.execute("DELETE FROM neurons WHERE id = ?", (neuron_id,))
        self.db.commit()

        # Rebuild search matrix
        self._rebuild_matrix()
        self._word_map_cache = None
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
        row = self.db.execute("SELECT COUNT(*) FROM neurons").fetchone()
        return row[0]

    def _rebuild_matrix(self):
        """Rebuild search matrix from SQLite. Used after deletes."""
        self._vectors = None
        self._id_to_row = {}
        self._row_to_id = {}
        self._load_from_sqlite()

    # --- Template persistence ---

    def save_template(self, template_id: int, pattern: str, slots_json: str,
                      confidence: float, vector: np.ndarray):
        vec_bytes = np.array(vector, dtype=np.float32).tobytes()
        self.db.execute(
            "INSERT OR REPLACE INTO templates (id, pattern, slots, confidence, vector) "
            "VALUES (?, ?, ?, ?, ?)",
            (template_id, pattern, slots_json, confidence, vec_bytes)
        )
        self.db.commit()

    def load_templates(self) -> list:
        rows = self.db.execute(
            "SELECT id, pattern, slots, confidence, vector FROM templates ORDER BY id"
        ).fetchall()
        result = []
        for tid, pattern, slots_json, conf, vec_bytes in rows:
            vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
            result.append((tid, pattern, slots_json, conf, vec))
        return result

    def delete_template(self, template_id: int) -> bool:
        cursor = self.db.execute(
            "DELETE FROM templates WHERE id = ?", (template_id,)
        )
        self.db.commit()
        return cursor.rowcount > 0

    # --- Word→neuron mapping ---

    def save_word_mapping(self, word: str, neuron_id: int):
        self.db.execute(
            "INSERT OR REPLACE INTO word_neurons (word, neuron_id) VALUES (?, ?)",
            (word, neuron_id)
        )
        self.db.commit()
        self._word_map_cache = None

    def load_word_mappings(self) -> dict:
        if self._word_map_cache is not None:
            return self._word_map_cache
        rows = self.db.execute(
            "SELECT word, neuron_id FROM word_neurons"
        ).fetchall()
        self._word_map_cache = {word: nid for word, nid in rows}
        return self._word_map_cache

    def delete_word_mapping(self, word: str) -> bool:
        cursor = self.db.execute(
            "DELETE FROM word_neurons WHERE word = ?", (word,)
        )
        self.db.commit()
        self._word_map_cache = None
        return cursor.rowcount > 0

    # --- Sentence-level association ---

    def record_sentence(self, neuron_ids: list) -> int:
        row = self.db.execute(
            "SELECT COALESCE(MAX(sentence_id), -1) + 1 FROM sentence_neurons"
        ).fetchone()
        sentence_id = row[0]

        for pos, nid in enumerate(neuron_ids):
            self.db.execute(
                "INSERT OR IGNORE INTO sentence_neurons "
                "(sentence_id, neuron_id, position) VALUES (?, ?, ?)",
                (sentence_id, nid, pos)
            )
        self.db.commit()
        return sentence_id

    def get_cooccurring_neurons(self, neuron_id: int) -> list:
        rows = self.db.execute("""
            SELECT sn2.neuron_id, sn2.position, sn2.sentence_id
            FROM sentence_neurons sn1
            JOIN sentence_neurons sn2 ON sn1.sentence_id = sn2.sentence_id
            WHERE sn1.neuron_id = ? AND sn2.neuron_id != ?
            ORDER BY sn2.sentence_id, sn2.position
        """, (neuron_id, neuron_id)).fetchall()
        return rows

    def get_sentences_for_neurons(self, neuron_ids: list) -> dict:
        if not neuron_ids:
            return {}
        placeholders = ",".join("?" * len(neuron_ids))
        rows = self.db.execute(f"""
            SELECT sentence_id, neuron_id, position
            FROM sentence_neurons
            WHERE neuron_id IN ({placeholders})
            ORDER BY sentence_id, position
        """, neuron_ids).fetchall()

        sentences = {}
        for sid, nid, pos in rows:
            if sid not in sentences:
                sentences[sid] = []
            sentences[sid].append((nid, pos))
        return sentences

    def get_sentence_neurons(self, sentence_id: int) -> list:
        rows = self.db.execute(
            "SELECT neuron_id, position FROM sentence_neurons "
            "WHERE sentence_id = ? ORDER BY position",
            (sentence_id,)
        ).fetchall()
        return rows

    # --- Miss logging (self-evolution) ---

    def log_miss(self, query_text: str, query_vector: np.ndarray) -> int:
        vec_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        now = int(time.time())
        cursor = self.db.execute(
            "INSERT INTO misses (query_text, query_vector, timestamp, resolved) "
            "VALUES (?, ?, ?, 0)",
            (query_text, vec_bytes, now)
        )
        self.db.commit()
        return cursor.lastrowid

    def resolve_miss(self, miss_id: int, answer_text: str):
        now = int(time.time())
        self.db.execute(
            "UPDATE misses SET resolved = 1, resolved_timestamp = ?, "
            "answer_text = ? WHERE id = ?",
            (now, answer_text, miss_id)
        )
        self.db.commit()

    def resolve_miss_by_query(self, query_text: str, answer_text: str) -> bool:
        row = self.db.execute(
            "SELECT id FROM misses WHERE query_text = ? AND resolved = 0 "
            "ORDER BY timestamp DESC LIMIT 1",
            (query_text,)
        ).fetchone()
        if row:
            self.resolve_miss(row[0], answer_text)
            return True
        return False

    def get_unresolved_misses(self, limit: int = 50) -> list:
        rows = self.db.execute(
            "SELECT id, query_text, timestamp FROM misses "
            "WHERE resolved = 0 ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return rows

    def miss_stats(self) -> dict:
        total = self.db.execute("SELECT COUNT(*) FROM misses").fetchone()[0]
        resolved = self.db.execute(
            "SELECT COUNT(*) FROM misses WHERE resolved = 1"
        ).fetchone()[0]
        unresolved = total - resolved
        return {
            "total_misses": total,
            "resolved": resolved,
            "unresolved": unresolved,
            "resolution_rate": resolved / total if total > 0 else 0.0,
        }

    def save_index(self):
        """No-op for compatibility. Matrix is rebuilt from SQLite."""
        pass

    def close(self):
        self.db.close()
