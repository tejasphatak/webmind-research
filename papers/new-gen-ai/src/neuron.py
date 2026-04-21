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


class Neuron:
    """A point in concept space with a trust score.

    Lightweight view object with __slots__ for speed. Returned by
    NeuronDB.get() for backward-compatible attribute access.
    Hot-path code should use NeuronDB's columnar array accessors instead.
    """
    __slots__ = ('id', 'vector', 'confidence', 'successors', 'predecessors',
                 'timestamp', 'temporal', 'level')

    def __init__(self, id: int, vector: np.ndarray,
                 confidence: float = DEFAULT_CONFIDENCE,
                 successors: list = None, predecessors: list = None,
                 timestamp: int = 0, temporal: bool = False,
                 level: 'Level' = None):
        self.id = id
        self.vector = vector
        self.confidence = confidence
        self.successors = successors if successors is not None else []
        self.predecessors = predecessors if predecessors is not None else []
        self.timestamp = timestamp
        self.temporal = temporal
        self.level = level if level is not None else Level.WORD

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


# Backward-compatible alias
NeuronView = Neuron


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

    # Pre-allocate matrix in chunks to avoid O(n^2) vstack
    _ALLOC_CHUNK = 256

    def __init__(self, path: Optional[str] = None, dim: int = VECTOR_DIM):
        self.dim = dim
        self.path = path
        self._next_id = 0

        # In-memory vector matrix for search (pre-allocated)
        self._vectors = None    # shape (alloc_rows, dim) or None
        self._n_rows = 0        # how many rows are actually used
        self._id_to_row = {}    # neuron_id → row index in matrix
        self._row_to_id = {}    # row index → neuron_id

        # Columnar arrays for hot-path scalar access (struct-of-arrays)
        self._confidences = np.zeros(self._ALLOC_CHUNK, dtype=np.float32)
        self._timestamps = np.zeros(self._ALLOC_CHUNK, dtype=np.int64)
        self._temporals = np.zeros(self._ALLOC_CHUNK, dtype=np.bool_)
        self._levels = np.zeros(self._ALLOC_CHUNK, dtype=np.int8)

        # Batch mode: defer commits until flush
        self._batch = False
        self._dirty = False

        # SQLite for metadata (check_same_thread=False for parallel queries)
        db_path = str(Path(path) / "neurons.db") if path else ":memory:"
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        # WAL mode: concurrent reads don't block each other
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

        # Word mapping cache
        self._word_map_cache = None

        if path:
            self._load_from_sqlite()

    def begin_batch(self):
        """Start batch mode — defers commits for speed."""
        self._batch = True
        self._dirty = False

    def end_batch(self):
        """End batch mode — flush pending writes."""
        self._batch = False
        if self._dirty:
            self.db.commit()
            self._dirty = False

    def _commit(self):
        """Commit unless in batch mode."""
        if self._batch:
            self._dirty = True
        else:
            self.db.commit()

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
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_misses_query
            ON misses(query_text, resolved)
        """)
        self._commit()

    def _add_vec_to_matrix(self, vec: np.ndarray) -> int:
        """Add a vector to the pre-allocated matrix. Returns row index."""
        if self._vectors is None:
            d = vec.shape[0]
            self._vectors = np.zeros((self._ALLOC_CHUNK, d), dtype=np.float32)
            self._n_rows = 0

        # Grow dimensions if needed
        if vec.shape[0] < self._vectors.shape[1]:
            vec = np.pad(vec, (0, self._vectors.shape[1] - vec.shape[0]))
        elif vec.shape[0] > self._vectors.shape[1]:
            pad_width = vec.shape[0] - self._vectors.shape[1]
            new_mat = np.zeros((self._vectors.shape[0], self._vectors.shape[1] + pad_width),
                               dtype=np.float32)
            new_mat[:, :self._vectors.shape[1]] = self._vectors
            self._vectors = new_mat

        # Grow rows if needed
        if self._n_rows >= self._vectors.shape[0]:
            extra = np.zeros((self._ALLOC_CHUNK, self._vectors.shape[1]), dtype=np.float32)
            self._vectors = np.vstack([self._vectors, extra])
            self._grow_columnar_arrays(self._vectors.shape[0])

        row_idx = self._n_rows
        self._vectors[row_idx] = vec
        self._n_rows += 1
        return row_idx

    def _grow_columnar_arrays(self, target_size: int):
        """Grow columnar arrays to match vector matrix allocation."""
        if len(self._confidences) >= target_size:
            return
        extra = target_size - len(self._confidences)
        self._confidences = np.concatenate([self._confidences, np.zeros(extra, dtype=np.float32)])
        self._timestamps = np.concatenate([self._timestamps, np.zeros(extra, dtype=np.int64)])
        self._temporals = np.concatenate([self._temporals, np.zeros(extra, dtype=np.bool_)])
        self._levels = np.concatenate([self._levels, np.zeros(extra, dtype=np.int8)])

    def _load_from_sqlite(self):
        """Rebuild search matrix and columnar arrays from SQLite on startup."""
        rows = self.db.execute(
            "SELECT id, confidence, timestamp, temporal, level, vector "
            "FROM neurons ORDER BY id"
        ).fetchall()
        if not rows:
            return

        # Find max dim first, then allocate once
        vecs = []
        nids = []
        confs = []
        timestamps = []
        temporals = []
        levels = []
        max_dim = 0
        for nid, conf, ts, temporal, level, vec_bytes in rows:
            vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vecs.append(vec)
            nids.append(nid)
            confs.append(conf)
            timestamps.append(ts)
            temporals.append(bool(temporal))
            levels.append(level)
            if vec.shape[0] > max_dim:
                max_dim = vec.shape[0]

        # Allocate matrix in one shot
        n = len(vecs)
        alloc = ((n // self._ALLOC_CHUNK) + 1) * self._ALLOC_CHUNK
        self._vectors = np.zeros((alloc, max_dim), dtype=np.float32)
        self._n_rows = n

        # Allocate columnar arrays
        self._confidences = np.zeros(alloc, dtype=np.float32)
        self._timestamps = np.zeros(alloc, dtype=np.int64)
        self._temporals = np.zeros(alloc, dtype=np.bool_)
        self._levels = np.zeros(alloc, dtype=np.int8)

        for row_idx, (nid, vec) in enumerate(zip(nids, vecs)):
            if vec.shape[0] < max_dim:
                vec = np.pad(vec, (0, max_dim - vec.shape[0]))
            self._vectors[row_idx] = vec
            self._id_to_row[nid] = row_idx
            self._row_to_id[row_idx] = nid
            self._confidences[row_idx] = confs[row_idx]
            self._timestamps[row_idx] = timestamps[row_idx]
            self._temporals[row_idx] = temporals[row_idx]
            self._levels[row_idx] = levels[row_idx]

        self._next_id = max(nids) + 1

    # --- Core operations ---

    def insert(self, vector: np.ndarray, confidence: float = DEFAULT_CONFIDENCE,
               level: Level = Level.WORD, temporal: bool = False) -> Neuron:
        """Insert a new neuron. Returns a NeuronView for backward compat."""
        nid = self._next_id
        self._next_id += 1

        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        now = int(time.time())

        # Add to pre-allocated search matrix
        row_idx = self._add_vec_to_matrix(vec)
        self._id_to_row[nid] = row_idx
        self._row_to_id[row_idx] = nid

        # Write to columnar arrays
        self._confidences[row_idx] = confidence
        self._timestamps[row_idx] = now
        self._temporals[row_idx] = temporal
        self._levels[row_idx] = int(level)

        # SQLite
        self.db.execute(
            "INSERT INTO neurons (id, confidence, successors, predecessors, "
            "timestamp, temporal, level, vector) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (nid, confidence, b'', b'', now, int(temporal), int(level), vec.tobytes())
        )
        self._commit()

        self._word_map_cache = None

        # Return a NeuronView for callers that need the full object
        return Neuron(
            id=nid, vector=vec, confidence=confidence,
            timestamp=now, temporal=temporal, level=level,
        )

    # --- Columnar convenience accessors (hot-path, no object allocation) ---

    def get_confidence(self, neuron_id: int) -> float:
        """Direct array read — no object allocation."""
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return 0.0
        return float(self._confidences[row])

    def get_vector(self, neuron_id: int) -> np.ndarray:
        """Direct array read — returns a view (no copy)."""
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return np.zeros(self.dim, dtype=np.float32)
        return self._vectors[row]

    def get_timestamp(self, neuron_id: int) -> int:
        """Direct array read."""
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return 0
        return int(self._timestamps[row])

    def get_level(self, neuron_id: int) -> int:
        """Direct array read."""
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return int(Level.WORD)
        return int(self._levels[row])

    def get_temporal(self, neuron_id: int) -> bool:
        """Direct array read."""
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return False
        return bool(self._temporals[row])

    def get(self, neuron_id: int) -> Optional[Neuron]:
        """Retrieve a neuron by ID. Builds NeuronView from columnar arrays + SQLite.

        For hot-path code, prefer get_confidence()/get_vector() to avoid
        object allocation. This method exists for backward compatibility.
        """
        row_idx = self._id_to_row.get(neuron_id)
        if row_idx is None:
            # Not in memory — try SQLite (handles edge cases / stale state)
            db_row = self.db.execute(
                "SELECT id, confidence, successors, predecessors, "
                "timestamp, temporal, level, vector "
                "FROM neurons WHERE id = ?", (neuron_id,)
            ).fetchone()
            if not db_row:
                return None
            return self._row_to_neuron(db_row)

        # Build NeuronView from columnar arrays (scalars) + SQLite (variable-length)
        succ_pred = self.db.execute(
            "SELECT successors, predecessors FROM neurons WHERE id = ?",
            (neuron_id,)
        ).fetchone()
        succ_bytes, pred_bytes = (b'', b'') if succ_pred is None else succ_pred

        return Neuron(
            id=neuron_id,
            vector=self._vectors[row_idx].copy(),
            confidence=round(float(self._confidences[row_idx]), 6),
            successors=_decode_successors(succ_bytes) if succ_bytes else [],
            predecessors=_decode_predecessors(pred_bytes) if pred_bytes else [],
            timestamp=int(self._timestamps[row_idx]),
            temporal=bool(self._temporals[row_idx]),
            level=Level(int(self._levels[row_idx])),
        )

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

    def search_ids(self, query_vector: np.ndarray, k: int = 5) -> list:
        """Find k nearest neuron IDs by cosine similarity. Returns [(neuron_id, similarity)].

        Hot-path version — no object allocation. Use with get_confidence()/get_vector()
        for zero-copy access to neuron data.
        """
        if self._vectors is None or self._n_rows == 0:
            return []

        vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        vec = vec / norm

        # Handle dimension mismatch
        mat = self._vectors[:self._n_rows]  # only used rows
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
                results.append((nid, float(sims[row_idx])))
        return results

    def search(self, query_vector: np.ndarray, k: int = 5) -> list:
        """
        Find k nearest neurons by cosine similarity.
        Pure numpy — brute-force matrix multiply.
        Sub-millisecond for < 100K neurons.

        Returns list of Neuron (NeuronView) objects for backward compat.
        For hot-path code, prefer search_ids() + get_confidence()/get_vector().
        """
        id_sims = self.search_ids(query_vector, k=k)
        results = []
        for nid, sim in id_sims:
            neuron = self.get(nid)
            if neuron is not None:
                results.append(neuron)
        return results

    def delete(self, neuron_id: int) -> bool:
        """Delete = gone. Immediately. Invariant #3."""
        if neuron_id not in self._id_to_row:
            # Check SQLite as fallback
            row = self.db.execute(
                "SELECT id FROM neurons WHERE id = ?", (neuron_id,)
            ).fetchone()
            if not row:
                return False

        self.db.execute("DELETE FROM neurons WHERE id = ?", (neuron_id,))
        self._commit()

        # Rebuild search matrix and columnar arrays from SQLite
        self._rebuild_matrix()
        self._word_map_cache = None
        return True

    def update_confidence(self, neuron_id: int, useful: bool):
        """Update confidence based on whether the neuron was useful.

        Direct array write — no object allocation on the hot path.
        """
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return

        conf = float(self._confidences[row])
        if useful:
            conf = min(conf * CONFIDENCE_BOOST, CONFIDENCE_CAP)
        else:
            conf = max(conf * CONFIDENCE_DECAY, CONFIDENCE_FLOOR)

        # Store and read back as float32 to ensure consistency
        self._confidences[row] = conf

        # Persist to SQLite
        self.db.execute(
            "UPDATE neurons SET confidence = ? WHERE id = ?",
            (conf, neuron_id)
        )
        self._commit()

    def update_successors(self, neuron_id: int, successor_id: int, conf: float):
        """Add a successor relationship."""
        if neuron_id not in self._id_to_row:
            return
        # Read current successors from SQLite (variable-length, not in columnar arrays)
        row = self.db.execute(
            "SELECT successors FROM neurons WHERE id = ?", (neuron_id,)
        ).fetchone()
        if not row:
            return
        successors = _decode_successors(row[0]) if row[0] else []

        # Apply the add_successor logic inline
        for i, (sid, _) in enumerate(successors):
            if sid == successor_id:
                successors[i] = (successor_id, conf)
                self.db.execute(
                    "UPDATE neurons SET successors = ? WHERE id = ?",
                    (_encode_successors(successors), neuron_id)
                )
                self._commit()
                return

        if len(successors) < MAX_SUCCESSORS:
            successors.append((successor_id, conf))
        else:
            min_idx = min(range(len(successors)), key=lambda i: successors[i][1])
            if conf > successors[min_idx][1]:
                successors[min_idx] = (successor_id, conf)
            # else: no change needed

        self.db.execute(
            "UPDATE neurons SET successors = ? WHERE id = ?",
            (_encode_successors(successors), neuron_id)
        )
        self._commit()

    def update_predecessors(self, neuron_id: int, predecessor_id: int):
        """Add a predecessor relationship."""
        if neuron_id not in self._id_to_row:
            return
        # Read current predecessors from SQLite (variable-length)
        row = self.db.execute(
            "SELECT predecessors FROM neurons WHERE id = ?", (neuron_id,)
        ).fetchone()
        if not row:
            return
        predecessors = _decode_predecessors(row[0]) if row[0] else []

        # Apply the add_predecessor logic inline
        if predecessor_id in predecessors:
            return
        if len(predecessors) < MAX_PREDECESSORS:
            predecessors.append(predecessor_id)
        else:
            predecessors.pop(0)
            predecessors.append(predecessor_id)

        self.db.execute(
            "UPDATE neurons SET predecessors = ? WHERE id = ?",
            (_encode_predecessors(predecessors), neuron_id)
        )
        self._commit()

    def count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM neurons").fetchone()
        return row[0]

    def _rebuild_matrix(self):
        """Rebuild search matrix and columnar arrays from SQLite. Used after deletes."""
        self._vectors = None
        self._n_rows = 0
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
        self._commit()

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
        self._commit()
        return cursor.rowcount > 0

    # --- Word→neuron mapping ---

    def save_word_mapping(self, word: str, neuron_id: int):
        self.db.execute(
            "INSERT OR REPLACE INTO word_neurons (word, neuron_id) VALUES (?, ?)",
            (word, neuron_id)
        )
        self._commit()
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
        self._commit()
        self._word_map_cache = None
        return cursor.rowcount > 0

    # --- Sentence-level association ---

    def record_sentence(self, neuron_ids: list) -> int:
        if not hasattr(self, '_next_sentence_id'):
            row = self.db.execute(
                "SELECT COALESCE(MAX(sentence_id), -1) + 1 FROM sentence_neurons"
            ).fetchone()
            self._next_sentence_id = row[0]
        sentence_id = self._next_sentence_id
        self._next_sentence_id += 1

        for pos, nid in enumerate(neuron_ids):
            self.db.execute(
                "INSERT OR IGNORE INTO sentence_neurons "
                "(sentence_id, neuron_id, position) VALUES (?, ?, ?)",
                (sentence_id, nid, pos)
            )
        self._commit()
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
        self._commit()
        return cursor.lastrowid

    def resolve_miss(self, miss_id: int, answer_text: str):
        now = int(time.time())
        self.db.execute(
            "UPDATE misses SET resolved = 1, resolved_timestamp = ?, "
            "answer_text = ? WHERE id = ?",
            (now, answer_text, miss_id)
        )
        self._commit()

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

    def health(self) -> dict:
        """Self-awareness: report resource usage and health metrics."""
        import os
        import resource

        # Memory: RSS of this process
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = rusage.ru_maxrss / 1024  # Linux reports KB

        # CPU time used by this process
        cpu_user = rusage.ru_utime
        cpu_sys = rusage.ru_stime

        # Database file size
        db_size_bytes = 0
        db_path_str = ""
        if self.path:
            db_file = Path(self.path) / "neurons.db"
            if db_file.exists():
                db_size_bytes = db_file.stat().st_size
                db_path_str = str(db_file)

        # Matrix memory
        matrix_bytes = 0
        if self._vectors is not None:
            matrix_bytes = self._vectors.nbytes

        # Disk free
        disk_free_bytes = 0
        if self.path:
            st = os.statvfs(self.path)
            disk_free_bytes = st.f_bavail * st.f_frsize

        # Neuron stats
        n_neurons = self._n_rows
        n_dims = self._vectors.shape[1] if self._vectors is not None else 0

        # Columnar array memory
        columnar_bytes = (
            self._confidences.nbytes + self._timestamps.nbytes +
            self._temporals.nbytes + self._levels.nbytes
        )

        return {
            "neurons": n_neurons,
            "dimensions": n_dims,
            "columnar_mb": round(columnar_bytes / (1024 * 1024), 2),
            "matrix_mb": round(matrix_bytes / (1024 * 1024), 2),
            "db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
            "db_path": db_path_str,
            "rss_mb": round(rss_mb, 1),
            "cpu_user_s": round(cpu_user, 2),
            "cpu_sys_s": round(cpu_sys, 2),
            "disk_free_gb": round(disk_free_bytes / (1024 ** 3), 2),
        }

    def close(self):
        # Flush any pending batch writes
        if self._dirty:
            self.db.commit()
        self.db.close()
