"""
NeuronDB backed by LMDB — drop-in replacement for the SQLite NeuronDB.

Storage layout (named sub-databases within one LMDB environment):
  neurons/      key: neuron_id (4B int32) → packed struct (confidence f32, timestamp i64, temporal u8, level i8)
  vectors/      key: neuron_id (4B)       → raw float32 bytes (dim * 4)
  successors/   key: neuron_id (4B)       → packed [(id i32, conf f32), ...]
  predecessors/ key: neuron_id (4B)       → packed [id i32, ...]
  words/        key: word (utf-8)         → neuron_id (4B)
  sentences/    key: sentence_id (4B)     → packed [(neuron_id i32, position i32), ...]
  sent_index/   key: neuron_id (4B)       → packed [sentence_id i32, ...]
  misses/       key: miss_id (4B)         → msgpack dict
  meta/         key: name (utf-8)         → value bytes

Hot path: numpy arrays in memory (vectors + confidences), loaded from LMDB on init.
LMDB is the persistence layer; numpy is the compute layer.
"""

import struct
import time
from pathlib import Path
from typing import Optional

import lmdb
import numpy as np

from neuron import (
    CONFIDENCE_BOOST,
    CONFIDENCE_CAP,
    CONFIDENCE_DECAY,
    CONFIDENCE_FLOOR,
    DEFAULT_CONFIDENCE,
    MAX_PREDECESSORS,
    MAX_SUCCESSORS,
    VECTOR_DIM,
    Level,
    Neuron,
)

# --- Struct formats (little-endian) ---
_NEURON_FMT = struct.Struct('<f q ? b')  # confidence(f32), timestamp(i64), temporal(bool), level(i8)
_ID_FMT = struct.Struct('<i')            # single int32
_ID_CONF_FMT = struct.Struct('<i f')     # (id, confidence)
_SENT_ENTRY_FMT = struct.Struct('<i i')  # (neuron_id, position)


def _pack_id(nid: int) -> bytes:
    return _ID_FMT.pack(nid)


def _unpack_id(data: bytes) -> int:
    return _ID_FMT.unpack(data)[0]


def _encode_successors(successors: list) -> bytes:
    parts = []
    for sid, conf in successors:
        parts.append(_ID_CONF_FMT.pack(sid, conf))
    return b''.join(parts)


def _decode_successors(data: bytes) -> list:
    sz = _ID_CONF_FMT.size
    return [_ID_CONF_FMT.unpack(data[i:i + sz]) for i in range(0, len(data), sz)]


def _encode_predecessors(predecessors: list) -> bytes:
    return b''.join(_ID_FMT.pack(pid) for pid in predecessors)


def _decode_predecessors(data: bytes) -> list:
    sz = _ID_FMT.size
    return [_ID_FMT.unpack(data[i:i + sz])[0] for i in range(0, len(data), sz)]


def _encode_sentence(entries: list) -> bytes:
    """Encode [(neuron_id, position), ...]."""
    return b''.join(_SENT_ENTRY_FMT.pack(nid, pos) for nid, pos in entries)


def _decode_sentence(data: bytes) -> list:
    sz = _SENT_ENTRY_FMT.size
    return [_SENT_ENTRY_FMT.unpack(data[i:i + sz]) for i in range(0, len(data), sz)]


def _encode_id_list(ids: list) -> bytes:
    return b''.join(_ID_FMT.pack(i) for i in ids)


def _decode_id_list(data: bytes) -> list:
    sz = _ID_FMT.size
    return [_ID_FMT.unpack(data[i:i + sz])[0] for i in range(0, len(data), sz)]


# Simple msgpack-like encoding for miss records (avoid external dependency).
# We use a minimal JSON-in-bytes approach.
import json as _json


def _encode_miss(d: dict) -> bytes:
    # Separate out the vector bytes, store rest as JSON + vector appended
    vec_bytes = d.get('query_vector', b'')
    meta = {k: v for k, v in d.items() if k != 'query_vector'}
    meta_bytes = _json.dumps(meta).encode('utf-8')
    # Format: 4-byte meta length + meta JSON + remaining = vector bytes
    return _ID_FMT.pack(len(meta_bytes)) + meta_bytes + vec_bytes


def _decode_miss(data: bytes) -> dict:
    meta_len = _ID_FMT.unpack(data[:4])[0]
    meta = _json.loads(data[4:4 + meta_len].decode('utf-8'))
    meta['query_vector'] = data[4 + meta_len:]
    return meta


class _LMDBCooccurrenceShim:
    """Minimal shim so brain_core.py's `self.db.db.execute(...)` calls work
    for the cooccurrence table. Stores cooccurrence in an LMDB sub-database."""

    def __init__(self, env, cooc_db):
        self._env = env
        self._db = cooc_db
        self._table_exists = False

    def execute(self, sql, params=None):
        """Route SQL-like calls to LMDB operations."""
        sql_lower = sql.strip().lower()

        if 'create table' in sql_lower or 'create index' in sql_lower:
            self._table_exists = True
            return _ShimCursor([])

        if 'select' in sql_lower and 'cooccurrence' in sql_lower:
            return self._select_all()

        if 'insert' in sql_lower and 'cooccurrence' in sql_lower:
            if params and len(params) == 3:
                self._insert(params[0], params[1], params[2])
            return _ShimCursor([])

        if 'delete' in sql_lower and 'cooccurrence' in sql_lower:
            return _ShimCursor([])

        # Fallback — return empty
        return _ShimCursor([])

    def _select_all(self):
        rows = []
        with self._env.begin(db=self._db) as txn:
            cursor = txn.cursor(db=self._db)
            for key, val in cursor:
                a, b = struct.unpack('<ii', key)
                w = struct.unpack('<f', val)[0]
                rows.append((a, b, w))
        return _ShimCursor(rows)

    def _insert(self, a, b, w):
        key = struct.pack('<ii', int(a), int(b))
        val = struct.pack('<f', float(w))
        with self._env.begin(write=True, db=self._db) as txn:
            txn.put(key, val, db=self._db)

    def commit(self):
        pass  # LMDB auto-commits per transaction


class _ShimCursor:
    """Minimal cursor returned by the cooccurrence shim."""
    def __init__(self, rows):
        self._rows = rows
        self.rowcount = len(rows)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class NeuronDBLMDB:
    """
    LMDB-backed neuron storage with numpy hot cache.

    Drop-in replacement for NeuronDB. Same public API.
    Vectors and confidences live in numpy arrays for search.
    Everything persists to LMDB.
    """

    _ALLOC_CHUNK = 4096  # Larger chunks for LMDB (less realloc)

    def __init__(self, path: str, dim: int = VECTOR_DIM,
                 map_size: int = 4 * 1024 * 1024 * 1024):
        self.dim = dim
        self.path = path
        self._next_id = 0

        # Ensure directory exists
        Path(path).mkdir(parents=True, exist_ok=True)

        # Open LMDB environment with named sub-databases
        self.env = lmdb.open(
            path,
            max_dbs=12,
            map_size=map_size,
            max_readers=126,
            readahead=False,      # We do our own caching
            meminit=False,        # Don't zero pages
            lock=True,
        )

        # Named sub-databases
        self._db_neurons = self.env.open_db(b'neurons')
        self._db_vectors = self.env.open_db(b'vectors')
        self._db_successors = self.env.open_db(b'successors')
        self._db_predecessors = self.env.open_db(b'predecessors')
        self._db_words = self.env.open_db(b'words')
        self._db_sentences = self.env.open_db(b'sentences')
        self._db_sent_index = self.env.open_db(b'sent_index')
        self._db_misses = self.env.open_db(b'misses')
        self._db_meta = self.env.open_db(b'meta')
        self._db_cooc = self.env.open_db(b'cooccurrence')

        # Compatibility shim for brain_core.py's direct db.db access
        self.db = _LMDBCooccurrenceShim(self.env, self._db_cooc)

        # In-memory hot cache
        self._vectors = None
        self._n_rows = 0
        self._id_to_row = {}
        self._row_to_id = {}

        self._confidences = np.zeros(self._ALLOC_CHUNK, dtype=np.float32)
        self._timestamps = np.zeros(self._ALLOC_CHUNK, dtype=np.int64)
        self._temporals = np.zeros(self._ALLOC_CHUNK, dtype=np.bool_)
        self._levels = np.zeros(self._ALLOC_CHUNK, dtype=np.int8)

        # Batch mode
        self._batch = False
        self._batch_txn = None

        # Word mapping cache
        self._word_map_cache = None

        # Sentence ID counter
        self._next_sentence_id = None

        # Miss ID counter
        self._next_miss_id = None

        # Load from LMDB
        self._load_meta()
        self._load_from_lmdb()

    # --- Meta helpers ---

    def _load_meta(self):
        with self.env.begin(db=self._db_meta) as txn:
            val = txn.get(b'next_id', db=self._db_meta)
            if val is not None:
                self._next_id = _unpack_id(val)

            val = txn.get(b'next_sentence_id', db=self._db_meta)
            if val is not None:
                self._next_sentence_id = _unpack_id(val)
            else:
                self._next_sentence_id = 0

            val = txn.get(b'next_miss_id', db=self._db_meta)
            if val is not None:
                self._next_miss_id = _unpack_id(val)
            else:
                self._next_miss_id = 0

    def _save_meta(self, txn=None):
        """Save meta counters. Uses provided txn or creates one."""
        if txn is not None:
            txn.put(b'next_id', _pack_id(self._next_id), db=self._db_meta)
            txn.put(b'next_sentence_id', _pack_id(self._next_sentence_id), db=self._db_meta)
            txn.put(b'next_miss_id', _pack_id(self._next_miss_id), db=self._db_meta)
        else:
            with self.env.begin(write=True, db=self._db_meta) as t:
                t.put(b'next_id', _pack_id(self._next_id), db=self._db_meta)
                t.put(b'next_sentence_id', _pack_id(self._next_sentence_id), db=self._db_meta)
                t.put(b'next_miss_id', _pack_id(self._next_miss_id), db=self._db_meta)

    # --- Batch mode ---

    def begin_batch(self):
        """Start batch mode — accumulate writes in a single transaction."""
        self._batch = True
        self._batch_txn = self.env.begin(write=True)

    def end_batch(self):
        """End batch mode — commit the accumulated transaction."""
        if self._batch_txn is not None:
            # Save meta inside the batch transaction
            self._batch_txn.put(b'next_id', _pack_id(self._next_id), db=self._db_meta)
            self._batch_txn.put(b'next_sentence_id', _pack_id(self._next_sentence_id), db=self._db_meta)
            self._batch_txn.put(b'next_miss_id', _pack_id(self._next_miss_id), db=self._db_meta)
            self._batch_txn.commit()
            self._batch_txn = None
        self._batch = False

    def _get_write_txn(self):
        """Get a write transaction — batch or new."""
        if self._batch and self._batch_txn is not None:
            return self._batch_txn
        return None

    def _write(self, fn):
        """Execute fn(txn) in the batch txn or a new one-shot txn."""
        if self._batch and self._batch_txn is not None:
            fn(self._batch_txn)
        else:
            with self.env.begin(write=True) as txn:
                fn(txn)

    # --- Load from LMDB ---

    def _load_from_lmdb(self):
        """Rebuild numpy hot cache from LMDB on startup."""
        with self.env.begin() as txn:
            # Count neurons
            cursor = txn.cursor(db=self._db_neurons)
            entries = []
            for key, val in cursor:
                nid = _unpack_id(key)
                conf, ts, temporal, level = _NEURON_FMT.unpack(val)
                entries.append((nid, conf, ts, temporal, level))

        if not entries:
            return

        # Load vectors
        n = len(entries)
        alloc = ((n // self._ALLOC_CHUNK) + 1) * self._ALLOC_CHUNK

        # Check if vectors sub-db has any data
        has_vectors = False
        with self.env.begin() as txn:
            vec_stat = txn.stat(db=self._db_vectors)
            if vec_stat['entries'] > 0:
                has_vectors = True
                # Determine dimension from first vector
                first_nid = entries[0][0]
                vec_bytes = txn.get(_pack_id(first_nid), db=self._db_vectors)
                if vec_bytes is not None:
                    actual_dim = len(vec_bytes) // 4
                    if actual_dim > 0:
                        self.dim = actual_dim

        self._vectors = np.zeros((alloc, self.dim), dtype=np.float32)
        self._confidences = np.zeros(alloc, dtype=np.float32)
        self._timestamps = np.zeros(alloc, dtype=np.int64)
        self._temporals = np.zeros(alloc, dtype=np.bool_)
        self._levels = np.zeros(alloc, dtype=np.int8)
        self._n_rows = n

        with self.env.begin() as txn:
            for row_idx, (nid, conf, ts, temporal, level) in enumerate(entries):
                self._id_to_row[nid] = row_idx
                self._row_to_id[row_idx] = nid
                self._confidences[row_idx] = conf
                self._timestamps[row_idx] = ts
                self._temporals[row_idx] = temporal
                self._levels[row_idx] = level

            # Only load vectors if the sub-db has data (skip 408K
            # empty lookups when vectors weren't stored by feed.py)
            if has_vectors:
                for row_idx, (nid, conf, ts, temporal, level) in enumerate(entries):
                    vec_bytes = txn.get(_pack_id(nid), db=self._db_vectors)
                    if vec_bytes is not None:
                        vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
                        if len(vec) < self.dim:
                            vec = np.pad(vec, (0, self.dim - len(vec)))
                        elif len(vec) > self.dim:
                            vec = vec[:self.dim]
                        self._vectors[row_idx] = vec

        max_nid = max(e[0] for e in entries)
        if max_nid + 1 > self._next_id:
            self._next_id = max_nid + 1

    # --- Matrix management ---

    def _grow_columnar_arrays(self, target_size: int):
        if len(self._confidences) >= target_size:
            return
        extra = target_size - len(self._confidences)
        self._confidences = np.concatenate([self._confidences, np.zeros(extra, dtype=np.float32)])
        self._timestamps = np.concatenate([self._timestamps, np.zeros(extra, dtype=np.int64)])
        self._temporals = np.concatenate([self._temporals, np.zeros(extra, dtype=np.bool_)])
        self._levels = np.concatenate([self._levels, np.zeros(extra, dtype=np.int8)])

    def _add_vec_to_matrix(self, vec: np.ndarray) -> int:
        if self._vectors is None:
            d = max(vec.shape[0], self.dim)
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

    # --- Core operations ---

    def insert(self, vector: np.ndarray, confidence: float = DEFAULT_CONFIDENCE,
               level: Level = Level.WORD, temporal: bool = False) -> Neuron:
        nid = self._next_id
        self._next_id += 1

        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        now = int(time.time())

        # Add to hot cache
        row_idx = self._add_vec_to_matrix(vec)
        self._id_to_row[nid] = row_idx
        self._row_to_id[row_idx] = nid
        self._confidences[row_idx] = confidence
        self._timestamps[row_idx] = now
        self._temporals[row_idx] = temporal
        self._levels[row_idx] = int(level)

        # Persist to LMDB
        key = _pack_id(nid)
        neuron_val = _NEURON_FMT.pack(confidence, now, temporal, int(level))
        vec_val = vec.tobytes()

        def do_write(txn):
            txn.put(key, neuron_val, db=self._db_neurons)
            txn.put(key, vec_val, db=self._db_vectors)
            txn.put(key, b'', db=self._db_successors)
            txn.put(key, b'', db=self._db_predecessors)
            if not self._batch:
                txn.put(b'next_id', _pack_id(self._next_id), db=self._db_meta)

        self._write(do_write)
        self._word_map_cache = None

        return Neuron(
            id=nid, vector=vec, confidence=confidence,
            timestamp=now, temporal=temporal, level=level,
        )

    def insert_bulk(self, vectors: np.ndarray,
                    confidences: np.ndarray = None,
                    levels: np.ndarray = None) -> int:
        """High-throughput bulk insert. Returns the first neuron ID.

        vectors: (N, dim) float32 array — will be L2-normalized.
        confidences: optional (N,) float32 array, defaults to DEFAULT_CONFIDENCE.
        levels: optional (N,) int8 array, defaults to Level.WORD.
        """
        n = vectors.shape[0]
        dim = vectors.shape[1]
        vecs = np.array(vectors, dtype=np.float32)

        # Bulk normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        if confidences is None:
            confidences = np.full(n, DEFAULT_CONFIDENCE, dtype=np.float32)
        if levels is None:
            levels = np.full(n, int(Level.WORD), dtype=np.int8)

        now = int(time.time())
        first_id = self._next_id

        # Pre-allocate matrix space
        if self._vectors is None:
            alloc = max(self._ALLOC_CHUNK, ((n // self._ALLOC_CHUNK) + 1) * self._ALLOC_CHUNK)
            self._vectors = np.zeros((alloc, dim), dtype=np.float32)
            self._n_rows = 0
        needed = self._n_rows + n
        alloc_target = ((needed // self._ALLOC_CHUNK) + 1) * self._ALLOC_CHUNK
        if needed > self._vectors.shape[0]:
            new_mat = np.zeros((alloc_target, self._vectors.shape[1]), dtype=np.float32)
            new_mat[:self._n_rows] = self._vectors[:self._n_rows]
            self._vectors = new_mat
        self._grow_columnar_arrays(alloc_target)

        # Bulk copy into arrays
        start_row = self._n_rows
        self._vectors[start_row:start_row + n] = vecs
        self._confidences[start_row:start_row + n] = confidences
        self._timestamps[start_row:start_row + n] = now
        self._levels[start_row:start_row + n] = levels

        # Build ID mappings
        for i in range(n):
            nid = first_id + i
            row = start_row + i
            self._id_to_row[nid] = row
            self._row_to_id[row] = nid

        self._n_rows += n
        self._next_id = first_id + n

        # Persist to LMDB in one transaction
        _empty = b''
        with self.env.begin(write=True) as txn:
            for i in range(n):
                nid = first_id + i
                key = _pack_id(nid)
                txn.put(key, _NEURON_FMT.pack(
                    float(confidences[i]), now, False, int(levels[i])
                ), db=self._db_neurons)
                txn.put(key, vecs[i].tobytes(), db=self._db_vectors)
                txn.put(key, _empty, db=self._db_successors)
                txn.put(key, _empty, db=self._db_predecessors)
            txn.put(b'next_id', _pack_id(self._next_id), db=self._db_meta)

        self._word_map_cache = None
        return first_id

    def get(self, neuron_id: int) -> Optional[Neuron]:
        row_idx = self._id_to_row.get(neuron_id)
        if row_idx is None:
            return None

        key = _pack_id(neuron_id)
        with self.env.begin() as txn:
            succ_bytes = txn.get(key, db=self._db_successors)
            pred_bytes = txn.get(key, db=self._db_predecessors)

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

    def get_confidence(self, neuron_id: int) -> float:
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return 0.0
        return float(self._confidences[row])

    def get_vector(self, neuron_id: int) -> np.ndarray:
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return np.zeros(self.dim, dtype=np.float32)
        return self._vectors[row]

    def get_timestamp(self, neuron_id: int) -> int:
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return 0
        return int(self._timestamps[row])

    def get_level(self, neuron_id: int) -> int:
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return int(Level.WORD)
        return int(self._levels[row])

    def get_temporal(self, neuron_id: int) -> bool:
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return False
        return bool(self._temporals[row])

    def delete(self, neuron_id: int) -> bool:
        if neuron_id not in self._id_to_row:
            return False

        key = _pack_id(neuron_id)

        def do_write(txn):
            txn.delete(key, db=self._db_neurons)
            txn.delete(key, db=self._db_vectors)
            txn.delete(key, db=self._db_successors)
            txn.delete(key, db=self._db_predecessors)

        self._write(do_write)

        # Rebuild hot cache (same as SQLite version — rebuild after delete)
        self._rebuild_cache()
        self._word_map_cache = None
        return True

    def _rebuild_cache(self):
        """Rebuild numpy cache from LMDB. Used after deletes."""
        self._vectors = None
        self._n_rows = 0
        self._id_to_row = {}
        self._row_to_id = {}
        self._load_from_lmdb()

    def count(self) -> int:
        return self._n_rows

    # --- Search ---

    def search_ids(self, query_vector: np.ndarray, k: int = 5) -> list:
        if self._vectors is None or self._n_rows == 0:
            return []

        vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        vec = vec / norm

        mat = self._vectors[:self._n_rows]
        if vec.shape[0] < mat.shape[1]:
            vec = np.pad(vec, (0, mat.shape[1] - vec.shape[0]))
        elif vec.shape[0] > mat.shape[1]:
            vec = vec[:mat.shape[1]]

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
        id_sims = self.search_ids(query_vector, k=k)
        results = []
        for nid, sim in id_sims:
            neuron = self.get(nid)
            if neuron is not None:
                results.append(neuron)
        return results

    # --- Confidence ---

    def update_confidence(self, neuron_id: int, useful: bool):
        row = self._id_to_row.get(neuron_id)
        if row is None:
            return

        conf = float(self._confidences[row])
        if useful:
            conf = min(conf * CONFIDENCE_BOOST, CONFIDENCE_CAP)
        else:
            conf = max(conf * CONFIDENCE_DECAY, CONFIDENCE_FLOOR)

        self._confidences[row] = conf

        # Persist — update the neuron struct in LMDB
        key = _pack_id(neuron_id)
        neuron_val = _NEURON_FMT.pack(
            conf,
            int(self._timestamps[row]),
            bool(self._temporals[row]),
            int(self._levels[row]),
        )

        def do_write(txn):
            txn.put(key, neuron_val, db=self._db_neurons)

        self._write(do_write)

    # --- Successors / Predecessors ---

    def update_successors(self, neuron_id: int, successor_id: int, conf: float):
        if neuron_id not in self._id_to_row:
            return

        key = _pack_id(neuron_id)
        with self.env.begin() as txn:
            data = txn.get(key, db=self._db_successors)

        successors = _decode_successors(data) if data else []

        # Update or insert
        for i, (sid, _) in enumerate(successors):
            if sid == successor_id:
                successors[i] = (successor_id, conf)
                def do_write(txn):
                    txn.put(key, _encode_successors(successors), db=self._db_successors)
                self._write(do_write)
                return

        if len(successors) < MAX_SUCCESSORS:
            successors.append((successor_id, conf))
        else:
            min_idx = min(range(len(successors)), key=lambda i: successors[i][1])
            if conf > successors[min_idx][1]:
                successors[min_idx] = (successor_id, conf)

        encoded = _encode_successors(successors)

        def do_write(txn):
            txn.put(key, encoded, db=self._db_successors)

        self._write(do_write)

    def update_predecessors(self, neuron_id: int, predecessor_id: int):
        if neuron_id not in self._id_to_row:
            return

        key = _pack_id(neuron_id)
        with self.env.begin() as txn:
            data = txn.get(key, db=self._db_predecessors)

        predecessors = _decode_predecessors(data) if data else []

        if predecessor_id in predecessors:
            return

        if len(predecessors) < MAX_PREDECESSORS:
            predecessors.append(predecessor_id)
        else:
            predecessors.pop(0)
            predecessors.append(predecessor_id)

        encoded = _encode_predecessors(predecessors)

        def do_write(txn):
            txn.put(key, encoded, db=self._db_predecessors)

        self._write(do_write)

    # --- Word mappings ---

    def save_word_mapping(self, word: str, neuron_id: int):
        key = word.encode('utf-8')
        val = _pack_id(neuron_id)

        def do_write(txn):
            txn.put(key, val, db=self._db_words)

        self._write(do_write)
        self._word_map_cache = None

    def load_word_mappings(self) -> dict:
        if self._word_map_cache is not None:
            return self._word_map_cache

        result = {}
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self._db_words)
            for key, val in cursor:
                word = key.decode('utf-8')
                nid = _unpack_id(val)
                result[word] = nid

        self._word_map_cache = result
        return result

    def delete_word_mapping(self, word: str) -> bool:
        key = word.encode('utf-8')
        deleted = False

        def do_write(txn):
            nonlocal deleted
            deleted = txn.delete(key, db=self._db_words)

        self._write(do_write)
        self._word_map_cache = None
        return deleted

    # --- Sentence associations ---

    def record_sentence(self, neuron_ids: list) -> int:
        if self._next_sentence_id is None:
            self._next_sentence_id = 0

        sid = self._next_sentence_id
        self._next_sentence_id += 1

        entries = [(nid, pos) for pos, nid in enumerate(neuron_ids)]
        sent_key = _pack_id(sid)
        sent_val = _encode_sentence(entries)

        def do_write(txn):
            txn.put(sent_key, sent_val, db=self._db_sentences)
            # Update reverse index: neuron_id → [sentence_ids]
            for nid, _ in entries:
                nid_key = _pack_id(nid)
                existing = txn.get(nid_key, db=self._db_sent_index)
                if existing:
                    ids = _decode_id_list(existing)
                    if sid not in ids:
                        ids.append(sid)
                    txn.put(nid_key, _encode_id_list(ids), db=self._db_sent_index)
                else:
                    txn.put(nid_key, _pack_id(sid), db=self._db_sent_index)

            if not self._batch:
                txn.put(b'next_sentence_id', _pack_id(self._next_sentence_id), db=self._db_meta)

        self._write(do_write)
        return sid

    def get_sentences_for_neurons(self, neuron_ids: list) -> dict:
        if not neuron_ids:
            return {}

        sentences = {}
        with self.env.begin() as txn:
            # Collect all sentence IDs for the given neurons
            all_sids = set()
            for nid in neuron_ids:
                data = txn.get(_pack_id(nid), db=self._db_sent_index)
                if data:
                    all_sids.update(_decode_id_list(data))

            # Fetch each sentence
            for sid in sorted(all_sids):
                data = txn.get(_pack_id(sid), db=self._db_sentences)
                if data:
                    entries = _decode_sentence(data)
                    sentences[sid] = entries

        return sentences

    def get_sentence_neurons(self, sentence_id: int) -> list:
        with self.env.begin() as txn:
            data = txn.get(_pack_id(sentence_id), db=self._db_sentences)
        if not data:
            return []
        return _decode_sentence(data)

    def get_cooccurring_neurons(self, neuron_id: int) -> list:
        """Find neurons that co-occur with neuron_id in sentences."""
        results = []
        with self.env.begin() as txn:
            # Get sentence IDs for this neuron
            idx_data = txn.get(_pack_id(neuron_id), db=self._db_sent_index)
            if not idx_data:
                return []
            sids = _decode_id_list(idx_data)

            for sid in sids:
                sent_data = txn.get(_pack_id(sid), db=self._db_sentences)
                if sent_data:
                    for co_nid, pos in _decode_sentence(sent_data):
                        if co_nid != neuron_id:
                            results.append((co_nid, pos, sid))

        results.sort(key=lambda x: (x[2], x[1]))
        return results

    # --- Miss logging ---

    def log_miss(self, query_text: str, query_vector: np.ndarray) -> int:
        if self._next_miss_id is None:
            self._next_miss_id = 0

        mid = self._next_miss_id
        self._next_miss_id += 1

        vec_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        now = int(time.time())

        miss_data = {
            'query_text': query_text,
            'timestamp': now,
            'resolved': 0,
            'resolved_timestamp': None,
            'answer_text': None,
            'query_vector': vec_bytes,
        }

        key = _pack_id(mid)
        val = _encode_miss(miss_data)

        def do_write(txn):
            txn.put(key, val, db=self._db_misses)
            if not self._batch:
                txn.put(b'next_miss_id', _pack_id(self._next_miss_id), db=self._db_meta)

        self._write(do_write)
        return mid

    def resolve_miss(self, miss_id: int, answer_text: str):
        key = _pack_id(miss_id)
        with self.env.begin() as txn:
            data = txn.get(key, db=self._db_misses)
        if not data:
            return

        miss = _decode_miss(data)
        miss['resolved'] = 1
        miss['resolved_timestamp'] = int(time.time())
        miss['answer_text'] = answer_text

        def do_write(txn):
            txn.put(key, _encode_miss(miss), db=self._db_misses)

        self._write(do_write)

    def resolve_miss_by_query(self, query_text: str, answer_text: str) -> bool:
        # Scan misses for matching unresolved query (newest first)
        best_id = None
        best_ts = -1
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self._db_misses)
            for key, val in cursor:
                miss = _decode_miss(val)
                if miss['query_text'] == query_text and miss['resolved'] == 0:
                    if miss['timestamp'] > best_ts:
                        best_ts = miss['timestamp']
                        best_id = _unpack_id(key)

        if best_id is not None:
            self.resolve_miss(best_id, answer_text)
            return True
        return False

    def get_unresolved_misses(self, limit: int = 50) -> list:
        results = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self._db_misses)
            for key, val in cursor:
                miss = _decode_miss(val)
                if miss['resolved'] == 0:
                    results.append((_unpack_id(key), miss['query_text'], miss['timestamp']))

        # Sort by timestamp descending, limit
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    def miss_stats(self) -> dict:
        total = 0
        resolved = 0
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self._db_misses)
            for key, val in cursor:
                total += 1
                miss = _decode_miss(val)
                if miss['resolved'] == 1:
                    resolved += 1

        unresolved = total - resolved
        return {
            "total_misses": total,
            "resolved": resolved,
            "unresolved": unresolved,
            "resolution_rate": resolved / total if total > 0 else 0.0,
        }

    # --- Templates (compatibility) ---

    def save_template(self, template_id: int, pattern: str, slots_json: str,
                      confidence: float, vector: np.ndarray):
        # Templates use the meta db with a prefix
        key = f"tmpl:{template_id}".encode('utf-8')
        vec_bytes = np.array(vector, dtype=np.float32).tobytes()
        val = _json.dumps({
            'id': template_id,
            'pattern': pattern,
            'slots': slots_json,
            'confidence': confidence,
        }).encode('utf-8')
        # Pack: 4-byte json len + json + vector bytes
        packed = _ID_FMT.pack(len(val)) + val + vec_bytes

        def do_write(txn):
            txn.put(key, packed, db=self._db_meta)

        self._write(do_write)

    def load_templates(self) -> list:
        results = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=self._db_meta)
            for key, val in cursor:
                k = key.decode('utf-8')
                if k.startswith('tmpl:'):
                    json_len = _ID_FMT.unpack(val[:4])[0]
                    meta = _json.loads(val[4:4 + json_len].decode('utf-8'))
                    vec = np.frombuffer(val[4 + json_len:], dtype=np.float32).copy()
                    results.append((
                        meta['id'], meta['pattern'], meta['slots'],
                        meta['confidence'], vec
                    ))
        results.sort(key=lambda x: x[0])
        return results

    def delete_template(self, template_id: int) -> bool:
        key = f"tmpl:{template_id}".encode('utf-8')
        deleted = False

        def do_write(txn):
            nonlocal deleted
            deleted = txn.delete(key, db=self._db_meta)

        self._write(do_write)
        return deleted

    # --- Compatibility ---

    def save_index(self):
        """No-op for compatibility."""
        pass

    def health(self) -> dict:
        import os
        import resource

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = rusage.ru_maxrss / 1024

        # LMDB environment info
        info = self.env.info()
        stat = self.env.stat()

        # DB file size
        db_size_bytes = 0
        db_path_str = ""
        if self.path:
            data_file = Path(self.path) / "data.mdb"
            if data_file.exists():
                db_size_bytes = data_file.stat().st_size
                db_path_str = str(data_file)

        matrix_bytes = 0
        if self._vectors is not None:
            matrix_bytes = self._vectors.nbytes

        columnar_bytes = (
            self._confidences.nbytes + self._timestamps.nbytes +
            self._temporals.nbytes + self._levels.nbytes
        )

        disk_free_bytes = 0
        if self.path:
            st = os.statvfs(self.path)
            disk_free_bytes = st.f_bavail * st.f_frsize

        return {
            "neurons": self._n_rows,
            "dimensions": self._vectors.shape[1] if self._vectors is not None else 0,
            "columnar_mb": round(columnar_bytes / (1024 * 1024), 2),
            "matrix_mb": round(matrix_bytes / (1024 * 1024), 2),
            "db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
            "db_path": db_path_str,
            "rss_mb": round(rss_mb, 1),
            "cpu_user_s": round(rusage.ru_utime, 2),
            "cpu_sys_s": round(rusage.ru_stime, 2),
            "disk_free_gb": round(disk_free_bytes / (1024 ** 3), 2),
            "backend": "lmdb",
            "lmdb_map_size_mb": round(info['map_size'] / (1024 * 1024), 2),
            "lmdb_pages_used": stat['psize'] * stat['leaf_pages'],
        }

    def close(self):
        if self._batch and self._batch_txn is not None:
            try:
                self._batch_txn.commit()
            except Exception:
                pass
            self._batch_txn = None
        self._save_meta()
        self.env.close()
