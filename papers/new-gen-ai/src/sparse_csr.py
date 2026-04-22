"""
Memory-mapped CSR (Compressed Sparse Row) storage for co-occurrence graphs.

Replaces Python dict-of-dicts with three numpy.memmap arrays:
  indptr.bin   — row pointers (V+1 int32)
  indices.bin  — column indices (nnz int32), sorted within each row
  data.bin     — edge weights (nnz float32)

43M edges: ~346MB on disk, ~50MB RSS (OS pages in only accessed rows).
Python dicts for the same: ~4GB. 11x reduction.

Write-Ahead Log (WAL) buffers real-time edge updates (teach/RLHF)
without rebuilding the CSR. Reads merge CSR + WAL transparently.
"""

import os
import math
import threading
import numpy as np

try:
    from scipy.sparse import csr_matrix as scipy_csr, coo_matrix as scipy_coo
    from scipy.sparse.linalg import norm as sparse_norm_scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MMapCSR:
    """Memory-mapped CSR matrix backed by three flat files.

    Zero-copy row access via numpy views into mmap'd arrays.
    OS page cache manages hot/cold rows automatically.
    """

    def __init__(self, path: str, readonly: bool = True):
        """Open an existing CSR from disk.

        Args:
            path: directory containing indptr.bin, indices.bin, data.bin
            readonly: if True, open read-only (no writes to mmap)
        """
        self._path = path
        mode = 'r' if readonly else 'r+'

        self._indptr = np.memmap(
            os.path.join(path, 'indptr.bin'),
            dtype=np.int32, mode=mode,
        )
        self._indices = np.memmap(
            os.path.join(path, 'indices.bin'),
            dtype=np.int32, mode=mode,
        )
        self._data = np.memmap(
            os.path.join(path, 'data.bin'),
            dtype=np.float32, mode=mode,
        )

        self.num_rows = len(self._indptr) - 1
        self.nnz = len(self._indices)

        # Pre-compute per-row norms for fast cosine
        self._row_norms = None  # lazy

        # scipy view (lazy)
        self._scipy_mat = None

    @property
    def scipy_matrix(self):
        """scipy.sparse.csr_matrix view over the same mmap'd data. No copy."""
        if self._scipy_mat is None:
            if not HAS_SCIPY:
                raise ImportError("scipy required for spmv. pip install scipy")
            self._scipy_mat = scipy_csr(
                (self._data, self._indices, self._indptr),
                shape=(self.num_rows, self.num_rows),
            )
        return self._scipy_mat

    def get_row(self, row_idx: int):
        """Get (col_indices, weights) for a row. Zero-copy numpy views.

        Returns:
            (np.ndarray[int32], np.ndarray[float32]) — column indices and weights
        """
        if row_idx < 0 or row_idx >= self.num_rows:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        start = int(self._indptr[row_idx])
        end = int(self._indptr[row_idx + 1])
        if start == end:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        return self._indices[start:end], self._data[start:end]

    def get_row_dict(self, row_idx: int) -> dict:
        """Get row as {col_idx: weight} dict. Compatibility shim for convergence."""
        cols, vals = self.get_row(row_idx)
        if len(cols) == 0:
            return {}
        return dict(zip(cols.astype(int), vals.astype(float)))

    def row_dot(self, row_idx: int, query: dict) -> float:
        """Dot product between a CSR row and a sparse query dict."""
        cols, vals = self.get_row(row_idx)
        if len(cols) == 0:
            return 0.0
        total = 0.0
        for i in range(len(cols)):
            c = int(cols[i])
            if c in query:
                total += float(vals[i]) * query[c]
        return total

    def row_norm(self, row_idx: int) -> float:
        """L2 norm of a row."""
        _, vals = self.get_row(row_idx)
        if len(vals) == 0:
            return 0.0
        return float(np.sqrt(np.dot(vals, vals)))

    def row_cosine(self, row_idx: int, query: dict, q_norm: float = 0.0) -> float:
        """Cosine similarity between a CSR row and a sparse query dict."""
        dot = self.row_dot(row_idx, query)
        if dot == 0.0:
            return 0.0
        rn = self.row_norm(row_idx)
        if rn == 0.0:
            return 0.0
        if q_norm == 0.0:
            q_norm = math.sqrt(sum(v * v for v in query.values()))
        if q_norm == 0.0:
            return 0.0
        return dot / (rn * q_norm)

    def spmv(self, query_vec: np.ndarray) -> np.ndarray:
        """Sparse matrix-vector multiply: CSR @ query_vec.

        Uses scipy BLAS if available, else manual row iteration.
        """
        if HAS_SCIPY:
            return np.asarray(self.scipy_matrix @ query_vec).ravel()
        # Fallback: manual
        result = np.zeros(self.num_rows, dtype=np.float32)
        for i in range(self.num_rows):
            start = int(self._indptr[i])
            end = int(self._indptr[i + 1])
            if start == end:
                continue
            cols = self._indices[start:end]
            vals = self._data[start:end]
            result[i] = np.dot(vals, query_vec[cols])
        return result

    @property
    def row_norms_vec(self):
        """Precomputed L2 norms for all rows. Lazy, cached."""
        if self._row_norms is None:
            if HAS_SCIPY:
                mat = self.scipy_matrix
                # Per-row L2 norm
                self._row_norms = np.sqrt(
                    np.asarray(mat.multiply(mat).sum(axis=1)).ravel()
                )
            else:
                norms = np.zeros(self.num_rows, dtype=np.float32)
                for i in range(self.num_rows):
                    _, vals = self.get_row(i)
                    if len(vals) > 0:
                        norms[i] = float(np.sqrt(np.dot(vals, vals)))
                self._row_norms = norms
        return self._row_norms

    def stats(self) -> dict:
        """Return size stats."""
        return {
            'num_rows': self.num_rows,
            'nnz': self.nnz,
            'avg_degree': self.nnz / max(self.num_rows, 1),
            'disk_mb': (
                os.path.getsize(os.path.join(self._path, 'indptr.bin')) +
                os.path.getsize(os.path.join(self._path, 'indices.bin')) +
                os.path.getsize(os.path.join(self._path, 'data.bin'))
            ) / (1024 * 1024),
        }


class CSRWriteAheadLog:
    """Buffers real-time edge updates without rebuilding CSR.

    teach() writes to WAL + LMDB. ask() reads CSR + WAL overlay merged.
    When WAL exceeds max_entries, signal for background CSR rebuild.
    """

    def __init__(self, max_entries: int = 100_000):
        self._log = {}       # word_idx → {neighbor_idx: weight}
        self._count = 0
        self._max = max_entries
        self._lock = threading.Lock()

    @property
    def needs_flush(self) -> bool:
        return self._count >= self._max

    @property
    def entry_count(self) -> int:
        return self._count

    # Maximum absolute edge weight. Prevents explosion from RLHF boosting.
    # At 100.0, cosine similarity stays numerically stable in float32.
    WEIGHT_CLAMP = 100.0

    def add_edge(self, word_a: int, word_b: int, weight: float):
        """Buffer a co-occurrence edge update. Weights clamped to ±WEIGHT_CLAMP."""
        with self._lock:
            if word_a not in self._log:
                self._log[word_a] = {}
            if word_b not in self._log:
                self._log[word_b] = {}
            old_a = self._log[word_a].get(word_b, 0)
            old_b = self._log[word_b].get(word_a, 0)
            new_a = max(-self.WEIGHT_CLAMP, min(self.WEIGHT_CLAMP, old_a + weight))
            new_b = max(-self.WEIGHT_CLAMP, min(self.WEIGHT_CLAMP, old_b + weight))
            self._log[word_a][word_b] = new_a
            self._log[word_b][word_a] = new_b
            self._count += 2

    def get_overlay(self, word_idx: int) -> dict:
        """Get pending WAL updates for one word. Returns {} if none."""
        with self._lock:
            return dict(self._log.get(word_idx, {}))

    def merge_read(self, csr: MMapCSR, word_idx: int) -> dict:
        """Read CSR row + WAL overlay, merged into one dict.

        This is the primary read path for convergence.
        """
        # Base from CSR (zero-copy read, converted to dict)
        if word_idx < csr.num_rows:
            base = csr.get_row_dict(word_idx)
        else:
            # New word beyond CSR — lives entirely in WAL
            base = {}

        # Overlay from WAL
        overlay = self.get_overlay(word_idx)
        if not overlay:
            # Self-connection if empty
            if not base:
                return {word_idx: 1.0}
            return base

        # Merge: WAL adds to base, clamped
        merged = dict(base)
        clamp = self.WEIGHT_CLAMP
        for k, v in overlay.items():
            merged[k] = max(-clamp, min(clamp, merged.get(k, 0) + v))
        if not merged:
            merged[word_idx] = 1.0
        return merged

    def clear(self):
        """Reset WAL after successful CSR rebuild."""
        with self._lock:
            self._log.clear()
            self._count = 0
            self._scipy_cache = None

    def snapshot(self) -> dict:
        """Get a frozen copy of all WAL entries (for CSR rebuild)."""
        with self._lock:
            return {k: dict(v) for k, v in self._log.items()}

    # --- Persistence (LMDB-backed, crash-safe) ---

    _EDGE_FMT = '<iif'  # word_a, word_b, weight — 12 bytes per edge
    _EDGE_SIZE = 12

    def persist_to_lmdb(self, env, wal_db):
        """Write all WAL edges to LMDB wal_edges sub-db. Atomic transaction."""
        import struct
        snapshot = self.snapshot()
        if not snapshot:
            return 0
        # Flatten to unique directed edges (a→b, not both a→b and b→a)
        seen = set()
        edges = []
        for word_a, neighbors in snapshot.items():
            for word_b, weight in neighbors.items():
                key = (min(word_a, word_b), max(word_a, word_b))
                if key not in seen:
                    seen.add(key)
                    # Clamp to float32 range to prevent overflow
                    clamped = max(-1e30, min(1e30, weight))
                    edges.append((word_a, word_b, clamped))
        with env.begin(write=True) as txn:
            # Clear old WAL entries
            txn.drop(wal_db, delete=False)
            # Write all edges as sequential keys
            for i, (a, b, w) in enumerate(edges):
                txn.put(
                    struct.pack('<i', i),
                    struct.pack(self._EDGE_FMT, a, b, w),
                    db=wal_db,
                )
            txn.put(b'__count__', struct.pack('<i', len(edges)), db=wal_db)
        return len(edges)

    def load_from_lmdb(self, env, wal_db):
        """Load persisted WAL edges from LMDB into memory. Called at startup."""
        import struct
        loaded = 0
        with env.begin(db=wal_db) as txn:
            count_raw = txn.get(b'__count__', db=wal_db)
            if not count_raw:
                return 0
            count = struct.unpack('<i', count_raw)[0]
            for i in range(count):
                val = txn.get(struct.pack('<i', i), db=wal_db)
                if val and len(val) == self._EDGE_SIZE:
                    a, b, w = struct.unpack(self._EDGE_FMT, val)
                    # Clamp on load — repair any exploded weights from prior sessions
                    w = max(-self.WEIGHT_CLAMP, min(self.WEIGHT_CLAMP, w))
                    # Add directly to log without double-counting
                    if a not in self._log:
                        self._log[a] = {}
                    if b not in self._log:
                        self._log[b] = {}
                    new_ab = max(-self.WEIGHT_CLAMP, min(self.WEIGHT_CLAMP, self._log[a].get(b, 0) + w))
                    new_ba = max(-self.WEIGHT_CLAMP, min(self.WEIGHT_CLAMP, self._log[b].get(a, 0) + w))
                    self._log[a][b] = new_ab
                    self._log[b][a] = new_ba
                    self._count += 2
                    loaded += 1
        return loaded

    # --- Background flush thread ---

    _flush_thread = None
    _flush_stop = None
    _flush_env = None
    _flush_db = None
    _last_persisted_count = 0

    def start_background_flush(self, env, wal_db, interval_sec: float = 5.0):
        """Start background thread that persists WAL to LMDB every interval_sec."""
        import time as _time
        self._flush_env = env
        self._flush_db = wal_db
        self._flush_stop = threading.Event()
        self._last_persisted_count = self._count

        def _flush_loop():
            while not self._flush_stop.is_set():
                self._flush_stop.wait(interval_sec)
                if self._flush_stop.is_set():
                    break
                if self._count != self._last_persisted_count:
                    self.persist_to_lmdb(env, wal_db)
                    self._last_persisted_count = self._count

        self._flush_thread = threading.Thread(target=_flush_loop, daemon=True)
        self._flush_thread.start()

    def stop_background_flush(self):
        """Stop background flush and do a final persist."""
        if self._flush_stop:
            self._flush_stop.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=5)
        # Final flush
        if self._flush_env and self._flush_db and self._count > self._last_persisted_count:
            self.persist_to_lmdb(self._flush_env, self._flush_db)

    _scipy_cache = None
    _scipy_cache_count = -1

    def effective_scipy_matrix(self, csr: 'MMapCSR'):
        """Get CSR + WAL merged as a single scipy sparse matrix. Cached.

        Rebuilds only when WAL has changed since last call.
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required")
        if self._scipy_cache is not None and self._scipy_cache_count == self._count:
            return self._scipy_cache
        base = csr.scipy_matrix
        if self._count == 0:
            self._scipy_cache = base
            self._scipy_cache_count = 0
            return base
        # Build COO from WAL entries
        rows, cols, vals = [], [], []
        with self._lock:
            for word_idx, neighbors in self._log.items():
                for neighbor_idx, weight in neighbors.items():
                    rows.append(word_idx)
                    cols.append(neighbor_idx)
                    vals.append(weight)
            count_now = self._count
        if not rows:
            self._scipy_cache = base
            self._scipy_cache_count = count_now
            return base
        # Expand shape if WAL has new words beyond CSR
        max_idx = max(max(rows), max(cols), base.shape[0] - 1)
        shape = (max_idx + 1, max_idx + 1)
        if shape != base.shape:
            base.resize(shape)  # in-place, returns None
        wal_sparse = scipy_coo(
            (np.array(vals, dtype=np.float32),
             (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
            shape=shape,
        ).tocsr()
        merged = base + wal_sparse

        # Clamp extreme values in merged matrix.
        # Don't row-normalize — cosine computation in convergence already normalizes.
        # Row-normalization destroys frequency information (rare words look as important as common ones).
        merged.data = np.clip(merged.data, -self.WEIGHT_CLAMP, self.WEIGHT_CLAMP)

        self._scipy_cache = merged
        self._scipy_cache_count = count_now
        return self._scipy_cache


def build_csr_from_dicts(cooc: dict, num_words: int, output_path: str):
    """Build CSR files from a dict-of-dicts co-occurrence graph.

    Args:
        cooc: {word_idx: {neighbor_idx: weight, ...}, ...}
        num_words: total vocabulary size (V)
        output_path: directory to write indptr.bin, indices.bin, data.bin
    """
    os.makedirs(output_path, exist_ok=True)

    # Pass 1: count edges per row
    row_counts = np.zeros(num_words, dtype=np.int32)
    for row_idx in range(num_words):
        if row_idx in cooc:
            row_counts[row_idx] = len(cooc[row_idx])

    # Build indptr
    indptr = np.zeros(num_words + 1, dtype=np.int32)
    np.cumsum(row_counts, out=indptr[1:])
    nnz = int(indptr[-1])

    # Pass 2: fill indices and data, sorted within rows
    indices = np.zeros(nnz, dtype=np.int32)
    data = np.zeros(nnz, dtype=np.float32)

    for row_idx in range(num_words):
        if row_idx not in cooc or not cooc[row_idx]:
            continue
        start = int(indptr[row_idx])
        edges = sorted(cooc[row_idx].items(), key=lambda x: x[0])
        for j, (col, weight) in enumerate(edges):
            indices[start + j] = col
            data[start + j] = weight

    # Write files
    indptr_path = os.path.join(output_path, 'indptr.bin')
    indices_path = os.path.join(output_path, 'indices.bin')
    data_path = os.path.join(output_path, 'data.bin')

    indptr.tofile(indptr_path)
    indices.tofile(indices_path)
    data.tofile(data_path)

    print(f"CSR built: {num_words:,} rows, {nnz:,} edges, "
          f"{(os.path.getsize(indptr_path) + os.path.getsize(indices_path) + os.path.getsize(data_path)) / 1024 / 1024:.1f} MB")

    return indptr_path, indices_path, data_path
