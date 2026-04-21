#!/usr/bin/env python3
"""
Build CSR files from LMDB co-occurrence database.

Reads the brain.lmdb cooccurrence/ sub-db and writes three flat files:
  indptr.bin   — row pointers (V+1 int32)
  indices.bin  — column indices (nnz int32), sorted within each row
  data.bin     — edge weights (nnz float32)

Usage:
  python3 src/build_csr.py ~/nexus-brain
  python3 src/build_csr.py ~/nexus-brain --max-edges 50
"""

import argparse
import os
import struct
import time

import numpy as np

try:
    import lmdb
except ImportError:
    print("pip install lmdb")
    raise


_ID_FMT = struct.Struct('<i')
_ID_CONF_FMT = struct.Struct('<if')


def build(db_path: str, max_edges_per_word: int = 50):
    lmdb_path = os.path.join(db_path, 'brain.lmdb')
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"No brain.lmdb at {lmdb_path}")

    env = lmdb.open(lmdb_path, max_dbs=10, readonly=True, lock=False,
                     map_size=4 * 1024 * 1024 * 1024)
    words_db = env.open_db(b'words')
    cooc_db = env.open_db(b'cooccurrence')

    # --- Count words ---
    t0 = time.time()
    num_words = 0
    with env.begin(db=words_db) as txn:
        cursor = txn.cursor(db=words_db)
        for _ in cursor:
            num_words += 1
    print(f"Vocabulary: {num_words:,} words")

    # --- Pass 1: count edges per row ---
    row_counts = np.zeros(num_words, dtype=np.int64)
    total_edges = 0

    with env.begin(db=cooc_db) as txn:
        cursor = txn.cursor(db=cooc_db)
        for key, val in cursor:
            word_idx = _ID_FMT.unpack(key)[0]
            if word_idx >= num_words:
                continue
            n_edges = len(val) // _ID_CONF_FMT.size
            capped = min(n_edges, max_edges_per_word)
            row_counts[word_idx] = capped
            total_edges += capped

    print(f"Pass 1: {total_edges:,} edges (capped at {max_edges_per_word}/word)")

    # --- Build indptr ---
    indptr = np.zeros(num_words + 1, dtype=np.int32)
    np.cumsum(row_counts.astype(np.int32), out=indptr[1:])
    nnz = int(indptr[-1])

    # --- Pass 2: fill indices and data ---
    indices = np.zeros(nnz, dtype=np.int32)
    data = np.zeros(nnz, dtype=np.float32)

    with env.begin(db=cooc_db) as txn:
        cursor = txn.cursor(db=cooc_db)
        for key, val in cursor:
            word_idx = _ID_FMT.unpack(key)[0]
            if word_idx >= num_words:
                continue

            # Parse all edges
            n_edges = len(val) // _ID_CONF_FMT.size
            if n_edges == 0:
                continue

            arr = np.frombuffer(val, dtype=np.dtype([('id', '<i4'), ('weight', '<f4')]))

            # Top-K by weight
            if n_edges > max_edges_per_word:
                top_idx = np.argpartition(-arr['weight'], max_edges_per_word)[:max_edges_per_word]
                arr = arr[top_idx]

            # Sort by column index (cache-friendly for spmv)
            order = np.argsort(arr['id'])
            arr = arr[order]

            start = int(indptr[word_idx])
            n = len(arr)
            indices[start:start + n] = arr['id']
            data[start:start + n] = arr['weight']

    env.close()

    # --- Write files ---
    csr_path = os.path.join(db_path, 'cooc_csr')
    os.makedirs(csr_path, exist_ok=True)

    indptr.tofile(os.path.join(csr_path, 'indptr.bin'))
    indices.tofile(os.path.join(csr_path, 'indices.bin'))
    data.tofile(os.path.join(csr_path, 'data.bin'))

    elapsed = time.time() - t0
    disk_mb = (
        os.path.getsize(os.path.join(csr_path, 'indptr.bin')) +
        os.path.getsize(os.path.join(csr_path, 'indices.bin')) +
        os.path.getsize(os.path.join(csr_path, 'data.bin'))
    ) / (1024 * 1024)

    print(f"CSR built in {elapsed:.1f}s: {num_words:,} rows, {nnz:,} edges, {disk_mb:.1f} MB")
    print(f"Output: {csr_path}/")
    return csr_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build CSR from LMDB')
    parser.add_argument('db_path', nargs='?', default=os.path.expanduser('~/nexus-brain'),
                        help='Path to nexus-brain directory')
    parser.add_argument('--max-edges', type=int, default=50,
                        help='Max edges per word (top-K by weight)')
    args = parser.parse_args()
    build(args.db_path, args.max_edges)
