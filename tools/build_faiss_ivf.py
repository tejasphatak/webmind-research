#!/usr/bin/env python3
"""
build_faiss_ivf.py
Rebuild saqt.faiss as IVF1024 (from flat IndexFlatIP) using all 305K vectors.
Also back-fills saqt.db with any qa_pairs.jsonl rows missing beyond the original 190K.

Usage:
    python3 tools/build_faiss_ivf.py [--repo-root PATH]
"""

import argparse
import json
import math
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "trained_model"

parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", type=Path, default=DEFAULT_ROOT,
                    help="Directory containing saqt.faiss, saqt.db, qa_embeddings.pt, qa_pairs.jsonl")
args = parser.parse_args()
ROOT = args.repo_root

FAISS_PATH   = ROOT / "saqt.faiss"
DB_PATH      = ROOT / "saqt.db"
EMB_PATH     = ROOT / "qa_embeddings.pt"
JSONL_PATH   = ROOT / "qa_pairs.jsonl"

print(f"[paths] root={ROOT}")
for p in (FAISS_PATH, DB_PATH, EMB_PATH, JSONL_PATH):
    print(f"  {'ok' if p.exists() else 'MISSING':6s}  {p}")

# ---------------------------------------------------------------------------
# 1. Load all embeddings
# ---------------------------------------------------------------------------
print("\n[1/5] Loading embeddings …")
t0 = time.perf_counter()
emb_tensor = torch.load(EMB_PATH, map_location="cpu", weights_only=True)
vectors_all = emb_tensor.numpy().astype(np.float32)
N, D = vectors_all.shape
print(f"  loaded {N:,} vectors, dim={D}  ({time.perf_counter()-t0:.1f}s)")

# L2-normalise so inner-product == cosine similarity (matches original build)
norms = np.linalg.norm(vectors_all, axis=1, keepdims=True)
norms = np.where(norms == 0, 1.0, norms)
vectors_norm = vectors_all / norms
print(f"  vectors L2-normalised")

# ---------------------------------------------------------------------------
# 2. Benchmark: existing flat index
# ---------------------------------------------------------------------------
print("\n[2/5] Benchmarking existing flat index …")
flat_index = faiss.read_index(str(FAISS_PATH))
print(f"  type={type(flat_index).__name__}  ntotal={flat_index.ntotal:,}  d={flat_index.d}")

rng = np.random.default_rng(42)
query_ids = rng.integers(0, N, size=100)
queries = vectors_norm[query_ids]

t0 = time.perf_counter()
for q in queries:
    flat_index.search(q.reshape(1, -1), 5)
flat_time_ms = (time.perf_counter() - t0) / 100 * 1000
print(f"  flat avg latency: {flat_time_ms:.3f} ms/query  (100 queries, k=5)")
del flat_index  # free memory

# ---------------------------------------------------------------------------
# 3. Build IVF index
# ---------------------------------------------------------------------------
nlist = max(1, round(math.sqrt(N)))        # ~553 for 305K; bump to nearest power-of-2-ish
# Using 1000 as instructed; clamp between 256 and 4096 for sanity
nlist = 1000
nprobe = 10

print(f"\n[3/5] Building IVFFlat index (nlist={nlist}, nprobe={nprobe}) …")
quantizer = faiss.IndexFlatIP(D)
ivf_index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
ivf_index.nprobe = nprobe

# Training requires >= nlist vectors; we have 305K so fine
print(f"  training on {N:,} vectors …")
t0 = time.perf_counter()
ivf_index.train(vectors_norm)
print(f"  trained  ({time.perf_counter()-t0:.1f}s)")

print(f"  adding {N:,} vectors …")
t0 = time.perf_counter()
BATCH = 50_000
for start in range(0, N, BATCH):
    ivf_index.add(vectors_norm[start:start+BATCH])
    print(f"    added {min(start+BATCH, N):,}/{N:,}", end="\r")
print(f"\n  done  ({time.perf_counter()-t0:.1f}s)  ntotal={ivf_index.ntotal:,}")

# ---------------------------------------------------------------------------
# 4. Benchmark: new IVF index
# ---------------------------------------------------------------------------
print(f"\n[4/5] Benchmarking IVF index (nprobe={nprobe}) …")
t0 = time.perf_counter()
for q in queries:
    ivf_index.search(q.reshape(1, -1), 5)
ivf_time_ms = (time.perf_counter() - t0) / 100 * 1000
print(f"  IVF avg latency: {ivf_time_ms:.3f} ms/query  (100 queries, k=5)")
print(f"  speedup vs flat: {flat_time_ms/ivf_time_ms:.1f}x")

# Save
print(f"  writing {FAISS_PATH} …")
faiss.write_index(ivf_index, str(FAISS_PATH))
print(f"  saved")

# ---------------------------------------------------------------------------
# 5. Back-fill SQLite with missing rows (IDs 190261 .. 305312)
# ---------------------------------------------------------------------------
print(f"\n[5/5] Syncing SQLite DB with all {N:,} qa_pairs …")
conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM qa")
existing = cur.fetchone()[0]
print(f"  existing rows: {existing:,}")

if existing >= N:
    print(f"  DB already has {existing:,} rows — nothing to insert")
else:
    # Read all pairs from jsonl; skip the first `existing` lines
    print(f"  reading qa_pairs.jsonl …")
    pairs_to_insert = []
    with open(JSONL_PATH) as f:
        for i, line in enumerate(f):
            if i < existing:
                continue
            obj = json.loads(line)
            pairs_to_insert.append((
                obj.get("question", ""),
                obj.get("answer", ""),
                obj.get("source", ""),
                1.0,
            ))

    print(f"  inserting {len(pairs_to_insert):,} new rows …")
    t0 = time.perf_counter()
    cur.executemany(
        "INSERT INTO qa (question, answer, source, weight) VALUES (?, ?, ?, ?)",
        pairs_to_insert,
    )
    conn.commit()
    elapsed = time.perf_counter() - t0

    cur.execute("SELECT COUNT(*) FROM qa")
    final_count = cur.fetchone()[0]
    print(f"  inserted in {elapsed:.1f}s — DB now has {final_count:,} rows")

conn.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
print(f"  Vectors indexed : {N:,}  (dim={D})")
print(f"  Index type      : IVFFlat  nlist={nlist}  nprobe={nprobe}")
print(f"  Flat latency    : {flat_time_ms:.3f} ms/query")
print(f"  IVF  latency    : {ivf_time_ms:.3f} ms/query")
print(f"  Speedup         : {flat_time_ms/ivf_time_ms:.1f}x")
print(f"  DB rows         : {N:,}")
print(f"  Index saved to  : {FAISS_PATH}")
print(f"  DB saved to     : {DB_PATH}")
