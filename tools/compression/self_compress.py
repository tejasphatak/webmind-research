#!/usr/bin/env python3
"""
self_compress.py — SAQT knowledge base self-compression tool.

Finds near-duplicate Q&A pairs using FAISS range_search (cosine similarity ≥ 0.95),
collapses each cluster to the single entry with the longest/most complete answer,
and writes a compressed JSONL without touching the originals.

Memory approach: L2-normalize all embeddings (converts cosine → L2), then use
FAISS IndexFlatL2 range_search with radius² = 2*(1 - 0.95) = 0.10 (since for
unit vectors: ||a-b||² = 2 - 2·cos(θ)).  Process in batches of 10K so the
result set stays manageable.

Usage:
    python3 self_compress.py [--threshold 0.95] [--batch-size 10000]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import faiss
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/tejasphatak/webmind-research/trained_model")
QA_PAIRS_PATH = REPO_ROOT / "qa_pairs.jsonl"
EMBEDDINGS_PATH = REPO_ROOT / "qa_embeddings.pt"
OUTPUT_PATH = REPO_ROOT / "qa_pairs_compressed.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_qa_pairs(path: Path) -> List[dict]:
    print(f"[load] Reading {path} ...", flush=True)
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"[load] {len(pairs):,} pairs loaded", flush=True)
    return pairs


def load_embeddings(path: Path) -> np.ndarray:
    print(f"[load] Reading {path} ...", flush=True)
    t = torch.load(path, map_location="cpu", weights_only=True)
    arr = t.numpy().astype(np.float32)
    print(f"[load] Embeddings shape: {arr.shape}", flush=True)
    return arr


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize each row to unit length in-place (returns new array)."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)          # avoid div-by-zero
    return arr / norms


def cosine_threshold_to_l2_radius_sq(cos_thresh: float) -> float:
    """
    For unit vectors: ||a-b||² = 2 - 2·cos(θ).
    cos(θ) ≥ T  ⟺  ||a-b||² ≤ 2*(1-T).
    """
    return 2.0 * (1.0 - cos_thresh)


# ---------------------------------------------------------------------------
# Union-Find (path compression + rank union) for cluster merging
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Main compression logic
# ---------------------------------------------------------------------------

def build_duplicate_clusters(
    embeddings_norm: np.ndarray,
    radius_sq: float,
    batch_size: int,
) -> UnionFind:
    n = embeddings_norm.shape[0]
    uf = UnionFind(n)

    index = faiss.IndexFlatL2(embeddings_norm.shape[1])
    index.add(embeddings_norm)

    print(f"[faiss] FAISS index built ({n:,} vectors, dim={embeddings_norm.shape[1]})", flush=True)
    print(f"[faiss] L2 radius² = {radius_sq:.4f}  (cosine ≥ {1 - radius_sq/2:.4f})", flush=True)

    total_pairs = 0
    t0 = time.time()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = embeddings_norm[start:end]

        # range_search returns all pairs within radius (including self)
        lims, D, I = index.range_search(batch, radius_sq)

        batch_pairs = 0
        for local_i in range(len(batch)):
            qi = start + local_i
            lo, hi = lims[local_i], lims[local_i + 1]
            for k in range(lo, hi):
                j = int(I[k])
                if j != qi:                     # skip self-match
                    uf.union(qi, j)
                    batch_pairs += 1

        total_pairs += batch_pairs
        elapsed = time.time() - t0
        pct = (end / n) * 100
        print(
            f"[faiss] {end:>7,}/{n:,} ({pct:5.1f}%)  "
            f"pairs-so-far: {total_pairs:,}  "
            f"elapsed: {elapsed:.1f}s",
            flush=True,
        )

    print(f"[faiss] Done. Total near-duplicate pairs found: {total_pairs:,}", flush=True)
    return uf


def best_in_cluster(indices: List[int], qa_pairs: List[dict]) -> int:
    """Return the index of the pair with the longest answer in the cluster."""
    return max(indices, key=lambda i: len(qa_pairs[i].get("answer", "")))


def compress(
    qa_pairs: List[dict],
    uf: UnionFind,
) -> tuple[List[dict], List[dict], List[dict]]:
    """
    Returns:
        kept      — the surviving (compressed) Q&A list
        removed   — the dropped entries
        examples  — up to 10 illustrative merge examples (list of dicts)
    """
    n = len(qa_pairs)

    # Group indices by cluster root
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)

    kept: List[dict] = []
    removed: List[dict] = []
    examples: List[dict] = []

    singleton_count = 0
    merged_count = 0

    for root, members in clusters.items():
        if len(members) == 1:
            singleton_count += 1
            kept.append(qa_pairs[members[0]])
        else:
            merged_count += 1
            winner_idx = best_in_cluster(members, qa_pairs)
            kept.append(qa_pairs[winner_idx])
            for idx in members:
                if idx != winner_idx:
                    removed.append(qa_pairs[idx])

            # Collect example (up to 10)
            if len(examples) < 10:
                examples.append({
                    "cluster_size": len(members),
                    "kept": qa_pairs[winner_idx],
                    "dropped_sample": qa_pairs[next(i for i in members if i != winner_idx)],
                })

    print(
        f"[compress] {n:,} input pairs → "
        f"{len(kept):,} kept  |  "
        f"{len(removed):,} removed  |  "
        f"{merged_count:,} clusters merged  |  "
        f"{singleton_count:,} singletons",
        flush=True,
    )

    return kept, removed, examples


def save_jsonl(records: List[dict], path: Path) -> None:
    print(f"[save] Writing {len(records):,} records to {path} ...", flush=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"[save] Done. File size: {size_mb:.1f} MB", flush=True)


def print_report(
    original_count: int,
    kept: List[dict],
    removed: List[dict],
    examples: List[dict],
    threshold: float,
    elapsed_total: float,
) -> None:
    removed_count = len(removed)
    ratio = removed_count / original_count * 100 if original_count else 0

    print("\n" + "=" * 65)
    print("SAQT SELF-COMPRESSION REPORT")
    print("=" * 65)
    print(f"  Cosine similarity threshold : {threshold:.2f}")
    print(f"  Original pairs              : {original_count:>10,}")
    print(f"  Pairs kept                  : {len(kept):>10,}")
    print(f"  Pairs removed               : {removed_count:>10,}")
    print(f"  Compression ratio           : {ratio:>9.2f}%")
    print(f"  Total wall time             : {elapsed_total:.1f}s")
    print(f"  Output file                 : {OUTPUT_PATH}")
    print()
    print(f"  {'EXAMPLES OF MERGED PAIRS':}")
    print("-" * 65)
    for i, ex in enumerate(examples, 1):
        print(f"\n  Example {i}  (cluster_size={ex['cluster_size']})")
        print(f"    KEPT  Q: {ex['kept']['question'][:90]!r}")
        print(f"    KEPT  A: {ex['kept']['answer'][:120]!r}")
        print(f"    DROP  Q: {ex['dropped_sample']['question'][:90]!r}")
        print(f"    DROP  A: {ex['dropped_sample']['answer'][:120]!r}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAQT knowledge base self-compressor")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Cosine similarity threshold for near-duplicates (default: 0.95)")
    parser.add_argument("--batch-size", type=int, default=10_000,
                        help="Batch size for FAISS range_search (default: 10000)")
    args = parser.parse_args()

    if OUTPUT_PATH.exists():
        print(f"[warn] Output file already exists: {OUTPUT_PATH}")
        print("[warn] It will be overwritten with fresh results.")

    t_start = time.time()

    qa_pairs = load_qa_pairs(QA_PAIRS_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH)

    # Sanity check: lengths must match (or truncate QA pairs if embeddings are shorter)
    if len(qa_pairs) != embeddings.shape[0]:
        if len(qa_pairs) > embeddings.shape[0]:
            print(
                f"[warn] QA pairs ({len(qa_pairs):,}) > embedding rows ({embeddings.shape[0]:,}). "
                f"Truncating QA pairs to match embeddings (trailing {len(qa_pairs) - embeddings.shape[0]} rows have no embedding).",
            )
            qa_pairs = qa_pairs[: embeddings.shape[0]]
        else:
            print(
                f"[error] Embeddings ({embeddings.shape[0]:,}) > QA pairs ({len(qa_pairs):,}). "
                f"Cannot proceed — missing QA data for some embeddings.",
                file=sys.stderr,
            )
            sys.exit(1)

    embeddings_norm = l2_normalize(embeddings)
    radius_sq = cosine_threshold_to_l2_radius_sq(args.threshold)

    uf = build_duplicate_clusters(embeddings_norm, radius_sq, args.batch_size)

    kept, removed, examples = compress(qa_pairs, uf)

    save_jsonl(kept, OUTPUT_PATH)

    elapsed_total = time.time() - t_start
    print_report(
        original_count=len(qa_pairs),
        kept=kept,
        removed=removed,
        examples=examples,
        threshold=args.threshold,
        elapsed_total=elapsed_total,
    )


if __name__ == "__main__":
    main()
