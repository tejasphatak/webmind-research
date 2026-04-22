"""
Semantic Bootstrap: extract top-K neighbors from model embeddings → edge file.

Takes word embeddings (from extract_model_weights.py or LSH index) and computes
top-K semantically similar neighbors for each word. Saves as bootstrap_edges.npz
that BrainCSR loads as a sparse matrix overlay on the co-occurrence CSR.

This gives the convergence loop semantic knowledge from day one — "shakespeare"
connects to "playwright" even before any sentence teaches it.

Usage:
    # From LSH index embeddings (MiniLM, 408K words)
    python3 build_semantic_bootstrap.py ~/nexus-brain

    # From extracted model weights (any model)
    python3 build_semantic_bootstrap.py ~/nexus-brain \
        --embeddings ~/nexus-brain/model_weights/gpt2/embeddings.npy

    # Custom parameters
    python3 build_semantic_bootstrap.py ~/nexus-brain \
        --top-k 10 --min-similarity 0.4 --bootstrap-factor 0.3

    # Inspect bootstrap for a word
    python3 build_semantic_bootstrap.py ~/nexus-brain --inspect shakespeare
"""

import os
import sys
import time
import struct
import argparse
import numpy as np
from typing import List, Tuple, Optional

_ID_FMT = struct.Struct('<i')


def load_csr_vocabulary(db_path: str) -> dict:
    """Load canonical word→index mapping from LMDB words_db.

    Returns dict of {word: csr_index} matching BrainCSR._word_idx ordering.
    """
    import lmdb

    lmdb_path = os.path.join(db_path, 'brain.lmdb')
    if not os.path.exists(lmdb_path):
        return {}

    env = lmdb.open(lmdb_path, max_dbs=4, readonly=True)
    words_db = env.open_db(b'words')

    word_neurons = {}
    with env.begin(db=words_db) as txn:
        cursor = txn.cursor(db=words_db)
        for key, val in cursor:
            word = key.decode('utf-8')
            nid = _ID_FMT.unpack(val)[0]
            word_neurons[word] = nid

    env.close()

    # Build word→csr_idx (same ordering as BrainCSR.__init__)
    sorted_words = sorted(word_neurons.items(), key=lambda x: x[1])
    word_to_csr = {}
    for idx, (word, nid) in enumerate(sorted_words):
        word_to_csr[word] = idx

    return word_to_csr


def load_embeddings(db_path: str, embeddings_path: Optional[str] = None
                    ) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings and word list.

    Priority:
    1. Explicit --embeddings path (from extract_model_weights.py)
    2. LSH index at {db_path}/lsh_index/
    """
    if embeddings_path and os.path.exists(embeddings_path):
        emb = np.load(embeddings_path)
        # Try to find words.txt alongside
        words_path = os.path.join(os.path.dirname(embeddings_path), 'words.txt')
        if not os.path.exists(words_path):
            # Check LSH index for word list
            words_path = os.path.join(db_path, 'lsh_index', 'words.txt')
        if os.path.exists(words_path):
            words = open(words_path).read().strip().split('\n')
        else:
            words = [f'word_{i}' for i in range(len(emb))]
            print(f"  Warning: no words.txt found, using indices")
        return emb, words

    # LSH index
    lsh_path = os.path.join(db_path, 'lsh_index')
    emb_path = os.path.join(lsh_path, 'embeddings.npy')
    words_path = os.path.join(lsh_path, 'words.txt')

    if os.path.exists(emb_path):
        emb = np.load(emb_path)
        words = open(words_path).read().strip().split('\n')
        return emb, words

    raise FileNotFoundError(
        f"No embeddings found. Run:\n"
        f"  python3 extract_model_weights.py --model <model> --output <path>\n"
        f"  OR build the LSH index first.")


def compute_neighbors_numpy(embeddings: np.ndarray, top_k: int = 10,
                            min_similarity: float = 0.4,
                            batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute top-K neighbors using numpy brute-force in batches.

    Embeddings assumed L2-normalized (dot product = cosine similarity).

    Returns: (rows, cols, similarities) arrays for all neighbor pairs.
    """
    n = len(embeddings)

    # Normalize if not already
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normed = embeddings / norms

    all_rows = []
    all_cols = []
    all_sims = []

    print(f"  Computing neighbors: {n:,} words × top-{top_k}, batch={batch_size}")
    t0 = time.time()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = normed[start:end]  # (batch, dim)

        # Cosine similarities: batch × all words
        sims = batch @ normed.T  # (batch, n)

        # Zero out self-similarity
        for i in range(end - start):
            sims[i, start + i] = -1.0

        # Top-K per row
        if top_k < n:
            # argpartition is O(n) vs O(n log n) for argsort
            top_indices = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
            # Get actual similarities for top indices
            top_sims = np.take_along_axis(sims, top_indices, axis=1)
        else:
            top_indices = np.argsort(-sims, axis=1)[:, :top_k]
            top_sims = np.take_along_axis(sims, top_indices, axis=1)

        # Filter by min_similarity
        for i in range(end - start):
            mask = top_sims[i] >= min_similarity
            valid_cols = top_indices[i][mask]
            valid_sims = top_sims[i][mask]

            if len(valid_cols) > 0:
                all_rows.append(np.full(len(valid_cols), start + i, dtype=np.int32))
                all_cols.append(valid_cols.astype(np.int32))
                all_sims.append(valid_sims.astype(np.float32))

        if (start // batch_size) % 50 == 0 and start > 0:
            elapsed = time.time() - t0
            pct = start / n * 100
            eta = elapsed / (start / n) - elapsed
            print(f"    {pct:.0f}% ({start:,}/{n:,}) - {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"  Neighbor search done: {elapsed:.1f}s")

    if not all_rows:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    return (np.concatenate(all_rows),
            np.concatenate(all_cols),
            np.concatenate(all_sims))


def compute_neighbors_scann(embeddings: np.ndarray, top_k: int = 10,
                            min_similarity: float = 0.4
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute top-K neighbors using ScaNN (fast approximate search)."""
    import scann

    n = len(embeddings)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normed = (embeddings / norms).astype(np.float32)

    print(f"  Building ScaNN index for {n:,} vectors...")
    t0 = time.time()

    searcher = scann.scann_ops_pybind.builder(
        normed, top_k + 1, "dot_product"
    ).score_ah(
        2, anisotropic_quantization_threshold=0.2
    ).reorder(
        min(200, n)
    ).build()

    print(f"  ScaNN index built: {time.time() - t0:.1f}s")
    print(f"  Batch searching...")
    t1 = time.time()

    indices, distances = searcher.search_batched(normed, final_num_neighbors=top_k + 1)
    print(f"  Batch search done: {time.time() - t1:.1f}s")

    # distances from ScaNN with dot_product are similarities (higher = more similar)
    all_rows = []
    all_cols = []
    all_sims = []

    for i in range(n):
        for j_pos in range(len(indices[i])):
            j = indices[i][j_pos]
            sim = distances[i][j_pos]
            if j != i and sim >= min_similarity:
                all_rows.append(i)
                all_cols.append(j)
                all_sims.append(sim)

    return (np.array(all_rows, dtype=np.int32),
            np.array(all_cols, dtype=np.int32),
            np.array(all_sims, dtype=np.float32))


def bootstrap(db_path: str, top_k: int = 10, bootstrap_factor: float = 0.3,
              min_similarity: float = 0.4, batch_size: int = 512,
              embeddings_path: Optional[str] = None) -> str:
    """Build semantic bootstrap edges from embeddings.

    Returns path to output file.
    """
    t0 = time.time()

    # 1. Load embeddings
    print("Loading embeddings...")
    embeddings, emb_words = load_embeddings(db_path, embeddings_path)
    print(f"  {len(emb_words):,} words, {embeddings.shape[1]}-dim")

    # 2. Load CSR vocabulary for index reconciliation
    csr_vocab = load_csr_vocabulary(db_path)
    if csr_vocab:
        print(f"  CSR vocabulary: {len(csr_vocab):,} words")
    else:
        print(f"  No LMDB found — using embedding indices directly")

    # 3. Build embedding→CSR index mapping
    emb_to_csr = {}
    if csr_vocab:
        for emb_idx, word in enumerate(emb_words):
            if word in csr_vocab:
                emb_to_csr[emb_idx] = csr_vocab[word]
        print(f"  Matched: {len(emb_to_csr):,} / {len(emb_words):,} words")
    else:
        # Direct mapping (embedding index = CSR index)
        emb_to_csr = {i: i for i in range(len(emb_words))}

    # 4. Compute top-K neighbors
    try:
        import scann
        rows, cols, sims = compute_neighbors_scann(embeddings, top_k, min_similarity)
    except ImportError:
        print("  ScaNN not available, using numpy brute-force")
        rows, cols, sims = compute_neighbors_numpy(
            embeddings, top_k, min_similarity, batch_size)

    print(f"  Raw neighbor pairs: {len(rows):,}")

    # 5. Remap to CSR indices + compute weights
    valid_mask = np.ones(len(rows), dtype=bool)
    csr_rows = np.empty(len(rows), dtype=np.int32)
    csr_cols = np.empty(len(rows), dtype=np.int32)

    for i in range(len(rows)):
        r_csr = emb_to_csr.get(int(rows[i]))
        c_csr = emb_to_csr.get(int(cols[i]))
        if r_csr is not None and c_csr is not None:
            csr_rows[i] = r_csr
            csr_cols[i] = c_csr
        else:
            valid_mask[i] = False

    csr_rows = csr_rows[valid_mask]
    csr_cols = csr_cols[valid_mask]
    valid_sims = sims[valid_mask]
    weights = valid_sims * bootstrap_factor

    print(f"  Valid CSR edges: {len(csr_rows):,}")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    # 6. Make symmetric (if (a,b), also add (b,a))
    sym_rows = np.concatenate([csr_rows, csr_cols])
    sym_cols = np.concatenate([csr_cols, csr_rows])
    sym_weights = np.concatenate([weights, weights])
    sym_sims = np.concatenate([valid_sims, valid_sims])

    # Deduplicate by taking max weight for each (row, col) pair
    # Use scipy sparse for efficient dedup
    from scipy.sparse import coo_matrix
    n_words = max(len(csr_vocab), len(emb_words))
    mat = coo_matrix((sym_weights, (sym_rows, sym_cols)),
                     shape=(n_words, n_words))
    # Convert to CSR and back to COO — this sums duplicates
    mat = mat.tocsr().tocoo()

    final_rows = mat.row.astype(np.int32)
    final_cols = mat.col.astype(np.int32)
    final_weights = mat.data.astype(np.float32)

    print(f"  Final edges (symmetric, deduped): {len(final_rows):,}")

    # 7. Save
    output_path = os.path.join(db_path, 'bootstrap_edges.npz')
    np.savez(output_path,
             rows=final_rows,
             cols=final_cols,
             weights=final_weights,
             bootstrap_factor=np.array([bootstrap_factor], dtype=np.float32),
             min_similarity=np.array([min_similarity], dtype=np.float32),
             top_k=np.array([top_k], dtype=np.int32),
             n_words=np.array([n_words], dtype=np.int32),
             timestamp=np.array([time.time()]),
             )

    elapsed = time.time() - t0
    file_size = os.path.getsize(output_path) / 1e6
    print(f"\nSaved: {output_path} ({file_size:.1f} MB)")
    print(f"Total time: {elapsed:.1f}s")

    return output_path


def inspect(db_path: str, word: str):
    """Show bootstrap neighbors for a word."""
    bootstrap_path = os.path.join(db_path, 'bootstrap_edges.npz')
    if not os.path.exists(bootstrap_path):
        print(f"No bootstrap file at {bootstrap_path}")
        return

    data = np.load(bootstrap_path)
    rows, cols, weights = data['rows'], data['cols'], data['weights']

    # Load vocabulary
    csr_vocab = load_csr_vocabulary(db_path)
    if not csr_vocab:
        print("No LMDB vocabulary found")
        return

    idx_to_word = {idx: w for w, idx in csr_vocab.items()}
    word_idx = csr_vocab.get(word)

    if word_idx is None:
        print(f"Word '{word}' not in vocabulary")
        return

    # Find edges where this word is source
    mask = rows == word_idx
    neighbor_cols = cols[mask]
    neighbor_weights = weights[mask]

    # Sort by weight descending
    order = np.argsort(-neighbor_weights)

    print(f"Bootstrap neighbors for '{word}' (idx={word_idx}):")
    for i in order[:20]:
        nword = idx_to_word.get(int(neighbor_cols[i]), f'?{neighbor_cols[i]}')
        print(f"  {nword:30s}  weight={neighbor_weights[i]:.4f}")

    print(f"\nTotal neighbors: {len(neighbor_cols)}")
    meta_keys = ['bootstrap_factor', 'min_similarity', 'top_k']
    for k in meta_keys:
        if k in data:
            print(f"  {k}: {data[k][0]}")


def main():
    parser = argparse.ArgumentParser(
        description='Build semantic bootstrap edges from model embeddings')
    parser.add_argument('db_path', help='Brain database path (e.g., ~/nexus-brain)')
    parser.add_argument('--embeddings', default=None,
                        help='Path to embeddings.npy (default: use LSH index)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of neighbors per word (default: 10)')
    parser.add_argument('--bootstrap-factor', type=float, default=0.3,
                        help='Edge weight = similarity × factor (default: 0.3)')
    parser.add_argument('--min-similarity', type=float, default=0.4,
                        help='Minimum cosine similarity threshold (default: 0.4)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for numpy computation (default: 512)')
    parser.add_argument('--inspect', metavar='WORD', default=None,
                        help='Inspect bootstrap neighbors for a word')

    args = parser.parse_args()
    args.db_path = os.path.expanduser(args.db_path)

    if args.inspect:
        inspect(args.db_path, args.inspect)
    else:
        bootstrap(args.db_path,
                  top_k=args.top_k,
                  bootstrap_factor=args.bootstrap_factor,
                  min_similarity=args.min_similarity,
                  batch_size=args.batch_size,
                  embeddings_path=args.embeddings)


if __name__ == '__main__':
    main()
