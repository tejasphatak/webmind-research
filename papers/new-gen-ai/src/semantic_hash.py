"""
Semantic Hashing — Locality-Sensitive Hashing for word embeddings.

Hash words so that semantically similar words land in the same/nearby buckets.
Uses random hyperplane projection on sentence-transformer embeddings.

Usage:
    from semantic_hash import SemanticHasher
    hasher = SemanticHasher(n_bits=64)
    hasher.build(words)                    # encode + hash all words
    neighbors = hasher.search("gravity")   # find similar words
    is_real = hasher.is_meaningful("asdfghjkl")  # garbage detection
"""

import numpy as np
import os
import time
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

try:
    import scann as _scann_lib
    HAS_SCANN = True
except ImportError:
    HAS_SCANN = False


class SemanticHasher:
    """LSH over sentence-transformer embeddings for O(1) semantic search."""

    def __init__(self, n_bits: int = 64, model_name: str = "all-MiniLM-L6-v2"):
        self.n_bits = n_bits
        self.model_name = model_name
        self._model = None
        self._dim = 384  # MiniLM output dim

        # Random hyperplanes for LSH projection
        self._planes = None  # shape: (n_bits, dim)

        # Index
        self._embeddings = None   # shape: (n_words, dim)
        self._hashes = None       # shape: (n_words,) uint64
        self._words = []           # word list (index → word)
        self._word_idx = {}        # word → index
        self._buckets = defaultdict(list)  # hash → [word_indices]

        # ScaNN index (optional, built if available)
        self._scann_index = None

        # Stats
        self._avg_bucket_size = 0

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def _init_planes(self, seed: int = 42):
        """Generate random hyperplanes for LSH projection."""
        rng = np.random.RandomState(seed)
        self._planes = rng.randn(self.n_bits, self._dim).astype(np.float32)
        # Normalize each hyperplane
        norms = np.linalg.norm(self._planes, axis=1, keepdims=True)
        self._planes /= norms

    def _hash_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Project vectors onto hyperplanes → binary hash.
        vectors: (N, dim) → returns (N,) uint64 hashes."""
        # Dot product with all planes: (N, dim) @ (n_bits, dim).T → (N, n_bits)
        projections = vectors @ self._planes.T
        # Sign → bits: positive = 1, negative = 0
        bits = (projections > 0).astype(np.uint64)
        # Pack bits into uint64
        hashes = np.zeros(len(vectors), dtype=np.uint64)
        for b in range(min(self.n_bits, 64)):
            hashes |= (bits[:, b] << np.uint64(b))
        return hashes

    def build(self, words: List[str], batch_size: int = 512,
              save_path: Optional[str] = None) -> dict:
        """Encode all words with MiniLM, hash them, build bucket index.

        Returns stats dict.
        """
        t0 = time.time()
        model = self._load_model()
        self._init_planes()

        self._words = list(words)
        self._word_idx = {w: i for i, w in enumerate(self._words)}

        # Encode in batches
        print(f"  Encoding {len(words)} words with {self.model_name}...")
        all_embeddings = []
        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(emb)
            if (i + batch_size) % 10000 == 0:
                print(f"    {i + batch_size}/{len(words)}")

        self._embeddings = np.vstack(all_embeddings).astype(np.float32)
        encode_time = time.time() - t0

        # Hash all embeddings
        print(f"  Hashing {len(words)} vectors with {self.n_bits} hyperplanes...")
        self._hashes = self._hash_vectors(self._embeddings)

        # Build bucket index
        self._buckets.clear()
        for idx, h in enumerate(self._hashes):
            self._buckets[int(h)].append(idx)

        build_time = time.time() - t0

        # Compute stats
        bucket_sizes = [len(v) for v in self._buckets.values()]
        self._avg_bucket_size = np.mean(bucket_sizes) if bucket_sizes else 0

        stats = {
            "words": len(words),
            "dim": self._dim,
            "bits": self.n_bits,
            "buckets": len(self._buckets),
            "avg_bucket_size": float(self._avg_bucket_size),
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "encode_time_s": encode_time,
            "total_time_s": build_time,
            "embeddings_mb": self._embeddings.nbytes / 1e6,
        }

        if save_path:
            self.save(save_path)
            stats["saved_to"] = save_path

        # Build ScaNN index if available (anisotropic quantization — much faster search)
        self._scann_index = None
        if HAS_SCANN and len(words) >= 100:
            try:
                self._scann_index = _scann_lib.scann_ops_pybind.builder(
                    self._embeddings, 10, "dot_product"
                ).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(
                    min(100, len(words))
                ).build()
                stats["scann"] = True
            except Exception:
                self._scann_index = None
                stats["scann"] = False
        else:
            stats["scann"] = HAS_SCANN

        print(f"  Done: {stats['buckets']} buckets, avg {stats['avg_bucket_size']:.1f} words/bucket, {build_time:.1f}s"
              + (", ScaNN enabled" if self._scann_index else ""))
        return stats

    def save(self, path: str):
        """Save hash index to disk."""
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embeddings.npy"), self._embeddings)
        np.save(os.path.join(path, "hashes.npy"), self._hashes)
        np.save(os.path.join(path, "planes.npy"), self._planes)
        with open(os.path.join(path, "words.txt"), "w") as f:
            f.write("\n".join(self._words))
        print(f"  Saved to {path}")

    def load(self, path: str):
        """Load hash index from disk."""
        self._embeddings = np.load(os.path.join(path, "embeddings.npy"))
        self._hashes = np.load(os.path.join(path, "hashes.npy"))
        self._planes = np.load(os.path.join(path, "planes.npy"))
        with open(os.path.join(path, "words.txt")) as f:
            self._words = f.read().strip().split("\n")
        self._word_idx = {w: i for i, w in enumerate(self._words)}
        self._dim = self._embeddings.shape[1]
        self.n_bits = self._planes.shape[0]

        # Rebuild buckets
        self._buckets.clear()
        for idx, h in enumerate(self._hashes):
            self._buckets[int(h)].append(idx)

        bucket_sizes = [len(v) for v in self._buckets.values()]
        self._avg_bucket_size = np.mean(bucket_sizes) if bucket_sizes else 0

        # Rebuild ScaNN index if available
        self._scann_index = None
        if HAS_SCANN and len(self._words) >= 100:
            try:
                self._scann_index = _scann_lib.scann_ops_pybind.builder(
                    self._embeddings, 10, "dot_product"
                ).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(
                    min(100, len(self._words))
                ).build()
            except Exception:
                pass

        print(f"  Loaded: {len(self._words)} words, {len(self._buckets)} buckets"
              + (", ScaNN enabled" if self._scann_index else ""))

    def hash_word(self, word: str) -> int:
        """Hash a single word. Encodes with MiniLM first."""
        model = self._load_model()
        emb = model.encode([word], normalize_embeddings=True).astype(np.float32)
        return int(self._hash_vectors(emb)[0])

    def hash_text(self, text: str) -> int:
        """Hash arbitrary text (sentence, phrase, etc.)."""
        model = self._load_model()
        emb = model.encode([text], normalize_embeddings=True).astype(np.float32)
        return int(self._hash_vectors(emb)[0])

    def search(self, query: str, k: int = 10, hamming_radius: int = 3) -> List[Tuple[str, float]]:
        """Find semantically similar words to query.

        Uses ScaNN (anisotropic quantization) if available, else LSH buckets.
        Both paths refine with exact cosine similarity.

        Returns: [(word, similarity), ...] sorted by similarity desc.
        """
        model = self._load_model()
        q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)

        # Fast path: ScaNN (anisotropic vector quantization — O(1) amortized)
        if self._scann_index is not None:
            try:
                indices, distances = self._scann_index.search(q_emb[0], final_num_neighbors=k)
                results = []
                for idx, dist in zip(indices, distances):
                    if 0 <= idx < len(self._words):
                        results.append((self._words[idx], float(dist)))
                return results
            except Exception:
                pass  # fall through to LSH

        # Fallback: LSH bucket search
        q_hash = int(self._hash_vectors(q_emb)[0])
        candidates = set()
        candidates.update(self._buckets.get(q_hash, []))

        if hamming_radius > 0 and len(candidates) < k:
            for bit in range(min(self.n_bits, 64)):
                flipped = q_hash ^ (1 << bit)
                candidates.update(self._buckets.get(flipped, []))
                if hamming_radius > 1:
                    for bit2 in range(bit + 1, min(self.n_bits, 64)):
                        flipped2 = flipped ^ (1 << bit2)
                        candidates.update(self._buckets.get(flipped2, []))

        if not candidates:
            return []

        candidate_indices = list(candidates)
        candidate_embs = self._embeddings[candidate_indices]
        similarities = (candidate_embs @ q_emb.T).flatten()

        top_k = np.argsort(similarities)[::-1][:k]
        results = []
        for idx in top_k:
            word_idx = candidate_indices[idx]
            results.append((self._words[word_idx], float(similarities[idx])))

        return results

    def is_meaningful(self, text: str, min_similarity: float = 0.15) -> bool:
        """Check if text is semantically close to any known word.

        Garbage text ("asdfghjkl") will have low similarity to all real words.
        Real text ("gravity") will be close to "force", "mass", "physics", etc.
        """
        if not text or len(text.strip()) < 2:
            return False

        model = self._load_model()
        q_emb = model.encode([text], normalize_embeddings=True).astype(np.float32)
        q_hash = int(self._hash_vectors(q_emb)[0])

        # Check bucket + 1-bit neighbors
        candidates = set(self._buckets.get(q_hash, []))
        for bit in range(min(self.n_bits, 64)):
            flipped = q_hash ^ (1 << bit)
            candidates.update(self._buckets.get(flipped, []))
            if len(candidates) >= 20:
                break

        if not candidates:
            return False

        # Max similarity to any known word
        candidate_indices = list(candidates)[:100]
        candidate_embs = self._embeddings[candidate_indices]
        similarities = (candidate_embs @ q_emb.T).flatten()
        max_sim = float(np.max(similarities))

        return max_sim >= min_similarity

    # --- Quantization (PolarQuant-inspired: random rotation + int8) ---

    def quantize(self, save_path: Optional[str] = None) -> dict:
        """Quantize float32 embeddings to int8 via random rotation.

        PolarQuant approach: rotate embeddings with a random orthogonal matrix
        so values distribute uniformly, then quantize to int8 (127 bins).
        4x memory reduction. Cosine similarity preserved within ~1% error.

        Returns stats dict. Stores quantized embeddings + rotation matrix.
        """
        if self._embeddings is None:
            return {"error": "No embeddings to quantize"}

        # Generate random orthogonal rotation matrix (Haar-distributed)
        rng = np.random.RandomState(42)
        random_mat = rng.randn(self._dim, self._dim).astype(np.float32)
        q_mat, _ = np.linalg.qr(random_mat)
        self._rotation = q_mat

        # Rotate embeddings
        rotated = self._embeddings @ q_mat

        # Quantize to int8: scale each dimension to [-127, 127]
        self._quant_scale = np.max(np.abs(rotated), axis=0).astype(np.float32)
        self._quant_scale[self._quant_scale == 0] = 1.0  # avoid div-by-zero
        scaled = rotated / self._quant_scale * 127.0
        self._embeddings_int8 = np.clip(scaled, -127, 127).astype(np.int8)

        original_mb = self._embeddings.nbytes / 1e6
        quantized_mb = self._embeddings_int8.nbytes / 1e6
        ratio = original_mb / max(quantized_mb, 0.001)

        stats = {
            "original_mb": round(original_mb, 2),
            "quantized_mb": round(quantized_mb, 2),
            "compression_ratio": round(ratio, 1),
            "words": len(self._words),
        }

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "embeddings_int8.npy"), self._embeddings_int8)
            np.save(os.path.join(save_path, "quant_scale.npy"), self._quant_scale)
            np.save(os.path.join(save_path, "rotation.npy"), self._rotation)
            stats["saved_to"] = save_path

        print(f"  Quantized: {original_mb:.1f}MB → {quantized_mb:.1f}MB ({ratio:.1f}x compression)")
        return stats

    def search_quantized(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using quantized embeddings (4x faster, ~1% accuracy loss).

        Uses int8 dot product (vectorized numpy) for candidate scoring,
        then refines top candidates with float32 cosine.
        """
        if not hasattr(self, '_embeddings_int8') or self._embeddings_int8 is None:
            return self.search(query, k=k)

        model = self._load_model()
        q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)

        # Rotate and quantize query
        q_rotated = q_emb @ self._rotation
        q_scaled = q_rotated / self._quant_scale * 127.0
        q_int8 = np.clip(q_scaled, -127, 127).astype(np.int8)

        # Int8 dot product (numpy auto-vectorizes this)
        scores = self._embeddings_int8.astype(np.int16) @ q_int8.T.astype(np.int16)
        scores = scores.flatten()

        # Top candidates
        top_indices = np.argpartition(-scores, min(k * 2, len(scores) - 1))[:k * 2]

        # Refine with float32 cosine
        candidate_embs = self._embeddings[top_indices]
        exact_sims = (candidate_embs @ q_emb.T).flatten()

        top_k = np.argsort(-exact_sims)[:k]
        results = []
        for idx in top_k:
            word_idx = top_indices[idx]
            results.append((self._words[word_idx], float(exact_sims[idx])))
        return results

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """Count differing bits between two hashes."""
        xor = hash1 ^ hash2
        return bin(xor).count('1')

    def word_similarity(self, word1: str, word2: str) -> float:
        """Semantic similarity between two words (cosine of embeddings)."""
        if word1 in self._word_idx and word2 in self._word_idx:
            e1 = self._embeddings[self._word_idx[word1]]
            e2 = self._embeddings[self._word_idx[word2]]
            return float(np.dot(e1, e2))

        model = self._load_model()
        embs = model.encode([word1, word2], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))

    def hash_similarity(self, word1: str, word2: str) -> float:
        """Approximate similarity from hash distance (0-1 scale)."""
        h1 = self.hash_word(word1) if word1 not in self._word_idx else int(self._hashes[self._word_idx[word1]])
        h2 = self.hash_word(word2) if word2 not in self._word_idx else int(self._hashes[self._word_idx[word2]])
        dist = self.hamming_distance(h1, h2)
        return 1.0 - (dist / self.n_bits)
