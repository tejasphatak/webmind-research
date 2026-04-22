"""
Sparse Convergence: multi-hop reasoning over sparse co-occurrence graphs.

The dense convergence loop (convergence.py) operates on numpy vectors via
NeuronDB.search(). This module does the same math — softmax weighting,
per-hop specialization, mutual attention, query anchoring — but natively
on sparse dicts (word_idx → weight). No dense vectors. No NeuronDB.

Transformer correspondence (same as convergence.py):
  - _softmax_blend()    → scaled dot-product attention
  - _mutual_attention() → token-to-token self-attention
  - query anchor        → residual connection
  - per-hop k/threshold → layer specialization
  - convergence check   → stopping criterion

Input: word indices (from tokenized query)
Output: list of (word_idx, confidence) pairs + inspectable trace
"""

import math
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.sparse import csr_matrix as _scipy_csr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class SparseHop:
    """One step in the sparse convergence trace."""
    hop_number: int
    neighbors: list       # [(word_idx, similarity)]
    query_profile: dict   # sparse profile at this hop (before anchor)
    current: dict         # sparse profile after anchor blend
    movement: float       # 1 - cosine(current, previous)


@dataclass
class SparseConvergenceResult:
    """Result of a sparse convergence loop."""
    converged: bool
    concepts: list              # [(word_idx, confidence)] — final concept set
    hops: list = field(default_factory=list)
    confidence: float = 0.0
    phase_timings: dict = field(default_factory=dict)  # {'sparse_ms', 'dense_ms', 'total_ms', 'hops'}

    def trace(self) -> str:
        lines = []
        for hop in self.hops:
            nb_str = ", ".join(
                f"w{idx}({sim:.3f})" for idx, sim in hop.neighbors[:5]
            )
            lines.append(
                f"  Hop {hop.hop_number}: [{nb_str}] movement={hop.movement:.4f}"
            )
        status = "CONVERGED" if self.converged else "DID NOT CONVERGE"
        lines.insert(0, f"SparseConvergence: {status} (confidence={self.confidence:.3f})")
        return "\n".join(lines)


@dataclass
class SparseMultiHopResult:
    """Result of multi-hop sparse reasoning across rounds."""
    converged: bool
    concepts: list
    rounds: list = field(default_factory=list)
    confidence: float = 0.0
    phase_timings: dict = field(default_factory=dict)

    def trace(self) -> str:
        lines = []
        for i, r in enumerate(self.rounds):
            lines.append(f"=== Round {i + 1} ===")
            lines.append(r.trace())
        status = "CONVERGED" if self.converged else "DID NOT CONVERGE"
        lines.insert(0,
            f"SparseMultiHop: {status} in {len(self.rounds)} round(s), "
            f"{len(self.concepts)} concepts (confidence={self.confidence:.3f})"
        )
        return "\n".join(lines)


# --- Sparse math utilities ---

_CLAMP = 1e18  # prevent overflow in float64 multiply (1e18² = 1e36, safe)


def sparse_norm(d: dict) -> float:
    """L2 norm of a sparse vector."""
    if not d:
        return 0.0
    return math.sqrt(sum(min(v * v, _CLAMP) for v in d.values()))


def sparse_cosine(a: dict, b: dict) -> float:
    """Cosine similarity between two sparse dicts. O(min(|a|, |b|))."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(min(v * b.get(k, 0), _CLAMP) for k, v in a.items())
    if dot == 0 or math.isnan(dot) or math.isinf(dot):
        return 0.0
    na = sparse_norm(a)
    nb = sparse_norm(b)
    if na == 0 or nb == 0 or math.isnan(na) or math.isnan(nb):
        return 0.0
    result = dot / (na * nb)
    return min(result, 1.0) if not math.isnan(result) else 0.0


def sparse_blend(profiles: list, weights: list = None) -> dict:
    """Weighted average of multiple sparse profiles."""
    if not profiles:
        return {}
    if weights is None:
        weights = [1.0] * len(profiles)
    total_w = sum(weights)
    if total_w == 0:
        return {}
    result = {}
    for profile, w in zip(profiles, weights):
        for k, v in profile.items():
            result[k] = result.get(k, 0) + w * v
    for k in result:
        result[k] /= total_w
    return result


def sparse_add(a: dict, b: dict, alpha: float, beta: float) -> dict:
    """Compute alpha*a + beta*b as sparse dicts."""
    result = {}
    for k, v in a.items():
        result[k] = alpha * v
    for k, v in b.items():
        result[k] = result.get(k, 0) + beta * v
    return result


def sparse_normalize(d: dict) -> dict:
    """L2-normalize a sparse dict."""
    n = sparse_norm(d)
    if n == 0:
        return dict(d)
    return {k: v / n for k, v in d.items()}


class SparseConvergenceLoop:
    """Multi-hop convergence over sparse co-occurrence graphs.

    Each hop:
      1. Build sparse query profile from current concepts
      2. Search for neighbors with highest cosine to query profile
      3. Filter by per-hop confidence threshold (specializing)
      4. NxN mutual attention among neighbors
      5. Softmax-weighted blend into activation profile
      6. Anchor back to original query (residual connection)
      7. Check convergence (cosine(current, previous) > threshold)

    The cooc dict is accessed by reference and supports lazy loading
    via ensure_cooc callback.
    """

    def __init__(self, cooc=None, word_idx=None, words=None, word_neurons=None,
                 ensure_cooc=None,
                 get_profile=None,
                 max_hops=10, k=5,
                 convergence_threshold=0.99,
                 min_confidence=0.1,
                 min_relevance=0.3,
                 temperature=1.0):
        """
        Args:
            cooc: dict of dicts — word_idx → {neighbor_idx: weight} (legacy)
            word_idx: dict — word → index
            words: list — index → word
            word_neurons: dict — word → neuron_id
            ensure_cooc: callable(word_idx) — lazy-loads cooc for a word (legacy)
            get_profile: callable(word_idx) → dict — replaces cooc+ensure_cooc.
                         If provided, cooc and ensure_cooc are ignored.
            max_hops: maximum reasoning steps
            k: neighbors per hop
            convergence_threshold: cosine threshold for "stable"
            min_confidence: minimum neighbor similarity to participate
            min_relevance: minimum cosine between query and best neighbor
            temperature: softmax temperature for blend weighting
        """
        self.word_idx = word_idx
        self.words = words
        self.word_neurons = word_neurons
        self.max_hops = max_hops
        self.k = k
        self.convergence_threshold = convergence_threshold
        self.min_confidence = min_confidence
        self.min_relevance = min_relevance
        self.temperature = temperature

        # Profile access: prefer get_profile callback, fall back to cooc dict
        if get_profile is not None:
            self._get_profile_fn = get_profile
        else:
            self.cooc = cooc or {}
            self.ensure_cooc = ensure_cooc or (lambda idx: None)
            self._get_profile_fn = None

    def _get_word_profile(self, idx: int) -> dict:
        """Get a word's co-occurrence profile.

        Uses get_profile callback if provided (CSR+WAL path),
        else falls back to cooc dict + ensure_cooc (legacy path).
        """
        if self._get_profile_fn is not None:
            return self._get_profile_fn(idx)
        self.ensure_cooc(idx)
        return self.cooc.get(idx, {})

    def _search(self, query_profile: dict, k: int) -> list:
        """Find top-k words by co-occurrence overlap with query profile.

        Strategy: score each candidate by how much weight the query profile
        assigns to it. This is O(|query_profile|) — no need to load cooc
        for candidates. Each key in query_profile is a word_idx with a
        weight; that weight IS the relevance score for that word.

        Then for the top candidates, optionally load their cooc for a
        more precise cosine (only for the top 2*k candidates).
        """
        if not query_profile:
            return []

        # Phase 1: fast scoring — each word_idx in query_profile gets
        # its weight as a direct relevance score. O(|query_profile|).
        fast_scores = []
        for widx, weight in query_profile.items():
            if weight > 0:
                fast_scores.append((widx, weight))

        fast_scores.sort(key=lambda x: x[1], reverse=True)

        # Phase 2: for top candidates, refine with full cosine.
        # Load cooc only for top 2*k candidates — bounded work.
        refine_count = min(2 * k, len(fast_scores))
        top_candidates = fast_scores[:refine_count]

        if not top_candidates:
            return []

        q_norm = sparse_norm(query_profile)
        if q_norm == 0:
            return []

        refined = []
        for widx, _ in top_candidates:
            word_profile = self._get_word_profile(widx)
            if not word_profile:
                continue
            dot = sum(query_profile.get(j, 0) * v for j, v in word_profile.items())
            if dot > 0:
                w_norm = sparse_norm(word_profile)
                if w_norm > 0:
                    sim = dot / (q_norm * w_norm)
                    refined.append((widx, sim))

        refined.sort(key=lambda x: x[1], reverse=True)
        return refined[:k]

    def _mutual_attention(self, neighbors: list) -> list:
        """NxN mutual attention among neighbors.

        Each neighbor's similarity score is boosted by how much it
        relates to the other neighbors (compositional reasoning).

        neighbors: [(word_idx, similarity)]
        Returns: [(word_idx, boosted_similarity)]
        """
        if len(neighbors) <= 1:
            return neighbors

        # Get profiles
        profiles = []
        for widx, sim in neighbors:
            profiles.append(self._get_word_profile(widx))

        n = len(neighbors)
        # Compute pairwise cosine
        mutual_scores = []
        for i in range(n):
            total_sim = 0.0
            for j in range(n):
                if i != j:
                    total_sim += sparse_cosine(profiles[i], profiles[j])
            mutual_scores.append(total_sim / max(n - 1, 1))

        # Boost: original_sim * (1 + mutual_score)
        boosted = []
        for i, (widx, sim) in enumerate(neighbors):
            boosted.append((widx, sim * (1.0 + mutual_scores[i])))

        return boosted

    def _softmax_blend(self, neighbors: list) -> dict:
        """Blend neighbor profiles using softmax over their similarity scores.

        neighbors: [(word_idx, similarity)]
        Returns: blended sparse profile
        """
        if not neighbors:
            return {}

        sims = [sim for _, sim in neighbors]
        max_sim = max(sims) if sims else 0

        if self.temperature == float('inf'):
            # Uniform weighting
            weights = [1.0 / len(neighbors)] * len(neighbors)
        elif self.temperature <= 0:
            # Winner-take-all
            weights = [0.0] * len(neighbors)
            weights[sims.index(max_sim)] = 1.0
        else:
            # Softmax with temperature
            scaled = [(s - max_sim) / self.temperature for s in sims]
            exp_scaled = [math.exp(s) for s in scaled]
            total = sum(exp_scaled)
            if total > 0:
                weights = [e / total for e in exp_scaled]
            else:
                weights = [1.0 / len(neighbors)] * len(neighbors)

        # Blend profiles
        profiles = []
        for widx, _ in neighbors:
            profiles.append(self._get_word_profile(widx))

        return sparse_blend(profiles, weights)

    def converge(self, query_word_indices: list,
                 query_weights: list = None) -> SparseConvergenceResult:
        """Run convergence on word indices.

        Args:
            query_word_indices: list of word indices from tokenized query
            query_weights: optional per-word weights (default: position decay)

        Returns:
            SparseConvergenceResult with concepts and trace
        """
        if not query_word_indices:
            return SparseConvergenceResult(
                converged=False, concepts=[], confidence=0.0
            )

        # Build initial query profile from word co-occurrence
        if query_weights is None:
            query_weights = [1.0 / (1.0 + 0.1 * i)
                             for i in range(len(query_word_indices))]

        profiles = [self._get_word_profile(idx) for idx in query_word_indices]
        query_profile = sparse_blend(profiles, query_weights)
        query_profile = sparse_normalize(query_profile)

        if not query_profile:
            return SparseConvergenceResult(
                converged=False, concepts=[], confidence=0.0
            )

        current = dict(query_profile)
        hops = []
        last_neighbors = []

        for hop_num in range(self.max_hops):
            previous = dict(current)

            # Per-hop specialization
            progress = hop_num / max(self.max_hops - 1, 1)

            # Early: explore broadly (more neighbors)
            # Later: focus narrowly (fewer neighbors)
            hop_k = max(2, int(self.k * (1.5 - 0.7 * progress)))

            # Early: accept weaker matches
            # Later: require stronger matches
            hop_min_sim = self.min_confidence * (1.0 + 0.5 * progress)

            # 1. Search for neighbors
            neighbors = self._search(current, k=hop_k)

            # 2. Filter by minimum similarity
            neighbors = [(widx, sim) for widx, sim in neighbors
                         if sim >= hop_min_sim]

            if not neighbors:
                return SparseConvergenceResult(
                    converged=False, concepts=last_neighbors,
                    hops=hops, confidence=0.0,
                )

            # 3. Mutual attention (NxN among neighbors)
            neighbors = self._mutual_attention(neighbors)

            # Re-sort after boosting
            neighbors.sort(key=lambda x: x[1], reverse=True)

            # 4. Softmax-weighted blend into activation
            activation = self._softmax_blend(neighbors)
            activation = sparse_normalize(activation)

            # 5. Anchor to query (residual connection)
            # Early hops: more activation (explore)
            # Later hops: more query anchor (contract)
            alpha = hop_num / self.max_hops  # 0 → 1
            current = sparse_add(activation, query_profile,
                                 1.0 - alpha, alpha)
            current = sparse_normalize(current)

            # 6. Compute movement
            movement = 1.0 - sparse_cosine(current, previous)

            hops.append(SparseHop(
                hop_number=hop_num,
                neighbors=[(widx, sim) for widx, sim in neighbors],
                query_profile=dict(activation),
                current=dict(current),
                movement=movement,
            ))

            last_neighbors = neighbors

            # 7. Check convergence
            sim = sparse_cosine(current, previous)
            if sim >= self.convergence_threshold:
                # Check relevance: are the neighbors actually related to query?
                best_relevance = max(
                    sparse_cosine(self._get_word_profile(widx), query_profile)
                    for widx, _ in neighbors
                )
                if best_relevance < self.min_relevance:
                    return SparseConvergenceResult(
                        converged=False,
                        concepts=[(widx, sim) for widx, sim in last_neighbors],
                        hops=hops, confidence=0.0,
                    )

                avg_sim = sum(s for _, s in neighbors) / len(neighbors)
                return SparseConvergenceResult(
                    converged=True,
                    concepts=[(widx, sim) for widx, sim in last_neighbors],
                    hops=hops,
                    confidence=avg_sim,
                )

        # Did not converge
        avg_sim = (sum(s for _, s in last_neighbors) / len(last_neighbors)
                   if last_neighbors else 0.0)
        return SparseConvergenceResult(
            converged=False,
            concepts=[(widx, sim) for widx, sim in last_neighbors],
            hops=hops,
            confidence=avg_sim * 0.5,  # penalize non-convergence
        )


class VectorizedConvergenceLoop:
    """Multi-hop convergence using scipy spmv — uses all CPUs via BLAS.

    Same algorithm as SparseConvergenceLoop but operates on numpy/scipy
    instead of Python dicts. 5-10x faster on 4+ CPUs.

    Requires a scipy.sparse.csr_matrix (from MMapCSR.scipy_matrix or
    CSRWriteAheadLog.effective_scipy_matrix).
    """

    def __init__(self, scipy_mat, words=None, word_idx=None, word_neurons=None,
                 max_hops=10, k=5,
                 convergence_threshold=0.99,
                 min_confidence=0.1,
                 min_relevance=0.3,
                 temperature=1.0,
                 dense_candidates=64,
                 use_dense_attention=True):
        self._mat = scipy_mat  # scipy.sparse.csr_matrix (V x V)
        self._V = scipy_mat.shape[0]
        self.words = words
        self.word_idx = word_idx
        self.word_neurons = word_neurons
        self.max_hops = max_hops
        self.k = k
        self.convergence_threshold = convergence_threshold
        self.min_confidence = min_confidence
        self.min_relevance = min_relevance
        self.temperature = temperature
        self.dense_candidates = dense_candidates
        self.use_dense_attention = use_dense_attention

        # Precompute row norms for cosine (clamp to prevent overflow)
        row_sq = np.asarray(scipy_mat.multiply(scipy_mat).sum(axis=1)).ravel()
        np.clip(row_sq, 0, 1e30, out=row_sq)
        self._row_norms = np.sqrt(row_sq).astype(np.float32)

    def update_matrix(self, scipy_mat):
        """Hot-swap the matrix (e.g., when WAL changes)."""
        self._mat = scipy_mat
        if scipy_mat.shape[0] != self._V:
            self._V = scipy_mat.shape[0]
            row_sq = np.asarray(scipy_mat.multiply(scipy_mat).sum(axis=1)).ravel()
            np.clip(row_sq, 0, 1e30, out=row_sq)
            self._row_norms = np.sqrt(row_sq).astype(np.float32)

    def _dense_attention(self, candidate_indices: np.ndarray,
                         query_vec: np.ndarray) -> tuple:
        """Phase 2: Dense transformer attention on narrowed candidate set.

        Given N candidate word indices (from sparse Phase 1 shortlisting),
        extracts the N×N submatrix and runs full scaled dot-product attention
        with N×N self-attention among candidates.

        For N=64: 64×64 matmul = 4,096 ops. Trivial on CPU (<0.01ms).

        Transformer correspondence:
          - Q: query_vec (the current convergence state)
          - K: candidate rows from CSR (co-occurrence profiles)
          - V: same candidate rows (self-attention)
          - Softmax: temperature-scaled over combined scores
          - Self-attention: N×N mutual coherence among candidates

        Returns: (output_vec, neighbors_list)
          output_vec: np.ndarray shape (V,) — attention output
          neighbors_list: [(word_idx, score)] — for trace (top 10)
        """
        N = len(candidate_indices)
        if N == 0:
            return np.zeros(self._V, dtype=np.float32), []

        # Extract N rows from CSR — the candidate co-occurrence profiles
        sub = self._mat[candidate_indices, :]  # N × V sparse

        # Q×K^T: each candidate scored against query
        scores = np.asarray(sub @ query_vec).ravel()  # (N,)

        # Normalize to cosine similarity
        cand_norms = self._row_norms[candidate_indices]
        q_norm = float(np.linalg.norm(query_vec))
        with np.errstate(divide='ignore', invalid='ignore'):
            cosines = scores / (cand_norms * q_norm + 1e-10)

        # N×N self-attention: mutual coherence among candidates
        # Candidates that co-occur with each other form coherent clusters
        sim_mat = (sub @ sub.T).toarray().astype(np.float32)  # N×N dense
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_outer = np.outer(cand_norms, cand_norms) + 1e-10
            sim_mat /= norm_outer

        # Mutual boost: mean similarity to other candidates
        np.fill_diagonal(sim_mat, 0.0)
        mutual = sim_mat.sum(axis=1) / max(N - 1, 1)

        # Combined score: query relevance × (1 + mutual coherence)
        combined = cosines * (1.0 + mutual)

        # Softmax attention weights (temperature-scaled)
        if self.temperature > 0:
            scaled = (combined - combined.max()) / self.temperature
            weights = np.exp(scaled)
            weights /= weights.sum() + 1e-10
        else:
            weights = np.zeros(N, dtype=np.float32)
            weights[combined.argmax()] = 1.0

        # V projection: weighted blend of candidate rows → output
        output = np.asarray(sub.T @ weights).ravel().astype(np.float32)

        # Build trace (top 10 candidates by score)
        scored = sorted(zip(candidate_indices.tolist(), combined.tolist()),
                        key=lambda x: x[1], reverse=True)

        return output, scored[:10]

    def _search(self, query_vec: np.ndarray, k: int) -> list:
        """Top-k words by spmv attention. Uses all CPUs via BLAS."""
        # scores = W @ q — sparse matrix-vector multiply
        scores = np.asarray(self._mat @ query_vec).ravel()

        q_norm = float(np.linalg.norm(query_vec))
        if q_norm == 0:
            return []

        # Cosine = dot / (row_norm * q_norm)
        with np.errstate(divide='ignore', invalid='ignore'):
            cosines = scores / (self._row_norms * q_norm + 1e-10)

        # Top-k
        if len(cosines) <= k:
            top_indices = np.argsort(-cosines)
        else:
            top_indices = np.argpartition(-cosines, k)[:k]
            top_indices = top_indices[np.argsort(-cosines[top_indices])]

        return [(int(idx), float(cosines[idx]))
                for idx in top_indices if cosines[idx] > 0]

    def _mutual_attention(self, neighbors: list) -> list:
        """NxN mutual attention via scipy submatrix multiply."""
        if len(neighbors) <= 1:
            return neighbors
        indices = np.array([idx for idx, _ in neighbors], dtype=np.int32)
        # Extract k rows as submatrix
        sub = self._mat[indices, :]
        # k×k similarity matrix in one call
        sim_mat = (sub @ sub.T).toarray()
        # Normalize to cosine
        norms = self._row_norms[indices]
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_outer = np.outer(norms, norms) + 1e-10
            sim_mat = sim_mat / norm_outer

        n = len(neighbors)
        boosted = []
        for i, (widx, sim) in enumerate(neighbors):
            mutual = 0.0
            for j in range(n):
                if i != j:
                    mutual += sim_mat[i, j]
            mutual /= max(n - 1, 1)
            boosted.append((widx, sim * (1.0 + mutual)))
        return boosted

    def _softmax_blend(self, neighbors: list) -> np.ndarray:
        """Blend neighbor rows using softmax weights. Returns dense vector."""
        if not neighbors:
            return np.zeros(self._V, dtype=np.float32)

        sims = np.array([s for _, s in neighbors], dtype=np.float32)
        max_sim = sims.max()

        if self.temperature <= 0:
            weights = np.zeros_like(sims)
            weights[sims.argmax()] = 1.0
        elif self.temperature == float('inf'):
            weights = np.ones_like(sims) / len(sims)
        else:
            scaled = (sims - max_sim) / self.temperature
            exp_scaled = np.exp(scaled)
            total = exp_scaled.sum()
            weights = exp_scaled / total if total > 0 else np.ones_like(sims) / len(sims)

        # Blend: profiles.T @ weights — one spmv
        indices = np.array([idx for idx, _ in neighbors], dtype=np.int32)
        profiles = self._mat[indices, :]  # k × V sparse
        blended = np.asarray(profiles.T @ weights).ravel()
        return blended.astype(np.float32)

    def converge(self, query_word_indices: list,
                 query_weights: list = None) -> SparseConvergenceResult:
        """Run vectorized convergence with two-phase attention.

        Phase 1 (sparse): spmv search narrows V words → N candidates.
        Phase 2 (dense): full Q×K^T attention + N×N self-attention on candidates.
        Residual: additive (output + query), standard transformer pattern.

        Falls back to original sparse-only path when use_dense_attention=False.
        """
        import time

        if not query_word_indices:
            return SparseConvergenceResult(converged=False, concepts=[], confidence=0.0)

        if query_weights is None:
            query_weights = [1.0 / (1.0 + 0.1 * i)
                             for i in range(len(query_word_indices))]

        # Build initial query vector from weighted blend of word rows
        weights_arr = np.array(query_weights, dtype=np.float32)
        weights_arr /= weights_arr.sum() + 1e-10
        valid_indices = [idx for idx in query_word_indices if idx < self._V]
        if not valid_indices:
            return SparseConvergenceResult(converged=False, concepts=[], confidence=0.0)

        query_rows = self._mat[valid_indices, :]
        query_w = weights_arr[:len(valid_indices)]
        query_profile = np.asarray(query_rows.T @ query_w).ravel().astype(np.float32)

        # Normalize
        qn = np.linalg.norm(query_profile)
        if qn > 0:
            query_profile /= qn

        current = query_profile.copy()
        hops = []
        last_neighbors = []

        t_start = time.perf_counter()
        t_sparse = 0.0
        t_dense = 0.0

        for hop_num in range(self.max_hops):
            previous = current.copy()
            progress = hop_num / max(self.max_hops - 1, 1)

            if self.use_dense_attention:
                # === TWO-PHASE DENSE ATTENTION PATH ===

                # Phase 1: Sparse shortlist — wider net than original k=5
                hop_N = max(16, int(self.dense_candidates * (1.3 - 0.3 * progress)))
                hop_min_sim = self.min_confidence * (1.0 + 0.3 * progress)

                t0 = time.perf_counter()
                neighbors = self._search(current, k=hop_N)
                neighbors = [(w, s) for w, s in neighbors if s >= hop_min_sim]
                t_sparse += time.perf_counter() - t0

                if not neighbors:
                    return SparseConvergenceResult(
                        converged=False, concepts=last_neighbors,
                        hops=hops, confidence=0.0,
                        phase_timings={
                            'sparse_ms': round(t_sparse * 1000, 2),
                            'dense_ms': round(t_dense * 1000, 2),
                            'total_ms': round((time.perf_counter() - t_start) * 1000, 2),
                            'hops': len(hops),
                        })

                candidate_indices = np.array([idx for idx, _ in neighbors],
                                             dtype=np.int32)

                # Phase 2: Dense attention on candidate set
                t0 = time.perf_counter()
                activation, scored = self._dense_attention(candidate_indices, current)
                t_dense += time.perf_counter() - t0

                # Normalize activation
                an = np.linalg.norm(activation)
                if an > 0:
                    activation /= an

                # Additive residual: output + query (standard transformer)
                current = activation + query_profile
                cn = np.linalg.norm(current)
                if cn > 0:
                    current /= cn

                last_neighbors = scored

            else:
                # === ORIGINAL SPARSE-ONLY PATH (fallback) ===
                hop_k = max(2, int(self.k * (1.5 - 0.7 * progress)))
                hop_min_sim = self.min_confidence * (1.0 + 0.5 * progress)

                t0 = time.perf_counter()
                neighbors = self._search(current, k=hop_k)
                neighbors = [(w, s) for w, s in neighbors if s >= hop_min_sim]
                t_sparse += time.perf_counter() - t0

                if not neighbors:
                    return SparseConvergenceResult(
                        converged=False, concepts=last_neighbors,
                        hops=hops, confidence=0.0,
                        phase_timings={
                            'sparse_ms': round(t_sparse * 1000, 2),
                            'dense_ms': 0.0,
                            'total_ms': round((time.perf_counter() - t_start) * 1000, 2),
                            'hops': len(hops),
                        })

                neighbors = self._mutual_attention(neighbors)
                neighbors.sort(key=lambda x: x[1], reverse=True)

                activation = self._softmax_blend(neighbors)
                an = np.linalg.norm(activation)
                if an > 0:
                    activation /= an

                alpha = hop_num / self.max_hops
                current = (1.0 - alpha) * activation + alpha * query_profile
                cn = np.linalg.norm(current)
                if cn > 0:
                    current /= cn

                last_neighbors = neighbors

            # Movement (shared by both paths)
            dot = float(np.dot(current, previous))
            movement = 1.0 - min(dot, 1.0)

            hops.append(SparseHop(
                hop_number=hop_num,
                neighbors=[(w, s) for w, s in last_neighbors],
                query_profile={},
                current={},
                movement=movement,
            ))

            # Convergence check (shared by both paths)
            if dot >= self.convergence_threshold:
                best_idx = last_neighbors[0][0] if last_neighbors else -1
                if best_idx >= 0 and best_idx < self._V:
                    row = np.asarray(self._mat[best_idx, :].todense()).ravel()
                    rn = self._row_norms[best_idx]
                    qn2 = np.linalg.norm(query_profile)
                    if rn > 0 and qn2 > 0:
                        best_relevance = float(np.dot(row, query_profile) / (rn * qn2))
                    else:
                        best_relevance = 0.0
                else:
                    best_relevance = 0.0

                timings = {
                    'sparse_ms': round(t_sparse * 1000, 2),
                    'dense_ms': round(t_dense * 1000, 2),
                    'total_ms': round((time.perf_counter() - t_start) * 1000, 2),
                    'hops': len(hops),
                }

                if best_relevance < self.min_relevance:
                    return SparseConvergenceResult(
                        converged=False,
                        concepts=[(w, s) for w, s in last_neighbors],
                        hops=hops, confidence=0.0,
                        phase_timings=timings)

                avg_sim = sum(s for _, s in last_neighbors) / len(last_neighbors)
                return SparseConvergenceResult(
                    converged=True,
                    concepts=[(w, s) for w, s in last_neighbors],
                    hops=hops, confidence=avg_sim,
                    phase_timings=timings)

        avg_sim = (sum(s for _, s in last_neighbors) / len(last_neighbors)
                   if last_neighbors else 0.0)
        return SparseConvergenceResult(
            converged=False,
            concepts=[(w, s) for w, s in last_neighbors],
            hops=hops, confidence=avg_sim * 0.5,
            phase_timings={
                'sparse_ms': round(t_sparse * 1000, 2),
                'dense_ms': round(t_dense * 1000, 2),
                'total_ms': round((time.perf_counter() - t_start) * 1000, 2),
                'hops': len(hops),
            })


class SparseMultiHop:
    """Chains multiple SparseConvergenceLoop rounds.

    Round 1: query → converge → concepts A
    Round 2: query + concepts_A blend → converge → concepts B
    Stop when: no new concepts or max rounds.
    """

    def __init__(self, loop: SparseConvergenceLoop,
                 max_rounds: int = 3,
                 concept_blend_weight: float = 0.4):
        self.loop = loop
        self.max_rounds = max_rounds
        self.concept_blend_weight = concept_blend_weight

    def reason(self, query_word_indices: list,
               query_weights: list = None) -> SparseMultiHopResult:
        """Run multi-hop sparse reasoning."""
        if not query_word_indices:
            return SparseMultiHopResult(
                converged=False, concepts=[], confidence=0.0,
            )

        all_concepts = []
        seen_indices = set()
        rounds = []

        # Build initial query profile
        if query_weights is None:
            query_weights = [1.0 / (1.0 + 0.1 * i)
                             for i in range(len(query_word_indices))]

        current_indices = list(query_word_indices)
        current_weights = list(query_weights)

        for round_num in range(self.max_rounds):
            result = self.loop.converge(current_indices, current_weights)
            rounds.append(result)

            # Collect new concepts
            new_concepts = []
            for widx, sim in result.concepts:
                if widx not in seen_indices:
                    new_concepts.append((widx, sim))
                    seen_indices.add(widx)

            all_concepts.extend(new_concepts)

            # Stop conditions
            if round_num == 0 and not result.converged and not result.concepts:
                break
            if not new_concepts and round_num > 0:
                break
            if round_num == self.max_rounds - 1:
                break

            # Prepare next round: add discovered concepts to query
            if new_concepts:
                w = self.concept_blend_weight
                # Build new query: original words + discovered concepts
                new_indices = [widx for widx, _ in new_concepts]
                new_weights_list = [sim * w for _, sim in new_concepts]
                # Original query words get (1-w) weight
                orig_w = [(1 - w) * qw for qw in query_weights]

                current_indices = list(query_word_indices) + new_indices
                current_weights = orig_w + new_weights_list

        # Aggregate phase timings across rounds
        agg_timings = {'sparse_ms': 0.0, 'dense_ms': 0.0, 'total_ms': 0.0, 'hops': 0}
        for r in rounds:
            for k in ('sparse_ms', 'dense_ms', 'total_ms', 'hops'):
                agg_timings[k] += r.phase_timings.get(k, 0)

        any_converged = any(r.converged for r in rounds)
        if all_concepts and any_converged:
            avg_conf = sum(s for _, s in all_concepts) / len(all_concepts)
            return SparseMultiHopResult(
                converged=True,
                concepts=all_concepts,
                rounds=rounds,
                confidence=avg_conf,
                phase_timings=agg_timings,
            )
        else:
            return SparseMultiHopResult(
                converged=False,
                concepts=all_concepts,
                rounds=rounds,
                confidence=0.0,
                phase_timings=agg_timings,
            )
