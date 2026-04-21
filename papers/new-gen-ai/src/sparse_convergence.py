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

def sparse_norm(d: dict) -> float:
    """L2 norm of a sparse vector."""
    if not d:
        return 0.0
    return math.sqrt(sum(v * v for v in d.values()))


def sparse_cosine(a: dict, b: dict) -> float:
    """Cosine similarity between two sparse dicts. O(min(|a|, |b|))."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    if dot == 0:
        return 0.0
    na = sparse_norm(a)
    nb = sparse_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


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

    def __init__(self, cooc, word_idx, words, word_neurons,
                 ensure_cooc=None,
                 max_hops=10, k=5,
                 convergence_threshold=0.99,
                 min_confidence=0.1,
                 min_relevance=0.3,
                 temperature=1.0):
        """
        Args:
            cooc: dict of dicts — word_idx → {neighbor_idx: weight}
            word_idx: dict — word → index
            words: list — index → word
            word_neurons: dict — word → neuron_id
            ensure_cooc: callable(word_idx) — lazy-loads cooc for a word
            max_hops: maximum reasoning steps
            k: neighbors per hop
            convergence_threshold: cosine threshold for "stable"
            min_confidence: minimum neighbor similarity to participate
            min_relevance: minimum cosine between query and best neighbor
            temperature: softmax temperature for blend weighting
        """
        self.cooc = cooc
        self.word_idx = word_idx
        self.words = words
        self.word_neurons = word_neurons
        self.ensure_cooc = ensure_cooc or (lambda idx: None)
        self.max_hops = max_hops
        self.k = k
        self.convergence_threshold = convergence_threshold
        self.min_confidence = min_confidence
        self.min_relevance = min_relevance
        self.temperature = temperature

    def _get_word_profile(self, idx: int) -> dict:
        """Get a word's co-occurrence profile, triggering lazy load."""
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

        # Load cooc for just the top candidates
        for widx, _ in top_candidates:
            self.ensure_cooc(widx)

        refined = []
        for widx, _ in top_candidates:
            word_profile = self.cooc.get(widx, {})
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
            profiles.append(self.cooc.get(widx, {}))

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
            profiles.append(self.cooc.get(widx, {}))

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

        # Ensure cooc loaded for query words
        for idx in query_word_indices:
            self.ensure_cooc(idx)

        profiles = [self.cooc.get(idx, {}) for idx in query_word_indices]
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
                    sparse_cosine(self.cooc.get(widx, {}), query_profile)
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

        any_converged = any(r.converged for r in rounds)
        if all_concepts and any_converged:
            avg_conf = sum(s for _, s in all_concepts) / len(all_concepts)
            return SparseMultiHopResult(
                converged=True,
                concepts=all_concepts,
                rounds=rounds,
                confidence=avg_conf,
            )
        else:
            return SparseMultiHopResult(
                converged=False,
                concepts=all_concepts,
                rounds=rounds,
                confidence=0.0,
            )
