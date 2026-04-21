"""
Convergence Loop: the core reasoning mechanism.

Replaces attention in transformers. Multi-hop spatial search with query anchor.

How it works:
  1. Encode query → vector
  2. Search nearest neurons in the DB
  3. Blend neighbors weighted by confidence → activation
  4. Mix activation with original query (anchor prevents drift)
  5. Repeat until vector stabilizes (converged) or max hops (abort)

Converged = answer found. Not converged = "I don't know." (Invariant #4)

Each hop is inspectable — the trace shows exactly why the answer was found.
(Invariant #2: every answer has a source.)
"""

from dataclasses import dataclass, field

import numpy as np

from neuron import Neuron, NeuronDB


@dataclass
class Hop:
    """One step in the convergence trace. For inspectability."""
    hop_number: int
    neighbors: list          # [(neuron_id, confidence, similarity)]
    activation: np.ndarray   # blended vector before anchor
    current: np.ndarray      # vector after anchor blend
    movement: float          # cosine distance from previous step


@dataclass
class ConvergenceResult:
    """Result of a convergence loop."""
    converged: bool
    vector: np.ndarray              # final vector position
    concepts: list                  # neurons that participated in final hop
    hops: list = field(default_factory=list)  # full trace
    confidence: float = 0.0         # aggregate confidence of result

    def trace(self) -> str:
        """Human-readable trace of the convergence path. Invariant #2."""
        lines = []
        for hop in self.hops:
            neighbors_str = ", ".join(
                f"n{nid}({conf:.2f})" for nid, conf, _ in hop.neighbors
            )
            lines.append(
                f"  Hop {hop.hop_number}: [{neighbors_str}] "
                f"movement={hop.movement:.4f}"
            )
        status = "CONVERGED" if self.converged else "DID NOT CONVERGE"
        lines.insert(0, f"Convergence: {status} (confidence={self.confidence:.3f})")
        return "\n".join(lines)


class ConvergenceLoop:
    """
    Multi-hop reasoning through spatial search.

    The convergence loop IS attention — but each hop is inspectable.
    Query anchor IS a residual connection — prevents drift.
    Convergence check IS the stopping criterion — no convergence = abstain.

    Transformer correspondence:
      - _weighted_blend() uses softmax (exponential sharpening) over
        confidence scores, identical to scaled dot-product attention.
      - Per-hop k and threshold schedules give functional layer
        specialization: early hops explore broadly, later hops focus.
      - Concept-to-concept attention (NxN among discovered neighbors)
        provides compositional reasoning — same as token-to-token
        attention in transformer self-attention.
    """

    def __init__(self, db: NeuronDB = None,
                 max_hops: int = 10,
                 k: int = 5,
                 convergence_threshold: float = 0.99,
                 min_confidence: float = 0.1,
                 min_relevance: float = 0.3,
                 temperature: float = 1.0,
                 search_fn=None,
                 blend_fn=None,
                 cosine_fn=None):
        """
        Args:
            db: NeuronDB to search in (optional if search_fn provided)
            max_hops: maximum reasoning steps before abort
            k: number of neighbors to retrieve per hop
            convergence_threshold: cosine sim threshold for "stable"
            min_confidence: minimum neuron confidence to participate
            min_relevance: minimum cosine similarity between query and
                          best neighbor to accept convergence. Below this,
                          the system says "I don't know" even if the vector
                          stabilized. Invariant #4: honest about failure.
            temperature: softmax temperature for confidence weighting.
                        Higher = more uniform, lower = sharper.
                        Default 1.0 gives true softmax behavior.
                        Use float('inf') to recover pre-softmax linear
                        normalization for backward compatibility.
            search_fn: optional callable(query, k) → list of Neuron-like objects.
                      Allows plugging in sparse search or any other backend.
                      Each returned object must have .id, .vector, .confidence.
            blend_fn: optional callable(neurons) → blended vector.
                     If None, uses default _weighted_blend.
            cosine_fn: optional callable(a, b) → float similarity.
                      If None, uses default numpy cosine.
        """
        self.db = db
        self.max_hops = max_hops
        self.k = k
        self.convergence_threshold = convergence_threshold
        self.min_confidence = min_confidence
        self.min_relevance = min_relevance
        self.temperature = temperature
        self._search_fn = search_fn
        self._blend_fn = blend_fn
        self._cosine_fn = cosine_fn

    def converge(self, query_vector: np.ndarray) -> ConvergenceResult:
        """
        Run the convergence loop.

        Returns ConvergenceResult with converged=True if stable,
        converged=False if max hops reached (honest abstention).

        Per-hop specialization (like transformer layers having different
        learned parameters): early hops explore broadly (higher k, lower
        confidence threshold), later hops focus narrowly (lower k, higher
        threshold). This gives functional layer specialization without
        learned weights.
        """
        query = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return ConvergenceResult(
                converged=False, vector=query, concepts=[], confidence=0.0
            )
        query = query / query_norm

        current = query.copy()
        hops = []
        last_concepts = []

        for hop_num in range(self.max_hops):
            previous = current.copy()

            # --- Per-hop specialization (transformer layer analogy) ---
            # Progress through the loop: 0.0 (first hop) → 1.0 (last hop)
            progress = hop_num / max(self.max_hops - 1, 1)

            # Early hops: explore broadly (more neighbors)
            # Later hops: focus narrowly (fewer neighbors)
            hop_k = max(2, int(self.k * (1.5 - 0.7 * progress)))

            # Early hops: accept lower confidence (explore)
            # Later hops: require higher confidence (focus)
            hop_min_conf = self.min_confidence * (1.0 + 0.5 * progress)

            # 1. Search nearest neurons with per-hop k
            if self._search_fn:
                neighbors = self._search_fn(current, k=hop_k)
            else:
                neighbors = self.db.search(current, k=hop_k)

            # Filter by per-hop minimum confidence
            neighbors = [n for n in neighbors if n.confidence >= hop_min_conf]

            if not neighbors:
                # No neurons above confidence threshold — honest abort
                return ConvergenceResult(
                    converged=False, vector=current,
                    concepts=[], hops=hops, confidence=0.0,
                )

            # 2. Concept-to-concept attention (NxN among neighbors)
            #    Transformers compute attention between all tokens.
            #    Here we compute pairwise similarity among discovered
            #    neighbors and boost those that are mutually relevant.
            #    This gives compositional reasoning: concepts that
            #    "attend to each other" get amplified.
            neighbors = self._mutual_attention(neighbors)

            # 3. Blend neighbors weighted by confidence → activation
            #    Uses softmax (exponential sharpening) over confidences.
            if self._blend_fn:
                activation = self._blend_fn(neighbors)
            else:
                activation = self._weighted_blend(neighbors)

            # 4. Anchor to query (prevents drift)
            #    Early hops: explore (more activation)
            #    Later hops: contract (more query anchor)
            alpha = hop_num / self.max_hops  # 0→1
            current = (1 - alpha) * activation + alpha * query

            # Re-normalize
            norm = np.linalg.norm(current)
            if norm > 0:
                current = current / norm

            # Compute movement (how much the vector changed)
            movement = 1.0 - float(self._cosine_sim(current, previous))

            # Compute similarities for the trace
            neighbor_info = []
            for n in neighbors:
                sim = float(self._cosine_sim(n.vector, current))
                neighbor_info.append((n.id, n.confidence, sim))

            hops.append(Hop(
                hop_number=hop_num,
                neighbors=neighbor_info,
                activation=activation.copy(),
                current=current.copy(),
                movement=movement,
            ))

            last_concepts = neighbors

            # 5. Check convergence: has the vector stopped moving?
            sim = self._cosine_sim(current, previous)
            if sim >= self.convergence_threshold:
                # Vector stabilized — but are the neighbors actually relevant?
                best_relevance = max(
                    self._cosine_sim(n.vector, query) for n in neighbors
                )
                if best_relevance < self.min_relevance:
                    # Converged on garbage — honest abstention
                    return ConvergenceResult(
                        converged=False,
                        vector=current,
                        concepts=last_concepts,
                        hops=hops,
                        confidence=0.0,
                    )

                # CONVERGED on relevant neurons
                avg_confidence = np.mean([n.confidence for n in neighbors])
                return ConvergenceResult(
                    converged=True,
                    vector=current,
                    concepts=last_concepts,
                    hops=hops,
                    confidence=float(avg_confidence),
                )

        # DID NOT CONVERGE — "I don't know" (Invariant #4)
        avg_confidence = (
            np.mean([n.confidence for n in last_concepts])
            if last_concepts else 0.0
        )
        return ConvergenceResult(
            converged=False,
            vector=current,
            concepts=last_concepts,
            hops=hops,
            confidence=float(avg_confidence) * 0.5,  # penalize non-convergence
        )

    def _weighted_blend(self, neurons: list) -> np.ndarray:
        """
        Blend neuron vectors weighted by softmax over confidence scores.

        This IS softmax attention: exp(c / T) / sum(exp(c / T)).
        Temperature controls sharpening:
          - T → 0: winner-take-all (hard attention)
          - T = 1: standard softmax
          - T → ∞: uniform weighting (recovers old linear normalization)
        """
        vectors = np.array([n.vector for n in neurons])
        confidences = np.array([n.confidence for n in neurons], dtype=np.float32)

        # Floor at 0 for weighting (negative confidence = no contribution)
        confidences = np.maximum(confidences, 0)

        if confidences.sum() == 0:
            weights = np.ones(len(neurons), dtype=np.float32) / len(neurons)
        elif self.temperature == float('inf'):
            # Backward compat: infinite temperature = linear normalization
            weights = confidences / confidences.sum()
        else:
            # Softmax with temperature: exp(c/T) / sum(exp(c/T))
            # Subtract max for numerical stability (standard softmax trick)
            scaled = confidences / max(self.temperature, 1e-8)
            scaled = scaled - scaled.max()
            exp_scaled = np.exp(scaled)
            weights = exp_scaled / exp_scaled.sum()

        blended = np.average(vectors, axis=0, weights=weights).astype(np.float32)

        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm

        return blended

    def _mutual_attention(self, neurons: list) -> list:
        """
        Concept-to-concept attention: NxN similarity among discovered
        neighbors. Boost neurons that are mutually relevant — they
        "attend to each other."

        This is the compositional reasoning step that makes transformers
        work: tokens don't just attend to the query, they attend to
        each other. Here, concepts that form a coherent cluster get
        boosted, while isolated concepts get dampened.

        Returns the same neurons with confidence adjusted by mutual
        relevance. Does NOT modify the original neuron objects — creates
        lightweight wrappers.
        """
        if len(neurons) <= 1:
            return neurons

        n = len(neurons)

        # Compute pairwise similarity — uses pluggable cosine if provided
        if self._cosine_fn:
            sim_matrix = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                for j in range(i + 1, n):
                    s = self._cosine_fn(neurons[i].vector, neurons[j].vector)
                    sim_matrix[i, j] = s
                    sim_matrix[j, i] = s
        else:
            vectors = np.array([nn.vector for nn in neurons])
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normed = vectors / norms
            sim_matrix = normed @ normed.T  # NxN
            np.fill_diagonal(sim_matrix, 0.0)

        # Each neuron's mutual relevance = mean similarity to all others
        mutual_scores = sim_matrix.sum(axis=1) / max(n - 1, 1)

        # Boost confidence by mutual relevance:
        # new_confidence = original * (1 + mutual_score)
        # This preserves ordering but amplifies coherent clusters.
        boosted = []
        for i, neuron in enumerate(neurons):
            boost_factor = 1.0 + float(mutual_scores[i])
            # Create a lightweight copy with boosted confidence
            boosted_neuron = Neuron(
                id=neuron.id,
                vector=neuron.vector,
                confidence=neuron.confidence * boost_factor,
                successors=neuron.successors,
                predecessors=neuron.predecessors,
                timestamp=neuron.timestamp,
                temporal=neuron.temporal,
                level=neuron.level,
            )
            boosted.append(boosted_neuron)

        return boosted

    def _cosine_sim(self, a, b) -> float:
        """Cosine similarity. Uses pluggable cosine_fn if provided."""
        if self._cosine_fn:
            return self._cosine_fn(a, b)
        dot = float(np.dot(a, b))
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


@dataclass
class MultiHopResult:
    """Result of multi-hop reasoning across convergence rounds."""
    converged: bool
    concepts: list                          # merged concept set from all rounds
    rounds: list = field(default_factory=list)  # list of ConvergenceResult per round
    confidence: float = 0.0
    vector: np.ndarray = None               # final vector position

    def trace(self) -> str:
        """Human-readable trace across all rounds. Invariant #2."""
        lines = []
        for i, r in enumerate(self.rounds):
            lines.append(f"=== Round {i + 1} ===")
            lines.append(r.trace())
        status = "CONVERGED" if self.converged else "DID NOT CONVERGE"
        concept_count = len(self.concepts)
        lines.insert(0,
            f"Multi-hop: {status} in {len(self.rounds)} round(s), "
            f"{concept_count} concepts (confidence={self.confidence:.3f})"
        )
        return "\n".join(lines)


class MultiHopConvergence:
    """
    Chained convergence: each round's discovered concepts shift the query
    for the next round, allowing reasoning to cross concept boundaries.

    Round 1: query → converge → concepts A
    Round 2: query + concepts_A blend → converge → concepts B
    ...
    Stop when: no new concepts found, or max rounds reached.

    This is iterative retrieval-generation (ITER-RETGEN) done without
    a neural component. Each round is inspectable. The query anchor
    prevents drift across rounds.
    """

    def __init__(self, loop: ConvergenceLoop,
                 max_rounds: int = 3,
                 concept_blend_weight: float = 0.4):
        """
        Args:
            loop: the underlying ConvergenceLoop
            max_rounds: maximum reasoning rounds
            concept_blend_weight: how much discovered concepts shift the query
                                 (0 = ignore concepts, 1 = ignore query)
        """
        self.loop = loop
        self.max_rounds = max_rounds
        self.concept_blend_weight = concept_blend_weight

    def reason(self, query_vector: np.ndarray) -> MultiHopResult:
        """
        Run multi-hop reasoning.

        Each round discovers concepts. Those concepts' vectors get blended
        into the query for the next round, shifting the search into new
        regions of concept space.
        """
        query = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm == 0:
            return MultiHopResult(
                converged=False, concepts=[], confidence=0.0,
                vector=query,
            )
        query = query / norm

        all_concepts = []       # merged across rounds
        seen_ids = set()        # avoid duplicates
        rounds = []
        current_query = query.copy()

        for round_num in range(self.max_rounds):
            # Run convergence with the current (possibly shifted) query
            result = self.loop.converge(current_query)
            rounds.append(result)

            # Collect new concepts from this round
            new_concepts = []
            for c in result.concepts:
                if c.id not in seen_ids:
                    new_concepts.append(c)
                    seen_ids.add(c.id)

            all_concepts.extend(new_concepts)

            # Stop conditions:
            # 1. First round didn't converge at all → no point continuing
            if round_num == 0 and not result.converged and not result.concepts:
                break

            # 2. No new concepts found → we've exhausted this reasoning chain
            if not new_concepts and round_num > 0:
                break

            # 3. Last round → don't prepare next query
            if round_num == self.max_rounds - 1:
                break

            # Prepare next round: blend discovered concepts into query
            # This shifts the search to a new region of concept space
            if new_concepts:
                concept_blend = self._blend_concepts(new_concepts)
                w = self.concept_blend_weight
                current_query = (1 - w) * query + w * concept_blend
                norm = np.linalg.norm(current_query)
                if norm > 0:
                    current_query = current_query / norm

        # Determine overall result
        any_converged = any(r.converged for r in rounds)
        if all_concepts and any_converged:
            avg_conf = float(np.mean([c.confidence for c in all_concepts]))
            final_vec = rounds[-1].vector if rounds else query
            return MultiHopResult(
                converged=True,
                concepts=all_concepts,
                rounds=rounds,
                confidence=avg_conf,
                vector=final_vec,
            )
        else:
            return MultiHopResult(
                converged=False,
                concepts=all_concepts,
                rounds=rounds,
                confidence=0.0,
                vector=rounds[-1].vector if rounds else query,
            )

    def _blend_concepts(self, concepts: list) -> np.ndarray:
        """Blend concept vectors weighted by confidence."""
        vectors = np.array([c.vector for c in concepts], dtype=np.float32)
        confs = np.array([max(c.confidence, 0.01) for c in concepts], dtype=np.float32)
        total = confs.sum()
        if total == 0:
            weights = np.ones(len(concepts), dtype=np.float32) / len(concepts)
        else:
            weights = confs / total
        blended = np.average(vectors, axis=0, weights=weights).astype(np.float32)
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm
        return blended
