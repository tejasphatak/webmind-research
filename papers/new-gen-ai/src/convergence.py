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
    """

    def __init__(self, db: NeuronDB,
                 max_hops: int = 10,
                 k: int = 5,
                 convergence_threshold: float = 0.99,
                 min_confidence: float = 0.1,
                 min_relevance: float = 0.3):
        """
        Args:
            db: NeuronDB to search in
            max_hops: maximum reasoning steps before abort
            k: number of neighbors to retrieve per hop
            convergence_threshold: cosine sim threshold for "stable"
            min_confidence: minimum neuron confidence to participate
            min_relevance: minimum cosine similarity between query and
                          best neighbor to accept convergence. Below this,
                          the system says "I don't know" even if the vector
                          stabilized. Invariant #4: honest about failure.
        """
        self.db = db
        self.max_hops = max_hops
        self.k = k
        self.convergence_threshold = convergence_threshold
        self.min_confidence = min_confidence
        self.min_relevance = min_relevance

    def converge(self, query_vector: np.ndarray) -> ConvergenceResult:
        """
        Run the convergence loop.

        Returns ConvergenceResult with converged=True if stable,
        converged=False if max hops reached (honest abstention).
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

            # 1. Search nearest neurons
            neighbors = self.db.search(current, k=self.k)

            # Filter by minimum confidence
            neighbors = [n for n in neighbors if n.confidence >= self.min_confidence]

            if not neighbors:
                # No neurons above confidence threshold — honest abort
                return ConvergenceResult(
                    converged=False, vector=current,
                    concepts=[], hops=hops, confidence=0.0,
                )

            # 2. Blend neighbors weighted by confidence → activation
            activation = self._weighted_blend(neighbors)

            # 3. Anchor to query (prevents drift)
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

            # 4. Check convergence: has the vector stopped moving?
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
        Blend neuron vectors weighted by confidence.
        This is what softmax does in transformers — but inspectable.
        """
        vectors = np.array([n.vector for n in neurons])
        confidences = np.array([n.confidence for n in neurons], dtype=np.float32)

        # Softmax-like normalization of confidences
        confidences = np.maximum(confidences, 0)  # floor at 0 for weighting
        total = confidences.sum()
        if total == 0:
            weights = np.ones(len(neurons), dtype=np.float32) / len(neurons)
        else:
            weights = confidences / total

        blended = np.average(vectors, axis=0, weights=weights).astype(np.float32)

        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm

        return blended

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
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
