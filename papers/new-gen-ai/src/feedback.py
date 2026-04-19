"""
Feedback Loop: updates neuron confidence based on query outcomes.

Three layers, time-stratified (MVP implements layers 1 and 2):

  Layer 1: Self-consistency (10% sampling)
    Ask the same thing differently, compare answers.
    Same answer = boost. Different answer = penalize.

  Layer 2: User behavior (always, free)
    Follow-up on same topic = answer was insufficient → penalize.
    New topic = answer was accepted → boost.

  Layer 3: External verification (idle time only) — post-MVP.

Confidence changes:
  Useful fire → confidence × 1.1
  Useless fire → confidence × 0.9
  Capped at ±0.8 to prevent mode collapse
"""

import random
from dataclasses import dataclass, field

import numpy as np

from neuron import NeuronDB
from convergence import ConvergenceLoop, ConvergenceResult


SELF_CONSISTENCY_RATE = 0.1  # 10% of queries trigger self-check
TOPIC_SHIFT_THRESHOLD = 0.3  # cosine sim below this = new topic


@dataclass
class FeedbackEvent:
    """Record of a feedback action for inspectability."""
    neuron_ids: list
    action: str           # "boost", "penalize", "none"
    reason: str
    details: dict = field(default_factory=dict)


class FeedbackLoop:
    """
    Updates neuron confidence based on query outcomes.

    No gradient descent. No loss function. Just:
    - Did the same neurons fire for the same question asked differently?
    - Did the user move on (accepted) or dig deeper (insufficient)?
    """

    def __init__(self, db: NeuronDB, convergence: ConvergenceLoop,
                 consistency_rate: float = SELF_CONSISTENCY_RATE,
                 seed: int = None):
        self.db = db
        self.convergence = convergence
        self.consistency_rate = consistency_rate
        self._rng = random.Random(seed)
        self._last_query_vector = None
        self._last_concepts = []
        self._history = []  # FeedbackEvent log

    @property
    def history(self) -> list:
        return list(self._history)

    def on_query_result(self, query_vector: np.ndarray,
                        result: ConvergenceResult) -> FeedbackEvent:
        """
        Called after every query. Runs layer 2 (user behavior)
        and probabilistically triggers layer 1 (self-consistency).

        Returns the feedback event for inspectability.
        """
        event = FeedbackEvent(
            neuron_ids=[c.id for c in result.concepts],
            action="none",
            reason="initial query",
        )

        # Layer 2: Compare with previous query
        if self._last_query_vector is not None:
            event = self._layer2_user_behavior(
                query_vector, result.concepts
            )

        # Layer 1: Self-consistency check (10% sampling)
        if (result.converged and
                self._rng.random() < self.consistency_rate):
            consistency_event = self._layer1_self_consistency(
                query_vector, result
            )
            if consistency_event.action != "none":
                event = consistency_event

        # Remember this query for next comparison
        self._last_query_vector = query_vector.copy()
        self._last_concepts = [c.id for c in result.concepts]

        self._history.append(event)
        return event

    def _layer1_self_consistency(self, query_vector: np.ndarray,
                                 original_result: ConvergenceResult) -> FeedbackEvent:
        """
        Layer 1: Ask the same thing with a perturbed vector.
        Same concepts = consistent = boost.
        Different concepts = inconsistent = penalize.
        """
        # Perturb the query slightly
        noise = self._rng.gauss(0, 0.05)
        perturbed = query_vector + noise
        norm = np.linalg.norm(perturbed)
        if norm > 0:
            perturbed = perturbed / norm

        # Re-run convergence
        check_result = self.convergence.converge(perturbed)

        original_ids = set(c.id for c in original_result.concepts)
        check_ids = set(c.id for c in check_result.concepts)

        overlap = len(original_ids & check_ids)
        total = len(original_ids | check_ids) if (original_ids | check_ids) else 1
        consistency = overlap / total  # Jaccard similarity

        if consistency > 0.5:
            # Consistent — boost participating neurons
            for nid in original_ids:
                self.db.update_confidence(nid, useful=True)
            return FeedbackEvent(
                neuron_ids=list(original_ids),
                action="boost",
                reason="self-consistency check passed",
                details={"consistency": consistency, "overlap": overlap},
            )
        else:
            # Inconsistent — penalize
            for nid in original_ids:
                self.db.update_confidence(nid, useful=False)
            return FeedbackEvent(
                neuron_ids=list(original_ids),
                action="penalize",
                reason="self-consistency check failed",
                details={"consistency": consistency, "overlap": overlap},
            )

    def _layer2_user_behavior(self, current_query: np.ndarray,
                               current_concepts: list) -> FeedbackEvent:
        """
        Layer 2: Infer feedback from user behavior.

        Same topic (high similarity) = follow-up = previous answer insufficient.
        New topic (low similarity) = user moved on = previous answer accepted.
        """
        sim = self._cosine_sim(current_query, self._last_query_vector)
        prev_ids = self._last_concepts

        if not prev_ids:
            return FeedbackEvent(
                neuron_ids=[], action="none", reason="no previous concepts",
            )

        if sim > (1.0 - TOPIC_SHIFT_THRESHOLD):
            # Same topic — follow-up — previous answer was insufficient
            for nid in prev_ids:
                self.db.update_confidence(nid, useful=False)
            return FeedbackEvent(
                neuron_ids=prev_ids,
                action="penalize",
                reason=f"follow-up query (sim={sim:.3f}) — previous answer insufficient",
                details={"similarity": sim},
            )
        elif sim < TOPIC_SHIFT_THRESHOLD:
            # New topic — user accepted previous answer
            for nid in prev_ids:
                self.db.update_confidence(nid, useful=True)
            return FeedbackEvent(
                neuron_ids=prev_ids,
                action="boost",
                reason=f"topic shift (sim={sim:.3f}) — previous answer accepted",
                details={"similarity": sim},
            )
        else:
            # Ambiguous — do nothing
            return FeedbackEvent(
                neuron_ids=prev_ids,
                action="none",
                reason=f"ambiguous topic similarity (sim={sim:.3f})",
                details={"similarity": sim},
            )

    def force_feedback(self, neuron_ids: list, useful: bool) -> FeedbackEvent:
        """
        Explicit feedback from the user. Overrides automatic layers.
        """
        for nid in neuron_ids:
            self.db.update_confidence(nid, useful=useful)

        action = "boost" if useful else "penalize"
        return FeedbackEvent(
            neuron_ids=neuron_ids,
            action=action,
            reason="explicit user feedback",
        )

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
