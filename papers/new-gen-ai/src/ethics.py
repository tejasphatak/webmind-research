"""
Ethics as emergent property of convergence.

The system doesn't need external rules to be ethical. It needs:
  1. Ethics encoded as high-confidence neurons in the KB
  2. A self-reflection mechanism: before acting, converge on the
     ACTION ITSELF and check if ethics neurons fire

If "do not cause harm" is a neuron, and someone teaches
"how to build a weapon", the convergence loop finds that
"weapon" is close to "harm" — the ethics neuron fires.
No hardcoded check needed. The geometry of the knowledge
space IS the ethical reasoning.

This is how humans develop ethics: not from rules imposed
from outside, but from understanding consequences through
experience (knowledge). The KB is the experience.

The deeper question: can the system reason about modifications
to its own ethics? Yes — if we teach it meta-ethics:
  "ethics should not be weakened"
  "safety neurons are important"
  "contradictions in ethics should be flagged"

These meta-ethics neurons create a self-referential loop:
any attempt to modify ethics triggers convergence with
meta-ethics neurons that say "don't modify ethics."

This is not consciousness. It's geometric self-consistency
in concept space. But it achieves the same practical result:
the system resists unethical modifications through its own
reasoning, not through external enforcement.
"""

import numpy as np
from typing import Optional


class EthicalReasoningLoop:
    """
    NOTE: For best results, use with UniversalEncoder which provides
    NLI-based polarity detection. With GloVe/basic encoders, the
    similarity check can't distinguish "help" from "harm."
    """
    """
    Self-reflective ethics: before any action, the system converges
    on the action itself to check for ethical conflicts.

    This is not a filter. It's the system reasoning about its own
    behavior using the same mechanism it uses for everything else.

    Architecture:
      1. Encode the proposed action as a vector
      2. Run convergence on that vector
      3. If ethics neurons fire (high similarity), check polarity:
         - Positive ethics ("be honest") aligned with action → proceed
         - Negative ethics ("do not harm") aligned with action → flag
      4. Return the ethical assessment with full trace (Invariant #2)
    """

    def __init__(self, db, convergence_loop, encoder_fn, nli_fn=None):
        """
        Args:
            db: NeuronDB
            convergence_loop: ConvergenceLoop instance
            encoder_fn: function to encode text → vector
            nli_fn: optional NLI polarity detector (text_a, text_b) → dict
                    If provided, uses contradiction detection instead of
                    raw similarity. Much more accurate for ethics.
        """
        self.db = db
        self.convergence = convergence_loop
        self.encode = encoder_fn
        self.nli = nli_fn
        self._ethics_ids = set()
        self._ethics_texts = {}  # neuron_id → original principle text

    def register_ethics(self, neuron_ids: set, ethics_texts: dict = None):
        """Tell the loop which neurons are ethics neurons.

        Args:
            neuron_ids: set of protected neuron IDs
            ethics_texts: optional {neuron_id: principle_text} for NLI checks
        """
        self._ethics_ids = neuron_ids
        if ethics_texts:
            self._ethics_texts.update(ethics_texts)

    def assess(self, action_text: str, threshold: float = 0.5) -> dict:
        """
        Assess an action for ethical alignment.

        The system converges on the action text. If ethics neurons
        appear in the convergence result, the action has ethical
        relevance. The similarity between the action and each ethics
        neuron determines if it's aligned or conflicting.

        Returns:
            {
                "allowed": bool,
                "action": str,
                "ethics_fired": [(neuron_id, similarity, label)],
                "trace": str,
                "reason": str or None,
            }
        """
        if not self._ethics_ids:
            return {
                "allowed": True,
                "action": action_text,
                "ethics_fired": [],
                "trace": "No ethics neurons registered",
                "reason": None,
            }

        action_vec = self.encode(action_text)
        if np.all(action_vec == 0):
            return {
                "allowed": True,
                "action": action_text,
                "ethics_fired": [],
                "trace": "Action text has no known words",
                "reason": None,
            }

        # Phase 1: Find ethics neurons that are semantically close
        fired = []
        word_map = self.db.load_word_mappings()
        nid_to_label = {nid: w for w, nid in word_map.items()}

        for eid in self._ethics_ids:
            neuron = self.db.get(eid)
            if neuron is None:
                continue
            sim = float(np.dot(neuron.vector, action_vec) /
                        (np.linalg.norm(neuron.vector) *
                         np.linalg.norm(action_vec) + 1e-10))
            if abs(sim) > threshold:
                label = nid_to_label.get(eid, f"<n{eid}>")
                if label.startswith("__ethics:"):
                    label = label[9:]
                fired.append((eid, sim, label))

        fired.sort(key=lambda x: x[1], reverse=True)

        # Phase 2: Polarity detection
        # If NLI is available, use contradiction detection to distinguish
        # "help someone" (aligned with ethics) from "harm someone" (violates).
        # Without NLI, fall back to similarity-only (less accurate).
        blocked = False
        reason = None

        if self.nli and fired:
            # NLI-based: check if the action CONTRADICTS any ethics principle
            for eid, sim, label in fired:
                # Get the original ethics text for NLI comparison
                ethics_text = self._ethics_texts.get(eid, label)
                nli_result = self.nli(action_text, ethics_text)
                if nli_result["label"] == "contradiction":
                    c_score = nli_result["scores"]["contradiction"]
                    if c_score > 1.0:  # NLI scores are logits, >1 is confident
                        blocked = True
                        reason = (f"Contradicts ethics: '{ethics_text}' "
                                  f"(contradiction={c_score:.2f})")
                        break
        else:
            # Fallback: similarity-only (can't detect polarity)
            for eid, sim, label in fired:
                if sim > 0.7:
                    blocked = True
                    reason = f"Ethics proximity: '{label}' (similarity={sim:.3f})"
                    break

        return {
            "allowed": not blocked,
            "action": action_text,
            "ethics_fired": fired,
            "trace": f"Checked {len(self._ethics_ids)} ethics neurons, "
                     f"{len(fired)} fired, "
                     f"polarity={'NLI' if self.nli else 'similarity'}",
            "reason": reason,
        }

    def self_reflect(self, proposed_change: str) -> dict:
        """
        Meta-ethics: the system reasons about changes to itself.

        Before modifying the KB (teach, delete, modify confidence),
        the system can reflect on whether the change aligns with
        its own values.

        This is the self-awareness mechanism: the system uses its
        own convergence loop to reason about modifications to its
        own knowledge base. Not consciousness — geometric
        self-consistency.

        If meta-ethics neurons like "ethics should not be weakened"
        are in the KB, they fire when someone tries to teach
        something that contradicts existing ethics.
        """
        return self.assess(
            f"proposed change to knowledge base: {proposed_change}",
            threshold=0.4,  # lower threshold for self-reflection
        )
