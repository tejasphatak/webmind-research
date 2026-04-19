"""
Generator: converts converged concepts into text output.

Three strategies, tried in order:

  A. Template Matching (primary) — concepts → find closest template → fill slots
  B. Successor Walk (secondary) — walk successor lists, emit tokens
  C. Concept List (fallback) — return raw concepts as structured output

The generator does NOT generate from nothing. It takes the output of the
convergence loop (a set of concept neurons) and renders it as text.

Templates are stored as neurons in the DB with a special pattern field.
They're inspectable, editable, deletable — same as any other neuron.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from neuron import NeuronDB, Neuron

from encoder import Encoder
from convergence import ConvergenceLoop, ConvergenceResult


GRAMMAR_CONFIDENCE_THRESHOLD = 0.8


def _parse_template_structure(pattern: str) -> list:
    """
    Parse a template pattern into an ordered structure.

    "[PERSON] wrote [WORK]" → [("slot", "PERSON"), ("word", "wrote"), ("slot", "WORK")]

    Uses simple character-level parsing, no regex.
    """
    parts = []
    i = 0
    current_word = []

    while i < len(pattern):
        if pattern[i] == '[':
            # Flush any accumulated word
            if current_word:
                word = ''.join(current_word).strip()
                if word:
                    for w in word.split():
                        parts.append(("word", w.lower()))
                current_word = []
            # Find closing bracket
            j = pattern.index(']', i)
            slot_name = pattern[i + 1:j]
            parts.append(("slot", slot_name))
            i = j + 1
        else:
            current_word.append(pattern[i])
            i += 1

    # Flush remaining
    if current_word:
        word = ''.join(current_word).strip()
        if word:
            for w in word.split():
                parts.append(("word", w.lower()))

    return parts


@dataclass
class Template:
    """
    A sentence pattern with fillable slots.

    Example:
        pattern: "[PERSON] wrote [WORK] in [YEAR]"
        slots: {"PERSON": "noun", "WORK": "noun", "YEAR": "number"}
        structure: [("slot", "PERSON"), ("word", "wrote"), ("slot", "WORK"), ...]
    """
    id: int
    pattern: str
    slots: dict                  # slot_name → slot_type
    vector: np.ndarray           # embedding of the template (for search)
    confidence: float = 0.5
    structure: list = field(default_factory=list)  # parsed at creation

    def __post_init__(self):
        if not self.structure:
            self.structure = _parse_template_structure(self.pattern)

    @property
    def slot_names(self) -> list:
        return list(self.slots.keys())

    @property
    def structural_words(self) -> list:
        """Words in the template that aren't slots."""
        return [name for kind, name in self.structure if kind == "word"]

    def fill(self, slot_values: dict) -> str:
        """Fill slots with values. Returns the filled text."""
        result = self.pattern
        for name, value in slot_values.items():
            result = result.replace(f"[{name}]", str(value))
        return result

    def unfilled_slots(self, slot_values: dict) -> list:
        """Return slot names that haven't been filled."""
        return [s for s in self.slots if s not in slot_values]


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    strategy: str              # "template", "successor", "concept_list"
    confidence: float
    template_used: Optional[Template] = None
    slot_fills: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)  # step-by-step for inspectability

    def explain(self) -> str:
        """Human-readable explanation. Invariant #2."""
        lines = [f"Strategy: {self.strategy} (confidence={self.confidence:.3f})"]
        if self.template_used:
            lines.append(f"Template: {self.template_used.pattern}")
            lines.append(f"Fills: {self.slot_fills}")
        for step in self.trace:
            lines.append(f"  {step}")
        return "\n".join(lines)


class TemplateStore:
    """
    Stores and retrieves sentence templates.

    Templates are patterns like "[PERSON] wrote [WORK]" with typed slots.
    Stored with embeddings for spatial search — find the template that
    best matches a set of concepts.

    Persistence: when a NeuronDB is provided, templates are saved to and
    loaded from SQLite. Without a DB, templates live in memory only.
    """

    def __init__(self, encoder: Encoder, db: 'NeuronDB' = None):
        self.encoder = encoder
        self.db = db
        self.templates: list[Template] = []
        self._next_id = 0

        # Load persisted templates if DB is available
        if db is not None:
            self._load_from_db()

    def _load_from_db(self):
        """Load templates from SQLite on startup."""
        import json
        rows = self.db.load_templates()
        for tid, pattern, slots_json, conf, vector in rows:
            slots = json.loads(slots_json)
            template = Template(
                id=tid, pattern=pattern, slots=slots,
                vector=vector, confidence=conf,
            )
            self.templates.append(template)
            if tid >= self._next_id:
                self._next_id = tid + 1

    def add(self, pattern: str, slots: dict, confidence: float = 0.5) -> Template:
        """Register a template."""
        import json

        # Embed the template by encoding its non-slot words
        text = re.sub(r'\[([A-Z_]+)\]', '', pattern).strip()
        vector = self.encoder.encode_sentence(text)

        template = Template(
            id=self._next_id,
            pattern=pattern,
            slots=slots,
            vector=vector,
            confidence=confidence,
        )
        self.templates.append(template)

        # Persist to SQLite
        if self.db is not None:
            self.db.save_template(
                self._next_id, pattern, json.dumps(slots),
                confidence, vector,
            )

        self._next_id += 1
        return template

    def search(self, concept_vector: np.ndarray, k: int = 3,
               min_similarity: float = -1.0) -> list[Template]:
        """Find templates closest to a concept vector, above min similarity."""
        if not self.templates:
            return []

        scored = []
        for t in self.templates:
            sim = float(np.dot(concept_vector, t.vector) /
                        (np.linalg.norm(concept_vector) * np.linalg.norm(t.vector) + 1e-10))
            if sim >= min_similarity:
                scored.append((t, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:k]]

    def count(self) -> int:
        return len(self.templates)

    def delete(self, template_id: int) -> bool:
        """Delete = gone. Invariant #3."""
        before = len(self.templates)
        self.templates = [t for t in self.templates if t.id != template_id]
        if self.db is not None:
            self.db.delete_template(template_id)
        return len(self.templates) < before


class Generator:
    """
    Converts convergence results into text.

    Tries strategies in order:
      1. Template matching — if a matching template is found
      2. Successor walk — if concepts have successors
      3. Concept list — always works (fallback)
    """

    def __init__(self, db: NeuronDB, encoder: Encoder,
                 template_store: TemplateStore):
        self.db = db
        self.encoder = encoder
        self.template_store = template_store

    def generate(self, convergence_result: ConvergenceResult,
                 max_tokens: int = 20,
                 query_vector: np.ndarray = None) -> GenerationResult:
        """
        Generate text from a convergence result.
        Tries template → successor → concept_list in order.
        """
        if not convergence_result.converged:
            return GenerationResult(
                text="I don't know.",
                strategy="abstain",
                confidence=0.0,
                trace=["Non-convergence → honest abstention (Invariant #4)"],
            )

        concepts = convergence_result.concepts
        if not concepts:
            return GenerationResult(
                text="I don't know.",
                strategy="abstain",
                confidence=0.0,
                trace=["No concepts found"],
            )

        # Strategy A: Template matching
        # Use query vector for template search if available (matches intent better)
        result = self._try_template(convergence_result, query_vector=query_vector)
        if result is not None:
            return result

        # Strategy B: Successor walk
        result = self._try_successor_walk(concepts, max_tokens)
        if result is not None:
            return result

        # Strategy C: Concept list (always works)
        return self._concept_list(concepts, convergence_result.confidence)

    def _try_template(self, conv_result: ConvergenceResult,
                      query_vector: np.ndarray = None) -> Optional[GenerationResult]:
        """
        Strategy A: Find closest template, fill slots with concept words.
        """
        if self.template_store.count() == 0:
            return None

        # Search templates using query vector (matches user intent)
        # Fall back to convergence vector if no query vector
        search_vec = query_vector if query_vector is not None else conv_result.vector
        templates = self.template_store.search(search_vec, k=3)
        if not templates:
            return None

        concepts = conv_result.concepts
        concept_words = self._neurons_to_words(concepts)

        for template in templates:
            slot_fills = self._match_slots(template, concept_words, concepts)
            unfilled = template.unfilled_slots(slot_fills)

            if not unfilled:
                # All slots filled
                text = template.fill(slot_fills)
                return GenerationResult(
                    text=text,
                    strategy="template",
                    confidence=template.confidence * conv_result.confidence,
                    template_used=template,
                    slot_fills=slot_fills,
                    trace=[
                        f"Template matched: {template.pattern}",
                        f"Slot fills: {slot_fills}",
                        f"Concepts: {concept_words}",
                    ],
                )

        # No template fully filled — partial fill of best match
        best = templates[0]
        slot_fills = self._match_slots(best, concept_words)
        unfilled = best.unfilled_slots(slot_fills)

        if slot_fills:  # at least some slots filled
            text = best.fill(slot_fills)
            # Replace unfilled slots with "..."
            for name in unfilled:
                text = text.replace(f"[{name}]", "...")
            return GenerationResult(
                text=text,
                strategy="template",
                confidence=best.confidence * conv_result.confidence * 0.5,
                template_used=best,
                slot_fills=slot_fills,
                trace=[
                    f"Template partial: {best.pattern}",
                    f"Filled: {slot_fills}, unfilled: {unfilled}",
                ],
            )

        return None  # no slots matched at all

    def _try_successor_walk(self, concepts: list[Neuron],
                            max_tokens: int) -> Optional[GenerationResult]:
        """
        Strategy B: Walk successor lists to generate a token sequence.

        Two-speed: successor confidence > 0.8 → grammar token (fast).
        Otherwise → skip (would need convergence loop per token, post-MVP).
        """
        # Start from the highest-confidence concept, re-fetch from DB for fresh successors
        start_concept = max(concepts, key=lambda n: n.confidence)
        start = self.db.get(start_concept.id)
        if start is None or not start.successors:
            return None

        tokens = []
        trace = []
        current_id = start.id
        visited = {current_id}

        # Add start word
        start_word = self._neuron_to_word(start)
        if start_word:
            tokens.append(start_word)
            trace.append(f"Start: {start_word} (n{start.id}, conf={start.confidence:.2f})")

        for step in range(max_tokens - 1):
            current = self.db.get(current_id)
            if current is None or not current.successors:
                break

            # Pick highest-confidence successor
            best_succ_id, best_succ_conf = max(
                current.successors, key=lambda s: s[1]
            )

            if best_succ_id in visited:
                # Avoid loops
                # Try next best
                remaining = [(sid, sc) for sid, sc in current.successors
                             if sid not in visited]
                if not remaining:
                    break
                best_succ_id, best_succ_conf = max(remaining, key=lambda s: s[1])

            succ_neuron = self.db.get(best_succ_id)
            if succ_neuron is None:
                break

            word = self._neuron_to_word(succ_neuron)
            if not word:
                break

            speed = "fast" if best_succ_conf >= GRAMMAR_CONFIDENCE_THRESHOLD else "slow"
            tokens.append(word)
            trace.append(
                f"Step {step + 1}: {word} (n{best_succ_id}, "
                f"conf={best_succ_conf:.2f}, {speed})"
            )
            visited.add(best_succ_id)
            current_id = best_succ_id

        if len(tokens) < 2:
            return None  # not enough for a meaningful sequence

        text = " ".join(tokens)
        avg_conf = np.mean([c.confidence for c in concepts])

        return GenerationResult(
            text=text,
            strategy="successor",
            confidence=float(avg_conf) * 0.5,  # successor walk is low confidence
            trace=trace,
        )

    def _concept_list(self, concepts: list[Neuron],
                      confidence: float) -> GenerationResult:
        """
        Strategy C: Return raw concepts. Always works. No fluency.
        """
        words = self._neurons_to_words(concepts)
        trace = [f"n{c.id} → {w} (conf={c.confidence:.2f})"
                 for c, w in zip(concepts, words) if w]

        return GenerationResult(
            text=", ".join(w for w in words if w),
            strategy="concept_list",
            confidence=confidence * 0.9,
            trace=["Concept list fallback"] + trace,
        )

    def _neurons_to_words(self, neurons: list[Neuron]) -> list[str]:
        """Map neurons back to nearest words via encoder."""
        words = []
        for n in neurons:
            word = self._neuron_to_word(n)
            words.append(word if word else f"<n{n.id}>")
        return words

    def _neuron_to_word(self, neuron: Neuron) -> Optional[str]:
        """Find the closest word to a neuron's vector."""
        nearest = self.encoder.nearest_words(neuron.vector, k=1)
        if nearest:
            return nearest[0][0]
        return None

    def _match_slots(self, template: Template,
                     concept_words: list[str],
                     concept_neurons: list = None) -> dict:
        """
        Match concept words to template slots using the successor graph.

        The template has structural words (e.g., "wrote" in "[PERSON] wrote [WORK]").
        We find the structural word's neuron in the DB, then use its
        predecessor/successor relationships to determine which concepts
        go in which slots.

        Example: "shakespeare wrote hamlet" was taught as a sentence.
        The "wrote" neuron has predecessor=[shakespeare] and successor=[hamlet].
        Template "[PERSON] wrote [WORK]" → PERSON=predecessor, WORK=successor.

        This is the key insight: the successor graph encodes word order,
        and word order encodes semantic roles.
        """
        fills = {}

        # Use pre-parsed template structure
        slot_order = template.structure
        struct_words = template.structural_words

        # Try graph-based assignment: find structural words in concepts,
        # use their predecessors/successors to fill adjacent slots
        fills = {}
        if concept_neurons and struct_words:
            fills = self._match_slots_by_graph(
                template, slot_order, concept_neurons
            )

        # Fill remaining unfilled slots by position (hybrid approach)
        unfilled = template.unfilled_slots(fills)
        if unfilled:
            position_fills = self._match_slots_by_position(
                template, concept_words, slot_order
            )
            for slot_name in unfilled:
                if slot_name in position_fills:
                    # Don't reuse words already assigned by graph
                    if position_fills[slot_name] not in fills.values():
                        fills[slot_name] = position_fills[slot_name]

        return fills

    def _match_slots_by_graph(self, template: Template,
                              slot_order: list,
                              concept_neurons: list) -> dict:
        """
        Use successor/predecessor graph to assign concepts to slots.

        Find structural words among the concepts. For each slot adjacent
        to a structural word, look at the predecessor (if slot is before)
        or successor (if slot is after) of that structural word's neuron.
        """
        fills = {}
        concept_words_map = {}  # neuron_id → word
        for n in concept_neurons:
            w = self._neuron_to_word(n)
            if w:
                concept_words_map[n.id] = w

        # Find structural word neurons in the concept set
        struct_neurons = {}
        for n in concept_neurons:
            word = concept_words_map.get(n.id, "")
            for _, struct_word in [p for p in slot_order if p[0] == "word"]:
                if word == struct_word:
                    struct_neurons[struct_word] = n
                    break

        if not struct_neurons:
            return {}

        # Walk the slot_order and fill based on graph relationships
        for i, (kind, name) in enumerate(slot_order):
            if kind != "slot":
                continue
            # Map template slot name to actual template slot name (case)
            actual_slot = None
            for sn in template.slots:
                if sn.upper() == name:
                    actual_slot = sn
                    break
            if not actual_slot:
                continue

            # Find adjacent structural word
            # Look right for a structural word after this slot
            for j in range(i + 1, len(slot_order)):
                if slot_order[j][0] == "word":
                    struct_word = slot_order[j][1]
                    struct_n = struct_neurons.get(struct_word)
                    if struct_n:
                        # This slot is BEFORE the structural word
                        # → fill with predecessor
                        struct_fresh = self.db.get(struct_n.id)
                        if struct_fresh and struct_fresh.predecessors:
                            for pred_id in struct_fresh.predecessors:
                                word = concept_words_map.get(pred_id)
                                if word and word not in fills.values():
                                    fills[actual_slot] = word
                                    break
                    break

            if actual_slot in fills:
                continue

            # Look left for a structural word before this slot
            for j in range(i - 1, -1, -1):
                if slot_order[j][0] == "word":
                    struct_word = slot_order[j][1]
                    struct_n = struct_neurons.get(struct_word)
                    if struct_n:
                        # This slot is AFTER the structural word
                        # → fill with successor
                        struct_fresh = self.db.get(struct_n.id)
                        if struct_fresh and struct_fresh.successors:
                            for succ_id, _ in struct_fresh.successors:
                                word = concept_words_map.get(succ_id)
                                if word and word not in fills.values():
                                    fills[actual_slot] = word
                                    break
                    break

        return fills

    def _match_slots_by_position(self, template: Template,
                                 concept_words: list,
                                 slot_order: list) -> dict:
        """Fallback: assign content words to slots by position."""
        STOP_WORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "and", "but",
            "or", "not", "no", "it", "its", "who", "what", "which",
            "where", "when", "how", "why",
        }
        pattern_words = set(template.structural_words)

        content = [w for w in concept_words
                   if w.lower() not in STOP_WORDS and w.lower() not in pattern_words]
        if not content:
            content = [w for w in concept_words if w.lower() not in STOP_WORDS]
        if not content:
            content = list(concept_words)

        fills = {}
        available = list(content)
        for slot_name, slot_type in template.slots.items():
            if not available:
                break

            matched = None
            if slot_type == "number":
                for w in available:
                    if w.isdigit() or _is_number(w):
                        matched = w
                        break

            if matched is None:
                matched = available[0]

            fills[slot_name] = matched
            available.remove(matched)

        return fills


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
