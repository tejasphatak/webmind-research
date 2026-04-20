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
from constants import (
    ABSTAIN_MESSAGE, FUNCTION_WORDS, STRUCTURAL_WORDS,
    GRAMMAR_CONFIDENCE_THRESHOLD, MAX_CONVERGENCE_JUMPS,
    QUERY_ANCHOR_FLOOR, PARAGRAPH_RELEVANCE_FLOOR,
)


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

        # Vectorized: batch cosine similarity
        template_matrix = np.array([t.vector for t in self.templates], dtype=np.float32)
        query_norm = np.linalg.norm(concept_vector)
        template_norms = np.linalg.norm(template_matrix, axis=1)
        sims = (template_matrix @ concept_vector) / (query_norm * template_norms + 1e-10)

        # Filter and sort
        mask = sims >= min_similarity
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []
        sorted_idx = indices[np.argsort(-sims[indices])]
        return [self.templates[i] for i in sorted_idx[:k]]

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
                 query_vector: np.ndarray = None,
                 query_words: list = None,
                 evaluate_all: bool = False) -> GenerationResult:
        """
        Generate text from a convergence result.

        Default mode: tries template → successor → concept_list, returns first success.
        Evaluate mode (evaluate_all=True): runs ALL strategies, picks highest confidence.
        The winning strategy and all candidates are in the trace for inspectability.
        """
        if not convergence_result.converged:
            return GenerationResult(
                text=ABSTAIN_MESSAGE,
                strategy="abstain",
                confidence=0.0,
                trace=["Non-convergence → honest abstention (Invariant #4)"],
            )

        concepts = convergence_result.concepts
        if not concepts:
            return GenerationResult(
                text=ABSTAIN_MESSAGE,
                strategy="abstain",
                confidence=0.0,
                trace=["No concepts found"],
            )

        if not evaluate_all:
            # Hierarchical: try simplest sufficient method first

            # Strategy A: Template matching
            result = self._try_template(convergence_result, query_vector=query_vector,
                                        query_words=query_words)
            if result is not None:
                return result

            # Strategy B: Sentence-constrained chain walk
            # Use co-occurrence graph to find the taught sentence that best
            # matches the query, then output its content words in taught order.
            result = self._try_sentence_chain(concepts, query_vector=query_vector,
                                               query_words=query_words)
            if result is not None:
                return result

            # Strategy C: Free successor walk (convergence-guided)
            result = self._try_successor_walk(concepts, max_tokens, query_vector=query_vector)
            if result is not None:
                return result

            # Strategy D: Concept list (always works)
            return self._concept_list(concepts, convergence_result.confidence)

        # Evaluate all strategies, pick best
        candidates = []

        template_result = self._try_template(
            convergence_result, query_vector=query_vector, query_words=query_words
        )
        if template_result is not None:
            candidates.append(template_result)

        sentence_result = self._try_sentence_chain(
            concepts, query_vector=query_vector, query_words=query_words
        )
        if sentence_result is not None:
            candidates.append(sentence_result)

        successor_result = self._try_successor_walk(
            concepts, max_tokens, query_vector=query_vector
        )
        if successor_result is not None:
            candidates.append(successor_result)

        concept_result = self._concept_list(concepts, convergence_result.confidence)
        candidates.append(concept_result)

        # Pick highest confidence
        best = max(candidates, key=lambda r: r.confidence)
        best.trace.insert(0, f"Evaluated {len(candidates)} strategies: "
                          + ", ".join(f"{c.strategy}({c.confidence:.3f})" for c in candidates))
        return best

    def _try_template(self, conv_result: ConvergenceResult,
                      query_vector: np.ndarray = None,
                      query_words: list = None) -> Optional[GenerationResult]:
        """
        Strategy A: Find closest template, fill slots with concept words.
        """
        if self.template_store.count() == 0:
            return None

        # Score templates by structural word overlap with the QUERY TEXT
        # (not concepts — concepts include noise like "the", "is").
        # A template's structural words should appear in what the user asked.
        query_word_set = set(query_words) if query_words else set()
        search_vec = query_vector if query_vector is not None else conv_result.vector

        scored_templates = []
        for t in self.template_store.templates:
            struct_words = t.structural_words
            struct_total = len(struct_words) or 1

            if query_word_set:
                # Match template structural words against query words
                struct_overlap = sum(
                    1 for w in struct_words if w in query_word_set
                )
                overlap_score = struct_overlap / struct_total
            else:
                # No query words available — use concept-based matching
                concept_word_set = set(self._neurons_to_words(conv_result.concepts))
                struct_overlap = sum(
                    1 for w in struct_words if w in concept_word_set
                )
                overlap_score = struct_overlap / struct_total

            # Vector similarity
            vec_sim = float(np.dot(search_vec, t.vector) /
                           (np.linalg.norm(search_vec) * np.linalg.norm(t.vector) + 1e-10))

            # Combined: overlap is dominant, vector similarity breaks ties
            combined = overlap_score * 0.75 + max(vec_sim, 0) * 0.25
            if overlap_score > 0:
                scored_templates.append((t, combined, overlap_score))

        scored_templates.sort(key=lambda x: x[1], reverse=True)
        templates = [t for t, _, _ in scored_templates[:3]]
        if not templates:
            # Fallback to pure vector search if no structural overlap
            templates = self.template_store.search(search_vec, k=3)
        if not templates:
            return None

        concepts = conv_result.concepts
        concept_words = self._neurons_to_words(concepts)

        # Sort concepts by TAUGHT sentence order when available.
        # The sentence_neurons table records the position each word was
        # taught in. Using this order preserves the original sentence
        # structure, so template slots get filled correctly:
        #   taught "paris is the capital of france" → positions [paris=0, capital=1, france=2]
        #   template "[S0] is the [S1] of [S2]" → S0=paris, S1=capital, S2=france
        # Without this, sorting by query similarity puts "capital" first
        # (highest sim to "capital of france") → garbled output.
        concept_ids = [c.id for c in concepts]
        sentence_order = self._get_sentence_order(concept_ids)
        if sentence_order:
            # Sort by taught position
            concept_pairs = list(zip(concepts, concept_words))
            concept_pairs.sort(
                key=lambda p: sentence_order.get(p[0].id, 999),
            )
            concepts = [p[0] for p in concept_pairs]
            concept_words = [p[1] for p in concept_pairs]
        elif query_vector is not None:
            # Fallback: sort by relevance to query
            concept_pairs = list(zip(concepts, concept_words))
            concept_pairs.sort(
                key=lambda p: float(np.dot(p[0].vector, query_vector)),
                reverse=True,
            )
            concepts = [p[0] for p in concept_pairs]
            concept_words = [p[1] for p in concept_pairs]

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

    def _try_sentence_chain(self, concepts: list, query_vector: np.ndarray = None,
                            query_words: list = None) -> Optional[GenerationResult]:
        """
        Strategy B: Sentence-constrained chain retrieval.

        Instead of walking the successor graph freely (which picks wrong chains
        at ambiguous nodes like "the"), use the sentence_neurons table to find
        which taught sentence best matches the query, then output that sentence's
        content words in their taught order.

        This is the key insight from the convergence analysis: the data IS in the
        graph (72% reconstructable), the problem is finding the right chain.
        The sentence table tells us which neurons were taught together — that's
        the constraint that eliminates ambiguity.
        """
        if not concepts:
            return None

        # Find concept neuron IDs
        concept_ids = [c.id for c in concepts]

        # Find sentences containing these concepts, scored by coverage
        sentences = self.db.get_sentences_for_neurons(concept_ids)
        if not sentences:
            return None

        # Score: how many of the query-relevant concepts does each sentence contain?
        scored = []
        for sid, matched_neurons in sentences.items():
            coverage = len(matched_neurons)
            # Also check query vector relevance if available
            sent_neurons = self.db.get_sentence_neurons(sid)
            if not sent_neurons:
                continue

            relevance = 0.0
            if query_vector is not None:
                # Centroid of sentence neurons vs query
                vecs = []
                for nid, pos in sent_neurons:
                    n = self.db.get(nid)
                    if n is not None:
                        vecs.append(n.vector)
                if vecs:
                    centroid = np.mean(vecs, axis=0).astype(np.float32)
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm
                    relevance = float(np.dot(centroid, query_vector))

            score = coverage * 0.4 + max(relevance, 0) * 0.6
            scored.append((sid, score, sent_neurons))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)

        # Get word mappings for neuron → word conversion
        word_map = self.db.load_word_mappings()
        neuron_to_word = {nid: w for w, nid in word_map.items()}

        # Try the top-scoring sentences
        for sid, score, sent_neurons in scored[:3]:
            # Output neurons in taught position order
            ordered = sorted(sent_neurons, key=lambda x: x[1])
            words = []
            for nid, pos in ordered:
                w = neuron_to_word.get(nid)
                if not w:
                    w = self._neuron_to_word(self.db.get(nid)) if self.db.get(nid) else None
                if w and not w.startswith("__"):
                    words.append(w)

            if len(words) >= 2:
                text = " ".join(words)
                avg_conf = np.mean([
                    self.db.get(nid).confidence
                    for nid, _ in ordered
                    if self.db.get(nid) is not None
                ]) if ordered else 0.5

                return GenerationResult(
                    text=text,
                    strategy="sentence_chain",
                    confidence=float(avg_conf) * 0.8,
                    trace=[
                        f"Sentence chain: sid={sid}, score={score:.3f}",
                        f"Coverage: {len(sentences[sid])} query concepts matched",
                        f"Words: {words}",
                    ],
                )

        return None

    def _try_successor_walk(self, concepts: list[Neuron],
                            max_tokens: int,
                            query_vector: np.ndarray = None) -> Optional[GenerationResult]:
        """
        Strategy B: Convergence-guided sentence generation.

        Each token position is a decision point:
          1. FAST PATH: current neuron has a high-confidence successor → emit it
             (grammar tokens: "is", "the", "of" — predictable from word order)
          2. SLOW PATH: run a mini convergence loop where the search vector
             blends query (what was asked) + context (tokens emitted so far).
             The convergence result is intersected with successor candidates
             to pick the token that is both grammatically valid AND relevant.

        This is the decoder from the design spec:
          - Successor graph = what CAN follow (grammar constraint)
          - Convergence = what SHOULD follow (semantic relevance)
          - Query anchor = stay on topic across the whole sentence
          - Context accumulation = each token enriches the next search

        Stop when: no successors, convergence fails, or max tokens.
        """
        if not concepts:
            return None

        # Pick starting concept: most relevant to the query, not just highest confidence
        if query_vector is not None:
            start_concept = max(
                concepts,
                key=lambda n: float(np.dot(n.vector, query_vector))
            )
        else:
            start_concept = max(concepts, key=lambda n: n.confidence)

        start = self.db.get(start_concept.id)
        if start is None:
            return None

        tokens = []
        trace = []
        emitted_neurons = []  # vectors of emitted tokens, for context
        current_id = start.id
        visited = {current_id}

        # Add start word
        start_word = self._neuron_to_word(start)
        if start_word:
            tokens.append(start_word)
            emitted_neurons.append(start)
            trace.append(f"Start: {start_word} (n{start.id}, conf={start.confidence:.2f})")

        consecutive_low_relevance = 0  # track drift from query
        had_jump = False  # whether we've crossed a sentence boundary
        jump_count = 0    # number of convergence jumps taken
        max_jumps = MAX_CONVERGENCE_JUMPS
        current_sentence_ids = None  # sentences the current neuron belongs to

        for step in range(max_tokens - 1):
            current = self.db.get(current_id)
            if current is None:
                break

            # Get unvisited successors
            candidates = [(sid, sc) for sid, sc in current.successors
                          if sid not in visited]
            if not candidates:
                # No successors — try convergence to find a continuation
                if (query_vector is not None
                        and len(tokens) < max_tokens - 1
                        and jump_count < max_jumps):
                    next_neuron = self._converge_next_token(
                        query_vector, emitted_neurons, visited
                    )
                    if next_neuron is not None:
                        word = self._neuron_to_word(next_neuron)
                        if word:
                            tokens.append(word)
                            emitted_neurons.append(next_neuron)
                            trace.append(
                                f"Step {step + 1}: {word} (n{next_neuron.id}, "
                                f"converge-jump {jump_count + 1}/{max_jumps}, "
                                f"conf={next_neuron.confidence:.2f})"
                            )
                            visited.add(next_neuron.id)
                            current_id = next_neuron.id
                            had_jump = True
                            jump_count += 1
                            consecutive_low_relevance = 0
                            # Track which sentence(s) the jump target belongs to
                            rows = self.db.get_cooccurring_neurons(next_neuron.id)
                            current_sentence_ids = {r[2] for r in rows} if rows else None
                            continue
                break

            # FAST PATH: best successor confidence above threshold → grammar token
            best_id, best_conf = max(candidates, key=lambda s: s[1])
            if best_conf >= GRAMMAR_CONFIDENCE_THRESHOLD:
                succ = self.db.get(best_id)
                if succ is None:
                    break
                word = self._neuron_to_word(succ)
                if not word:
                    break

                # Sentence-aware filtering: after a jump, check if this
                # successor belongs to the same sentence as the jump target.
                # If it's from a different sentence, we've crossed a boundary
                # into unrelated content — stop.
                if had_jump and current_sentence_ids is not None:
                    succ_rows = self.db.get_cooccurring_neurons(best_id)
                    succ_sentences = {r[2] for r in succ_rows} if succ_rows else set()
                    if succ_sentences and not (succ_sentences & current_sentence_ids):
                        # Different sentence — stop here
                        trace.append(
                            f"Stop: sentence boundary at step {step + 1} "
                            f"({word} belongs to different sentence)"
                        )
                        break

                # Relevance drift check (backup for neurons without sentence data)
                if had_jump and current_sentence_ids is None and query_vector is not None:
                    token_rel = float(np.dot(succ.vector, query_vector) /
                                     (np.linalg.norm(succ.vector) *
                                      np.linalg.norm(query_vector) + 1e-10))
                    if token_rel < 0.25:
                        consecutive_low_relevance += 1
                    else:
                        consecutive_low_relevance = 0
                    if consecutive_low_relevance >= 2:
                        trim = min(consecutive_low_relevance, len(tokens))
                        tokens = tokens[:-trim]
                        emitted_neurons = emitted_neurons[:-trim]
                        trace.append(f"Stop: post-jump drift after {len(tokens)} tokens")
                        break

                tokens.append(word)
                emitted_neurons.append(succ)
                trace.append(
                    f"Step {step + 1}: {word} (n{best_id}, "
                    f"fast, succ_conf={best_conf:.2f})"
                )
                visited.add(best_id)
                current_id = best_id
                continue

            # SLOW PATH: convergence-guided selection among successors
            if query_vector is not None:
                chosen = self._convergence_pick(
                    query_vector, emitted_neurons, candidates
                )
            else:
                chosen = None

            if chosen is None:
                # Fallback: just take the best successor
                chosen_id, chosen_conf = best_id, best_conf
            else:
                chosen_id, chosen_conf = chosen

            succ = self.db.get(chosen_id)
            if succ is None:
                break
            word = self._neuron_to_word(succ)
            if not word:
                break

            speed = "converge" if chosen is not None else "fallback"
            tokens.append(word)

            # Relevance stopping: if content tokens drift away from query, stop.
            # Grammar tokens (fast path) don't count — "the", "of" are always low-relevance.
            if query_vector is not None:
                token_relevance = float(np.dot(succ.vector, query_vector) /
                                        (np.linalg.norm(succ.vector) *
                                         np.linalg.norm(query_vector) + 1e-10))
                if token_relevance < 0.2:
                    consecutive_low_relevance += 1
                else:
                    consecutive_low_relevance = 0
                # Two consecutive low-relevance content tokens = we've drifted off topic
                if consecutive_low_relevance >= 2:
                    # Remove the drifted tokens
                    tokens = tokens[:-consecutive_low_relevance]
                    emitted_neurons = emitted_neurons[:-consecutive_low_relevance]
                    trace.append(f"Stop: relevance drift after {len(tokens)} tokens")
                    break
            emitted_neurons.append(succ)
            trace.append(
                f"Step {step + 1}: {word} (n{chosen_id}, "
                f"{speed}, conf={chosen_conf:.2f})"
            )
            visited.add(chosen_id)
            current_id = chosen_id

        if len(tokens) < 2:
            return None

        text = " ".join(tokens)
        avg_conf = np.mean([n.confidence for n in emitted_neurons])

        return GenerationResult(
            text=text,
            strategy="successor",
            confidence=float(avg_conf) * 0.6,
            trace=trace,
        )

    def _build_context_vector(self, query_vector: np.ndarray,
                              emitted: list) -> np.ndarray:
        """
        Build a search vector that blends query intent with generation context.

        Early in generation: mostly query (stay on topic).
        Later: more context (maintain coherence with what's been said).
        But query never drops below 40% — it's the anchor.
        """
        if not emitted:
            return query_vector

        # Context = average of emitted neuron vectors
        context = np.mean([n.vector for n in emitted], axis=0).astype(np.float32)
        norm = np.linalg.norm(context)
        if norm > 0:
            context = context / norm

        # Query weight decreases but floors at 0.4
        query_weight = max(QUERY_ANCHOR_FLOOR, 1.0 - len(emitted) * 0.1)
        blended = query_weight * query_vector + (1 - query_weight) * context
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm
        return blended

    def _convergence_pick(self, query_vector: np.ndarray,
                          emitted: list,
                          candidates: list) -> Optional[tuple]:
        """
        Pick the best successor using convergence-guided scoring.

        Scores each candidate by how well it fits the query+context blend.
        Returns (neuron_id, score) or None if no candidate is relevant.
        """
        search_vec = self._build_context_vector(query_vector, emitted)

        # Fetch all candidate neurons
        valid = []
        for cand_id, succ_conf in candidates:
            cand = self.db.get(cand_id)
            if cand is not None:
                valid.append((cand_id, succ_conf, cand))

        if not valid:
            return None

        # Vectorized scoring
        vectors = np.array([c.vector for _, _, c in valid], dtype=np.float32)
        succ_confs = np.array([sc for _, sc, _ in valid], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        search_norm = np.linalg.norm(search_vec)
        sims = (vectors @ search_vec) / (norms * search_norm + 1e-10)
        scores = sims * 0.6 + succ_confs * 0.4

        best_idx = int(np.argmax(scores))
        if scores[best_idx] > 0.0:
            return (valid[best_idx][0], float(scores[best_idx]))
        return None

    def _converge_next_token(self, query_vector: np.ndarray,
                             emitted: list,
                             visited: set) -> Optional[Neuron]:
        """
        When successor chain runs out, use convergence to jump to
        a new concept that's relevant to the query+context.

        This enables cross-sentence reasoning: the system can chain
        concepts that weren't explicitly connected by successor edges.

        Tighter than general search — requires meaningful query relevance
        to prevent drifting into unrelated KB regions. Also filters out
        generic/function words (the, is, of) which have high similarity
        to everything in GloVe but carry no content.
        """
        search_vec = self._build_context_vector(query_vector, emitted)

        # Search the DB for neurons near the blended vector
        neighbors = self.db.search(search_vec, k=10)
        best_candidate = None
        best_score = 0.0

        # Pre-load word mappings to avoid expensive nearest_words calls
        word_map = self.db.load_word_mappings()
        neuron_to_word = {nid: w for w, nid in word_map.items()}

        # Filter candidates: not visited, not function words
        valid = []
        for n in neighbors:
            if n.id in visited:
                continue
            word = neuron_to_word.get(n.id)
            if word and self._is_function_word(word):
                continue
            if n.confidence > 0.1:
                valid.append(n)

        if not valid:
            return None

        # Vectorized: batch cosine similarities against query and context
        vectors = np.array([n.vector for n in valid], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        query_sims = (vectors @ query_vector) / (norms * np.linalg.norm(query_vector) + 1e-10)
        ctx_sims = (vectors @ search_vec) / (norms * np.linalg.norm(search_vec) + 1e-10)

        # Apply thresholds and score
        mask = (query_sims > 0.3) & (ctx_sims > 0.25)
        if not np.any(mask):
            return None

        scores = query_sims * 0.6 + ctx_sims * 0.4
        scores = np.where(mask, scores, -np.inf)
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > 0:
            return valid[best_idx]
        return None

    @staticmethod
    def _is_function_word(word: str) -> bool:
        """Check if a word is a function/grammar word that shouldn't start a jump."""
        return word.lower() in FUNCTION_WORDS

    # --- Paragraph Generation ---

    def generate_paragraph(self, query_vector: np.ndarray,
                           convergence_loop: ConvergenceLoop,
                           max_sentences: int = 5,
                           max_tokens_per_sentence: int = 15,
                           query_words: list = None,
                           sentence_separator: str = ". ") -> GenerationResult:
        """
        Generate a multi-sentence paragraph via convergence-driven retrieval.

        The key insight: convergence finds WHAT to say. The sentence_neurons
        table (taught word order) determines HOW to say it. No separate
        decoder needed — convergence IS the decoder at the sentence level.

        Phase 1 — CONVERGE: find concepts relevant to the query.
        Phase 2 — RETRIEVE SENTENCES: find taught sentences containing
                  those concepts, ranked by query coverage.
        Phase 3 — RENDER: output each sentence's neurons in taught order,
                  mapped back to words. The original word order IS grammar.

        This avoids the drift problem entirely — we don't generate token
        by token, we retrieve whole sentences that were taught correctly
        and output them in the order they were taught.
        """
        # Phase 1: Converge to find relevant concepts
        result = convergence_loop.converge(query_vector)
        if not result.converged or not result.concepts:
            return GenerationResult(
                text=ABSTAIN_MESSAGE,
                strategy="abstain",
                confidence=0.0,
                trace=["Paragraph planning: convergence failed"],
            )

        # Enrich with per-word matches
        all_concepts = list(result.concepts)
        seen_ids = {c.id for c in all_concepts}
        if query_words:
            for token in query_words:
                wv = self.encoder.encode_word(token)
                if not np.all(wv == 0):
                    for n in self.db.search(wv, k=3):
                        if n.id not in seen_ids:
                            sim = float(np.dot(n.vector, wv))
                            if sim > 0.3:
                                all_concepts.append(n)
                                seen_ids.add(n.id)

        # Phase 2: Find sentences containing these concepts
        concept_ids = [c.id for c in all_concepts]
        sentences_map = self.db.get_sentences_for_neurons(concept_ids)

        if not sentences_map:
            # No sentence structure — fall back to single-sentence generation
            conv_result = ConvergenceResult(
                converged=True, vector=result.vector,
                concepts=all_concepts, confidence=result.confidence,
            )
            return self.generate(conv_result, max_tokens=max_tokens_per_sentence,
                                 query_vector=query_vector, query_words=query_words)

        # Score sentences by:
        #   1. Query coverage (how many query-relevant concepts are in this sentence)
        #   2. Relevance of the sentence's centroid to the query
        scored_sentences = []
        for sid, matched_neurons in sentences_map.items():
            # Get ALL neurons in this sentence (not just the ones that matched)
            full_sentence = self.db.get_sentence_neurons(sid)
            if not full_sentence:
                continue

            # Coverage score: what fraction of the query concepts does this sentence contain?
            coverage = len(matched_neurons) / max(len(concept_ids), 1)

            # Relevance: centroid of sentence neurons vs query
            sent_neurons = []
            for nid, pos in full_sentence:
                n = self.db.get(nid)
                if n is not None:
                    sent_neurons.append((n, pos))

            if not sent_neurons:
                continue

            centroid = np.mean([n.vector for n, _ in sent_neurons], axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            relevance = float(np.dot(centroid, query_vector))

            score = coverage * 0.4 + max(relevance, 0) * 0.6
            scored_sentences.append((sid, score, sent_neurons))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Phase 3: Render top sentences in taught word order
        rendered = []
        trace = [f"Plan: {len(scored_sentences)} candidate sentences from {len(all_concepts)} concepts"]
        used_sids = set()

        # Relevance floor: only include sentences scoring at least 50%
        # of the best sentence's score. Prevents noise sentences.
        best_score = scored_sentences[0][1] if scored_sentences else 0
        relevance_floor = best_score * PARAGRAPH_RELEVANCE_FLOOR

        for sid, score, sent_neurons in scored_sentences[:max_sentences]:
            if sid in used_sids:
                continue
            if score < relevance_floor:
                break  # sorted by score, so all remaining are worse
            used_sids.add(sid)

            # Sort by position (the taught order)
            sent_neurons.sort(key=lambda x: x[1])

            # Map neurons to words
            words = []
            for n, pos in sent_neurons:
                word = self._neuron_to_word(n)
                if word:
                    words.append(word)

            if words:
                sentence_text = " ".join(words)
                # Skip duplicates
                if sentence_text not in rendered:
                    rendered.append(sentence_text)
                    trace.append(f"S{len(rendered)} (score={score:.3f}): {sentence_text}")

        if not rendered:
            return GenerationResult(
                text=ABSTAIN_MESSAGE,
                strategy="abstain",
                confidence=0.0,
                trace=trace + ["No sentences rendered"],
            )

        # Join sentences using the provided separator.
        # Default is ". " — but this is configurable, not hardcoded behavior.
        # Future: teach punctuation as neurons so boundaries emerge from data.
        paragraph = sentence_separator.join(rendered)
        avg_conf = float(np.mean([c.confidence for c in all_concepts]))

        return GenerationResult(
            text=paragraph,
            strategy="paragraph",
            confidence=avg_conf * 0.7,
            trace=trace,
        )

    def _cluster_by_sentence(self, concepts: list) -> list:
        """
        Group concepts by sentence co-occurrence.

        Neurons taught together in the same sentence belong to the same
        cluster. This gives natural topic boundaries — each taught sentence
        is one "idea unit."

        Returns list of clusters, each cluster = list of neurons.
        """
        # Map neuron_id → set of sentence_ids
        neuron_sentences = {}
        for c in concepts:
            rows = self.db.get_cooccurring_neurons(c.id)
            sentence_ids = {r[2] for r in rows}  # r = (neuron_id, position, sentence_id)
            if sentence_ids:
                neuron_sentences[c.id] = sentence_ids

        if not neuron_sentences:
            return []

        # Group: neurons sharing any sentence_id go together
        # Build adjacency from shared sentences
        concept_map = {c.id: c for c in concepts}
        assigned = set()
        clusters = []

        for c in concepts:
            if c.id in assigned:
                continue
            if c.id not in neuron_sentences:
                continue

            # BFS to find all neurons connected by shared sentences
            cluster = []
            queue = [c.id]
            while queue:
                nid = queue.pop(0)
                if nid in assigned:
                    continue
                assigned.add(nid)
                if nid in concept_map:
                    cluster.append(concept_map[nid])
                # Find co-occurring neurons
                for other_id, sids in neuron_sentences.items():
                    if other_id not in assigned:
                        if sids & neuron_sentences.get(nid, set()):
                            queue.append(other_id)

            if cluster:
                clusters.append(cluster)

        # Add unclustered concepts as individual clusters
        for c in concepts:
            if c.id not in assigned:
                clusters.append([c])

        return clusters

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

    def _get_sentence_order(self, concept_ids: list) -> dict:
        """
        Find the taught sentence that best covers these concepts and
        return a position map: {neuron_id: taught_position}.

        When concepts come from a taught sentence, this preserves the
        original word order for template slot filling. If no sentence
        covers enough concepts, returns empty dict (fall back to other
        ordering methods).
        """
        if len(concept_ids) < 2:
            return {}

        sentences = self.db.get_sentences_for_neurons(concept_ids)
        if not sentences:
            return {}

        # Score by coverage: how many of our concepts are in this sentence?
        best_sid = None
        best_coverage = 0
        for sid, matched_nids in sentences.items():
            coverage = len(matched_nids)
            if coverage > best_coverage:
                best_coverage = coverage
                best_sid = sid

        # Need at least 2 concepts matched to trust the ordering
        if best_coverage < 2:
            return {}

        # Get positions from the best sentence
        sent_neurons = self.db.get_sentence_neurons(best_sid)
        return {nid: pos for nid, pos in sent_neurons}

    def _neurons_to_words(self, neurons: list[Neuron]) -> list[str]:
        """Map neurons back to nearest words via encoder."""
        words = []
        for n in neurons:
            word = self._neuron_to_word(n)
            words.append(word if word else f"<n{n.id}>")
        return words

    def _neuron_to_word(self, neuron: Neuron) -> Optional[str]:
        """Find the closest word to a neuron's vector.

        First checks the word→neuron mapping (works for any dimension).
        Falls back to encoder nearest_words (requires matching dimensions).
        """
        # Fast path: direct lookup from DB mapping
        word_map = self.db.load_word_mappings()
        neuron_to_word = {nid: w for w, nid in word_map.items()}
        label = neuron_to_word.get(neuron.id)
        if label and not label.startswith("__"):
            return label

        # Fallback: encoder nearest word (only works if dimensions match)
        try:
            nearest = self.encoder.nearest_words(neuron.vector, k=1)
            if nearest:
                return nearest[0][0]
        except (ValueError, RuntimeError):
            pass  # dimension mismatch (e.g., CLIP 512 vs GloVe 300)
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
                                 slot_order: list,
                                 query_vector: np.ndarray = None) -> dict:
        """
        Fallback: assign content words to slots by relevance to query.

        Filters out stop words and template structural words, then
        assigns remaining content words to slots. If a query vector
        is provided, sorts candidates by relevance to query.
        """
        STOP_WORDS = FUNCTION_WORDS
        pattern_words = set(template.structural_words)

        content = [w for w in concept_words
                   if w.lower() not in STOP_WORDS and w.lower() not in pattern_words]
        if not content:
            content = [w for w in concept_words if w.lower() not in STOP_WORDS]
        if not content:
            content = list(concept_words)

        # Deduplicate while preserving order
        seen = set()
        unique_content = []
        for w in content:
            if w not in seen:
                seen.add(w)
                unique_content.append(w)
        content = unique_content

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
