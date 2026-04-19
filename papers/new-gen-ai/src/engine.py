"""
Engine: wires all components together into a working reasoning system.

Flow:
  Input text
    → Encoder (text → vector)
    → Convergence Loop (vector → concepts)
    → Generator (concepts → text output)
    → Feedback Loop (update confidence)
    → Output with trace

Also provides:
  - teach(): add facts to the knowledge base
  - delete(): remove facts (invariant #3)
  - inspect(): show what the system knows about a topic
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from neuron import NeuronDB, Neuron, VECTOR_DIM

from encoder import Encoder, GLOVE_DIM
from convergence import ConvergenceLoop, ConvergenceResult, MultiHopConvergence, MultiHopResult
from generator import Generator, TemplateStore, GenerationResult
from feedback import FeedbackLoop


@dataclass
class QueryResult:
    """Complete result of a query, with full trace."""
    query: str
    answer: str
    confidence: float
    converged: bool
    strategy: str
    trace: str
    generation: GenerationResult = None
    convergence: ConvergenceResult = None


class Engine:
    """
    The reasoning engine. Wires encoder + neurons + convergence + generator + feedback.

    Not a neural network. Not trained with gradient descent.
    Every answer traceable. Every fact editable. Every failure honest.
    """

    def __init__(self, data_dir: str = None, dim: int = GLOVE_DIM):
        self.dim = dim
        self.data_dir = data_dir

        # Components
        self.db = NeuronDB(path=data_dir, dim=dim)
        self.encoder = Encoder(
            data_dir=data_dir or "/tmp/newgen-data",
            dim=dim,
        )
        self.template_store = TemplateStore(self.encoder, db=self.db)
        self.convergence = ConvergenceLoop(self.db, max_hops=10, k=5)
        self.multi_hop = MultiHopConvergence(self.convergence, max_rounds=3)
        self.generator = Generator(self.db, self.encoder, self.template_store)
        self.feedback = FeedbackLoop(self.db, self.convergence)

        # Word → neuron_id map: load from DB if persisted
        self._word_neurons = self.db.load_word_mappings()

    def load_embeddings(self, path: str = None):
        """Load pretrained word embeddings."""
        if path:
            self.encoder.load(path)
        else:
            self.encoder.download()
            self.encoder.load()

    def load_embeddings_from_dict(self, word_vectors: dict):
        """Load from dict — useful for testing."""
        self.encoder.load_from_dict(word_vectors)

    def query(self, text: str) -> QueryResult:
        """
        Ask the system a question.

        Returns a QueryResult with the answer, confidence, and full trace.
        """
        # Encode
        query_vector = self.encoder.encode_sentence(text)

        # Check for zero vector (all OOV)
        if np.all(query_vector == 0):
            return QueryResult(
                query=text,
                answer="I don't know — none of those words are in my vocabulary.",
                confidence=0.0,
                converged=False,
                strategy="abstain",
                trace="All words OOV → zero vector → honest abstention",
            )

        # Multi-hop convergence: chains reasoning rounds so concepts
        # discovered in round N shift the query for round N+1
        multi_result = self.multi_hop.reason(query_vector)

        # Convert MultiHopResult → ConvergenceResult for downstream compat
        conv_result = ConvergenceResult(
            converged=multi_result.converged,
            vector=multi_result.vector,
            concepts=multi_result.concepts,
            hops=[h for r in multi_result.rounds for h in r.hops],
            confidence=multi_result.confidence,
        )
        # Keep multi-hop trace for full inspectability
        conv_result._multi_hop_trace = multi_result.trace()

        # Multi-word enrichment: also search for each content word directly
        # and merge any neurons found into the concept set
        tokens = self.encoder._tokenize(text)
        seen_ids = {c.id for c in conv_result.concepts}
        extra_concepts = []
        for token in tokens:
            word_vec = self.encoder.encode_word(token)
            if np.all(word_vec == 0):
                continue
            neighbors = self.db.search(word_vec, k=3)
            for n in neighbors:
                if n.id not in seen_ids:
                    sim = float(np.dot(n.vector, word_vec))
                    if sim > 0.3:
                        extra_concepts.append(n)
                        seen_ids.add(n.id)

        if extra_concepts:
            conv_result.concepts = conv_result.concepts + extra_concepts
            if not conv_result.converged and conv_result.concepts:
                conv_result.converged = True
                conv_result.confidence = np.mean(
                    [c.confidence for c in conv_result.concepts]
                )

        # Filter concepts: only keep neurons relevant to the query
        # This prevents unrelated neurons from polluting slot filling
        if conv_result.concepts:
            filtered = []
            for c in conv_result.concepts:
                sim = float(np.dot(c.vector, query_vector))
                if sim > self.convergence.min_relevance:
                    # Skip generic/function words using embedding specificity.
                    # But keep words that are template structural words —
                    # they're needed for graph-based slot filling.
                    word = self.encoder.nearest_words(c.vector, k=1)
                    word_str = word[0][0] if word else ""
                    is_template_word = any(
                        word_str in t.structural_words
                        for t in self.template_store.templates
                    )
                    if not is_template_word and self._is_generic_word(c):
                        continue
                    filtered.append(c)
            # Sort by relevance to query (most relevant first)
            filtered.sort(key=lambda c: float(np.dot(c.vector, query_vector)), reverse=True)
            if filtered:
                conv_result.concepts = filtered
            else:
                # No relevant content concepts → honest abstention
                conv_result.converged = False
                conv_result.confidence = 0.0
                conv_result.concepts = []

        # Generate — pass query vector for template matching
        gen_result = self.generator.generate(conv_result, query_vector=query_vector)

        # Feedback
        self.feedback.on_query_result(query_vector, conv_result)

        # Build trace — use multi-hop trace if available
        hop_trace = getattr(conv_result, '_multi_hop_trace', None) or conv_result.trace()
        trace_parts = [hop_trace, "", gen_result.explain()]
        trace = "\n".join(trace_parts)

        return QueryResult(
            query=text,
            answer=gen_result.text,
            confidence=gen_result.confidence,
            converged=conv_result.converged,
            strategy=gen_result.strategy,
            trace=trace,
            generation=gen_result,
            convergence=conv_result,
        )

    def teach(self, word: str, confidence: float = 0.5) -> Neuron:
        """
        Teach the system a word by inserting its embedding as a neuron.

        Returns the created neuron.
        """
        vec = self.encoder.encode_word(word)
        if np.all(vec == 0):
            raise ValueError(f"Word '{word}' not in vocabulary — can't teach OOV")

        neuron = self.db.insert(vec, confidence=confidence)
        self._word_neurons[word.lower()] = neuron.id
        self.db.save_word_mapping(word.lower(), neuron.id)
        return neuron

    def _is_generic_word(self, neuron: Neuron) -> bool:
        """
        Detect if a neuron represents a generic/function word.

        Generic words (the, a, is, in) are close to many other neurons
        in the DB because they co-occur with everything. Content words
        (shakespeare, hamlet, relativity) occupy specific regions.

        Method: compare this neuron's average neighbor similarity against
        the KB-wide baseline. Generic = significantly more connected than
        average. This adapts to KB size automatically — in a small KB
        where everything is close, the baseline is high so nothing is
        falsely flagged as generic.

        This works for any language — no hardcoded word list needed.
        """
        if self.db.count() < 10:
            # Too few neurons for the statistical test to be meaningful
            return False

        k = min(10, self.db.count())
        neighbors = self.db.search(neuron.vector, k=k)
        if len(neighbors) < 3:
            return False

        # Average similarity to neighbors (excluding self)
        sims = []
        for n in neighbors:
            if n.id != neuron.id:
                sim = float(np.dot(neuron.vector, n.vector))
                sims.append(sim)

        if not sims:
            return False

        avg_sim = sum(sims) / len(sims)

        # Compare against KB baseline: compute average similarity for
        # a sample of other neurons. This adapts to KB size.
        baseline = self._kb_baseline_similarity(k)

        # Generic = notably more connected than average
        return avg_sim > baseline + 0.08

    def _kb_baseline_similarity(self, k: int) -> float:
        """
        Compute average neighbor similarity across a sample of neurons.
        This is the KB-wide baseline — how connected is a typical neuron.
        Cached per query to avoid recomputation.
        """
        if hasattr(self, '_baseline_cache'):
            cache_count, cached_val = self._baseline_cache
            if cache_count == self.db.count():
                return cached_val

        # Sample up to 10 neurons from the DB
        sample_size = min(10, self.db.count())
        # Use deterministic sampling via first N neurons
        sample_sims = []
        rows = self.db.db.execute(
            "SELECT id FROM neurons ORDER BY id LIMIT ?", (sample_size,)
        ).fetchall()

        for (nid,) in rows:
            neuron = self.db.get(nid)
            if neuron is None:
                continue
            neighbors = self.db.search(neuron.vector, k=k)
            for n in neighbors:
                if n.id != neuron.id:
                    sim = float(np.dot(neuron.vector, n.vector))
                    sample_sims.append(sim)

        baseline = sum(sample_sims) / len(sample_sims) if sample_sims else 0.0
        self._baseline_cache = (self.db.count(), baseline)
        return baseline

    def teach_sentence(self, sentence: str, confidence: float = 0.5) -> list:
        """
        Teach a sentence: insert each known word as a neuron,
        wire successor relationships between consecutive words.
        """
        tokens = self.encoder._tokenize(sentence)
        neurons = []

        for token in tokens:
            if self.encoder.has_word(token):
                if token in self._word_neurons:
                    # Already taught — reuse
                    n = self.db.get(self._word_neurons[token])
                    if n:
                        neurons.append(n)
                        continue
                n = self.teach(token, confidence=confidence)
                neurons.append(n)

        # Wire successors
        for i in range(len(neurons) - 1):
            self.db.update_successors(neurons[i].id, neurons[i + 1].id, 0.8)
            self.db.update_predecessors(neurons[i + 1].id, neurons[i].id)

        return neurons

    def teach_template(self, pattern: str, slots: dict,
                       confidence: float = 0.7):
        """Add a template for text generation."""
        return self.template_store.add(pattern, slots, confidence)

    def delete_word(self, word: str) -> bool:
        """
        Delete a word from the knowledge base. Invariant #3: gone immediately.
        """
        word = word.lower()
        nid = self._word_neurons.get(word)
        if nid is None:
            return False
        result = self.db.delete(nid)
        if result:
            del self._word_neurons[word]
            self.db.delete_word_mapping(word)
        return result

    def inspect(self, text: str, k: int = 5) -> dict:
        """
        Show what the system knows about a topic.
        Returns nearest neurons with their words and confidence.
        Invariant #2: every answer has a source.
        """
        vec = self.encoder.encode_sentence(text)
        if np.all(vec == 0):
            return {"query": text, "neighbors": [], "note": "all words OOV"}

        neighbors = self.db.search(vec, k=k)
        results = []
        for n in neighbors:
            word = self.encoder.nearest_words(n.vector, k=1)
            word_str = word[0][0] if word else f"<n{n.id}>"
            results.append({
                "id": n.id,
                "word": word_str,
                "confidence": n.confidence,
                "successors": len(n.successors),
            })

        return {"query": text, "neighbors": results}

    def stats(self) -> dict:
        """System statistics."""
        return {
            "neurons": self.db.count(),
            "templates": self.template_store.count(),
            "vocab_size": self.encoder.vocab_size,
            "dim": self.dim,
            "feedback_events": len(self.feedback.history),
        }

    def close(self):
        self.db.close()


def main():
    """Interactive CLI for the reasoning engine."""
    print("New-Gen-AI Reasoning Engine")
    print("=" * 40)
    print("Commands:")
    print("  query <text>     — ask a question")
    print("  teach <word>     — add a word to KB")
    print("  teach_s <text>   — teach a sentence")
    print("  delete <word>    — remove from KB")
    print("  inspect <text>   — show what system knows")
    print("  stats            — show system stats")
    print("  trace            — toggle trace output")
    print("  quit             — exit")
    print()

    engine = Engine()
    show_trace = False

    # Check if embeddings need loading
    if engine.encoder.vocab_size == 0:
        print("No embeddings loaded. Use 'load' to download GloVe,")
        print("or the engine will work with an empty vocabulary.")
        print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue

        parts = line.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "quit" or cmd == "exit":
            break
        elif cmd == "query" or cmd == "q":
            if not arg:
                print("Usage: query <text>")
                continue
            result = engine.query(arg)
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Strategy: {result.strategy}")
            if show_trace:
                print(f"\nTrace:\n{result.trace}")
            print()
        elif cmd == "teach":
            if not arg:
                print("Usage: teach <word>")
                continue
            try:
                n = engine.teach(arg)
                print(f"Taught '{arg}' → neuron {n.id}")
            except ValueError as e:
                print(f"Error: {e}")
        elif cmd == "teach_s":
            if not arg:
                print("Usage: teach_s <sentence>")
                continue
            neurons = engine.teach_sentence(arg)
            print(f"Taught {len(neurons)} words: {[engine.encoder.nearest_words(n.vector, k=1)[0][0] for n in neurons]}")
        elif cmd == "delete":
            if not arg:
                print("Usage: delete <word>")
                continue
            if engine.delete_word(arg):
                print(f"Deleted '{arg}' — gone.")
            else:
                print(f"'{arg}' not found in KB.")
        elif cmd == "inspect":
            if not arg:
                print("Usage: inspect <text>")
                continue
            info = engine.inspect(arg)
            for n in info["neighbors"]:
                print(f"  n{n['id']}: {n['word']} (conf={n['confidence']:.3f}, succ={n['successors']})")
        elif cmd == "stats":
            s = engine.stats()
            for k, v in s.items():
                print(f"  {k}: {v}")
        elif cmd == "trace":
            show_trace = not show_trace
            print(f"Trace output: {'ON' if show_trace else 'OFF'}")
        elif cmd == "load":
            print("Downloading GloVe 6B 300d (822MB)...")
            engine.load_embeddings()
            print(f"Loaded {engine.encoder.vocab_size} words.")
        else:
            print(f"Unknown command: {cmd}")

    engine.close()
    print("Goodbye.")


if __name__ == "__main__":
    main()
