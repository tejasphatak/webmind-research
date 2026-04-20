"""
Brain: the complete reasoning system in one file.

A growing matrix that learns from what it's taught.
No pretrained embeddings. No gradient descent. No FAISS.

    brain = Brain()
    brain.teach("paris is the capital of france")
    brain.teach("london is the capital of england")
    brain.ask("capital of france")  # → "paris capital france"

The matrix IS the understanding. Each word is a dimension.
Co-occurring words pull toward each other. Query = search the matrix.
The DB (SQLite) is the portable brain — copy it anywhere.
"""

import re
import hashlib
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np

from neuron import NeuronDB, Neuron

# How much co-occurring words pull toward each other
COOCCURRENCE_PULL = 0.3

# Words that are grammar, not knowledge
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "of", "in", "to",
    "for", "with", "on", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "and", "but", "or", "nor", "not", "so",
    "yet", "both", "either", "neither", "each", "every", "all", "any",
    "few", "more", "most", "other", "some", "such", "no", "only", "own",
    "same", "than", "too", "very", "just", "about", "up", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "if", "then",
    "because", "while", "although", "though", "even", "also", "it", "its",
    "how", "when", "where", "why", "there", "here",
})

STRUCTURAL_WORDS = frozenset({
    "is", "are", "was", "were", "wrote", "discovered", "invented",
    "created", "built", "made", "found", "said", "called", "known",
    "born", "died", "has", "had", "have", "been", "became", "became",
    "the", "a", "an", "of", "in", "for", "to", "on", "at", "by",
    "with", "from", "as", "and", "or", "but",
})


class Brain:
    """
    A self-growing reasoning system.

    The brain is a matrix + a database.
    The matrix stores word relationships (encoder).
    The database stores metadata (confidence, links, sentences).

    Teaching grows the matrix. Querying searches it.
    """

    def __init__(self, db_path: str = None):
        """
        Create a brain. Optionally persist to disk.

        Args:
            db_path: directory for SQLite storage. None = in-memory.
        """
        # The growing matrix: words × words
        self._words = []           # index → word
        self._word_idx = {}        # word → index
        self._matrix = None        # N×N relationship matrix

        # The database: metadata + persistence
        self.db = NeuronDB(path=db_path, dim=1)  # dim doesn't matter, we manage vectors
        self._word_neurons = self.db.load_word_mappings()

        # Templates for fluent output
        self._templates = []       # [(pattern, slots, vector)]

    # --- Learning ---

    def teach(self, sentence: str, confidence: float = 0.5) -> list:
        """
        Teach the brain a sentence. Returns list of neuron IDs created.

        "paris is the capital of france" →
          1. Learns words: paris, capital, france (skips function words)
          2. Records co-occurrence: paris↔capital, paris↔france, capital↔france
          3. Creates neurons in DB with successor links
          4. Records sentence association
          5. Extracts template: "[S0] is the [S1] of [S2]"
          6. Reindexes all vectors to current dimensions
        """
        tokens = self._tokenize(sentence)
        content = [t for t in tokens if t not in FUNCTION_WORDS]

        if not content:
            return []

        # Grow the matrix
        dim_before = len(self._words)
        for word in content:
            self._learn_word(word)
        if len(content) >= 2:
            self._learn_cooccurrence(content)

        # Create neurons in DB
        neurons = []
        for word in content:
            if word in self._word_neurons:
                n = self.db.get(self._word_neurons[word])
                if n:
                    neurons.append(n)
                    continue
            vec = self._encode_word(word)
            if np.any(vec != 0):
                n = self.db.insert(vec, confidence=confidence)
                self._word_neurons[word] = n.id
                self.db.save_word_mapping(word, n.id)
                neurons.append(n)

        # Wire successors
        for i in range(len(neurons) - 1):
            self.db.update_successors(neurons[i].id, neurons[i + 1].id, 0.8)
            self.db.update_predecessors(neurons[i + 1].id, neurons[i].id)

        # Record sentence
        if len(neurons) >= 2:
            self.db.record_sentence([n.id for n in neurons])

        # Extract template
        if len(tokens) >= 3:
            self._extract_template(tokens)

        # Reindex if dimensions changed
        if len(self._words) != dim_before:
            self._reindex()

        return [n.id for n in neurons]

    def correct(self, question: str, answer: str):
        """Learn from a failure. Teach the answer, resolve the miss."""
        self.teach(answer, confidence=0.6)
        self.db.resolve_miss_by_query(question, answer)

    # --- Querying ---

    def ask(self, question: str) -> dict:
        """
        Ask the brain a question. Returns dict with answer + trace.

        Flow:
          1. Encode question → vector
          2. Find nearest neurons in DB
          3. Disambiguate via sentence table
          4. Output answer (template or concept list)
        """
        tokens = self._tokenize(question)
        content = [t for t in tokens if t not in FUNCTION_WORDS]
        query_vec = self._encode_sentence(question)

        if np.all(query_vec == 0) or len(self._words) == 0:
            self.db.log_miss(question, query_vec if query_vec is not None
                             else np.zeros(1, dtype=np.float32))
            return {"answer": "I don't know.", "confidence": 0.0,
                    "strategy": "abstain", "trace": "No knowledge"}

        # Search: find neurons near the query
        neighbors = self.db.search(query_vec, k=10)
        if not neighbors:
            self.db.log_miss(question, query_vec)
            return {"answer": "I don't know.", "confidence": 0.0,
                    "strategy": "abstain", "trace": "No matching neurons"}

        # Disambiguate: which taught sentence best matches?
        content_nids = [self._word_neurons[t] for t in content
                        if t in self._word_neurons]

        best_sentence_nids = None
        if content_nids:
            sentences = self.db.get_sentences_for_neurons(content_nids)
            if sentences:
                scored = [(sid, len(nids)) for sid, nids in sentences.items()]
                scored.sort(key=lambda x: x[1], reverse=True)
                best_score = scored[0][1]
                best_sentence_nids = set()
                for sid, score in scored:
                    if score < best_score:
                        break
                    for nid, pos in self.db.get_sentence_neurons(sid):
                        best_sentence_nids.add(nid)

        # Collect concept neurons
        nid_to_word = {nid: w for w, nid in self._word_neurons.items()
                       if not w.startswith("__")}
        concepts = []
        seen = set()

        # From search results
        for n in neighbors:
            if n.id not in seen:
                word = nid_to_word.get(n.id)
                if word and word not in FUNCTION_WORDS:
                    concepts.append((n, word))
                    seen.add(n.id)

        # From sentence disambiguation
        if best_sentence_nids:
            for nid in best_sentence_nids:
                if nid not in seen:
                    n = self.db.get(nid)
                    word = nid_to_word.get(nid)
                    if n and word:
                        concepts.append((n, word))
                        seen.add(nid)

        if not concepts:
            self.db.log_miss(question, query_vec)
            return {"answer": "I don't know.", "confidence": 0.0,
                    "strategy": "abstain", "trace": "No relevant concepts"}

        # Filter: keep only concepts from best sentence (if disambiguated)
        if best_sentence_nids and len(content_nids) >= 2:
            concepts = [(n, w) for n, w in concepts
                        if n.id in best_sentence_nids
                        or n.id in set(content_nids)]

        # Generate answer
        return self._generate(concepts, query_vec, tokens, question)

    # --- Generation ---

    def _generate(self, concepts, query_vec, query_tokens, question) -> dict:
        """Generate an answer from concepts."""
        concept_neurons = [n for n, w in concepts]
        concept_words = [w for n, w in concepts]
        concept_ids = [n.id for n, w in concepts]

        # Try template
        template_result = self._try_template(
            concept_neurons, concept_words, concept_ids, query_vec, query_tokens
        )
        if template_result:
            return template_result

        # Try sentence chain: output taught sentence in word order
        chain_result = self._try_sentence_chain(concept_ids, query_vec)
        if chain_result:
            return chain_result

        # Fallback: concept list
        avg_conf = float(np.mean([n.confidence for n in concept_neurons]))
        return {
            "answer": " ".join(concept_words),
            "confidence": avg_conf,
            "strategy": "concept_list",
            "trace": f"Concepts: {concept_words}",
        }

    def _try_template(self, neurons, words, nids, query_vec, query_tokens) -> dict:
        """Try to fill a template with concepts."""
        if not self._templates:
            return None

        query_word_set = set(query_tokens)

        # Score templates by structural word overlap with query
        best_template = None
        best_score = 0
        for pattern, slots, tvec in self._templates:
            struct_words = [w for w in re.findall(r'[a-z]+', pattern.lower())
                            if w not in [s.lower() for s in slots]]
            overlap = sum(1 for w in struct_words if w in query_word_set)
            if overlap > best_score:
                best_score = overlap
                best_template = (pattern, slots)

        if not best_template or best_score == 0:
            return None

        pattern, slots = best_template

        # Order concepts by taught sentence position
        sentence_order = self._get_sentence_order(nids)
        if sentence_order:
            ordered = sorted(zip(neurons, words),
                             key=lambda p: sentence_order.get(p[0].id, 999))
        else:
            ordered = list(zip(neurons, words))

        # Fill slots
        content_words = [w for n, w in ordered
                         if w.lower() not in STRUCTURAL_WORDS]
        fills = {}
        available = list(content_words)
        for slot_name in slots:
            if available:
                fills[slot_name] = available.pop(0)

        if not fills:
            return None

        text = pattern
        for name, value in fills.items():
            text = text.replace(f"[{name}]", value)
        # Replace unfilled slots
        for name in slots:
            if name not in fills:
                text = text.replace(f"[{name}]", "...")

        avg_conf = float(np.mean([n.confidence for n in neurons]))
        return {
            "answer": text,
            "confidence": avg_conf * 0.8,
            "strategy": "template",
            "trace": f"Template: {pattern}, fills: {fills}",
        }

    def _try_sentence_chain(self, concept_ids, query_vec) -> dict:
        """Find the best matching taught sentence, output in word order."""
        sentences = self.db.get_sentences_for_neurons(concept_ids)
        if not sentences:
            return None

        nid_to_word = {nid: w for w, nid in self._word_neurons.items()
                       if not w.startswith("__")}

        scored = []
        for sid, matched in sentences.items():
            sent_neurons = self.db.get_sentence_neurons(sid)
            if not sent_neurons:
                continue
            score = len(matched)
            scored.append((sid, score, sent_neurons))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        sid, score, sent_neurons = scored[0]

        # Output in taught order
        ordered = sorted(sent_neurons, key=lambda x: x[1])
        words = [nid_to_word.get(nid, "") for nid, pos in ordered]
        words = [w for w in words if w]

        if len(words) < 2:
            return None

        return {
            "answer": " ".join(words),
            "confidence": 0.5,
            "strategy": "sentence_chain",
            "trace": f"Sentence {sid}: {words}",
        }

    def _get_sentence_order(self, concept_ids) -> dict:
        """Get taught position order for concepts."""
        if len(concept_ids) < 2:
            return {}
        sentences = self.db.get_sentences_for_neurons(concept_ids)
        if not sentences:
            return {}
        best_sid = max(sentences, key=lambda sid: len(sentences[sid]))
        if len(sentences[best_sid]) < 2:
            return {}
        return {nid: pos for nid, pos in self.db.get_sentence_neurons(best_sid)}

    # --- Matrix operations ---

    def _learn_word(self, word: str) -> int:
        """Add a word to the matrix. Returns its dimension index."""
        word = word.lower().strip()
        if word in self._word_idx:
            return self._word_idx[word]

        idx = len(self._words)
        self._words.append(word)
        self._word_idx[word] = idx

        if self._matrix is None:
            self._matrix = np.array([[1.0]], dtype=np.float32)
        else:
            n = self._matrix.shape[0]
            new_col = np.zeros((n, 1), dtype=np.float32)
            self._matrix = np.hstack([self._matrix, new_col])
            new_row = np.zeros((1, n + 1), dtype=np.float32)
            new_row[0, idx] = 1.0
            self._matrix = np.vstack([self._matrix, new_row])

        return idx

    def _learn_cooccurrence(self, words: list):
        """Pull co-occurring words toward each other."""
        indices = [self._word_idx[w.lower().strip()] for w in words
                   if w.lower().strip() in self._word_idx]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                self._matrix[a, b] += COOCCURRENCE_PULL
                self._matrix[b, a] += COOCCURRENCE_PULL

    def _encode_word(self, word: str) -> np.ndarray:
        """Get a word's vector from the matrix."""
        word = word.lower().strip()
        idx = self._word_idx.get(word)
        if idx is None:
            return np.zeros(len(self._words) or 1, dtype=np.float32)
        vec = self._matrix[idx].copy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _encode_sentence(self, text: str) -> np.ndarray:
        """Encode a sentence as weighted average of word vectors."""
        tokens = self._tokenize(text)
        if not tokens:
            d = len(self._words) or 1
            return np.zeros(d, dtype=np.float32)

        vectors = []
        weights = []
        for i, token in enumerate(tokens):
            vec = self._encode_word(token)
            if np.any(vec != 0):
                vectors.append(vec)
                weights.append(1.0 / (1.0 + 0.1 * i))

        if not vectors:
            d = len(self._words) or 1
            return np.zeros(d, dtype=np.float32)

        vectors = np.array(vectors)
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        result = np.average(vectors, axis=0, weights=weights).astype(np.float32)
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result

    def _reindex(self):
        """Re-encode all neurons to current dimensions."""
        for word, nid in self._word_neurons.items():
            if word.startswith("__"):
                continue
            vec = self._encode_word(word)
            if np.any(vec != 0):
                # Update search matrix
                row = self.db._id_to_row.get(nid)
                if row is not None and self.db._vectors is not None:
                    if vec.shape[0] != self.db._vectors.shape[1]:
                        # Dimension changed — full rebuild
                        self._rebuild_search_matrix()
                        return
                    self.db._vectors[row] = vec
                self.db.db.execute(
                    "UPDATE neurons SET vector = ? WHERE id = ?",
                    (vec.tobytes(), nid)
                )
        self.db.db.commit()

        # Re-encode templates
        for i, (pattern, slots, old_vec) in enumerate(self._templates):
            text = re.sub(r'\[[A-Z_0-9]+\]', '', pattern).strip()
            vec = self._encode_sentence(text)
            self._templates[i] = (pattern, slots, vec)

    def _rebuild_search_matrix(self):
        """Rebuild the DB's search matrix from scratch."""
        self.db._vectors = None
        self.db._id_to_row = {}
        self.db._row_to_id = {}

        for word, nid in self._word_neurons.items():
            if word.startswith("__"):
                continue
            vec = self._encode_word(word)
            if np.any(vec != 0):
                row = len(self.db._id_to_row)
                if self.db._vectors is None:
                    self.db._vectors = vec.reshape(1, -1)
                else:
                    self.db._vectors = np.vstack(
                        [self.db._vectors, vec.reshape(1, -1)]
                    )
                self.db._id_to_row[nid] = row
                self.db._row_to_id[row] = nid
                self.db.db.execute(
                    "UPDATE neurons SET vector = ? WHERE id = ?",
                    (vec.tobytes(), nid)
                )
        self.db.db.commit()

    # --- Template extraction ---

    def _extract_template(self, tokens: list):
        """Auto-extract a template from a sentence."""
        structural = set()
        content = []
        for i, token in enumerate(tokens):
            if token in STRUCTURAL_WORDS:
                structural.add(i)
            else:
                content.append(i)

        if not content or not structural:
            return

        pattern_parts = []
        slots = {}
        slot_idx = 0
        for i, token in enumerate(tokens):
            if i in structural:
                pattern_parts.append(token)
            else:
                name = f"S{slot_idx}"
                pattern_parts.append(f"[{name}]")
                slots[name] = "noun"
                slot_idx += 1

        pattern = " ".join(pattern_parts)

        # Check for duplicates
        for p, s, v in self._templates:
            if p == pattern:
                return

        text = " ".join(t for t in tokens if t in STRUCTURAL_WORDS)
        vec = self._encode_sentence(text) if text.strip() else np.zeros(
            len(self._words) or 1, dtype=np.float32
        )
        self._templates.append((pattern, slots, vec))

    # --- Utilities ---

    @staticmethod
    def _tokenize(text: str) -> list:
        return re.findall(r'[a-z0-9]+', text.lower())

    def inspect(self, word: str) -> dict:
        """Show what the brain knows about a word."""
        word = word.lower().strip()
        if word not in self._word_idx:
            return {"word": word, "known": False}

        idx = self._word_idx[word]
        row = self._matrix[idx]

        # Find strongest relationships
        connections = []
        for i, val in enumerate(row):
            if i != idx and val > 0:
                connections.append((self._words[i], float(val)))
        connections.sort(key=lambda x: x[1], reverse=True)

        nid = self._word_neurons.get(word)
        neuron = self.db.get(nid) if nid else None

        return {
            "word": word,
            "known": True,
            "dimension": idx,
            "connections": connections[:10],
            "confidence": neuron.confidence if neuron else None,
            "successors": len(neuron.successors) if neuron else 0,
        }

    def stats(self) -> dict:
        return {
            "words": len(self._words),
            "dimensions": len(self._words),
            "neurons": self.db.count(),
            "templates": len(self._templates),
            "matrix_size": f"{len(self._words)}x{len(self._words)}",
        }

    def close(self):
        self.db.close()


# --- CLI ---

if __name__ == "__main__":
    brain = Brain()

    print("Brain — self-growing reasoning system")
    print("Commands: teach <sentence>, ask <question>, inspect <word>, stats, quit")
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

        if cmd in ("quit", "exit"):
            break
        elif cmd == "teach":
            ids = brain.teach(arg)
            print(f"Learned {len(ids)} concepts. Dimensions: {len(brain._words)}")
        elif cmd == "ask":
            result = brain.ask(arg)
            print(f"A: {result['answer']}")
            print(f"   [{result['strategy']}, conf={result['confidence']:.3f}]")
        elif cmd == "inspect":
            info = brain.inspect(arg)
            if info["known"]:
                print(f"  dim={info['dimension']}, conf={info['confidence']}")
                for w, v in info["connections"]:
                    print(f"    → {w}: {v:.3f}")
            else:
                print(f"  Unknown word: {arg}")
        elif cmd == "stats":
            for k, v in brain.stats().items():
                print(f"  {k}: {v}")
        else:
            print(f"Unknown: {cmd}")

    brain.close()
    print("Done.")
