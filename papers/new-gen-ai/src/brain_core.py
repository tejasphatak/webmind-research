"""
BrainCore: the clean reasoning engine. No performance hacks.

A growing matrix that learns from what it's taught.
No pretrained embeddings. No gradient descent. No FAISS.

    brain = BrainCore()
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np

from neuron import NeuronDB, Neuron
from convergence import ConvergenceLoop, MultiHopConvergence

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


class BrainCore:
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
        self._words = []           # index → word
        self._word_idx = {}        # word → index
        self._cooc = {}            # word_idx → {word_idx: weight} (sparse co-occurrence)
        self._matrix = None        # legacy compat — derived from _cooc when needed
        self._db_path = db_path

        # The database: metadata + persistence
        self.db = NeuronDB(path=db_path, dim=1)
        self._word_neurons = self.db.load_word_mappings()

        # Templates for fluent output
        self._templates = []

        # Convergence loop for multi-hop reasoning
        self._convergence = ConvergenceLoop(
            self.db, max_hops=10, k=5,
            convergence_threshold=0.99,
            min_confidence=0.1, min_relevance=0.3,
        )
        self._multi_hop = MultiHopConvergence(
            self._convergence, max_rounds=3, concept_blend_weight=0.4,
        )

        # Nid-to-word cache
        self._nid_to_word_cache = None

        # Thread pools
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._batch_pool = ThreadPoolExecutor(max_workers=8)
        self._lock = Lock()

        # Load co-occurrence from DB
        self._load_cooc()

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

        with self._lock:
            # Batch all DB writes for this sentence
            self.db.begin_batch()

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

            # Flush all writes at once
            self.db.end_batch()

            # Reindex if dimensions changed
            if len(self._words) != dim_before:
                self._reindex()

        return [n.id for n in neurons]

    def teach_batch(self, sentences: list, confidence: float = 0.5) -> list:
        """Teach multiple sentences in parallel. Returns list of neuron ID lists."""
        futures = [self._batch_pool.submit(self.teach, s, confidence) for s in sentences]
        return [f.result() for f in futures]

    def correct(self, question: str, answer: str):
        """Learn from a failure. Teach the answer, resolve the miss."""
        self.teach(answer, confidence=0.6)
        self.db.resolve_miss_by_query(question, answer)

    # --- Querying ---

    def ask_batch(self, questions: list) -> list:
        """Ask multiple questions in parallel. Returns list of result dicts.
        Uses separate batch pool to avoid thread starvation (each ask()
        uses the internal pool for its own parallelism)."""
        futures = [self._batch_pool.submit(self.ask, q) for q in questions]
        return [f.result() for f in futures]

    def ask(self, question: str) -> dict:
        """
        Ask the brain a question. Returns dict with answer + trace.

        Flow:
          1. Encode question → vector
          2. Multi-hop convergence: iteratively search the neuron DB,
             blending discovered concepts back into the query each round.
             This allows reasoning to cross concept boundaries.
          3. Sparse co-occurrence search as complementary signal
          4. Disambiguate via sentence table
          5. Output answer (template or concept list)
        """
        tokens = self._tokenize(question)
        content = [t for t in tokens if t not in FUNCTION_WORDS]
        query_vec = self._encode_sentence(question)

        if np.all(query_vec == 0) or len(self._words) == 0:
            self.db.log_miss(question, query_vec if query_vec is not None
                             else np.zeros(1, dtype=np.float32))
            return {"answer": "I don't know.", "confidence": 0.0,
                    "strategy": "abstain", "trace": "No knowledge"}

        content_indices = [self._word_idx[t] for t in content
                           if t in self._word_idx]
        content_nids = [self._word_neurons[t] for t in content
                        if t in self._word_neurons]

        if not content_indices:
            self.db.log_miss(question, query_vec if query_vec is not None
                             else np.zeros(1, dtype=np.float32))
            return {"answer": "I don't know.", "confidence": 0.0,
                    "strategy": "abstain", "trace": "No known words in query"}

        # --- Multi-hop convergence ---
        # Run convergence loop on the dense neuron vectors in the DB.
        # Each round discovers concepts, blends them into the query,
        # and searches again — allowing reasoning across concept gaps.
        multi_hop_result = self._multi_hop.reason(query_vec)
        convergence_trace = multi_hop_result.trace()

        # Build nid→word lookup
        if self._nid_to_word_cache is None:
            self._nid_to_word_cache = {nid: w for w, nid in self._word_neurons.items()
                                       if not w.startswith("__")}
        nid_to_word = self._nid_to_word_cache
        word_to_nid = {w: nid for nid, w in nid_to_word.items()}

        # Collect concepts from multi-hop convergence
        concepts = []
        seen = set()
        for n in multi_hop_result.concepts:
            word = nid_to_word.get(n.id)
            if word and n.id not in seen and word not in FUNCTION_WORDS:
                concepts.append((n, word))
                seen.add(n.id)

        # --- Sparse co-occurrence search (complementary) ---
        weights = [1.0 / (1.0 + 0.1 * i) for i in range(len(content_indices))]
        query_cooc = self._sparse_blend(content_indices, weights)
        search_results = self._sparse_search(query_cooc, k=10)

        for word_idx, sim in search_results:
            if word_idx < len(self._words):
                word = self._words[word_idx]
                nid = word_to_nid.get(word)
                if nid and nid not in seen and word not in FUNCTION_WORDS:
                    n = self.db.get(nid)
                    if n:
                        concepts.append((n, word))
                        seen.add(nid)

        # --- Sentence disambiguation ---
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

        # Add concepts from sentence disambiguation
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

        # Generate answer (pass convergence trace for inspectability)
        result = self._generate(concepts, query_vec, tokens, question)
        result["convergence_trace"] = convergence_trace
        result["converged"] = multi_hop_result.converged
        result["convergence_rounds"] = len(multi_hop_result.rounds)
        return result

    # --- Generation ---

    def _generate(self, concepts, query_vec, query_tokens, question) -> dict:
        """Generate an answer from concepts. Template and chain race in parallel."""
        concept_neurons = [n for n, w in concepts]
        concept_words = [w for n, w in concepts]
        concept_ids = [n.id for n, w in concepts]

        # Race: template vs sentence chain — first valid result wins
        template_future = self._pool.submit(
            self._try_template,
            concept_neurons, concept_words, concept_ids, query_vec, query_tokens
        )
        chain_future = self._pool.submit(
            self._try_sentence_chain, concept_ids, query_vec
        )

        # Wait for both (fast enough that ordering doesn't matter much)
        template_result = template_future.result()
        chain_result = chain_future.result()

        if template_result:
            return template_result
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

        # Convergence: does the filled template live near the query?
        answer_vec = self._encode_sentence(text)
        q_norm = np.linalg.norm(query_vec)
        a_norm = np.linalg.norm(answer_vec)
        if q_norm > 0 and a_norm > 0:
            convergence = float(np.dot(query_vec, answer_vec) / (q_norm * a_norm))
        else:
            convergence = 0.0
        if convergence <= 0:
            return None

        return {
            "answer": text,
            "confidence": convergence,
            "strategy": "template",
            "trace": f"Template: {pattern}, fills: {fills}, convergence={convergence:.3f}",
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

        # Sparse convergence check: do answer words' co-occurrence overlap with query?
        answer_indices = [self._word_idx[w] for w in words if w in self._word_idx]
        if not answer_indices:
            return None
        answer_cooc = self._sparse_blend(answer_indices)
        # concept_ids are neuron IDs — convert to word indices
        nid_to_word = self._nid_to_word_cache or {nid: w for w, nid in self._word_neurons.items()
                                                   if not w.startswith("__")}
        query_words = [nid_to_word.get(nid) for nid in concept_ids]
        query_indices = [self._word_idx[w] for w in query_words
                         if w and w in self._word_idx]
        if not query_indices:
            query_indices = answer_indices
        query_cooc = self._sparse_blend(query_indices)
        convergence = self._sparse_cosine(query_cooc, answer_cooc)

        if convergence <= 0:
            return None

        return {
            "answer": " ".join(words),
            "confidence": convergence,
            "strategy": "sentence_chain",
            "trace": f"Sentence {sid} (convergence={convergence:.3f}): {words}",
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

    # --- Co-occurrence operations (dict-based, no N×N matrix) ---

    def _learn_word(self, word: str) -> int:
        """Add a word. Returns its index."""
        word = word.lower().strip()
        if word in self._word_idx:
            return self._word_idx[word]

        idx = len(self._words)
        self._words.append(word)
        self._word_idx[word] = idx
        self._cooc[idx] = {idx: 1.0}  # self-connection
        return idx

    def _load_cooc(self):
        """Load co-occurrence from the cooccurrence table, or rebuild from neuron vectors."""
        # Build word list from DB
        words_in_db = sorted(
            [(w, nid) for w, nid in self._word_neurons.items()
             if not w.startswith("__")],
            key=lambda x: x[1]
        )
        for word, nid in words_in_db:
            if word not in self._word_idx:
                idx = len(self._words)
                self._words.append(word)
                self._word_idx[word] = idx
                self._cooc[idx] = {idx: 1.0}

        if not words_in_db:
            return

        # Try loading from cooccurrence table first
        try:
            rows = self.db.db.execute(
                "SELECT word_a, word_b, weight FROM cooccurrence"
            ).fetchall()
            if rows:
                for a, b, w in rows:
                    if a not in self._cooc:
                        self._cooc[a] = {a: 1.0}
                    if b not in self._cooc:
                        self._cooc[b] = {b: 1.0}
                    self._cooc[a][b] = w
                    self._cooc[b][a] = w
                self._rebuild_search_matrix()
                return
        except Exception:
            pass  # table doesn't exist yet

        # Create cooccurrence table
        self.db.db.execute("""
            CREATE TABLE IF NOT EXISTS cooccurrence (
                word_a INTEGER NOT NULL,
                word_b INTEGER NOT NULL,
                weight REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (word_a, word_b)
            )
        """)
        self.db.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cooc_a ON cooccurrence(word_a)")
        self.db.db.commit()

        # Rebuild from neuron vectors (migration from old N×N format)
        n = len(self._words)
        nid_to_idx = {nid: self._word_idx[w]
                      for w, nid in words_in_db if w in self._word_idx}

        for nid, vec_bytes in self.db.db.execute("SELECT id, vector FROM neurons"):
            idx = nid_to_idx.get(nid)
            if idx is None:
                continue
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            for j, val in enumerate(vec):
                if j < n and val > 0 and j != idx:
                    if idx not in self._cooc:
                        self._cooc[idx] = {idx: 1.0}
                    self._cooc[idx][j] = float(val)

        # Persist to cooccurrence table
        self._save_cooc()
        self._rebuild_search_matrix()

    def _save_cooc(self):
        """Persist co-occurrence dict to SQLite."""
        self.db.db.execute("""
            CREATE TABLE IF NOT EXISTS cooccurrence (
                word_a INTEGER NOT NULL,
                word_b INTEGER NOT NULL,
                weight REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (word_a, word_b)
            )
        """)
        pairs = []
        for a, neighbors in self._cooc.items():
            for b, w in neighbors.items():
                if a != b and w > 0:
                    pairs.append((a, b, w))
        self.db.db.execute("DELETE FROM cooccurrence")
        self.db.db.executemany(
            "INSERT INTO cooccurrence (word_a, word_b, weight) VALUES (?, ?, ?)",
            pairs)
        self.db.db.commit()

    def _save_matrix(self):
        """Persist co-occurrence to DB."""
        self._save_cooc()

    def _learn_cooccurrence(self, words: list):
        """Strengthen connections between co-occurring words."""
        indices = [self._word_idx[w.lower().strip()] for w in words
                   if w.lower().strip() in self._word_idx]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                if a not in self._cooc:
                    self._cooc[a] = {a: 1.0}
                if b not in self._cooc:
                    self._cooc[b] = {b: 1.0}
                self._cooc[a][b] = self._cooc[a].get(b, 0) + COOCCURRENCE_PULL
                self._cooc[b][a] = self._cooc[b].get(a, 0) + COOCCURRENCE_PULL

    # --- Sparse operations (no N-dimensional vectors) ---

    def _get_cooc(self, word: str) -> dict:
        """Get a word's co-occurrence dict. Sparse. O(K) where K = connections."""
        word = word.lower().strip()
        idx = self._word_idx.get(word)
        if idx is None:
            return {}
        return self._cooc.get(idx, {})

    def _sparse_norm(self, d: dict) -> float:
        """L2 norm of a sparse dict."""
        return sum(v * v for v in d.values()) ** 0.5

    def _sparse_cosine(self, a: dict, b: dict) -> float:
        """Cosine similarity between two sparse dicts. O(min(|a|, |b|))."""
        if not a or not b:
            return 0.0
        # Iterate over the smaller dict
        if len(a) > len(b):
            a, b = b, a
        dot = sum(v * b.get(k, 0) for k, v in a.items())
        if dot == 0:
            return 0.0
        na = self._sparse_norm(a)
        nb = self._sparse_norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _sparse_blend(self, word_indices: list, weights: list = None) -> dict:
        """Blend multiple words' co-occurrence dicts. Weighted average."""
        result = {}
        if weights is None:
            weights = [1.0] * len(word_indices)
        total_w = sum(weights)
        if total_w == 0:
            return result
        for idx, w in zip(word_indices, weights):
            for k, v in self._cooc.get(idx, {}).items():
                result[k] = result.get(k, 0) + w * v
        # Normalize
        for k in result:
            result[k] /= total_w
        return result

    def _sparse_search(self, query_cooc: dict, k: int = 5) -> list:
        """Search all words by sparse cosine with query. O(N × K)."""
        if not query_cooc:
            return []
        scores = []
        q_norm = self._sparse_norm(query_cooc)
        if q_norm == 0:
            return []
        for word_idx, word_cooc in self._cooc.items():
            if not word_cooc:
                continue
            dot = sum(query_cooc.get(j, 0) * v for j, v in word_cooc.items())
            if dot > 0:
                w_norm = self._sparse_norm(word_cooc)
                if w_norm > 0:
                    sim = dot / (q_norm * w_norm)
                    scores.append((word_idx, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _encode_word(self, word: str) -> np.ndarray:
        """Get a word's vector. Dense form for backward compat."""
        word = word.lower().strip()
        n = len(self._words)
        idx = self._word_idx.get(word)
        if idx is None or n == 0:
            return np.zeros(n or 1, dtype=np.float32)
        vec = np.zeros(n, dtype=np.float32)
        for j, w in self._cooc.get(idx, {}).items():
            if j < n:
                vec[j] = w
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _encode_sentence(self, text: str) -> np.ndarray:
        """Encode a sentence. Dense form for backward compat."""
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
        """No-op. Sparse search uses _cooc dict directly. No matrix needed."""
        pass

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

        # Find strongest relationships from co-occurrence dict
        connections = []
        for j, val in self._cooc.get(idx, {}).items():
            if j != idx and j < len(self._words):
                connections.append((self._words[j], float(val)))
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

    def health(self) -> dict:
        """
        Self-awareness: how much resource am I using?
        Returns CPU, memory, DB size, matrix size, disk free.
        The brain should know its own cost and not exploit the machine.
        """
        h = self.db.health()

        # Add brain-level matrix stats
        # Estimate co-occurrence memory: entries × ~50 bytes
        cooc_entries = sum(len(v) for v in self._cooc.values())
        h["brain_matrix_mb"] = round(cooc_entries * 50 / (1024 * 1024), 2)
        h["cooc_entries"] = cooc_entries
        h["words"] = len(self._words)
        h["templates"] = len(self._templates)

        # Pressure signals — is the brain getting too big?
        h["memory_pressure"] = h["rss_mb"] > 512  # over 512MB = pressure
        h["disk_pressure"] = h["disk_free_gb"] < 1.0  # under 1GB = pressure
        h["matrix_pressure"] = len(self._words) > 10000  # 10K dims = O(100M) matrix

        return h

    def close(self):
        self._pool.shutdown(wait=False)
        self._batch_pool.shutdown(wait=False)
        self._save_matrix()
        self.db.close()


# --- CLI ---

if __name__ == "__main__":
    brain = BrainCore()

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
