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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

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

    # Max words per expert matrix (pre-allocated mmap size)
    MAX_WORDS = 8000

    def __init__(self, db_path: str = None, embed_dim: int = 0,
                 use_mmap: bool = True):
        """
        Create a brain. Optionally persist to disk.

        Args:
            db_path: directory for SQLite storage. None = in-memory.
            embed_dim: fixed embedding dimension (0 = N×N matrix mode).
            use_mmap: if True, matrix lives on disk via mmap (instant boot).
        """
        self._words = []
        self._word_idx = {}
        self._db_path = db_path
        self._embed_dim = embed_dim
        self._use_mmap = use_mmap and db_path is not None

        self._embeddings = None
        self._matrix = None
        self._matrix_capacity = 0

        # The database: metadata + persistence
        self.db = NeuronDB(path=db_path, dim=1)
        self._word_neurons = self.db.load_word_mappings()

        # Templates for fluent output
        self._templates = []

        # Thread pools
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._batch_pool = ThreadPoolExecutor(max_workers=8)
        self._lock = Lock()

        # Bulk mode
        self._bulk_mode = False
        self._bulk_dirty = False
        self._search_dirty = False
        self._nid_to_word_cache = None

        # Boot: mmap = instant, else try cache, else rebuild from DB
        if self._use_mmap and embed_dim == 0:
            self._boot_mmap()
        elif not self._load_cache():
            self._load_matrix()
            self._save_cache()

    def _boot_mmap(self):
        """Boot from memory-mapped matrix file. Instant — no deserialization."""
        d = Path(self._db_path)
        mmap_file = d / '_matrix.mmap'
        words_file = d / '_words.json'
        cap = self.MAX_WORDS

        # Load word list
        if words_file.exists():
            import json
            data = json.loads(words_file.read_text())
            self._words = data.get('words', [])
            self._word_idx = {w: i for i, w in enumerate(self._words)}
        else:
            # First boot — build word list from DB
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

        n = len(self._words)

        # Open or create mmap
        if mmap_file.exists():
            self._matrix = np.memmap(str(mmap_file), dtype=np.float32,
                                     mode='r+', shape=(cap, cap))
        else:
            self._matrix = np.memmap(str(mmap_file), dtype=np.float32,
                                     mode='w+', shape=(cap, cap))
            # Initialize diagonal
            for i in range(n):
                self._matrix[i, i] = 1.0
            # If we have neuron vectors in DB, load them into mmap
            if n > 0:
                nid_to_idx = {}
                for word, nid in self._word_neurons.items():
                    idx = self._word_idx.get(word)
                    if idx is not None:
                        nid_to_idx[nid] = idx
                rows = self.db.db.execute("SELECT id, vector FROM neurons").fetchall()
                for nid, vec_bytes in rows:
                    idx = nid_to_idx.get(nid)
                    if idx is None:
                        continue
                    vec = np.frombuffer(vec_bytes, dtype=np.float32)
                    vlen = min(len(vec), cap)
                    self._matrix[idx, :vlen] = vec[:vlen]
                    self._matrix[:vlen, idx] = vec[:vlen]
            self._matrix.flush()

        self._matrix_capacity = cap

        # Save word list
        self._save_words()

        # Build search index directly from matrix (no separate copy)
        self._rebuild_search_matrix()

    def _save_words(self):
        """Save word list to JSON sidecar."""
        if self._db_path:
            import json
            d = Path(self._db_path)
            (d / '_words.json').write_text(json.dumps({
                'words': self._words,
                'count': len(self._words),
            }))

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
            # In bulk mode, keep one long transaction open
            if not self._bulk_mode:
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
                    self._nid_to_word_cache = None  # invalidate
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

            if not self._bulk_mode:
                self.db.end_batch()

            # Lazy reindex: mark dirty, rebuild before next ask()
            if len(self._words) != dim_before:
                self._search_dirty = True

            # In bulk mode, periodic commit to avoid unbounded WAL
            if self._bulk_mode:
                self._bulk_count = getattr(self, '_bulk_count', 0) + 1
                if self._bulk_count % 500 == 0:
                    self.db.db.commit()

        return [n.id for n in neurons]

    def begin_bulk(self):
        """Enter bulk feed mode. Skips reindex + batches commits."""
        self._bulk_mode = True
        self._bulk_dirty = False
        self._bulk_count = 0
        self.db.begin_batch()  # one long transaction

    def end_bulk(self):
        """Exit bulk feed mode. Commits + runs one reindex for everything taught."""
        self._bulk_mode = False
        self.db.end_batch()  # final commit
        if self._bulk_dirty:
            self._rebuild_search_matrix()
            self._bulk_dirty = False
        self._save_cache()  # cache for fast boot next time

    def teach_batch(self, sentences: list, confidence: float = 0.5) -> list:
        """Teach multiple sentences. Sequential — lock serializes anyway."""
        return [self.teach(s, confidence) for s in sentences]

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
          1. Lazy reindex if dirty (teaches since last ask)
          2. Encode question → vector
          3. Find nearest neurons in DB
          4. Disambiguate via sentence table
          4. Output answer (template or concept list)
        """
        # Lazy reindex: rebuild search matrix if teaches happened since last ask
        if getattr(self, '_search_dirty', False):
            self._rebuild_search_matrix()
            self._search_dirty = False

        tokens = self._tokenize(question)
        content = [t for t in tokens if t not in FUNCTION_WORDS]
        query_vec = self._encode_sentence(question)

        if np.all(query_vec == 0) or len(self._words) == 0:
            self.db.log_miss(question, query_vec if query_vec is not None
                             else np.zeros(1, dtype=np.float32))
            return {"answer": "I don't know.", "confidence": 0.0,
                    "strategy": "abstain", "trace": "No knowledge"}

        # Parallel: search + disambiguate at the same time
        content_nids = [self._word_neurons[t] for t in content
                        if t in self._word_neurons]

        search_future = self._pool.submit(self.db.search, query_vec, 10)

        def _disambiguate():
            if not content_nids:
                return None
            sentences = self.db.get_sentences_for_neurons(content_nids)
            if not sentences:
                return None
            scored = [(sid, len(nids)) for sid, nids in sentences.items()]
            scored.sort(key=lambda x: x[1], reverse=True)
            best_score = scored[0][1]
            result = set()
            for sid, score in scored:
                if score < best_score:
                    break
                for nid, pos in self.db.get_sentence_neurons(sid):
                    result.add(nid)
            return result

        disambig_future = self._pool.submit(_disambiguate)

        neighbors = search_future.result()
        if not neighbors:
            disambig_future.cancel()
            self.db.log_miss(question, query_vec)
            return {"answer": "I don't know.", "confidence": 0.0,
                    "strategy": "abstain", "trace": "No matching neurons"}

        best_sentence_nids = disambig_future.result()

        # Collect concept neurons
        if self._nid_to_word_cache is None:
            self._nid_to_word_cache = {nid: w for w, nid in self._word_neurons.items()
                                       if not w.startswith("__")}
        nid_to_word = self._nid_to_word_cache
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
        """Find the best matching taught sentence, output in word order.

        Confidence = convergence: cosine similarity between query vector
        and answer vector. If the answer lives in the same region of the
        co-occurrence space as the query, it converged. If not, it's noise.
        """
        sentences = self.db.get_sentences_for_neurons(concept_ids)
        if not sentences:
            return None

        if self._nid_to_word_cache is None:
            self._nid_to_word_cache = {nid: w for w, nid in self._word_neurons.items()
                                       if not w.startswith("__")}
        nid_to_word = self._nid_to_word_cache

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

        # Convergence check: does the answer map back to the query?
        answer_text = " ".join(words)
        answer_vec = self._encode_sentence(answer_text)
        q_norm = np.linalg.norm(query_vec)
        a_norm = np.linalg.norm(answer_vec)
        if q_norm > 0 and a_norm > 0:
            convergence = float(np.dot(query_vec, answer_vec) / (q_norm * a_norm))
        else:
            convergence = 0.0

        # No convergence = no answer. The space decides, not a threshold.
        if convergence <= 0:
            return None

        return {
            "answer": answer_text,
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

    # --- Matrix operations ---

    def _learn_word(self, word: str) -> int:
        """Add a word to the brain. Returns its index."""
        word = word.lower().strip()
        if word in self._word_idx:
            return self._word_idx[word]

        idx = len(self._words)
        self._words.append(word)
        self._word_idx[word] = idx

        if self._embed_dim > 0:
            # Fixed-dim: start with a sparse identity-like vector
            # (one-hot at a deterministic position based on word hash)
            # This ensures unrelated words start orthogonal, not random-close
            new_vec = np.zeros(self._embed_dim, dtype=np.float32)
            h = hash(word) % self._embed_dim
            new_vec[h] = 1.0
            # Add small noise so same-hash words aren't identical
            new_vec += np.random.randn(self._embed_dim).astype(np.float32) * 0.01
            new_vec /= (np.linalg.norm(new_vec) + 1e-10)
            if self._embeddings is None:
                self._embeddings = new_vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, new_vec.reshape(1, -1)])
        else:
            # N×N mode — mmap is pre-allocated, in-memory needs growth
            if self._use_mmap:
                # mmap already allocated at MAX_WORDS — just set diagonal
                if idx < self._matrix_capacity:
                    self._matrix[idx, idx] = 1.0
                # Save updated word list periodically
                if idx % 100 == 0:
                    self._save_words()
            elif self._matrix is None:
                self._matrix = np.zeros((256, 256), dtype=np.float32)
                self._matrix[0, 0] = 1.0
                self._matrix_capacity = 256
            else:
                cap = self._matrix_capacity
                if idx >= cap:
                    new_cap = max(cap * 2, idx + 256)
                    new_matrix = np.zeros((new_cap, new_cap), dtype=np.float32)
                    n = self._matrix.shape[0]
                    new_matrix[:n, :n] = self._matrix[:n, :n]
                    self._matrix = new_matrix
                    self._matrix_capacity = new_cap
                self._matrix[idx, idx] = 1.0

        return idx

    def _load_matrix(self):
        """Rebuild embeddings/matrix from persisted word mappings and neuron vectors."""
        words_in_db = sorted(
            [(w, nid) for w, nid in self._word_neurons.items()
             if not w.startswith("__")],
            key=lambda x: x[1]
        )
        if not words_in_db:
            return

        for word, nid in words_in_db:
            if word not in self._word_idx:
                idx = len(self._words)
                self._words.append(word)
                self._word_idx[word] = idx

        n = len(self._words)
        if n == 0:
            return

        # Bulk load neuron vectors
        nid_to_idx = {}
        for word, nid in words_in_db:
            idx = self._word_idx.get(word)
            if idx is not None:
                nid_to_idx[nid] = idx

        rows = self.db.db.execute("SELECT id, vector FROM neurons").fetchall()

        if self._embed_dim > 0:
            # Fixed-dim mode: load vectors directly as embeddings
            self._embeddings = np.random.randn(n, self._embed_dim).astype(np.float32)
            # Normalize initial random vectors
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            self._embeddings /= (norms + 1e-10)
            # Overwrite with stored vectors where they fit
            for nid, vec_bytes in rows:
                idx = nid_to_idx.get(nid)
                if idx is None:
                    continue
                vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
                if len(vec) == self._embed_dim:
                    self._embeddings[idx] = vec
        else:
            # Legacy N×N mode
            cap = max(n * 2, 256)
            self._matrix = np.zeros((cap, cap), dtype=np.float32)
            for i in range(n):
                self._matrix[i, i] = 1.0
            self._matrix_capacity = cap
            for nid, vec_bytes in rows:
                idx = nid_to_idx.get(nid)
                if idx is None:
                    continue
                vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
                if len(vec) <= n:
                    self._matrix[idx, :len(vec)] = vec
                    self._matrix[:len(vec), idx] = vec

        self._rebuild_search_matrix()

    def _cache_dir(self):
        """Cache directory — same as DB path."""
        if self._db_path:
            return Path(self._db_path)
        return None

    def _save_cache(self):
        """Save embeddings/matrix + words to .npy/.txt for fast boot."""
        d = self._cache_dir()
        if d is None or len(self._words) == 0:
            return
        try:
            n = len(self._words)
            # Save the active data
            if self._embed_dim > 0 and self._embeddings is not None:
                np.save(d / '_matrix_cache.npy', self._embeddings[:n])
            elif self._matrix is not None:
                np.save(d / '_matrix_cache.npy', self._matrix[:n, :n])
            # Save word list
            (d / '_words_cache.txt').write_text('\n'.join(self._words))
            # Save search matrix if it exists
            if self.db._vectors is not None:
                np.save(d / '_vectors_cache.npy', self.db._vectors)
                # Save ID mappings
                import json
                (d / '_idmap_cache.json').write_text(json.dumps({
                    'id_to_row': {str(k): v for k, v in self.db._id_to_row.items()},
                    'row_to_id': {str(k): v for k, v in self.db._row_to_id.items()},
                }))
            # Version = word count + hash of word list
            word_hash = hashlib.md5('\n'.join(self._words).encode()).hexdigest()
            (d / '_cache_version.txt').write_text(f"{n}:{word_hash}")
        except Exception:
            pass  # cache is optional, never crash on save failure

    def _load_cache(self) -> bool:
        """Try to load matrix from cache. Returns True if successful."""
        d = self._cache_dir()
        if d is None:
            return False

        version_file = d / '_cache_version.txt'
        matrix_file = d / '_matrix_cache.npy'
        words_file = d / '_words_cache.txt'
        vectors_file = d / '_vectors_cache.npy'
        idmap_file = d / '_idmap_cache.json'

        if not all(f.exists() for f in [version_file, matrix_file, words_file]):
            return False

        try:
            # Check version matches DB — count + hash
            version_str = version_file.read_text().strip()
            if ':' not in version_str:
                return False  # old format
            cached_n, cached_hash = version_str.split(':', 1)
            cached_n = int(cached_n)

            db_words = sorted(w for w in self._word_neurons if not w.startswith("__"))
            if len(db_words) != cached_n:
                return False

            # Load and verify word list
            words = words_file.read_text().strip().split('\n')
            if len(words) != cached_n:
                return False

            word_hash = hashlib.md5('\n'.join(words).encode()).hexdigest()
            if word_hash != cached_hash:
                return False  # words changed

            # Load matrix
            matrix_data = np.load(matrix_file)
            if matrix_data.shape[0] != cached_n:
                return False

            # Success — populate state
            self._words = words
            self._word_idx = {w: i for i, w in enumerate(words)}
            n = len(words)

            if self._embed_dim > 0 and matrix_data.ndim == 2 and matrix_data.shape[1] == self._embed_dim:
                # Fixed-dim: load embeddings directly
                self._embeddings = matrix_data
            elif self._embed_dim == 0 and matrix_data.ndim == 2 and matrix_data.shape[0] == matrix_data.shape[1]:
                # Legacy N×N mode
                cap = max(n * 2, 256)
                self._matrix = np.zeros((cap, cap), dtype=np.float32)
                self._matrix[:n, :n] = matrix_data
                self._matrix_capacity = cap
            else:
                return False  # dimension mismatch — stale cache from different mode

            # Load search matrix if cached
            if vectors_file.exists() and idmap_file.exists():
                import json
                self.db._vectors = np.load(vectors_file)
                idmap = json.loads(idmap_file.read_text())
                self.db._id_to_row = {int(k): v for k, v in idmap['id_to_row'].items()}
                self.db._row_to_id = {int(k): v for k, v in idmap['row_to_id'].items()}
            else:
                self._rebuild_search_matrix()

            return True
        except Exception:
            return False  # cache corrupt, fall back to full rebuild

    def _save_matrix(self):
        """Persist current matrix state to neuron vectors in SQLite."""
        for word, nid in self._word_neurons.items():
            if word.startswith("__"):
                continue
            vec = self._encode_word(word)
            if np.any(vec != 0):
                self.db.db.execute(
                    "UPDATE neurons SET vector = ? WHERE id = ?",
                    (vec.tobytes(), nid)
                )
        self.db.db.commit()

    def _learn_cooccurrence(self, words: list):
        """Pull co-occurring words toward each other."""
        indices = [self._word_idx[w.lower().strip()] for w in words
                   if w.lower().strip() in self._word_idx]

        if self._embed_dim > 0:
            # Fixed-dim: pull embeddings closer (Hebbian)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]
                    va = self._embeddings[a]
                    vb = self._embeddings[b]
                    self._embeddings[a] = va + COOCCURRENCE_PULL * vb
                    self._embeddings[b] = vb + COOCCURRENCE_PULL * va
                    # Renormalize
                    na = np.linalg.norm(self._embeddings[a])
                    nb = np.linalg.norm(self._embeddings[b])
                    if na > 0:
                        self._embeddings[a] /= na
                    if nb > 0:
                        self._embeddings[b] /= nb
        else:
            # Legacy N×N mode
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]
                    self._matrix[a, b] += COOCCURRENCE_PULL
                    self._matrix[b, a] += COOCCURRENCE_PULL

    def _encode_word(self, word: str) -> np.ndarray:
        """Get a word's vector."""
        word = word.lower().strip()
        idx = self._word_idx.get(word)

        if self._embed_dim > 0:
            # Fixed-dim: return the embedding directly
            if idx is None or self._embeddings is None:
                return np.zeros(self._embed_dim, dtype=np.float32)
            return self._embeddings[idx].copy()
        else:
            # Legacy N×N mode
            n = len(self._words)
            if idx is None or n == 0:
                return np.zeros(n or 1, dtype=np.float32)
            vec = self._matrix[idx, :n].copy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

    def _encode_sentence(self, text: str) -> np.ndarray:
        """Encode a sentence as weighted average of word vectors. In-place accumulation."""
        tokens = self._tokenize(text)
        d = self._embed_dim if self._embed_dim > 0 else (len(self._words) or 1)
        if not tokens:
            return np.zeros(d, dtype=np.float32)

        result = np.zeros(d, dtype=np.float32)
        total_weight = 0.0
        for i, token in enumerate(tokens):
            token = token.lower().strip()
            idx = self._word_idx.get(token)
            if idx is not None:
                if self._embed_dim > 0 and self._embeddings is not None:
                    vec = self._embeddings[idx]
                elif self._matrix is not None:
                    n = len(self._words)
                    vec = self._matrix[idx, :n]
                else:
                    continue
                if np.any(vec != 0):
                    w = 1.0 / (1.0 + 0.1 * i)
                    result += w * vec  # accumulate directly, no copy
                    total_weight += w

        if total_weight > 0:
            result /= total_weight
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
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
        """Rebuild the DB's search matrix from current embeddings/matrix."""
        words = [(w, nid) for w, nid in self._word_neurons.items()
                 if not w.startswith("__")]
        if not words:
            self.db._vectors = None
            self.db._id_to_row = {}
            self.db._row_to_id = {}
            return

        vectors_list = []
        id_to_row = {}
        row_to_id = {}

        for word, nid in words:
            vec = self._encode_word(word)
            if np.any(vec != 0):
                row_idx = len(vectors_list)
                vectors_list.append(vec)
                id_to_row[nid] = row_idx
                row_to_id[row_idx] = nid

        if vectors_list:
            self.db._vectors = np.array(vectors_list, dtype=np.float32)
        else:
            self.db._vectors = None
        self.db._id_to_row = id_to_row
        self.db._row_to_id = row_to_id

        # Batch update DB vectors
        if self.db._vectors is not None:
            updates = [(self.db._vectors[id_to_row[nid]].tobytes(), nid)
                       for word, nid in words if nid in id_to_row]
            self.db.db.executemany(
                "UPDATE neurons SET vector = ? WHERE id = ?", updates)
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

        # Find strongest relationships
        connections = []
        if self._embed_dim > 0 and self._embeddings is not None:
            # Fixed-dim: cosine similarity to all other words
            vec = self._embeddings[idx]
            sims = self._embeddings @ vec
            for i, sim in enumerate(sims):
                if i != idx and sim > 0.1:
                    connections.append((self._words[i], float(sim)))
        elif self._matrix is not None:
            n = len(self._words)
            for i in range(n):
                val = self._matrix[idx, i]
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

    def health(self) -> dict:
        """
        Self-awareness: how much resource am I using?
        Returns CPU, memory, DB size, matrix size, disk free.
        The brain should know its own cost and not exploit the machine.
        """
        h = self.db.health()

        # Add brain-level matrix stats
        matrix_bytes = 0
        if self._embeddings is not None:
            matrix_bytes = self._embeddings.nbytes
        elif self._matrix is not None:
            matrix_bytes = self._matrix.nbytes
        h["brain_matrix_mb"] = round(matrix_bytes / (1024 * 1024), 2)
        h["words"] = len(self._words)
        h["templates"] = len(self._templates)

        # System-wide memory check — OOM awareness
        try:
            with open('/proc/meminfo') as f:
                meminfo = f.read()
            for line in meminfo.splitlines():
                if line.startswith('MemAvailable:'):
                    avail_kb = int(line.split()[1])
                    h["system_avail_mb"] = round(avail_kb / 1024, 1)
                    h["system_avail_pct"] = round(avail_kb / 1024 / (h.get("rss_mb", 1) + avail_kb / 1024) * 100, 1)
                    break
        except Exception:
            h["system_avail_mb"] = -1

        # Pressure signals
        h["memory_pressure"] = h["rss_mb"] > 512
        h["oom_risk"] = h.get("system_avail_mb", 9999) < 1024
        h["disk_pressure"] = h["disk_free_gb"] < 1.0
        h["matrix_pressure"] = len(self._words) > 10000

        # CPU load
        try:
            load1, load5, load15 = os.getloadavg()
            ncpu = os.cpu_count() or 1
            h["load_1m"] = round(load1, 2)
            h["load_5m"] = round(load5, 2)
            h["load_15m"] = round(load15, 2)
            h["cpu_count"] = ncpu
            h["cpu_saturated"] = load1 > ncpu * 0.9
        except Exception:
            h["load_1m"] = -1
            h["cpu_saturated"] = False

        # Swap
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('SwapTotal:'):
                        h["swap_total_mb"] = int(line.split()[1]) / 1024
                    elif line.startswith('SwapFree:'):
                        h["swap_free_mb"] = int(line.split()[1]) / 1024
        except Exception:
            pass
        swap_used = h.get("swap_total_mb", 0) - h.get("swap_free_mb", 0)
        h["swap_used_mb"] = round(swap_used, 1)
        h["swapping"] = swap_used > 100  # using >100MB swap = thrashing

        # Death risk score: 0 = healthy, 100 = about to die
        # Weighted across all vitals
        risk = 0
        avail = h.get("system_avail_mb", 9999)
        if avail < 2048:
            risk += int((2048 - avail) / 2048 * 40)   # 0-40 from RAM
        if h.get("swapping"):
            risk += min(25, int(swap_used / 500 * 25)) # 0-25 from swap
        if h["disk_free_gb"] < 5:
            risk += int((5 - h["disk_free_gb"]) / 5 * 20)  # 0-20 from disk
        if h.get("cpu_saturated"):
            risk += 10                                  # +10 from CPU saturation
        if h["rss_mb"] > 512:
            risk += min(5, int((h["rss_mb"] - 512) / 1024 * 5))  # 0-5 from process bloat
        h["death_risk"] = min(100, risk)

        return h

    def close(self):
        self._pool.shutdown(wait=False)
        self._batch_pool.shutdown(wait=False)
        if self._use_mmap and self._matrix is not None and hasattr(self._matrix, 'flush'):
            self._matrix.flush()
            self._save_words()
        else:
            self._save_matrix()
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
