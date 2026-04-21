"""
BrainLMDB: adapter that loads a trained LMDB model into BrainCore.

The LMDB was created by feed.py with this layout:
  words/        key: word (utf-8)       -> neuron_id (4B int32)
  cooccurrence/ key: word_idx (4B int32) -> packed [(int32 neighbor, float32 weight), ...]
  sentences/    key: sentence_id (4B)   -> packed [(int32 neuron_id, int32 position), ...]
  sent_index/   key: neuron_id (4B)     -> packed [int32 sentence_id, ...]
  successors/   key: int32              -> packed [(int32 id, float32 conf), ...]
  neurons/      key: int32              -> packed struct (float32 conf, int64 ts, bool temporal, int8 level)
  meta/         key: bytes              -> bytes

Uses SparseConvergenceLoop for reasoning -- no dense vectors, no NeuronDB.
"""

import os
import struct
import re
import math
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

import lmdb

# Must be importable from src/
import sys
sys.path.insert(0, os.path.dirname(__file__))

from neuron import Neuron, Level, DEFAULT_CONFIDENCE
from sparse_convergence import (
    SparseConvergenceLoop, SparseMultiHop,
    sparse_cosine, sparse_blend, sparse_norm, sparse_normalize,
)

# Co-occurrence pull strength (matches brain_core.py)
COOCCURRENCE_PULL = 0.3

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
    "born", "died", "has", "had", "have", "been", "became",
    "the", "a", "an", "of", "in", "for", "to", "on", "at", "by",
    "with", "from", "as", "and", "or", "but",
})

# Struct formats matching feed.py
_ID_FMT = struct.Struct('<i')
_ID_CONF_FMT = struct.Struct('<if')
_SENT_ENTRY_FMT = struct.Struct('<ii')
_NEURON_FMT = struct.Struct('<fq?b')


class _LMDBReader:
    """Lightweight reader for LMDB sub-databases.

    Provides the methods that BrainLMDB needs from the DB layer:
    get_sentences_for_neurons, get_sentence_neurons, get (neuron),
    log_miss, count, health, close.

    No NeuronDB dependency. No dense vectors.
    """

    def __init__(self, env, sentences_db, sent_index_db,
                 neurons_db, successors_db):
        self._env = env
        self._sentences_db = sentences_db
        self._sent_index_db = sent_index_db
        self._neurons_db = neurons_db
        self._successors_db = successors_db
        self._neuron_count = None

    def get(self, neuron_id: int):
        """Load a single neuron from LMDB. Returns Neuron or None."""
        key = _ID_FMT.pack(neuron_id)
        with self._env.begin() as txn:
            meta_bytes = txn.get(key, db=self._neurons_db)
            if meta_bytes is None:
                return None
            conf, ts, temporal, level = _NEURON_FMT.unpack(meta_bytes)

            succ_bytes = txn.get(key, db=self._successors_db)
            successors = []
            if succ_bytes:
                sz = _ID_CONF_FMT.size
                for i in range(0, len(succ_bytes), sz):
                    sid, sconf = _ID_CONF_FMT.unpack(succ_bytes[i:i+sz])
                    successors.append((sid, sconf))

        # Provide a zero vector -- callers that need sparse ops use cooc directly
        import numpy as np
        return Neuron(
            id=neuron_id,
            vector=np.zeros(1, dtype=np.float32),
            confidence=conf,
            successors=successors,
            predecessors=[],
            timestamp=ts,
            temporal=temporal,
            level=Level(level) if 0 <= level <= 2 else Level.WORD,
        )

    def get_sentences_for_neurons(self, neuron_ids: list) -> dict:
        """Return {sentence_id: [(neuron_id, position), ...]} for sentences
        containing any of the given neuron_ids."""
        if not neuron_ids:
            return {}

        sentences = {}
        with self._env.begin() as txn:
            all_sids = set()
            for nid in neuron_ids:
                data = txn.get(_ID_FMT.pack(nid), db=self._sent_index_db)
                if data:
                    sz = _ID_FMT.size
                    for i in range(0, len(data), sz):
                        all_sids.add(_ID_FMT.unpack(data[i:i+sz])[0])

            for sid in sorted(all_sids):
                data = txn.get(_ID_FMT.pack(sid), db=self._sentences_db)
                if data:
                    entries = []
                    sz = _SENT_ENTRY_FMT.size
                    for i in range(0, len(data), sz):
                        entries.append(_SENT_ENTRY_FMT.unpack(data[i:i+sz]))
                    sentences[sid] = entries

        return sentences

    def get_sentence_neurons(self, sentence_id: int) -> list:
        """Return [(neuron_id, position), ...] for a sentence."""
        with self._env.begin() as txn:
            data = txn.get(_ID_FMT.pack(sentence_id), db=self._sentences_db)
        if not data:
            return []
        entries = []
        sz = _SENT_ENTRY_FMT.size
        for i in range(0, len(data), sz):
            entries.append(_SENT_ENTRY_FMT.unpack(data[i:i+sz]))
        return entries

    def count(self) -> int:
        """Count neurons in the DB."""
        if self._neuron_count is not None:
            return self._neuron_count
        with self._env.begin() as txn:
            stat = txn.stat(db=self._neurons_db)
            self._neuron_count = stat['entries']
        return self._neuron_count

    def log_miss(self, question: str, query_vec=None):
        """No-op for read-only LMDB. Misses not persisted."""
        pass

    def begin_batch(self):
        pass

    def end_batch(self):
        pass

    def insert(self, *args, **kwargs):
        raise NotImplementedError("BrainLMDB is read-only for neuron inserts")

    def save_word_mapping(self, *args, **kwargs):
        pass

    def update_successors(self, *args, **kwargs):
        pass

    def update_predecessors(self, *args, **kwargs):
        pass

    def record_sentence(self, *args, **kwargs):
        pass

    def resolve_miss_by_query(self, *args, **kwargs):
        pass

    def health(self) -> dict:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = rusage.ru_maxrss / 1024

        disk_free_bytes = 0
        try:
            st = os.statvfs('/')
            disk_free_bytes = st.f_bavail * st.f_frsize
        except Exception:
            pass

        return {
            "neurons": self.count(),
            "rss_mb": round(rss_mb, 1),
            "disk_free_gb": round(disk_free_bytes / (1024 ** 3), 2),
            "backend": "lmdb",
        }

    def close(self):
        pass  # env closed by BrainLMDB


class BrainLMDB:
    """Brain backed by LMDB data from feed.py.

    Loads co-occurrence, words, sentences, successors directly from LMDB.
    Uses SparseConvergenceLoop for multi-hop reasoning.
    No dense vectors. No NeuronDB dependency.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.expanduser('~/nexus-brain')

        self._db_path = db_path

        lmdb_path = os.path.join(db_path, 'brain.lmdb')
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"No brain.lmdb at {lmdb_path}")

        # Core data structures
        self._words = []
        self._word_idx = {}
        self._cooc = {}
        self._word_neurons = {}
        self._templates = []
        self._nid_to_word_cache = None

        # Thread pools
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._batch_pool = ThreadPoolExecutor(max_workers=8)
        self._lock = Lock()

        # Open LMDB (read-only)
        self._env = lmdb.open(lmdb_path, max_dbs=10, readonly=True, lock=False,
                              map_size=4 * 1024 * 1024 * 1024)
        self._words_db = self._env.open_db(b'words')
        self._sentences_db = self._env.open_db(b'sentences')
        self._sent_index_db = self._env.open_db(b'sent_index')
        self._successors_db = self._env.open_db(b'successors')
        self._neurons_db = self._env.open_db(b'neurons')

        # LMDB reader for sentence/neuron lookups
        self.db = _LMDBReader(self._env, self._sentences_db,
                              self._sent_index_db, self._neurons_db,
                              self._successors_db)

        # Load word mappings (fast key-value scan)
        with self._env.begin(db=self._words_db) as txn:
            cursor = txn.cursor(db=self._words_db)
            for key, val in cursor:
                word = key.decode('utf-8')
                nid = _ID_FMT.unpack(val)[0]
                self._word_neurons[word] = nid

        # Build word list sorted by neuron_id for consistent ordering
        sorted_words = sorted(self._word_neurons.items(), key=lambda x: x[1])
        for word, nid in sorted_words:
            idx = len(self._words)
            self._words.append(word)
            self._word_idx[word] = idx
            self._cooc[idx] = {idx: 1.0}  # self-connection

        # Lazy co-occurrence from LMDB
        self._cooc_db_handle = self._env.open_db(b'cooccurrence')
        self._cooc_loaded = set()

        # Sparse convergence loop
        self._convergence = SparseConvergenceLoop(
            cooc=self._cooc,
            word_idx=self._word_idx,
            words=self._words,
            word_neurons=self._word_neurons,
            ensure_cooc=self._ensure_cooc,
            max_hops=10, k=5,
            convergence_threshold=0.99,
            min_confidence=0.1,
            min_relevance=0.3,
        )
        self._multi_hop = SparseMultiHop(
            self._convergence, max_rounds=3, concept_blend_weight=0.4,
        )

        print(f"BrainLMDB loaded: {len(self._words):,} words (lazy cooc, sparse convergence)")

    # --- Lazy co-occurrence loading ---

    # Max edges to load per word (top-K by weight). Prevents memory explosion.
    _MAX_EDGES_PER_WORD = 50

    @staticmethod
    def _parse_cooc_topk(val: bytes, k: int) -> list:
        """Parse packed cooc bytes, return top-k (neighbor, weight) by weight.

        Uses numpy for bulk unpack -- 10x faster than struct loop for 2K+ edges.
        """
        import numpy as np
        sz = _ID_CONF_FMT.size  # 8 bytes
        n_edges = len(val) // sz
        if n_edges == 0:
            return []
        if n_edges <= k:
            # Small enough to just unpack all
            return [_ID_CONF_FMT.unpack(val[i:i+sz]) for i in range(0, len(val), sz)]
        # Bulk unpack with numpy
        arr = np.frombuffer(val, dtype=np.dtype([('id', '<i4'), ('weight', '<f4')]))
        # Partial sort: get indices of top-k by weight
        top_indices = np.argpartition(-arr['weight'], k)[:k]
        result = [(int(arr['id'][i]), float(arr['weight'][i])) for i in top_indices]
        return result

    def _ensure_cooc(self, word_idx: int):
        """Lazy-load top-K co-occurrence edges for a word from LMDB."""
        if word_idx in self._cooc_loaded:
            return
        self._cooc_loaded.add(word_idx)

        with self._env.begin(db=self._cooc_db_handle) as txn:
            val = txn.get(_ID_FMT.pack(word_idx), db=self._cooc_db_handle)
            if val is None:
                return
            top_edges = self._parse_cooc_topk(val, self._MAX_EDGES_PER_WORD)
            if word_idx not in self._cooc:
                self._cooc[word_idx] = {word_idx: 1.0}
            for neighbor, weight in top_edges:
                self._cooc[word_idx][neighbor] = weight

    def _ensure_cooc_batch(self, word_indices: list):
        """Lazy-load top-K co-occurrence for multiple words at once."""
        needed = [idx for idx in word_indices if idx not in self._cooc_loaded]
        if not needed:
            return
        with self._env.begin(db=self._cooc_db_handle) as txn:
            for word_idx in needed:
                self._cooc_loaded.add(word_idx)
                val = txn.get(_ID_FMT.pack(word_idx), db=self._cooc_db_handle)
                if val is None:
                    continue
                top_edges = self._parse_cooc_topk(val, self._MAX_EDGES_PER_WORD)
                if word_idx not in self._cooc:
                    self._cooc[word_idx] = {word_idx: 1.0}
                for neighbor, weight in top_edges:
                    self._cooc[word_idx][neighbor] = weight

    # --- Sparse operations ---

    @staticmethod
    def _tokenize(text: str) -> list:
        return re.findall(r'[a-z0-9]+', text.lower())

    def _sparse_norm(self, d: dict) -> float:
        return sparse_norm(d)

    def _sparse_cosine(self, a: dict, b: dict) -> float:
        return sparse_cosine(a, b)

    def _sparse_blend(self, word_indices: list, weights: list = None) -> dict:
        self._ensure_cooc_batch(word_indices)
        profiles = [self._cooc.get(idx, {}) for idx in word_indices]
        return sparse_blend(profiles, weights)

    def _sparse_search(self, query_cooc: dict, k: int = 5) -> list:
        """Search by scoring words present in query profile.

        Phase 1: Each word_idx in query_cooc gets its weight as a fast
        relevance score — O(|query_cooc|), no extra LMDB reads.
        Phase 2: Refine top 2*k candidates with full cosine.
        """
        if not query_cooc:
            return []

        # Phase 1: fast scoring from query profile weights
        fast_scores = [(widx, weight) for widx, weight in query_cooc.items()
                       if weight > 0]
        fast_scores.sort(key=lambda x: x[1], reverse=True)

        # Phase 2: refine top candidates with cosine
        refine_count = min(2 * k, len(fast_scores))
        top_candidates = fast_scores[:refine_count]
        if not top_candidates:
            return []

        self._ensure_cooc_batch([widx for widx, _ in top_candidates])

        q_norm = sparse_norm(query_cooc)
        if q_norm == 0:
            return []

        scores = []
        for word_idx, _ in top_candidates:
            word_cooc = self._cooc.get(word_idx, {})
            if not word_cooc:
                continue
            dot = sum(query_cooc.get(j, 0) * v for j, v in word_cooc.items())
            if dot > 0:
                w_norm = sparse_norm(word_cooc)
                if w_norm > 0:
                    sim = dot / (q_norm * w_norm)
                    scores.append((word_idx, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # --- Querying ---

    def ask(self, question: str) -> dict:
        """Ask the brain a question using sparse convergence.

        Flow:
          1. Tokenize query, get word indices
          2. Multi-hop sparse convergence (iterative search over cooc graph)
          3. Sparse co-occurrence search (complementary)
          4. Sentence disambiguation
          5. Generate answer
        """
        tokens = self._tokenize(question)
        content = [t for t in tokens if t not in FUNCTION_WORDS]

        content_indices = [self._word_idx[t] for t in content
                           if t in self._word_idx]
        content_nids = [self._word_neurons[t] for t in content
                        if t in self._word_neurons]

        if not content_indices or len(self._words) == 0:
            return {"answer": "I don't know.", "confidence": 0.0,
                    "strategy": "abstain", "trace": "No known words in query",
                    "convergence_trace": "", "converged": False,
                    "convergence_rounds": 0}

        # --- Multi-hop sparse convergence ---
        multi_hop_result = self._multi_hop.reason(content_indices)
        convergence_trace = multi_hop_result.trace()

        # Build nid->word lookup
        if self._nid_to_word_cache is None:
            self._nid_to_word_cache = {nid: w for w, nid in self._word_neurons.items()
                                       if not w.startswith("__")}
        nid_to_word = self._nid_to_word_cache
        word_to_nid = {w: nid for nid, w in nid_to_word.items()}

        # Collect concepts from convergence
        concepts = []
        seen = set()
        for widx, sim in multi_hop_result.concepts:
            if widx < len(self._words):
                word = self._words[widx]
                nid = word_to_nid.get(word)
                if nid and nid not in seen and word not in FUNCTION_WORDS:
                    n = self.db.get(nid)
                    if n:
                        concepts.append((n, word))
                        seen.add(nid)

        # --- Sparse co-occurrence search (complementary) ---
        weights = [1.0 / (1.0 + 0.1 * i) for i in range(len(content_indices))]
        query_cooc = self._sparse_blend(content_indices, weights)
        search_results = self._sparse_search(query_cooc, k=10)

        # Positional encoding
        query_positions = {}
        for i, t in enumerate(content):
            nid = self._word_neurons.get(t)
            if nid is not None:
                query_positions[nid] = i

        nid_taught_positions = {}
        if content_nids:
            all_sentences = self.db.get_sentences_for_neurons(content_nids)
            for sid, nid_pos_list in all_sentences.items():
                for nid, pos in nid_pos_list:
                    if nid not in nid_taught_positions:
                        nid_taught_positions[nid] = set()
                    nid_taught_positions[nid].add(pos)

        def _position_bias(nid: int) -> float:
            taught_pos = nid_taught_positions.get(nid)
            if not taught_pos or not query_positions:
                return 1.0
            total_sim = 0.0
            count = 0
            for q_nid, q_pos in query_positions.items():
                min_dist = min(abs(q_pos - tp) for tp in taught_pos)
                total_sim += 1.0 / (1.0 + min_dist)
                count += 1
            if count == 0:
                return 1.0
            avg_sim = total_sim / count
            return 1.0 + 0.5 * avg_sim

        # Apply position bias to search results
        if nid_taught_positions:
            biased_results = []
            for word_idx, sim in search_results:
                if word_idx < len(self._words):
                    word = self._words[word_idx]
                    nid = word_to_nid.get(word)
                    if nid:
                        biased_results.append((word_idx, sim * _position_bias(nid)))
                    else:
                        biased_results.append((word_idx, sim))
                else:
                    biased_results.append((word_idx, sim))
            biased_results.sort(key=lambda x: x[1], reverse=True)
            search_results = biased_results

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
            return {"answer": "I don't know.", "confidence": 0.0,
                    "strategy": "abstain", "trace": "No relevant concepts",
                    "convergence_trace": convergence_trace,
                    "converged": multi_hop_result.converged,
                    "convergence_rounds": len(multi_hop_result.rounds)}

        if best_sentence_nids and len(content_nids) >= 2:
            concepts = [(n, w) for n, w in concepts
                        if n.id in best_sentence_nids
                        or n.id in set(content_nids)]

        result = self._generate_answer(concepts, content_indices, tokens, question)
        result["convergence_trace"] = convergence_trace
        result["converged"] = multi_hop_result.converged
        result["convergence_rounds"] = len(multi_hop_result.rounds)
        return result

    # --- Answer Generation ---

    def _generate_answer(self, concepts, query_indices, query_tokens, question) -> dict:
        concept_neurons = [n for n, w in concepts]
        concept_words = [w for n, w in concepts]
        concept_ids = [n.id for n, w in concepts]

        template_future = self._pool.submit(
            self._try_template,
            concept_neurons, concept_words, concept_ids, query_indices, query_tokens
        )
        chain_future = self._pool.submit(
            self._try_sentence_chain, concept_ids, query_indices
        )

        template_result = template_future.result()
        chain_result = chain_future.result()

        if template_result:
            return template_result
        if chain_result:
            return chain_result

        import numpy as np
        avg_conf = float(np.mean([n.confidence for n in concept_neurons]))
        return {
            "answer": " ".join(concept_words),
            "confidence": avg_conf,
            "strategy": "concept_list",
            "trace": f"Concepts: {concept_words}",
        }

    def _try_template(self, neurons, words, nids, query_indices, query_tokens) -> dict:
        if not self._templates:
            return None
        query_word_set = set(query_tokens)
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
        sentence_order = self._get_sentence_order(nids)
        if sentence_order:
            ordered = sorted(zip(neurons, words),
                             key=lambda p: sentence_order.get(p[0].id, 999))
        else:
            ordered = list(zip(neurons, words))

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
        for name in slots:
            if name not in fills:
                text = text.replace(f"[{name}]", "...")

        # Convergence check: sparse cosine between answer and query profiles
        answer_tokens = [t for t in self._tokenize(text) if t in self._word_idx]
        answer_indices = [self._word_idx[t] for t in answer_tokens]
        if not answer_indices:
            return None

        answer_cooc = self._sparse_blend(answer_indices)
        query_cooc = self._sparse_blend(query_indices)
        convergence = sparse_cosine(query_cooc, answer_cooc)

        if convergence <= 0:
            return None
        return {
            "answer": text,
            "confidence": convergence,
            "strategy": "template",
            "trace": f"Template: {pattern}, fills: {fills}, convergence={convergence:.3f}",
        }

    def _try_sentence_chain(self, concept_ids, query_indices) -> dict:
        sentences = self.db.get_sentences_for_neurons(concept_ids)
        if not sentences:
            return None

        nid_to_word = self._nid_to_word_cache or {
            nid: w for w, nid in self._word_neurons.items()
            if not w.startswith("__")
        }

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

        ordered = sorted(sent_neurons, key=lambda x: x[1])
        words = [nid_to_word.get(nid, "") for nid, pos in ordered]
        words = [w for w in words if w]

        if len(words) < 2:
            return None

        answer_indices = [self._word_idx[w] for w in words if w in self._word_idx]
        if not answer_indices:
            return None

        answer_cooc = self._sparse_blend(answer_indices)

        # Build query cooc from concept nids -> word indices
        query_words = [nid_to_word.get(nid) for nid in concept_ids]
        q_indices = [self._word_idx[w] for w in query_words
                     if w and w in self._word_idx]
        if not q_indices:
            q_indices = list(query_indices)

        query_cooc = self._sparse_blend(q_indices)
        convergence = sparse_cosine(query_cooc, answer_cooc)

        if convergence <= 0:
            return None
        return {
            "answer": " ".join(words),
            "confidence": convergence,
            "strategy": "sentence_chain",
            "trace": f"Sentence {sid} (convergence={convergence:.3f}): {words}",
        }

    def _get_sentence_order(self, concept_ids) -> dict:
        if len(concept_ids) < 2:
            return {}
        sentences = self.db.get_sentences_for_neurons(concept_ids)
        if not sentences:
            return {}
        best_sid = max(sentences, key=lambda sid: len(sentences[sid]))
        if len(sentences[best_sid]) < 2:
            return {}
        return {nid: pos for nid, pos in self.db.get_sentence_neurons(best_sid)}

    # --- Generation ---

    def generate(self, query: str, max_tokens: int = 30, temperature: float = 0.7,
                 min_score: float = 0.01) -> dict:
        """Generate text via sparse successor walk."""
        if not self._words:
            return {"text": "", "trace": ["No vocabulary"], "tokens_generated": 0}

        ask_result = self.ask(query)
        query_tokens_set = set(self._tokenize(query))

        nid_to_word = self._nid_to_word_cache or {
            nid: w for w, nid in self._word_neurons.items()
            if not w.startswith("__")
        }

        # Find starting word from ask result
        start_word = None
        start_score = -1.0
        answer_words = self._tokenize(ask_result.get("answer", ""))
        for w in answer_words:
            if w in self._word_idx and w not in FUNCTION_WORDS and w not in query_tokens_set:
                nid = self._word_neurons.get(w)
                conf = 0.5
                if nid:
                    n = self.db.get(nid)
                    if n:
                        conf = n.confidence
                if conf > start_score:
                    start_score = conf
                    start_word = w

        # Fallback: best connected word to query
        if start_word is None:
            query_content = [t for t in self._tokenize(query) if t in self._word_idx]
            if query_content:
                q_indices = [self._word_idx[t] for t in query_content]
                q_cooc = self._sparse_blend(q_indices)
                search = self._sparse_search(q_cooc, k=20)
                for widx, sim in search:
                    w = self._words[widx]
                    if w not in FUNCTION_WORDS and w not in query_tokens_set:
                        start_word = w
                        start_score = sim
                        break

        if start_word is None:
            return {"text": "", "trace": ["No starting word found"], "tokens_generated": 0}

        generated = [start_word]
        trace = [{"token": start_word, "score": start_score, "reason": "start (best concept)"}]
        recent_window = 6

        start_idx = self._word_idx[start_word]
        self._ensure_cooc(start_idx)
        context_profile = dict(self._cooc.get(start_idx, {}))

        for pos in range(1, max_tokens):
            prev_word = generated[-1]

            # Get successors for sequence ordering
            prev_successors = {}
            if prev_word in self._word_neurons:
                prev_nid = self._word_neurons[prev_word]
                prev_neuron = self.db.get(prev_nid)
                if prev_neuron and prev_neuron.successors:
                    for succ_nid, succ_conf in prev_neuron.successors:
                        succ_word = nid_to_word.get(succ_nid)
                        if succ_word:
                            prev_successors[succ_word] = succ_conf

            # Score candidates from context profile directly
            # Use the fast approach: words in context_profile ARE the candidates
            # Plus successors for sequence ordering
            candidate_indices = set(context_profile.keys())
            # Add successor targets if any
            for sw in prev_successors:
                if sw in self._word_idx:
                    candidate_indices.add(self._word_idx[sw])
            # Load cooc only for top candidates by weight
            top_by_weight = sorted(
                ((idx, context_profile.get(idx, 0)) for idx in candidate_indices),
                key=lambda x: x[1], reverse=True
            )[:100]
            self._ensure_cooc_batch([idx for idx, _ in top_by_weight])
            candidate_indices = {idx for idx, _ in top_by_weight}

            scores = []
            for widx in candidate_indices:
                if widx >= len(self._words):
                    continue
                word = self._words[widx]
                if word in FUNCTION_WORDS:
                    continue
                word_cooc = self._cooc.get(widx, {})
                ctx_sim = sparse_cosine(word_cooc, context_profile)
                succ_conf = prev_successors.get(word, 0.0)
                raw_score = ctx_sim * (1.0 + succ_conf)
                if raw_score > 0:
                    scores.append((widx, word, raw_score))

            if not scores:
                trace.append({"token": "[STOP]", "score": 0, "reason": "no candidates"})
                break

            scores.sort(key=lambda x: x[2], reverse=True)
            top_n = min(50, len(scores))
            top_scores = scores[:top_n]

            if temperature > 0:
                max_raw = top_scores[0][2]
                exp_scores = []
                for widx, word, raw in top_scores:
                    exp_scores.append((widx, word, raw, math.exp((raw - max_raw) / temperature)))
                total_exp = sum(e for _, _, _, e in exp_scores)
                if total_exp > 0:
                    softmax_scores = [(widx, word, raw, e / total_exp)
                                      for widx, word, raw, e in exp_scores]
                else:
                    softmax_scores = [(widx, word, raw, 0.0)
                                      for widx, word, raw, _ in exp_scores]
            else:
                softmax_scores = [(top_scores[0][0], top_scores[0][1],
                                   top_scores[0][2], 1.0)]

            chosen = None
            recent_set = (set(generated[-recent_window:])
                          if len(generated) >= recent_window else set(generated))

            for widx, word, raw, prob in softmax_scores:
                if word in recent_set:
                    continue
                if raw < min_score:
                    continue
                chosen = (widx, word, raw, prob)
                break

            if chosen is None:
                trace.append({"token": "[STOP]", "score": 0,
                              "reason": "all candidates below threshold or in loop"})
                break

            widx, word, raw, prob = chosen
            generated.append(word)
            trace.append({"token": word, "score": round(raw, 4),
                          "prob": round(prob, 4),
                          "reason": f"ctx_sim*(1+succ), prob={prob:.3f}"})

            # Update context profile
            self._ensure_cooc(widx)
            new_cooc = self._cooc.get(widx, {})
            blend_weight = 0.3
            for k, v in new_cooc.items():
                context_profile[k] = (context_profile.get(k, 0) * (1 - blend_weight)
                                      + v * blend_weight)
            for k in list(context_profile.keys()):
                if k not in new_cooc:
                    context_profile[k] *= (1 - blend_weight)

        return {
            "text": " ".join(generated),
            "trace": trace,
            "tokens_generated": len(generated),
        }

    # --- Learning (for RLHF/benchmark compatibility) ---

    def _learn_word(self, word: str) -> int:
        word = word.lower().strip()
        if word in self._word_idx:
            return self._word_idx[word]
        idx = len(self._words)
        self._words.append(word)
        self._word_idx[word] = idx
        self._cooc[idx] = {idx: 1.0}
        return idx

    def _learn_cooccurrence(self, words: list):
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

    def teach(self, sentence: str, confidence: float = 0.5) -> list:
        """Teach a new sentence (in-memory only, not persisted to LMDB)."""
        tokens = self._tokenize(sentence)
        content = [t for t in tokens if t not in FUNCTION_WORDS]
        if not content:
            return []

        with self._lock:
            for word in content:
                self._learn_word(word)
            if len(content) >= 2:
                self._learn_cooccurrence(content)

            neurons = []
            for word in content:
                nid = self._word_neurons.get(word)
                if nid:
                    n = self.db.get(nid)
                    if n:
                        neurons.append(n)

            if len(tokens) >= 3:
                self._extract_template(tokens)

        return [n.id for n in neurons]

    def teach_batch(self, sentences, confidence=0.5):
        return [self.teach(s, confidence) for s in sentences]

    def correct(self, question: str, answer: str):
        self.teach(answer, confidence=0.6)

    # --- Template extraction ---

    def _extract_template(self, tokens: list):
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
        for p, s, v in self._templates:
            if p == pattern:
                return
        # Store None for template vector -- not used in sparse path
        self._templates.append((pattern, slots, None))

    # --- Utility ---

    def inspect(self, word: str) -> dict:
        word = word.lower().strip()
        if word not in self._word_idx:
            return {"word": word, "known": False}
        idx = self._word_idx[word]
        self._ensure_cooc(idx)
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
            "backend": "lmdb",
        }

    def health(self) -> dict:
        h = self.db.health()
        cooc_entries = sum(len(v) for v in self._cooc.values())
        h["brain_matrix_mb"] = round(cooc_entries * 50 / (1024 * 1024), 2)
        h["cooc_entries"] = cooc_entries
        h["words"] = len(self._words)
        h["templates"] = len(self._templates)
        h["memory_pressure"] = h["rss_mb"] > 512
        h["disk_pressure"] = h["disk_free_gb"] < 1.0
        h["matrix_pressure"] = len(self._words) > 10000

        risk = 0
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        avail = int(line.split()[1]) / 1024
                        h["system_avail_mb"] = round(avail, 1)
                        if avail < 2048:
                            risk += int((2048 - avail) / 2048 * 40)
        except Exception:
            pass
        h["death_risk"] = min(100, risk)
        h["word_count"] = len(self._words)
        h["neuron_count"] = self.db.count()
        h["persist_queue"] = 0

        return h

    # --- Compatibility stubs ---

    def begin_bulk(self):
        pass

    def end_bulk(self):
        pass

    def flush(self):
        pass

    def close(self):
        self._pool.shutdown(wait=False)
        self._batch_pool.shutdown(wait=False)
        self._env.close()
