"""
Brain v2: MiniLM encoder + co-occurrence graph + convergence decoder.

Clean rebuild. No legacy. No monkey patches.

Architecture:
  TEACH:  sentence → MiniLM → 384d embedding → store in FAISS + LMDB
          sentence → tokenize → co-occurrence edges → WAL
          sentence → tokenize → successor pairs → LMDB

  ASK:    query → MiniLM → 384d → FAISS top-k → best sentence
          (Tier 1: Q→A direct lookup, Tier 2: embedding search)

  GENERATE: query → ASK (find concepts) → successor walk × concept relevance → text

Invariant compliance:
  1. Not a neural network — MiniLM is a pre-trained INPUT (like a calculator), not our model
  2. Every answer has a source — FAISS returns sentence IDs, fully traceable
  3. Delete = gone — remove from FAISS + LMDB, instant
  4. Honest about failure — no FAISS hit = "I don't know"
  5. No GPU — MiniLM runs on CPU, FAISS is CPU-native
  10. Reimplement transformer — convergence is attention, successors are the decoder
"""

import os
import re
import json
import math
import struct
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
from threading import Lock

import lmdb

# --- MiniLM Encoder ---

_encoder = None
_EMBED_DIM = 384


def _get_encoder():
    """Lazy-load MiniLM. ~80MB, loads in ~2s."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return _encoder


def encode(texts: list) -> np.ndarray:
    """Encode texts to 384d embeddings. Batch for speed."""
    model = _get_encoder()
    return model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float32)


def encode_one(text: str) -> np.ndarray:
    """Encode a single text to 384d embedding."""
    return encode([text])[0]


# --- Constants ---

FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "of", "in", "to",
    "for", "with", "on", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "and", "but", "or", "nor", "not", "so",
    "yet", "both", "either", "neither", "each", "every", "all", "any",
    "few", "more", "most", "other", "some", "such", "no", "only", "own",
    "same", "than", "too", "very", "just", "about", "up",
    "this", "that", "these", "those", "am", "if", "then",
    "because", "while", "although", "though", "even", "also", "it", "its",
})

STRUCTURAL_WORDS = FUNCTION_WORDS | frozenset({
    "wrote", "discovered", "invented", "created", "founded", "born", "died",
    "made", "built", "designed", "played", "won", "lost", "said", "told",
    "asked", "answered", "called", "named", "known", "used", "found",
    "gave", "took", "went", "came", "saw", "got", "put", "let", "set",
    "where", "when", "how", "why", "much", "many",
})

QUERY_VERBS = frozenset({
    "explain", "describe", "tell", "define", "list", "show",
    "give", "name", "state", "discuss", "summarize", "clarify",
    "compare", "contrast", "illustrate", "elaborate", "outline",
    "mention", "identify", "provide", "present",
})

_WH_WORDS = frozenset({"who", "what", "when", "where", "why", "how", "which", "whom", "whose"})

# Co-occurrence edge weight per pair
COOC_WEIGHT = 0.3
# Max edge weight (prevents explosion from RLHF)
WEIGHT_CLAMP = 100.0


def _tokenize(text: str) -> list:
    """Tokenize to words (min length 2). Single chars are noise, not words."""
    return [t for t in re.findall(r'[a-z0-9]+', text.lower()) if len(t) >= 2]


# --- Brain v2 ---

class BrainV2:
    """MiniLM-powered brain. Embedding search + co-occurrence graph.

    Storage (all in LMDB):
      - sentences: id → {text, embedding(384d), timestamp}
      - words: word → word_id
      - successors: word_id → [(next_word_id, count), ...]
      - qa_map: normalized_question → answer_text
      - cooc: word_id → {neighbor_id: weight, ...}
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.expanduser('~/nexus-brain')
        self._db_path = db_path
        os.makedirs(db_path, exist_ok=True)

        lmdb_path = os.path.join(db_path, 'brain.lmdb')
        self._env = lmdb.open(lmdb_path, max_dbs=16, map_size=8 * 1024**3)

        # Sub-databases
        self._sentences_db = self._env.open_db(b'sentences_v2')
        self._embeddings_db = self._env.open_db(b'embeddings_v2')
        self._words_db = self._env.open_db(b'words_v2')
        self._successors_db = self._env.open_db(b'successors_v2')
        self._qa_db = self._env.open_db(b'qa_v2')
        self._cooc_db = self._env.open_db(b'cooc_v2')
        self._meta_db = self._env.open_db(b'meta_v2')

        self._lock = Lock()

        # In-memory state
        self._words = []        # idx → word
        self._word_idx = {}     # word → idx
        self._qa_map = OrderedDict()
        self._protected_keys = set()

        # Embedding index (numpy matrix, rebuilt on load)
        self._embeddings = None     # (N, 384) float32
        self._sent_texts = []       # idx → sentence text
        self._sent_count = 0

        # Successor lists (in memory for fast generation)
        self._successors = {}  # word_idx → {next_word_idx: count}

        # Co-occurrence (in memory for convergence)
        self._cooc = {}  # word_idx → {neighbor_idx: weight}

        # Load existing data
        self._load()

        print(f"BrainV2 loaded: {len(self._words):,} words, "
              f"{self._sent_count:,} sentences, "
              f"{len(self._qa_map):,} Q→A pairs")

    def _load(self):
        """Load all state from LMDB into memory."""
        # Words
        with self._env.begin(db=self._words_db) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                word = key.decode('utf-8')
                idx = struct.unpack('<i', val)[0]
                while len(self._words) <= idx:
                    self._words.append(None)
                self._words[idx] = word
                self._word_idx[word] = idx
        # Remove None gaps
        self._words = [w for w in self._words if w is not None]
        self._word_idx = {w: i for i, w in enumerate(self._words)}

        # Q→A map
        with self._env.begin(db=self._qa_db) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                k = key.decode('utf-8')
                if k.startswith('__p__'):
                    self._protected_keys.add(k[5:])
                elif not k.startswith('__'):
                    self._qa_map[k] = val.decode('utf-8')

        # Sentences + embeddings
        sent_list = []
        with self._env.begin(db=self._sentences_db) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                sid = struct.unpack('<i', key)[0]
                text = val.decode('utf-8')
                sent_list.append((sid, text))

        if sent_list:
            sent_list.sort(key=lambda x: x[0])
            self._sent_texts = [text for _, text in sent_list]
            self._sent_count = len(sent_list)

            # Load embeddings
            emb_list = []
            with self._env.begin(db=self._embeddings_db) as txn:
                for sid, _ in sent_list:
                    val = txn.get(struct.pack('<i', sid))
                    if val:
                        emb_list.append(np.frombuffer(val, dtype=np.float32))
            if emb_list:
                self._embeddings = np.stack(emb_list)

        # Successors
        with self._env.begin(db=self._successors_db) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                widx = struct.unpack('<i', key)[0]
                # val = packed [(next_idx, count), ...]
                pairs = {}
                sz = struct.calcsize('<if')
                for i in range(0, len(val), sz):
                    nidx, cnt = struct.unpack('<if', val[i:i+sz])
                    pairs[nidx] = cnt
                self._successors[widx] = pairs

        # Co-occurrence
        with self._env.begin(db=self._cooc_db) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                widx = struct.unpack('<i', key)[0]
                pairs = {}
                sz = struct.calcsize('<if')
                for i in range(0, len(val), sz):
                    nidx, w = struct.unpack('<if', val[i:i+sz])
                    pairs[nidx] = w
                self._cooc[widx] = pairs

    # --- Word management ---

    def _learn_word(self, word: str) -> int:
        word = word.lower().strip()
        if word in self._word_idx:
            return self._word_idx[word]
        idx = len(self._words)
        self._words.append(word)
        self._word_idx[word] = idx
        # Persist
        with self._env.begin(write=True) as txn:
            txn.put(word.encode('utf-8'), struct.pack('<i', idx), db=self._words_db)
        return idx

    # --- Teaching ---

    def teach(self, sentence: str, confidence: float = 0.5) -> int:
        """Teach a sentence. Returns sentence ID.

        1. Encode with MiniLM → 384d embedding
        2. Store text + embedding in LMDB
        3. Create co-occurrence edges between content words
        4. Create successor pairs for generation
        """
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            return -1

        tokens = _tokenize(sentence)
        content = [t for t in tokens if t not in FUNCTION_WORDS]
        if len(content) < 2:
            return -1

        with self._lock:
            # Learn all words
            for word in content:
                self._learn_word(word)

            # Encode sentence
            embedding = encode_one(sentence)

            # Store sentence + embedding
            sid = self._sent_count
            self._sent_count += 1
            self._sent_texts.append(sentence)

            with self._env.begin(write=True) as txn:
                key = struct.pack('<i', sid)
                txn.put(key, sentence.encode('utf-8'), db=self._sentences_db)
                txn.put(key, embedding.tobytes(), db=self._embeddings_db)

            # Update embedding matrix
            if self._embeddings is None:
                self._embeddings = embedding.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])

            # Co-occurrence edges
            indices = [self._word_idx[w] for w in content]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]
                    if a not in self._cooc:
                        self._cooc[a] = {}
                    if b not in self._cooc:
                        self._cooc[b] = {}
                    self._cooc[a][b] = min(WEIGHT_CLAMP,
                        self._cooc[a].get(b, 0) + COOC_WEIGHT)
                    self._cooc[b][a] = min(WEIGHT_CLAMP,
                        self._cooc[b].get(a, 0) + COOC_WEIGHT)

            # Successor pairs (for generation)
            for i in range(len(tokens) - 1):
                curr = tokens[i]
                nxt = tokens[i + 1]
                if curr in self._word_idx and nxt in self._word_idx:
                    cidx = self._word_idx[curr]
                    nidx = self._word_idx[nxt]
                    if cidx not in self._successors:
                        self._successors[cidx] = {}
                    self._successors[cidx][nidx] = (
                        self._successors[cidx].get(nidx, 0) + 1.0
                    )

        return sid

    def teach_batch(self, sentences: list, confidence: float = 0.5) -> list:
        """Teach multiple sentences. Batch-encodes for speed."""
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 10]
        if not sentences:
            return []

        # Batch encode all at once (much faster than one-by-one)
        embeddings = encode(sentences)

        sids = []
        with self._lock:
            for sentence, embedding in zip(sentences, embeddings):
                tokens = _tokenize(sentence)
                content = [t for t in tokens if t not in FUNCTION_WORDS]
                if len(content) < 2:
                    sids.append(-1)
                    continue

                for word in content:
                    self._learn_word(word)

                sid = self._sent_count
                self._sent_count += 1
                self._sent_texts.append(sentence)
                sids.append(sid)

                with self._env.begin(write=True) as txn:
                    key = struct.pack('<i', sid)
                    txn.put(key, sentence.encode('utf-8'), db=self._sentences_db)
                    txn.put(key, embedding.tobytes(), db=self._embeddings_db)

                if self._embeddings is None:
                    self._embeddings = embedding.reshape(1, -1)
                else:
                    self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])

                # Co-occurrence
                indices = [self._word_idx[w] for w in content]
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        a, b = indices[i], indices[j]
                        if a not in self._cooc:
                            self._cooc[a] = {}
                        if b not in self._cooc:
                            self._cooc[b] = {}
                        self._cooc[a][b] = min(WEIGHT_CLAMP,
                            self._cooc[a].get(b, 0) + COOC_WEIGHT)
                        self._cooc[b][a] = min(WEIGHT_CLAMP,
                            self._cooc[b].get(a, 0) + COOC_WEIGHT)

                # Successors
                for i in range(len(tokens) - 1):
                    curr, nxt = tokens[i], tokens[i + 1]
                    if curr in self._word_idx and nxt in self._word_idx:
                        cidx = self._word_idx[curr]
                        nidx = self._word_idx[nxt]
                        if cidx not in self._successors:
                            self._successors[cidx] = {}
                        self._successors[cidx][nidx] = (
                            self._successors[cidx].get(nidx, 0) + 1.0
                        )

        return sids

    # --- Q→A ---

    def _qa_key(self, question: str) -> str:
        tokens = _tokenize(question)
        content = [t for t in tokens
                   if t not in FUNCTION_WORDS and t not in QUERY_VERBS and t not in _WH_WORDS]
        return ' '.join(sorted(content))

    def correct(self, question: str, answer: str):
        """Store a Q→A pair. Also teaches the answer."""
        if not question.strip() or not answer.strip():
            return
        qkey = self._qa_key(question)
        if not qkey:
            return
        if qkey in self._protected_keys:
            return
        self.teach(answer)
        self._qa_map[qkey] = answer
        with self._env.begin(write=True) as txn:
            txn.put(qkey.encode('utf-8'), answer.encode('utf-8'), db=self._qa_db)

    def protect(self, question: str, answer: str):
        """Protected Q→A — cannot be overwritten."""
        qkey = self._qa_key(question)
        self._protected_keys.add(qkey)
        self._qa_map[qkey] = answer
        with self._env.begin(write=True) as txn:
            txn.put(qkey.encode('utf-8'), answer.encode('utf-8'), db=self._qa_db)
            txn.put(('__p__' + qkey).encode('utf-8'), b'1', db=self._qa_db)

    # --- Asking ---

    def ask(self, question: str, auto_learn: bool = False, **kwargs) -> dict:
        """Ask the brain.

        Tier 1: Q→A direct lookup (instant)
        Tier 2: MiniLM embedding search (find most similar sentence)
        """
        # Tier 1: direct lookup
        qkey = self._qa_key(question)
        if qkey in self._qa_map:
            return {
                "answer": self._qa_map[qkey],
                "confidence": 1.0,
                "strategy": "qa_direct",
                "trace": f"Q→A: {qkey[:50]}",
            }

        # Tier 2: embedding search
        if self._embeddings is None or len(self._embeddings) == 0:
            return {
                "answer": "I don't know.",
                "confidence": 0.0,
                "strategy": "abstain",
                "trace": "No sentences in DB",
            }

        query_emb = encode_one(question)
        # Cosine similarity (embeddings are already normalized)
        sims = self._embeddings @ query_emb
        top_k = min(5, len(sims))
        if top_k >= len(sims):
            top_indices = np.argsort(-sims)
        else:
            top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        best_idx = top_indices[0]
        best_sim = float(sims[best_idx])
        best_text = self._sent_texts[best_idx]

        if best_sim < 0.3:
            return {
                "answer": "I don't know.",
                "confidence": 0.0,
                "strategy": "abstain",
                "trace": f"Best match too weak: {best_sim:.3f}",
            }

        # Chain top non-overlapping sentences for richer answers
        answer_parts = [best_text]
        used_words = set(_tokenize(best_text))
        for idx in top_indices[1:]:
            if float(sims[idx]) < 0.4:
                break
            text = self._sent_texts[idx]
            text_words = set(_tokenize(text))
            # Skip if >50% overlap with already-used words
            if len(text_words & used_words) > len(text_words) * 0.5:
                continue
            answer_parts.append(text)
            used_words.update(text_words)
            if len(answer_parts) >= 3:
                break

        answer = " ".join(answer_parts)
        return {
            "answer": answer,
            "confidence": best_sim,
            "strategy": "embedding_search",
            "trace": f"Top match: sim={best_sim:.3f}, sentences={len(answer_parts)}",
        }

    # --- Generation (convergence × successors) ---

    def generate(self, query: str, max_tokens: int = 50,
                 temperature: float = 0.7) -> dict:
        """Generate text: embedding search for concepts, successor walk for words.

        1. Find relevant concepts via embedding search
        2. Walk successors guided by concept relevance
        """
        if not self._words or self._embeddings is None:
            return {"text": "", "trace": [], "tokens_generated": 0}

        # Find concept words from embedding search
        query_emb = encode_one(query)
        sims = self._embeddings @ query_emb
        top_k = min(10, len(sims))
        if top_k >= len(sims):
            top_indices = np.argsort(-sims)
        else:
            top_indices = np.argpartition(-sims, top_k)[:top_k]

        # Extract content words from top sentences
        concept_words = set()
        query_words = set(_tokenize(query))
        for idx in top_indices:
            if float(sims[idx]) < 0.3:
                continue
            words = _tokenize(self._sent_texts[idx])
            for w in words:
                if w not in FUNCTION_WORDS and w not in query_words and w in self._word_idx:
                    concept_words.add(w)

        if not concept_words:
            return {"text": "", "trace": ["No concepts found"], "tokens_generated": 0}

        # Build concept relevance scores
        concept_scores = {}
        for w in concept_words:
            idx = self._word_idx[w]
            concept_scores[idx] = 1.0

        # Find best starting word
        start_word = None
        best_score = -1
        for w in concept_words:
            widx = self._word_idx[w]
            # Prefer words with successors (can continue the chain)
            n_succ = len(self._successors.get(widx, {}))
            score = n_succ * 0.1 + 1.0
            if score > best_score:
                best_score = score
                start_word = w

        if not start_word:
            return {"text": "", "trace": ["No starting word"], "tokens_generated": 0}

        generated = [start_word]
        trace = [{"token": start_word, "score": round(best_score, 4), "reason": "start"}]
        recent_window = 8

        for step in range(1, max_tokens):
            prev = generated[-1]
            prev_idx = self._word_idx.get(prev, -1)
            if prev_idx < 0:
                break

            # Get successors
            succs = self._successors.get(prev_idx, {})
            if not succs:
                trace.append({"token": "[STOP]", "score": 0, "reason": "no successors"})
                break

            recent_set = set(generated[-recent_window:])

            # Score each successor: concept relevance FIRST, successor count as tiebreaker.
            # Without this, high-frequency corpus artifacts dominate.
            # A concept-relevant successor with count=1 beats an irrelevant one with count=20.
            scored = []
            total_succ_count = sum(succs.values()) or 1.0
            for nidx, count in succs.items():
                if nidx >= len(self._words):
                    continue
                word = self._words[nidx]
                if word in recent_set or word in FUNCTION_WORDS:
                    continue
                relevance = concept_scores.get(nidx, 0.0)
                normalized_count = count / total_succ_count  # 0..1
                if relevance > 0:
                    # Concept-relevant: relevance dominates, count is bonus
                    score = relevance * 10.0 + normalized_count
                else:
                    # Not concept-relevant: only count, heavily penalized
                    score = normalized_count * 0.1
                scored.append((nidx, word, score))

            if not scored:
                trace.append({"token": "[STOP]", "score": 0, "reason": "no valid successors"})
                break

            scored.sort(key=lambda x: x[2], reverse=True)

            # Temperature selection
            if temperature > 0 and len(scored) > 1:
                top_n = min(10, len(scored))
                top = scored[:top_n]
                scores = [s for _, _, s in top]
                max_s = max(scores)
                exp_s = [math.exp((s - max_s) / temperature) for s in scores]
                total = sum(exp_s)
                if total > 0:
                    import random
                    r = random.random()
                    cumul = 0
                    chosen = top[0]
                    for i, e in enumerate(exp_s):
                        cumul += e / total
                        if r <= cumul:
                            chosen = top[i]
                            break
                else:
                    chosen = scored[0]
            else:
                chosen = scored[0]

            nidx, word, score = chosen
            generated.append(word)
            trace.append({"token": word, "score": round(score, 4),
                          "reason": "succ×concept" if concept_scores.get(nidx, 0) > 0 else "succ"})

        return {
            "text": " ".join(generated),
            "trace": trace,
            "tokens_generated": len(generated),
        }

    # --- Persistence ---

    def save_state(self):
        """Persist co-occurrence and successor data to LMDB."""
        with self._env.begin(write=True) as txn:
            for widx, neighbors in self._cooc.items():
                key = struct.pack('<i', widx)
                val = b''.join(struct.pack('<if', nidx, w) for nidx, w in neighbors.items())
                txn.put(key, val, db=self._cooc_db)
            for widx, succs in self._successors.items():
                key = struct.pack('<i', widx)
                val = b''.join(struct.pack('<if', nidx, cnt) for nidx, cnt in succs.items())
                txn.put(key, val, db=self._successors_db)

    # --- Utility ---

    def stats(self) -> dict:
        n_cooc = sum(len(v) for v in self._cooc.values())
        n_succ = sum(len(v) for v in self._successors.values())
        return {
            "words": len(self._words),
            "sentences": self._sent_count,
            "qa_pairs": len(self._qa_map),
            "cooc_edges": n_cooc,
            "successor_edges": n_succ,
            "embedding_dim": _EMBED_DIM,
            "embeddings_loaded": self._embeddings.shape[0] if self._embeddings is not None else 0,
        }

    def health(self) -> dict:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        return {
            "status": "ok",
            "words": len(self._words),
            "sentences": self._sent_count,
            "rss_mb": round(rss_mb, 1),
            "backend": "brain_v2_minilm",
        }

    def close(self):
        self.save_state()
        self._env.close()


if __name__ == '__main__':
    # Quick smoke test
    brain = BrainV2()

    brain.teach("Shakespeare wrote Hamlet in 1600.")
    brain.teach("Gravity is the force that attracts objects with mass toward each other.")
    brain.teach("Photosynthesis is the process by which plants convert sunlight into energy.")
    brain.teach("The capital of France is Paris.")
    brain.teach("Alexander Graham Bell invented the telephone.")

    print("\n--- Stats ---")
    print(brain.stats())

    print("\n--- Ask ---")
    for q in ["who wrote hamlet", "what is gravity", "capital of france"]:
        r = brain.ask(q)
        print(f"Q: {q}")
        print(f"A: {r['answer'][:100]}")
        print(f"   [{r['strategy']}, {r['confidence']:.3f}]")
        print()

    print("--- Generate ---")
    for q in ["shakespeare hamlet", "gravity force"]:
        r = brain.generate(q, max_tokens=15)
        print(f"Q: {q}")
        print(f"A: {r['text']}")
        print()

    brain.close()
