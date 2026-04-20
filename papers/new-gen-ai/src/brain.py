"""
Brain: performance layer over brain_core.

Adds mmap-backed matrix, caching, bulk mode, lazy reindex, and
LRU eviction on top of the clean core. The core handles all reasoning.
This file handles all speed.

    from brain import Brain  # same API, faster
    brain = Brain(db_path="./my_brain")
    brain.teach("paris is the capital of france")
    brain.ask("capital of france")
"""

import json
import os
import hashlib
import numpy as np
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from brain_core import BrainCore, FUNCTION_WORDS, STRUCTURAL_WORDS, COOCCURRENCE_PULL

# Re-export for backward compatibility
__all__ = ['Brain', 'FUNCTION_WORDS', 'STRUCTURAL_WORDS', 'COOCCURRENCE_PULL']


class Brain(BrainCore):
    """Brain with performance optimizations. Same API as BrainCore.

    Adds:
    - mmap-backed matrix (instant boot from disk)
    - .npy cache fallback (for non-mmap mode)
    - Bulk mode (batched commits + deferred reindex)
    - Lazy reindex (dirty flag, rebuild before next ask)
    - Cached nid→word dict
    """

    MAX_WORDS = 16000  # mmap pre-allocation (resizes dynamically)

    def __init__(self, db_path=None, use_mmap=True):
        # Performance state (before super().__init__ which calls _load_matrix)
        self._use_mmap = use_mmap and db_path is not None
        self._bulk_mode = False
        self._bulk_dirty = False
        self._search_dirty = False
        self._nid_to_word_cache = None
        self._matrix_capacity = 0

        if self._use_mmap and db_path:
            # Skip parent's _load_matrix — we boot from mmap instead
            self._words = []
            self._word_idx = {}
            self._db_path = db_path
            self._matrix = None
            self._embed_dim = 0
            self._embeddings = None
            self._templates = []
            self._pool = ThreadPoolExecutor(max_workers=4)
            self._batch_pool = ThreadPoolExecutor(max_workers=8)
            self._lock = Lock()

            from neuron import NeuronDB
            self.db = NeuronDB(path=db_path, dim=1)
            self._word_neurons = self.db.load_word_mappings()

            self._boot_mmap()
        else:
            # Normal boot — let core handle it
            super().__init__(db_path=db_path)
            self._matrix_capacity = len(self._words) if self._matrix is not None else 0

    # --- MMAP Boot ---

    def _boot_mmap(self):
        """Boot from memory-mapped matrix file. <10ms."""
        d = Path(self._db_path)
        mmap_file = d / '_matrix.mmap'
        words_file = d / '_words.json'
        cap = self.MAX_WORDS

        # Load word list
        if words_file.exists():
            data = json.loads(words_file.read_text())
            self._words = data.get('words', [])
            self._word_idx = {w: i for i, w in enumerate(self._words)}
        else:
            # First boot — build from DB
            words_in_db = sorted(
                [(w, nid) for w, nid in self._word_neurons.items()
                 if not w.startswith("__")],
                key=lambda x: x[1]
            )
            for word, nid in words_in_db:
                if word not in self._word_idx:
                    self._words.append(word)
                    self._word_idx[word] = len(self._words) - 1

        n = len(self._words)

        # Ensure cap is large enough
        if n > cap:
            cap = n + 4000

        # Open or create mmap
        if mmap_file.exists():
            # Check if existing mmap is big enough
            expected_size = cap * cap * 4  # float32
            actual_size = mmap_file.stat().st_size
            if actual_size < expected_size:
                # Need to resize — old mmap was smaller
                old_cap = int((actual_size / 4) ** 0.5)
                old_matrix = np.memmap(str(mmap_file), dtype=np.float32,
                                       mode='r+', shape=(old_cap, old_cap))
                new_file = d / '_matrix.mmap.new'
                new_matrix = np.memmap(str(new_file), dtype=np.float32,
                                       mode='w+', shape=(cap, cap))
                copy_n = min(old_cap, n)
                new_matrix[:copy_n, :copy_n] = old_matrix[:copy_n, :copy_n]
                new_matrix.flush()
                del old_matrix, new_matrix
                new_file.rename(mmap_file)

            self._matrix = np.memmap(str(mmap_file), dtype=np.float32,
                                     mode='r+', shape=(cap, cap))
        else:
            self._matrix = np.memmap(str(mmap_file), dtype=np.float32,
                                     mode='w+', shape=(cap, cap))
            for i in range(n):
                self._matrix[i, i] = 1.0
            # Load vectors from DB into mmap
            if n > 0:
                nid_to_idx = {nid: self._word_idx[w]
                              for w, nid in self._word_neurons.items()
                              if w in self._word_idx}
                for nid, vec_bytes in self.db.db.execute("SELECT id, vector FROM neurons"):
                    idx = nid_to_idx.get(nid)
                    if idx is None:
                        continue
                    vec = np.frombuffer(vec_bytes, dtype=np.float32)
                    vlen = min(len(vec), cap)
                    self._matrix[idx, :vlen] = vec[:vlen]
                    self._matrix[:vlen, idx] = vec[:vlen]
            self._matrix.flush()

        self._matrix_capacity = cap
        self._save_words()
        self._rebuild_search_matrix()

    def _save_words(self):
        if self._db_path:
            d = Path(self._db_path)
            (d / '_words.json').write_text(json.dumps({
                'words': self._words, 'count': len(self._words)
            }))

    def _resize_mmap(self, new_cap):
        """Resize mmap file dynamically."""
        d = Path(self._db_path)
        old_file = d / '_matrix.mmap'
        new_file = d / '_matrix.mmap.new'
        n = min(len(self._words), self._matrix_capacity)

        new_matrix = np.memmap(str(new_file), dtype=np.float32,
                               mode='w+', shape=(new_cap, new_cap))
        new_matrix[:n, :n] = self._matrix[:n, :n]
        new_matrix.flush()
        del self._matrix
        new_file.rename(old_file)
        self._matrix = np.memmap(str(old_file), dtype=np.float32,
                                 mode='r+', shape=(new_cap, new_cap))
        self._matrix_capacity = new_cap

    # --- Override _learn_word for mmap ---

    def _learn_word(self, word):
        """Add word, with mmap-aware growth."""
        word = word.lower().strip()
        if word in self._word_idx:
            return self._word_idx[word]

        idx = len(self._words)
        self._words.append(word)
        self._word_idx[word] = idx

        if self._use_mmap:
            if idx >= self._matrix_capacity:
                self._resize_mmap(max(self._matrix_capacity * 2, idx + 4000))
            self._matrix[idx, idx] = 1.0
            if idx % 100 == 0:
                self._save_words()
        else:
            # In-memory: pre-allocated doubling (from core)
            return super()._learn_word(word)

        return idx

    # --- Override _encode_word for dimension clamping ---

    def _encode_word(self, word):
        word = word.lower().strip()
        idx = self._word_idx.get(word)
        n = min(len(self._words), self._matrix_capacity) if self._matrix_capacity > 0 else len(self._words)
        if idx is None or n == 0 or idx >= n:
            return np.zeros(n or 1, dtype=np.float32)
        vec = self._matrix[idx, :n].copy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # --- Override _encode_sentence for dimension clamping ---

    def _encode_sentence(self, text):
        tokens = self._tokenize(text)
        if self._matrix_capacity > 0:
            d = min(len(self._words), self._matrix_capacity) or 1
        else:
            d = len(self._words) or 1
        if not tokens:
            return np.zeros(d, dtype=np.float32)

        result = np.zeros(d, dtype=np.float32)
        total_weight = 0.0
        for i, token in enumerate(tokens):
            token = token.lower().strip()
            idx = self._word_idx.get(token)
            if idx is not None and idx < d and self._matrix is not None:
                vec = self._matrix[idx, :d]
                if np.any(vec != 0):
                    w = 1.0 / (1.0 + 0.1 * i)
                    result += w * vec
                    total_weight += w
        if total_weight > 0:
            result /= total_weight
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        return result

    # --- Lazy reindex ---

    def teach(self, sentence, confidence=0.5):
        """Teach with lazy reindex + cached dict invalidation."""
        with self._lock:
            if not self._bulk_mode:
                self.db.begin_batch()

            dim_before = len(self._words)
            result = self._teach_inner(sentence, confidence)

            if not self._bulk_mode:
                self.db.end_batch()

            if len(self._words) != dim_before:
                self._search_dirty = True
                self._nid_to_word_cache = None

            if self._bulk_mode:
                self._bulk_count = getattr(self, '_bulk_count', 0) + 1
                if self._bulk_count % 500 == 0:
                    self.db.db.commit()

        return result

    def _teach_inner(self, sentence, confidence):
        """Core teach logic without locking/batching."""
        tokens = self._tokenize(sentence)
        content = [t for t in tokens if t not in FUNCTION_WORDS]
        if not content:
            return []

        for word in content:
            self._learn_word(word)
        if len(content) >= 2:
            self._learn_cooccurrence(content)

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
                self._nid_to_word_cache = None
                self.db.save_word_mapping(word, n.id)
                neurons.append(n)

        for i in range(len(neurons) - 1):
            self.db.update_successors(neurons[i].id, neurons[i + 1].id, 0.8)
            self.db.update_predecessors(neurons[i + 1].id, neurons[i].id)

        if len(neurons) >= 2:
            self.db.record_sentence([n.id for n in neurons])

        if len(tokens) >= 3:
            self._extract_template(tokens)

        return [n.id for n in neurons]

    def ask(self, question):
        """Ask with lazy reindex."""
        if getattr(self, '_search_dirty', False):
            self._rebuild_search_matrix()
            self._search_dirty = False
        return super().ask(question)

    def teach_batch(self, sentences, confidence=0.5):
        """Sequential batch — lock serializes anyway."""
        return [self.teach(s, confidence) for s in sentences]

    # --- Bulk mode ---

    def begin_bulk(self):
        self._bulk_mode = True
        self._bulk_dirty = False
        self._bulk_count = 0
        self.db.begin_batch()

    def end_bulk(self):
        self._bulk_mode = False
        self.db.end_batch()
        self._rebuild_search_matrix()
        if self._use_mmap and self._matrix is not None and hasattr(self._matrix, 'flush'):
            self._matrix.flush()
        self._save_words()

    # --- Rebuild search matrix ---

    def _rebuild_search_matrix(self):
        """Build search index from current matrix state."""
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

        # Batch update DB
        if self.db._vectors is not None:
            updates = [(self.db._vectors[id_to_row[nid]].tobytes(), nid)
                       for word, nid in words if nid in id_to_row]
            self.db.db.executemany(
                "UPDATE neurons SET vector = ? WHERE id = ?", updates)
            self.db.db.commit()

    # --- Cached nid_to_word ---

    def _get_nid_to_word(self):
        if self._nid_to_word_cache is None:
            self._nid_to_word_cache = {nid: w for w, nid in self._word_neurons.items()
                                       if not w.startswith("__")}
        return self._nid_to_word_cache

    # --- Close ---

    def close(self):
        self._pool.shutdown(wait=False)
        self._batch_pool.shutdown(wait=False)
        if self._use_mmap and self._matrix is not None and hasattr(self._matrix, 'flush'):
            self._matrix.flush()
            self._save_words()
        self.db.close()

    # --- Health with death risk ---

    def health(self):
        h = super().health()

        # System-wide memory
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        h["system_avail_mb"] = round(int(line.split()[1]) / 1024, 1)
                    elif line.startswith('SwapTotal:'):
                        h["swap_total_mb"] = round(int(line.split()[1]) / 1024, 1)
                    elif line.startswith('SwapFree:'):
                        h["swap_free_mb"] = round(int(line.split()[1]) / 1024, 1)
        except Exception:
            pass

        h["swap_used_mb"] = round(h.get("swap_total_mb", 0) - h.get("swap_free_mb", 0), 1)

        # Death risk: 0-100
        risk = 0
        avail = h.get("system_avail_mb", 9999)
        if avail < 2048:
            risk += int((2048 - avail) / 2048 * 40)
        if h.get("swap_used_mb", 0) > 100:
            risk += min(25, int(h["swap_used_mb"] / 500 * 25))
        if h["disk_free_gb"] < 5:
            risk += int((5 - h["disk_free_gb"]) / 5 * 20)
        try:
            load1 = os.getloadavg()[0]
            ncpu = os.cpu_count() or 1
            if load1 > ncpu * 0.9:
                risk += 10
        except Exception:
            pass
        if h["rss_mb"] > 512:
            risk += min(5, int((h["rss_mb"] - 512) / 1024 * 5))
        h["death_risk"] = min(100, risk)

        return h
