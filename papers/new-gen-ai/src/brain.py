"""
Brain: performance layer over brain_core.

Adds bulk mode, lazy reindex, cached lookups, and death risk
on top of the clean core. The core handles all reasoning.
This file handles all speed.

    from brain import Brain  # same API, faster
    brain = Brain(db_path="./my_brain")
    brain.teach("paris is the capital of france")
    brain.ask("capital of france")
"""

import os
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
    - Bulk mode (batched commits + deferred reindex)
    - Lazy reindex (dirty flag, rebuild before next ask)
    - Cached nid→word dict
    - Death risk score
    """

    def __init__(self, db_path=None):
        self._bulk_mode = False
        self._bulk_dirty = False
        self._search_dirty = False
        self._nid_to_word_cache = None
        super().__init__(db_path=db_path)

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
        self._save_cooc()

    # --- Close ---

    def close(self):
        self._pool.shutdown(wait=False)
        self._batch_pool.shutdown(wait=False)
        self._save_cooc()
        self.db.close()

    # --- Health with death risk ---

    def health(self):
        h = super().health()

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
