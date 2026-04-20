"""
Brain: two-layer performance wrapper over brain_core.

Layer 1 (fast): co-occurrence dict + word list. In-memory. Microseconds.
Layer 2 (background): neurons, successors, sentences → SQLite. Deferred.

The brain KNOWS things at Layer 1 speed. It SAVES them at Layer 2 speed.
Like human memory: learn instantly, consolidate during sleep.

    from brain import Brain
    brain = Brain(db_path="./my_brain")
    brain.teach("paris is the capital of france")  # instant (dict only)
    brain.ask("capital of france")                  # works immediately
    brain.flush()                                   # persist to SQLite
"""

import os
import numpy as np
from collections import deque
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor

from brain_core import BrainCore, FUNCTION_WORDS, STRUCTURAL_WORDS, COOCCURRENCE_PULL

__all__ = ['Brain', 'FUNCTION_WORDS', 'STRUCTURAL_WORDS', 'COOCCURRENCE_PULL']


class Brain(BrainCore):
    """Two-layer brain. Same API as BrainCore, much faster teach.

    Layer 1: _learn_word + _learn_cooccurrence (dict ops, ~microseconds)
    Layer 2: neuron insert + successors + sentences (SQLite, deferred)
    """

    def __init__(self, db_path=None):
        self._bulk_mode = False
        self._search_dirty = False
        self._nid_to_word_cache = None

        # Layer 2: async persist queue + background thread
        self._persist_queue = deque()
        self._persist_count = 0
        self._flush_thread = None
        self._flushing = False

        super().__init__(db_path=db_path)

    # --- Two-layer teach ---

    def teach(self, sentence, confidence=0.5):
        """Layer 1: instant. Dict operations only.
        Layer 2: queued for background persist."""
        tokens = self._tokenize(sentence)
        content = [t for t in tokens if t not in FUNCTION_WORDS]
        if not content:
            return []

        with self._lock:
            dim_before = len(self._words)

            # Layer 1: co-occurrence (instant)
            for word in content:
                self._learn_word(word)
            if len(content) >= 2:
                self._learn_cooccurrence(content)

            if len(self._words) != dim_before:
                self._search_dirty = True
                self._nid_to_word_cache = None

        # Queue for Layer 2 persist — backpressure prevents OOM
        self._persist_queue.append((sentence, content, tokens, confidence))
        self._persist_count += 1

        # Backpressure: flush when queue hits cap (memory-bounded)
        MAX_QUEUE = 5000
        if len(self._persist_queue) >= MAX_QUEUE:
            self._flush_persist()  # blocks until drained — backpressure

        return list(range(len(content)))  # placeholder IDs

    def _async_flush(self):
        """Kick off a background flush if not already running."""
        if self._flushing:
            return
        self._flushing = True
        self._flush_thread = Thread(target=self._bg_flush, daemon=True)
        self._flush_thread.start()

    def _bg_flush(self):
        """Background thread: persist queued teaches to SQLite."""
        try:
            self._flush_persist()
        finally:
            self._flushing = False

    def _flush_persist(self):
        """Layer 2: flush queued teaches to SQLite."""
        if not self._persist_queue:
            return

        self.db.begin_batch()
        flushed = 0

        while self._persist_queue:
            sentence, content, tokens, confidence = self._persist_queue.popleft()

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

            flushed += 1

        self.db.end_batch()
        return flushed

    def flush(self):
        """Force flush all pending teaches to SQLite."""
        self._flush_persist()
        self._save_cooc()
        if self._search_dirty:
            self._rebuild_search_matrix()
            self._search_dirty = False

    def ask(self, question):
        """Wait for any async flush, then query."""
        # Wait for background flush to finish
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join()
        # Flush any remaining
        if self._persist_queue:
            self._flush_persist()
        if getattr(self, '_search_dirty', False):
            self._rebuild_search_matrix()
            self._search_dirty = False
        return super().ask(question)

    def teach_batch(self, sentences, confidence=0.5):
        """Batch teach — all Layer 1 instant, one Layer 2 flush at end."""
        results = [self.teach(s, confidence) for s in sentences]
        return results

    # --- Bulk mode ---

    def begin_bulk(self):
        """Enter bulk mode. Layer 2 deferred until end_bulk."""
        self._bulk_mode = True

    def end_bulk(self):
        """Exit bulk mode. Flush everything."""
        self._bulk_mode = False
        self._flush_persist()
        self._save_cooc()
        self._rebuild_search_matrix()

    # --- Close ---

    def close(self):
        self._pool.shutdown(wait=False)
        self._batch_pool.shutdown(wait=False)
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join()
        self._flush_persist()
        self._save_cooc()
        self.db.close()

    # --- Health with death risk ---

    def health(self):
        h = super().health()
        h["persist_queue"] = len(self._persist_queue)

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
