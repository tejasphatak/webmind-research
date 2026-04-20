"""
MoE Brain: Mixture of Expert matrices.

Instead of one giant N×N co-occurrence matrix, split into K smaller
expert matrices. Each expert handles a cluster of related words.
A router maps words to experts.

    One 50K×50K matrix = 10GB (doesn't fit)
    32 experts × 1.5K×1.5K = 32 × 9MB = 288MB (fits on a phone)

Same interface as Brain — drop-in replacement.

Routing: words that co-occur go to the same expert.
The sentence is the unit of clustering. All words in a sentence
share an expert. Related topics naturally cluster because they
share vocabulary.

Cross-domain queries activate multiple experts, merge by confidence.
"""

import os
import sqlite3
import time
import numpy as np
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from brain import Brain, FUNCTION_WORDS


class MoEBrain:
    """Mixture of Expert brains with a semantic router."""

    def __init__(self, db_path: str, num_experts: int = 16,
                 max_expert_words: int = 5000):
        self.db_path = db_path
        self.num_experts = num_experts
        self.max_expert_words = max_expert_words  # split threshold

        # Expert directory
        expert_dir = os.path.join(db_path, 'experts')
        os.makedirs(expert_dir, exist_ok=True)

        # Router DB: word → expert_id
        router_path = os.path.join(db_path, 'router.db')
        self._router_db = sqlite3.connect(router_path, check_same_thread=False)
        self._router_db.execute("PRAGMA journal_mode=WAL")
        self._router_db.execute("PRAGMA busy_timeout=5000")
        self._router_db.execute("""
            CREATE TABLE IF NOT EXISTS routes (
                word TEXT PRIMARY KEY,
                expert_id INTEGER NOT NULL
            )
        """)
        self._router_db.execute("""
            CREATE TABLE IF NOT EXISTS expert_stats (
                expert_id INTEGER PRIMARY KEY,
                word_count INTEGER DEFAULT 0
            )
        """)
        self._router_db.commit()

        # Load router into memory — auto-detect spawned experts
        self._routes = {}
        max_eid = num_experts - 1
        for word, eid in self._router_db.execute("SELECT word, expert_id FROM routes"):
            self._routes[word] = eid
            if eid > max_eid:
                max_eid = eid

        # Expand to fit all known experts (including previously spawned ones)
        self.num_experts = max(num_experts, max_eid + 1)

        # Expert word counts (for load balancing)
        self._expert_counts = [0] * self.num_experts
        for eid, count in self._router_db.execute("SELECT expert_id, word_count FROM expert_stats"):
            if eid < self.num_experts:
                self._expert_counts[eid] = count

        # Lazy-load experts
        self._experts = [None] * self.num_experts
        self._expert_dir = expert_dir
        self._lock = Lock()

        # DB proxy for daemon compatibility
        self._db_proxy = self._DBProxy(self)

        # Pool for parallel expert queries
        self._pool = ThreadPoolExecutor(max_workers=min(num_experts, 8))

    # Max experts to keep loaded — the rest sleep on disk
    MAX_LOADED = 4

    def _get_expert(self, expert_id: int) -> Brain:
        """Load an expert, evicting LRU if at capacity. O(1) via OrderedDict."""
        if not hasattr(self, '_expert_lru'):
            self._expert_lru = OrderedDict()

        if self._experts[expert_id] is not None:
            # Touch — move to end (most recently used)
            if expert_id in self._expert_lru:
                self._expert_lru.move_to_end(expert_id)
            return self._experts[expert_id]

        with self._lock:
            if self._experts[expert_id] is not None:
                return self._experts[expert_id]

            # Evict oldest if at capacity
            while len(self._expert_lru) >= self.MAX_LOADED:
                evict_id, _ = self._expert_lru.popitem(last=False)
                if self._experts[evict_id] is not None:
                    self._experts[evict_id]._save_cache()
                    self._experts[evict_id].close()
                    self._experts[evict_id] = None

            # Load the requested expert
            path = os.path.join(self._expert_dir, f'expert_{expert_id:02d}')
            os.makedirs(path, exist_ok=True)
            brain = Brain(db_path=path)
            if getattr(self, '_bulk_mode', False):
                brain.begin_bulk()
            self._experts[expert_id] = brain
            self._expert_lru[expert_id] = True

        return self._experts[expert_id]

    def _route_word(self, word: str) -> int:
        """Get or assign an expert for a word."""
        word = word.lower().strip()
        if word in self._routes:
            return self._routes[word]
        return -1  # unassigned

    def _spawn_expert(self) -> int:
        """Create a new expert. Thread-safe."""
        with self._lock:
            new_id = self.num_experts
            self.num_experts += 1
            self._experts.append(None)
            self._expert_counts.append(0)
            return new_id

    def _route_sentence(self, tokens: list) -> int:
        """Route a sentence to an expert. Self-organizing:

        1. Vote from already-routed rare words → existing expert
        2. If best expert is full (>max_expert_words) → spawn new expert
        3. If no words routed yet → lightest expert, or spawn if all full
        """
        content = [t.lower().strip() for t in tokens if t.lower().strip() not in FUNCTION_WORDS]
        if not content:
            return 0

        # Vote from rare words
        votes = {}
        for word in content:
            eid = self._routes.get(word, -1)
            if eid >= 0:
                freq = self._expert_counts[eid]
                if freq < self.max_expert_words:
                    votes[eid] = votes.get(eid, 0) + 1

        if votes:
            best = max(votes, key=votes.get)
            # If the winning expert is near capacity, still use it
            # (the words already live there)
            return best

        # No signal — find lightest expert with room
        for eid in sorted(range(len(self._expert_counts)),
                          key=lambda i: self._expert_counts[i]):
            if self._expert_counts[eid] < self.max_expert_words:
                return eid

        # All experts full — spawn a new one
        return self._spawn_expert()

    def _assign_words(self, words: list, expert_id: int):
        """Assign unrouted words to an expert."""
        new_words = []
        for word in words:
            word = word.lower().strip()
            if word not in self._routes and word not in FUNCTION_WORDS:
                self._routes[word] = expert_id
                new_words.append(word)

        if new_words:
            self._expert_counts[expert_id] += len(new_words)
            self._router_db.executemany(
                "INSERT OR IGNORE INTO routes (word, expert_id) VALUES (?, ?)",
                [(w, expert_id) for w in new_words]
            )
            # Batch commits: only flush every 500 assignments
            self._route_pending = getattr(self, '_route_pending', 0) + len(new_words)
            if self._route_pending >= 500:
                self._flush_router()

    def _flush_router(self):
        """Flush pending router writes to disk."""
        for eid in range(self.num_experts):
            if self._expert_counts[eid] > 0:
                self._router_db.execute(
                    "INSERT OR REPLACE INTO expert_stats (expert_id, word_count) VALUES (?, ?)",
                    (eid, self._expert_counts[eid])
                )
        self._router_db.commit()
        self._route_pending = 0

    # --- Public API (same as Brain) ---

    def teach(self, sentence: str, confidence: float = 0.5) -> list:
        """Teach a sentence. Routes to the right expert automatically."""
        tokens = sentence.lower().split()
        content = [t for t in tokens if t not in FUNCTION_WORDS]

        if not content:
            return []

        expert_id = self._route_sentence(tokens)
        self._assign_words(content, expert_id)

        expert = self._get_expert(expert_id)
        return expert.teach(sentence, confidence)

    def ask(self, question: str) -> dict:
        """Ask a question. Routes to relevant expert(s), merges results."""
        tokens = question.lower().split()
        content = [t for t in tokens if t not in FUNCTION_WORDS]

        if not content:
            return {"answer": "I don't know.", "confidence": 0.0,
                    "strategy": "abstain", "trace": "Empty query"}

        # Find which experts own the query words
        expert_votes = {}
        for word in content:
            eid = self._routes.get(word, -1)
            if eid >= 0:
                expert_votes[eid] = expert_votes.get(eid, 0) + 1

        if not expert_votes:
            # No words routed — query all loaded experts
            active = [i for i, e in enumerate(self._experts) if e is not None]
            if not active:
                return {"answer": "I don't know.", "confidence": 0.0,
                        "strategy": "abstain", "trace": "No experts loaded"}
            expert_ids = active[:3]  # cap at 3
        else:
            # Top-2 experts by vote count
            sorted_experts = sorted(expert_votes.items(), key=lambda x: x[1], reverse=True)
            expert_ids = [eid for eid, _ in sorted_experts[:2]]

        # Query experts in parallel
        futures = {}
        for eid in expert_ids:
            expert = self._get_expert(eid)
            futures[self._pool.submit(expert.ask, question)] = eid

        best = {"answer": "I don't know.", "confidence": 0.0,
                "strategy": "abstain", "trace": "No expert answered"}

        for future in as_completed(futures):
            result = future.result()
            if result["confidence"] > best["confidence"]:
                best = result
                best["trace"] = f"expert_{futures[future]:02d}: {result.get('trace', '')}"

        return best

    def ask_batch(self, questions: list) -> list:
        """Ask multiple questions in parallel — different questions hit different experts."""
        futures = [self._pool.submit(self.ask, q) for q in questions]
        return [f.result() for f in futures]

    def teach_batch(self, sentences: list, confidence: float = 0.5) -> list:
        """Teach multiple sentences. Returns list of neuron ID lists."""
        return [self.teach(s, confidence) for s in sentences]

    # --- Daemon compatibility (brain.db.* interface) ---

    @property
    def _words(self):
        """Words that actually exist as neurons in some expert."""
        words = []
        for i in range(self.num_experts):
            if self._experts[i] is not None:
                words.extend(w for w in self._experts[i]._words
                             if w not in FUNCTION_WORDS and not w.startswith("__"))
        return words

    @property
    def _word_neurons(self):
        """Word → (expert_id) mapping. Daemon uses this for introspection."""
        return dict(self._routes)

    @property
    def db(self):
        """Proxy for daemon's brain.db.* calls. Delegates to primary expert."""
        return self._db_proxy

    class _DBProxy:
        """Compatibility layer so daemon can call brain.db.count(), brain.db.miss_stats(), etc."""
        def __init__(self, moe):
            self._moe = moe

        def count(self):
            total = 0
            for i in range(self._moe.num_experts):
                if self._moe._experts[i] is not None:
                    total += self._moe._experts[i].db.count()
            return total

        def miss_stats(self):
            total = {"total_misses": 0, "resolved": 0, "unresolved": 0}
            for i in range(self._moe.num_experts):
                if self._moe._experts[i] is not None:
                    s = self._moe._experts[i].db.miss_stats()
                    total["total_misses"] += s["total_misses"]
                    total["resolved"] += s["resolved"]
                    total["unresolved"] += s["unresolved"]
            t = total["total_misses"]
            total["resolution_rate"] = total["resolved"] / t if t > 0 else 0
            return total

        def get_unresolved_misses(self, limit=10):
            """Gather unresolved misses from all experts."""
            all_misses = []
            per_expert = max(1, limit // max(1, sum(
                1 for e in self._moe._experts if e is not None)))
            for i in range(self._moe.num_experts):
                if self._moe._experts[i] is not None:
                    misses = self._moe._experts[i].db.get_unresolved_misses(limit=per_expert)
                    all_misses.extend(misses)
            return all_misses[:limit]

        def resolve_miss(self, miss_id, answer):
            """Resolve a miss. Miss IDs are per-expert, so try each loaded
            expert but stop after first successful resolution to avoid
            cross-expert ID collision."""
            for i in range(self._moe.num_experts):
                if self._moe._experts[i] is not None:
                    try:
                        cursor = self._moe._experts[i].db.db.execute(
                            "SELECT id FROM misses WHERE id = ?", (miss_id,))
                        if cursor.fetchone():
                            self._moe._experts[i].db.resolve_miss(miss_id, answer)
                            return  # found the right expert
                    except Exception:
                        pass

        def resolve_miss_by_query(self, query, answer):
            for i in range(self._moe.num_experts):
                if self._moe._experts[i] is not None:
                    self._moe._experts[i].db.resolve_miss_by_query(query, answer)

        def log_miss(self, query, vec):
            """Log a miss to the first active expert."""
            for i in range(self._moe.num_experts):
                if self._moe._experts[i] is not None:
                    self._moe._experts[i].db.log_miss(query, vec)
                    return

        def get_sentences_for_neurons(self, nids):
            """Aggregate across experts."""
            result = {}
            for i in range(self._moe.num_experts):
                if self._moe._experts[i] is not None:
                    r = self._moe._experts[i].db.get_sentences_for_neurons(nids)
                    result.update(r)
            return result

        def get_sentence_neurons(self, sid):
            for i in range(self._moe.num_experts):
                if self._moe._experts[i] is not None:
                    r = self._moe._experts[i].db.get_sentence_neurons(sid)
                    if r:
                        return r
            return []

    def correct(self, question: str, answer: str):
        """Learn from a failure."""
        self.teach(answer, confidence=0.6)
        # Try to resolve miss in relevant experts
        tokens = question.lower().split()
        content = [t for t in tokens if t not in FUNCTION_WORDS]
        expert_ids = set()
        for word in content:
            eid = self._routes.get(word, -1)
            if eid >= 0:
                expert_ids.add(eid)
        for eid in expert_ids:
            expert = self._get_expert(eid)
            expert.db.resolve_miss_by_query(question, answer)

    def inspect(self, word: str) -> dict:
        """Inspect a word — find its expert and inspect there."""
        word = word.lower().strip()
        eid = self._routes.get(word, -1)
        if eid < 0:
            return {"known": False, "word": word}
        expert = self._get_expert(eid)
        result = expert.inspect(word)
        result["expert"] = eid
        result["expert_words"] = self._expert_counts[eid] if eid < len(self._expert_counts) else 0
        return result

    def introspect(self) -> dict:
        """Full self-knowledge: what experts exist, what do they know, how healthy are they."""
        experts = []
        for i in range(self.num_experts):
            info = {"id": i, "words": self._expert_counts[i] if i < len(self._expert_counts) else 0}
            if self._experts[i] is not None:
                s = self._experts[i].stats()
                h = self._experts[i].health()
                info["neurons"] = s["neurons"]
                info["loaded"] = True
                info["matrix_mb"] = h.get("brain_matrix_mb", 0)
                info["db_mb"] = h.get("db_size_mb", 0)
                # Sample top words by connection count
                words = [w for w in self._experts[i]._words
                         if w not in FUNCTION_WORDS and not w.startswith("__")]
                if words:
                    import random
                    sample = random.sample(words, min(5, len(words)))
                    info["sample_words"] = sample
            else:
                info["loaded"] = False
            experts.append(info)
        return {
            "total_experts": self.num_experts,
            "total_routes": len(self._routes),
            "total_neurons": sum(e["words"] for e in experts),
            "experts": experts,
        }

    def explain(self, question: str) -> str:
        """Explain how the brain would process a question — full observability.

        Returns a human-readable trace of: routing → expert selection →
        search → confidence check → answer.
        """
        tokens = question.lower().split()
        content = [t for t in tokens if t not in FUNCTION_WORDS]

        lines = [f"Query: \"{question}\""]
        lines.append(f"Content words: {content}")

        # Routing
        expert_votes = {}
        word_routes = {}
        for word in content:
            eid = self._routes.get(word, -1)
            word_routes[word] = eid
            if eid >= 0:
                expert_votes[eid] = expert_votes.get(eid, 0) + 1

        lines.append(f"Word routes: {word_routes}")
        lines.append(f"Expert votes: {expert_votes}")

        if not expert_votes:
            lines.append("No words routed → would query all active experts")
            return "\n".join(lines)

        sorted_experts = sorted(expert_votes.items(), key=lambda x: x[1], reverse=True)
        target_ids = [eid for eid, _ in sorted_experts[:2]]
        lines.append(f"Selected experts: {target_ids}")

        # Query each expert
        for eid in target_ids:
            expert = self._get_expert(eid)
            result = expert.ask(question)
            lines.append(f"\n  Expert {eid:02d}:")
            lines.append(f"    Strategy: {result['strategy']}")
            lines.append(f"    Confidence: {result['confidence']:.3f}")
            lines.append(f"    Answer: {result['answer'][:80]}")
            lines.append(f"    Trace: {result.get('trace', 'none')}")

        return "\n".join(lines)

    def stats(self) -> dict:
        """Aggregate stats across all experts."""
        total_neurons = 0
        total_words = 0
        expert_details = []
        for i in range(self.num_experts):
            if self._experts[i] is not None:
                s = self._experts[i].stats()
                total_neurons += s['neurons']
                total_words += s['words']
                expert_details.append((i, s['neurons'], s['words']))
        return {
            "neurons": total_neurons,
            "words": total_words,
            "num_experts": self.num_experts,
            "active_experts": len(expert_details),
            "routed_words": len(self._routes),
            "expert_details": expert_details,
        }

    def health(self) -> dict:
        """Aggregate health across all experts."""
        import resource
        import shutil
        ru = resource.getrusage(resource.RUSAGE_SELF)
        disk = shutil.disk_usage(self.db_path)
        total_matrix_mb = 0
        total_db_mb = 0
        for i in range(self.num_experts):
            if self._experts[i] is not None:
                h = self._experts[i].health()
                total_matrix_mb += h.get('brain_matrix_mb', 0)
                total_db_mb += h.get('db_size_mb', 0)
        return {
            "neurons": sum(self._expert_counts),
            "routed_words": len(self._routes),
            "active_experts": sum(1 for e in self._experts if e is not None),
            "total_matrix_mb": round(total_matrix_mb, 1),
            "total_db_mb": round(total_db_mb, 1),
            "rss_mb": round(ru.ru_maxrss / 1024, 1),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "cpu_user_s": round(ru.ru_utime, 2),
            "memory_pressure": ru.ru_maxrss > 800 * 1024,
            "oom_risk": self._check_oom(),
            "disk_pressure": disk.free < 5 * (1024**3),
            "death_risk": self._death_risk(ru, disk),
        }

    @staticmethod
    def _death_risk(ru, disk):
        """0 = healthy, 100 = about to die."""
        risk = 0
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        avail_mb = int(line.split()[1]) / 1024
                        if avail_mb < 2048:
                            risk += int((2048 - avail_mb) / 2048 * 50)
                        break
        except Exception:
            pass
        free_gb = disk.free / (1024**3)
        if free_gb < 5:
            risk += int((5 - free_gb) / 5 * 30)
        rss_mb = ru.ru_maxrss / 1024
        if rss_mb > 800:
            risk += min(20, int((rss_mb - 800) / 800 * 20))
        return min(100, risk)

    @staticmethod
    def _check_oom():
        """Check system memory — are we approaching OOM?"""
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        avail_mb = int(line.split()[1]) / 1024
                        return avail_mb < 1024  # under 1GB = danger
        except Exception:
            return False

    def begin_bulk(self):
        """Enter bulk mode on all loaded experts."""
        self._bulk_mode = True
        for e in self._experts:
            if e is not None:
                e.begin_bulk()

    def end_bulk(self):
        """Exit bulk mode — reindex all experts."""
        self._bulk_mode = False
        for e in self._experts:
            if e is not None:
                e.end_bulk()

    def rebalance(self):
        """Self-organize: move word ROUTES to better experts based on co-occurrence.

        This is lightweight neuroplasticity — it changes which expert
        future teaches/queries route to, but does NOT move existing
        neuron data between experts. Full neuron migration (deleting
        from one expert's matrix and inserting into another) is the
        next level — that's brain surgery, needs to happen during REST.

        For each word, check if its co-occurring neighbors mostly live
        in a different expert. If so, reroute future queries there.
        """
        moves = []
        for word, current_eid in list(self._routes.items()):
            # Find where this word's co-occurring neighbors live
            expert = self._get_expert(current_eid)
            info = expert.inspect(word)
            if not info.get("known"):
                continue

            # Check connections — where do they live?
            neighbor_experts = {}
            for neighbor_word, strength in info.get("connections", [])[:10]:
                n_eid = self._routes.get(neighbor_word, -1)
                if n_eid >= 0 and n_eid != current_eid:
                    neighbor_experts[n_eid] = neighbor_experts.get(n_eid, 0) + strength

            if not neighbor_experts:
                continue

            # If majority of strong connections are in another expert, migrate
            best_other = max(neighbor_experts, key=neighbor_experts.get)
            if neighbor_experts[best_other] > 2.0:  # strong signal
                moves.append((word, current_eid, best_other))

        # Execute moves
        for word, from_eid, to_eid in moves[:50]:  # cap per cycle
            self._routes[word] = to_eid
            self._expert_counts[from_eid] = max(0, self._expert_counts[from_eid] - 1)
            self._expert_counts[to_eid] += 1

        if moves:
            self._flush_router()

        return len(moves)

    def close(self):
        """Close all experts and router."""
        self._flush_router()
        for e in self._experts:
            if e is not None:
                e.close()
        self._router_db.close()
