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
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from brain import Brain, FUNCTION_WORDS


class MoEBrain:
    """Mixture of Expert brains with a semantic router."""

    def __init__(self, db_path: str, num_experts: int = 16):
        self.db_path = db_path
        self.num_experts = num_experts

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

        # Load router into memory
        self._routes = {}  # word → expert_id
        for word, eid in self._router_db.execute("SELECT word, expert_id FROM routes"):
            self._routes[word] = eid

        # Expert word counts (for load balancing)
        self._expert_counts = [0] * num_experts
        for eid, count in self._router_db.execute("SELECT expert_id, word_count FROM expert_stats"):
            if eid < num_experts:
                self._expert_counts[eid] = count

        # Lazy-load experts — only create Brain instances when needed
        self._experts = [None] * num_experts
        self._expert_dir = expert_dir
        self._lock = Lock()

        # Pool for parallel expert queries
        self._pool = ThreadPoolExecutor(max_workers=min(num_experts, 8))

    def _get_expert(self, expert_id: int) -> Brain:
        """Lazy-load an expert brain."""
        if self._experts[expert_id] is None:
            path = os.path.join(self._expert_dir, f'expert_{expert_id:02d}')
            os.makedirs(path, exist_ok=True)
            brain = Brain(db_path=path)
            if getattr(self, '_bulk_mode', False):
                brain.begin_bulk()
            self._experts[expert_id] = brain
        return self._experts[expert_id]

    def _route_word(self, word: str) -> int:
        """Get or assign an expert for a word."""
        word = word.lower().strip()
        if word in self._routes:
            return self._routes[word]
        return -1  # unassigned

    def _route_sentence(self, tokens: list) -> int:
        """Route a sentence to an expert.

        Strategy: vote from already-routed RARE words (not high-frequency).
        High-frequency words appear everywhere and don't signal domain.
        If no signal, assign to the lightest expert (load balance).
        """
        content = [t.lower().strip() for t in tokens if t.lower().strip() not in FUNCTION_WORDS]
        if not content:
            return 0

        # Only vote with rare words (appear in ≤2 experts = domain-specific)
        votes = {}
        for word in content:
            eid = self._routes.get(word, -1)
            if eid >= 0:
                # Weight by inverse frequency: rare words vote louder
                freq = self._expert_counts[eid]
                if freq < 500:  # rare enough to be a signal
                    votes[eid] = votes.get(eid, 0) + 1

        if votes:
            return max(votes, key=votes.get)

        # No rare-word signal — assign to lightest expert
        return int(np.argmin(self._expert_counts))

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
        return result

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

    def close(self):
        """Close all experts and router."""
        self._flush_router()
        for e in self._experts:
            if e is not None:
                e.close()
        self._router_db.close()
