#!/usr/bin/env python3
"""
SAQT Database — SQLite + FAISS
===============================
Migrates from JSONL+PyTorch to SQLite+FAISS.
- SQLite: stores Q&A pairs, compressed, ACID, single file
- FAISS: stores embeddings, <1ms search

Usage:
  from saqt_db import SAQTDB
  db = SAQTDB("/path/to/saqt.db")
  db.add("What is DNA?", "DNA is a double helix molecule...")
  results = db.search("What is DNA?", top_k=5)  # <1ms
"""

import sqlite3
import numpy as np
import faiss
import json, os, time
from pathlib import Path


class SAQTDB:
    def __init__(self, db_path, encoder=None):
        self.db_path = db_path
        self.index_path = db_path.replace('.db', '.faiss')
        self.encoder = encoder
        self.dim = 384

        # SQLite
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT DEFAULT '',
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON qa(source)")
        self.conn.commit()

        # FAISS
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # inner product (cosine after normalization)

        count = self.conn.execute("SELECT COUNT(*) FROM qa").fetchone()[0]
        print(f"[saqt-db] {count} pairs in SQLite, {self.index.ntotal} in FAISS", flush=True)

    def add(self, question, answer, source=""):
        """Add a Q&A pair. Encodes and indexes immediately."""
        self.conn.execute("INSERT INTO qa (question, answer, source) VALUES (?, ?, ?)",
                         (question, answer, source))
        self.conn.commit()

        if self.encoder:
            emb = self.encoder.encode([question], normalize_embeddings=True)
            self.index.add(emb.astype(np.float32))

    def add_batch(self, pairs):
        """Add multiple Q&A pairs at once. Much faster."""
        self.conn.executemany(
            "INSERT INTO qa (question, answer, source) VALUES (?, ?, ?)",
            [(p["question"], p["answer"], p.get("source", "")) for p in pairs])
        self.conn.commit()

        if self.encoder and pairs:
            questions = [p["question"] for p in pairs]
            embs = self.encoder.encode(questions, normalize_embeddings=True,
                                      batch_size=256, show_progress_bar=len(questions) > 1000)
            self.index.add(embs.astype(np.float32))

    def search(self, query, top_k=5):
        """Search for nearest Q&A pairs. Returns list of (id, question, answer, score)."""
        if self.encoder is None:
            raise RuntimeError("No encoder set")

        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            row_id = int(idx) + 1  # SQLite IDs start at 1
            row = self.conn.execute("SELECT question, answer, source FROM qa WHERE id=?",
                                   (row_id,)).fetchone()
            if row:
                results.append({
                    "id": row_id,
                    "question": row[0],
                    "answer": row[1],
                    "source": row[2],
                    "score": float(score),
                })
        return results

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM qa").fetchone()[0]

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        print(f"[saqt-db] Saved FAISS index: {self.index.ntotal} vectors", flush=True)

    def close(self):
        self.save_index()
        self.conn.close()


def migrate_from_jsonl(jsonl_path, db_path, encoder):
    """Migrate from JSONL to SQLite+FAISS."""
    print(f"[migrate] {jsonl_path} → {db_path}", flush=True)

    db = SAQTDB(db_path, encoder=encoder)

    if db.count() > 0:
        print(f"[migrate] DB already has {db.count()} pairs, skipping", flush=True)
        return db

    pairs = []
    with open(jsonl_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"[migrate] Loading {len(pairs)} pairs...", flush=True)
    t0 = time.time()

    # Batch insert
    batch_size = 5000
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        db.add_batch(batch)
        if (i + batch_size) % 20000 == 0:
            print(f"  {i+len(batch)}/{len(pairs)}", flush=True)

    db.save_index()
    elapsed = time.time() - t0
    print(f"[migrate] Done in {elapsed:.0f}s. {db.count()} pairs, {db.index.ntotal} vectors", flush=True)
    return db


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    home = os.environ.get("HOME", "/home/tejasphatak")
    jsonl = os.path.join(home, "webmind-research/trained_model/qa_pairs.jsonl")
    db_path = os.path.join(home, "webmind-research/trained_model/saqt.db")

    db = migrate_from_jsonl(jsonl, db_path, encoder)

    # Test search speed
    print("\n[test] Search speed:", flush=True)
    queries = ["What is DNA?", "Who is Narendra Modi?", "347 * 29",
               "What causes rain?", "How are you?"]
    for q in queries:
        t0 = time.time()
        results = db.search(q, top_k=3)
        ms = (time.time() - t0) * 1000
        best = results[0] if results else {}
        print(f"  {ms:.1f}ms  Q: {q[:30]:30s} → {best.get('answer','?')[:50]}", flush=True)

    db.close()
