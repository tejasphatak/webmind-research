#!/usr/bin/env python3
"""
Migrate an existing SQLite neurons.db to LMDB format.

Usage:
    python3 migrate_to_lmdb.py /path/to/sqlite_dir /path/to/lmdb_dir

The SQLite dir should contain neurons.db.
The LMDB dir will be created if it doesn't exist.
"""

import sqlite3
import struct
import sys
import time
from pathlib import Path

import lmdb
import numpy as np

# Reuse struct formats from neuron_lmdb
_NEURON_FMT = struct.Struct('<f q ? b')
_ID_FMT = struct.Struct('<i')
_ID_CONF_FMT = struct.Struct('<i f')
_SENT_ENTRY_FMT = struct.Struct('<i i')


def _pack_id(nid: int) -> bytes:
    return _ID_FMT.pack(nid)


def _encode_id_list(ids: list) -> bytes:
    return b''.join(_ID_FMT.pack(i) for i in ids)


def migrate(sqlite_dir: str, lmdb_dir: str, map_size: int = 4 * 1024 * 1024 * 1024):
    sqlite_path = Path(sqlite_dir) / "neurons.db"
    if not sqlite_path.exists():
        print(f"Error: {sqlite_path} not found")
        sys.exit(1)

    Path(lmdb_dir).mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(str(sqlite_path))
    env = lmdb.open(lmdb_dir, max_dbs=12, map_size=map_size)

    db_neurons = env.open_db(b'neurons')
    db_vectors = env.open_db(b'vectors')
    db_successors = env.open_db(b'successors')
    db_predecessors = env.open_db(b'predecessors')
    db_words = env.open_db(b'words')
    db_sentences = env.open_db(b'sentences')
    db_sent_index = env.open_db(b'sent_index')
    db_misses = env.open_db(b'misses')
    db_meta = env.open_db(b'meta')

    t0 = time.time()

    # --- Neurons ---
    print("Migrating neurons...")
    rows = db.execute(
        "SELECT id, confidence, successors, predecessors, "
        "timestamp, temporal, level, vector FROM neurons ORDER BY id"
    ).fetchall()

    max_id = 0
    with env.begin(write=True) as txn:
        for nid, conf, succ_bytes, pred_bytes, ts, temporal, level, vec_bytes in rows:
            key = _pack_id(nid)
            txn.put(key, _NEURON_FMT.pack(conf, ts, bool(temporal), level), db=db_neurons)
            txn.put(key, vec_bytes, db=db_vectors)
            txn.put(key, succ_bytes if succ_bytes else b'', db=db_successors)
            txn.put(key, pred_bytes if pred_bytes else b'', db=db_predecessors)
            if nid > max_id:
                max_id = nid

    print(f"  {len(rows)} neurons migrated")

    # --- Word mappings ---
    print("Migrating word mappings...")
    word_rows = db.execute("SELECT word, neuron_id FROM word_neurons").fetchall()
    with env.begin(write=True) as txn:
        for word, nid in word_rows:
            txn.put(word.encode('utf-8'), _pack_id(nid), db=db_words)
    print(f"  {len(word_rows)} word mappings migrated")

    # --- Sentence associations ---
    print("Migrating sentence associations...")
    sent_rows = db.execute(
        "SELECT sentence_id, neuron_id, position FROM sentence_neurons "
        "ORDER BY sentence_id, position"
    ).fetchall()

    sentences = {}
    for sid, nid, pos in sent_rows:
        if sid not in sentences:
            sentences[sid] = []
        sentences[sid].append((nid, pos))

    # Build reverse index: neuron_id → [sentence_ids]
    neuron_sents = {}
    max_sid = -1

    with env.begin(write=True) as txn:
        for sid, entries in sentences.items():
            packed = b''.join(_SENT_ENTRY_FMT.pack(nid, pos) for nid, pos in entries)
            txn.put(_pack_id(sid), packed, db=db_sentences)
            if sid > max_sid:
                max_sid = sid
            for nid, _ in entries:
                if nid not in neuron_sents:
                    neuron_sents[nid] = []
                neuron_sents[nid].append(sid)

        # Write reverse index
        for nid, sids in neuron_sents.items():
            txn.put(_pack_id(nid), _encode_id_list(sids), db=db_sent_index)

    print(f"  {len(sentences)} sentences migrated")

    # --- Misses ---
    print("Migrating misses...")
    import json as _json
    miss_rows = db.execute(
        "SELECT id, query_text, query_vector, timestamp, resolved, "
        "resolved_timestamp, answer_text FROM misses ORDER BY id"
    ).fetchall()

    max_mid = -1
    with env.begin(write=True) as txn:
        for mid, qtext, qvec, ts, resolved, rts, answer in miss_rows:
            meta = {
                'query_text': qtext,
                'timestamp': ts,
                'resolved': resolved,
                'resolved_timestamp': rts,
                'answer_text': answer,
            }
            meta_bytes = _json.dumps(meta).encode('utf-8')
            val = _ID_FMT.pack(len(meta_bytes)) + meta_bytes + (qvec if qvec else b'')
            txn.put(_pack_id(mid), val, db=db_misses)
            if mid > max_mid:
                max_mid = mid
    print(f"  {len(miss_rows)} misses migrated")

    # --- Templates ---
    print("Migrating templates...")
    try:
        tmpl_rows = db.execute(
            "SELECT id, pattern, slots, confidence, vector FROM templates ORDER BY id"
        ).fetchall()
        with env.begin(write=True) as txn:
            for tid, pattern, slots, conf, vec_bytes in tmpl_rows:
                key = f"tmpl:{tid}".encode('utf-8')
                meta = _json.dumps({
                    'id': tid, 'pattern': pattern,
                    'slots': slots, 'confidence': conf,
                }).encode('utf-8')
                packed = _ID_FMT.pack(len(meta)) + meta + (vec_bytes if vec_bytes else b'')
                txn.put(key, packed, db=db_meta)
        print(f"  {len(tmpl_rows)} templates migrated")
    except Exception:
        print("  No templates table found, skipping")

    # --- Meta ---
    with env.begin(write=True) as txn:
        txn.put(b'next_id', _pack_id(max_id + 1), db=db_meta)
        txn.put(b'next_sentence_id', _pack_id(max_sid + 1), db=db_meta)
        txn.put(b'next_miss_id', _pack_id(max_mid + 1), db=db_meta)

    elapsed = time.time() - t0
    print(f"\nMigration complete in {elapsed:.2f}s")
    print(f"  LMDB at: {lmdb_dir}")

    db.close()
    env.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <sqlite_dir> <lmdb_dir>")
        sys.exit(1)
    migrate(sys.argv[1], sys.argv[2])
