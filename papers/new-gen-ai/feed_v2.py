#!/usr/bin/env python3
"""
Parallel data feeder for BrainV2.

Encodes sentences with MiniLM using multiprocessing (4 workers),
then writes to LMDB sequentially.

Usage:
    python3 feed_v2.py                    # all quality datasets
    python3 feed_v2.py --limit 5000       # cap per dataset
"""

import os, sys, json, time, argparse, re
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

DATA_DIR = Path.home() / "webmind-research" / "data"

# Quality natural language datasets only
DATASETS = [
    'natural_questions', 'triviaqa', 'wikiqa', 'webq',
    'dolly_conversations', 'wikipedia_en', 'wikipedia_simple',
    'hotpotqa', 'squad',
    'medicine', 'psychology', 'philosophy_ethics', 'literature',
    'history_politics', 'science_extended', 'law_economics',
    'arc', 'arc_challenge', 'strategyqa',
    'mmlu_philosophy', 'mmlu_physics', 'mmlu_astronomy', 'mmlu_religions',
    'mmlu_algebra', 'mmlu_compsec',
]


def load_records(ds_name, limit):
    """Load records from a dataset, return (sentences, qa_pairs)."""
    path = DATA_DIR / f"{ds_name}.jsonl"
    if not path.exists():
        return [], []

    sentences = []
    qa_pairs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                item = json.loads(line.strip())
            except:
                continue
            q = item.get('question', '')
            a = item.get('answer', '')
            text = item.get('text', '')

            if a and len(a) >= 15:
                sentences.append(a)
            if text and len(text) >= 15 and len(text) <= 1000:
                sentences.append(text)
            if q and a and len(a) >= 5:
                qa_pairs.append((q, a))

    return sentences, qa_pairs


def encode_batch(sentences):
    """Encode a batch of sentences with MiniLM. Runs in worker process."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences, normalize_embeddings=True,
                               show_progress_bar=False, batch_size=128)
    return embeddings.astype(np.float32)


def chunk_list(lst, n):
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()))
    args = parser.parse_args()

    print(f"Workers: {args.workers}, Limit: {args.limit}/dataset")

    # Phase 1: Load all records
    t0 = time.time()
    all_sentences = []
    all_qa = []
    for ds in DATASETS:
        sents, qas = load_records(ds, args.limit)
        all_sentences.extend(sents)
        all_qa.extend(qas)
        if sents or qas:
            print(f"  {ds}: {len(sents):,} sentences, {len(qas):,} Q&A")

    # Deduplicate sentences
    seen = set()
    unique = []
    for s in all_sentences:
        key = s.strip().lower()[:100]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    all_sentences = unique

    print(f"\nLoaded: {len(all_sentences):,} unique sentences, {len(all_qa):,} Q&A pairs ({time.time()-t0:.1f}s)")

    # Phase 2: Parallel MiniLM encoding
    print(f"\nEncoding {len(all_sentences):,} sentences with {args.workers} workers...")
    t0 = time.time()

    chunks = chunk_list(all_sentences, args.workers)
    with Pool(args.workers) as pool:
        results = pool.map(encode_batch, chunks)

    all_embeddings = np.vstack(results)
    encode_time = time.time() - t0
    rate = len(all_sentences) / encode_time
    print(f"Encoded in {encode_time:.1f}s ({rate:.0f} sentences/sec)")

    # Phase 3: Write to BrainV2 (sequential — LMDB single writer)
    print(f"\nWriting to BrainV2...")
    t0 = time.time()

    from brain_v2 import BrainV2, _tokenize, FUNCTION_WORDS, COOC_WEIGHT, WEIGHT_CLAMP
    import struct, lmdb

    brain = BrainV2()

    # Batch write sentences + embeddings
    for i, (sentence, embedding) in enumerate(zip(all_sentences, all_embeddings)):
        tokens = _tokenize(sentence)
        content = [t for t in tokens if t not in FUNCTION_WORDS]
        if len(content) < 2:
            continue

        # Learn words
        for word in content:
            brain._learn_word(word)

        # Store sentence + embedding
        sid = brain._sent_count
        brain._sent_count += 1
        brain._sent_texts.append(sentence)

        with brain._env.begin(write=True) as txn:
            key = struct.pack('<i', sid)
            txn.put(key, sentence.encode('utf-8'), db=brain._sentences_db)
            txn.put(key, embedding.tobytes(), db=brain._embeddings_db)

        # Update embedding matrix
        if brain._embeddings is None:
            brain._embeddings = embedding.reshape(1, -1)
        else:
            brain._embeddings = np.vstack([brain._embeddings, embedding.reshape(1, -1)])

        # Co-occurrence edges
        indices = [brain._word_idx[w] for w in content]
        for a_i in range(len(indices)):
            for b_i in range(a_i + 1, len(indices)):
                a, b = indices[a_i], indices[b_i]
                if a not in brain._cooc:
                    brain._cooc[a] = {}
                if b not in brain._cooc:
                    brain._cooc[b] = {}
                brain._cooc[a][b] = min(WEIGHT_CLAMP, brain._cooc[a].get(b, 0) + COOC_WEIGHT)
                brain._cooc[b][a] = min(WEIGHT_CLAMP, brain._cooc[b].get(a, 0) + COOC_WEIGHT)

        # Successor pairs
        for t_i in range(len(tokens) - 1):
            curr, nxt = tokens[t_i], tokens[t_i + 1]
            if curr in brain._word_idx and nxt in brain._word_idx:
                cidx = brain._word_idx[curr]
                nidx = brain._word_idx[nxt]
                if cidx not in brain._successors:
                    brain._successors[cidx] = {}
                brain._successors[cidx][nidx] = brain._successors[cidx].get(nidx, 0) + 1.0

        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,}/{len(all_sentences):,} sentences written")

    # Write Q→A pairs
    print(f"Writing {len(all_qa):,} Q→A pairs...")
    for q, a in all_qa:
        brain.correct(q, a)

    # Persist co-occurrence + successors
    brain.save_state()
    write_time = time.time() - t0

    print(f"\nDone in {write_time:.1f}s")
    print(brain.stats())
    brain.close()


if __name__ == '__main__':
    main()
