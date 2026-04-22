#!/usr/bin/env python3
"""
Bulk-teach all rich-text datasets to the live brain.

Feeds Wikipedia, OASST conversations, HotPotQA contexts, NQ passages,
and Q&A pairs. Skips audio/image-only datasets.

Usage:
    python3 teach_bulk.py                     # teach to live brain directly
    python3 teach_bulk.py --limit 10000       # cap per dataset
    python3 teach_bulk.py --dataset wikipedia_en  # single dataset
"""

import sys, os, json, time, argparse, re
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

DATA_DIR = Path.home() / "webmind-research" / "data"

# Skip datasets that are audio/image captions (no reasoning content)
SKIP = {"audiocaps", "coco_captions", "vggsound", "stackoverflow"}

# Datasets with full text paragraphs (highest value — teach as sentences)
TEXT_DATASETS = {
    "wikipedia_en", "wikipedia_simple", "hotpotqa", "squad",
    "natural_questions", "oasst_conversations", "triviaqa", "wikiqa", "webq",
    "literature",
}

# Datasets with Q&A pairs (teach via correct() for direct lookup)
QA_DATASETS = {
    "arc", "arc_challenge", "bbh_causal", "bbh_logic", "codealpaca",
    "codesearchnet_python", "dolly_conversations", "extracted_qa",
    "gsm8k", "hle", "mmlu_algebra", "mmlu_astronomy", "mmlu_compsec",
    "mmlu_philosophy", "mmlu_physics", "mmlu_pro", "mmlu_religions",
    "stackoverflow_qa", "strategyqa",
    # New knowledge domains
    "medicine", "psychology", "philosophy_ethics", "history_politics",
    "science_extended", "math_reasoning", "law_economics",
}


def split_sentences(text):
    """Split text into sentences for teaching."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter: min 10 chars, max 500 chars
    return [s.strip() for s in sentences if 10 <= len(s.strip()) <= 500]


def teach_text_dataset(brain, ds_path, limit):
    """Teach a text-heavy dataset (Wikipedia, OASST, etc.)."""
    taught = 0
    with open(ds_path) as f:
        for line in f:
            if limit and taught >= limit:
                break
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            text = item.get('text', '').strip()
            if not text or len(text) < 20:
                continue

            # Teach each sentence individually for granular retrieval
            sentences = split_sentences(text)
            for sent in sentences:
                brain.teach(sent, confidence=0.5)
                taught += 1
                if limit and taught >= limit:
                    break

    return taught


def teach_qa_dataset(brain, ds_path, limit):
    """Teach Q&A pairs via correct() for direct lookup + edge boosting."""
    taught = 0
    with open(ds_path) as f:
        for line in f:
            if limit and taught >= limit:
                break
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()

            if not question or not answer:
                continue

            # Teach the answer text (creates co-occurrence edges)
            brain.teach(answer, confidence=0.6)
            # Store direct Q→A mapping
            brain.correct(question, answer)
            taught += 1

    return taught


def main():
    parser = argparse.ArgumentParser(description="Bulk teach datasets to brain")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max records per dataset (None=all)")
    parser.add_argument("--dataset", default="all",
                        help="Single dataset name or 'all'")
    parser.add_argument("--db-path", default=os.path.expanduser("~/nexus-brain"),
                        help="Brain DB path")
    args = parser.parse_args()

    # Load brain
    from brain_csr_adapter import BrainCSR
    brain = BrainCSR(db_path=args.db_path)

    if args.dataset == 'all':
        ds_paths = sorted(DATA_DIR.glob("*.jsonl"))
    else:
        ds_paths = [DATA_DIR / f"{args.dataset}.jsonl"]

    total_taught = 0
    t_start = time.time()

    for ds_path in ds_paths:
        name = ds_path.stem
        if name in SKIP:
            continue
        if not ds_path.exists() or ds_path.stat().st_size == 0:
            continue

        t0 = time.time()
        if name in TEXT_DATASETS:
            n = teach_text_dataset(brain, ds_path, args.limit)
            kind = "text"
        elif name in QA_DATASETS:
            n = teach_qa_dataset(brain, ds_path, args.limit)
            kind = "Q&A"
        else:
            continue

        elapsed = time.time() - t0
        total_taught += n
        print(f"  {name}: {n:,} {kind} records ({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\nDone: {total_taught:,} records taught in {elapsed:.1f}s")
    print(f"Brain: {len(brain._words):,} words, WAL: {brain._wal.entry_count:,} edges")
    brain.close()


if __name__ == '__main__':
    main()
