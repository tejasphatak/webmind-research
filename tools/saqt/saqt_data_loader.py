#!/usr/bin/env python3
"""
SAQT Data Loader — Download and prepare knowledge datasets
============================================================
Downloads key datasets, chunks them, and prepares for SAQT vector DB ingestion.

Datasets:
1. Wikipedia (Simple English — manageable size, good coverage)
2. HotpotQA (multi-hop QA benchmark)
3. SQuAD 2.0 (reading comprehension)
4. HLE (Humanity's Last Exam — frontier benchmark)
5. Semantic Scholar / arXiv (scientific papers)

Output: JSONL files with {text, topic, source} per chunk
"""

import json, os, sys, time
from pathlib import Path

OUT_DIR = Path(os.environ.get("SAQT_DATA_DIR", os.path.expanduser("~/webmind-research/data")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def load_simple_wikipedia(max_articles=10000):
    """Simple English Wikipedia — clean, factual, manageable."""
    from datasets import load_dataset
    print(f"Loading Simple Wikipedia (max {max_articles} articles)...", flush=True)
    ds = load_dataset("wikipedia", "20220301.simple", split="train",
                      trust_remote_code=True)

    out_file = OUT_DIR / "wikipedia_simple.jsonl"
    count = 0
    with open(out_file, "w") as f:
        for i, row in enumerate(ds):
            if i >= max_articles:
                break
            text = row["text"].strip()
            if len(text) < 50:
                continue
            # Chunk into paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
            for para in paragraphs[:5]:  # max 5 paragraphs per article
                if len(para) > 500:
                    para = para[:500]
                f.write(json.dumps({
                    "text": para,
                    "topic": row.get("title", "general"),
                    "source": "wikipedia_simple"
                }) + "\n")
                count += 1

    print(f"  Saved {count} chunks to {out_file}", flush=True)
    return count


def load_hotpotqa(max_examples=5000):
    """HotpotQA — multi-hop QA with supporting facts."""
    from datasets import load_dataset
    print(f"Loading HotpotQA (max {max_examples})...", flush=True)
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train",
                      trust_remote_code=True)

    out_file = OUT_DIR / "hotpotqa.jsonl"
    count = 0
    with open(out_file, "w") as f:
        for i, row in enumerate(ds):
            if i >= max_examples:
                break
            # Store the question + answer as a fact
            q = row.get("question", "")
            a = row.get("answer", "")
            if q and a:
                f.write(json.dumps({
                    "text": f"{q} Answer: {a}",
                    "topic": "qa",
                    "source": "hotpotqa"
                }) + "\n")
                count += 1
            # Also store supporting context paragraphs
            contexts = row.get("context", {})
            if isinstance(contexts, dict):
                titles = contexts.get("title", [])
                sentences_list = contexts.get("sentences", [])
                for title, sents in zip(titles, sentences_list):
                    text = " ".join(sents)
                    if len(text) > 30:
                        f.write(json.dumps({
                            "text": text[:500],
                            "topic": title,
                            "source": "hotpotqa_context"
                        }) + "\n")
                        count += 1

    print(f"  Saved {count} chunks to {out_file}", flush=True)
    return count


def load_squad(max_examples=5000):
    """SQuAD 2.0 — reading comprehension."""
    from datasets import load_dataset
    print(f"Loading SQuAD 2.0 (max {max_examples})...", flush=True)
    ds = load_dataset("rajpurkar/squad_v2", split="train",
                      trust_remote_code=True)

    out_file = OUT_DIR / "squad.jsonl"
    count = 0
    seen_contexts = set()
    with open(out_file, "w") as f:
        for i, row in enumerate(ds):
            if count >= max_examples:
                break
            context = row.get("context", "").strip()
            if context and context not in seen_contexts:
                seen_contexts.add(context)
                f.write(json.dumps({
                    "text": context[:500],
                    "topic": row.get("title", "general"),
                    "source": "squad"
                }) + "\n")
                count += 1
            # Also store Q+A
            q = row.get("question", "")
            answers = row.get("answers", {}).get("text", [])
            if q and answers:
                f.write(json.dumps({
                    "text": f"{q} Answer: {answers[0]}",
                    "topic": row.get("title", "general"),
                    "source": "squad_qa"
                }) + "\n")
                count += 1

    print(f"  Saved {count} chunks to {out_file}", flush=True)
    return count


def load_hle(max_examples=500):
    """HLE — Humanity's Last Exam (frontier benchmark)."""
    from datasets import load_dataset
    print(f"Loading HLE (max {max_examples})...", flush=True)
    try:
        ds = load_dataset("cais/hle", split="test", token=HF_TOKEN,
                          trust_remote_code=True)
        out_file = OUT_DIR / "hle.jsonl"
        count = 0
        with open(out_file, "w") as f:
            for i, row in enumerate(ds):
                if i >= max_examples:
                    break
                q = row.get("question", "")
                a = row.get("answer", "")
                category = row.get("category", "general")
                if q:
                    entry = {"text": q, "topic": category, "source": "hle_question"}
                    if a:
                        entry["answer"] = a
                    f.write(json.dumps(entry) + "\n")
                    count += 1
        print(f"  Saved {count} chunks to {out_file}", flush=True)
        return count
    except Exception as e:
        print(f"  HLE failed: {e}", flush=True)
        return 0


def load_arxiv_abstracts(max_papers=10000):
    """arXiv abstracts — scientific knowledge."""
    from datasets import load_dataset
    print(f"Loading arXiv abstracts (max {max_papers})...", flush=True)
    try:
        ds = load_dataset("ccdv/arxiv-summarization", split="train",
                          trust_remote_code=True)
        out_file = OUT_DIR / "arxiv.jsonl"
        count = 0
        with open(out_file, "w") as f:
            for i, row in enumerate(ds):
                if i >= max_papers:
                    break
                abstract = row.get("abstract", "").strip()
                if len(abstract) > 50:
                    f.write(json.dumps({
                        "text": abstract[:500],
                        "topic": "science",
                        "source": "arxiv"
                    }) + "\n")
                    count += 1
        print(f"  Saved {count} chunks to {out_file}", flush=True)
        return count
    except Exception as e:
        print(f"  arXiv failed: {e}", flush=True)
        return 0


def summary():
    """Print summary of all downloaded data."""
    print(f"\n{'='*50}", flush=True)
    print("SAQT DATA SUMMARY", flush=True)
    print(f"{'='*50}", flush=True)
    total = 0
    for f in sorted(OUT_DIR.glob("*.jsonl")):
        lines = sum(1 for _ in open(f))
        size = f.stat().st_size
        total += lines
        print(f"  {f.name:30s}  {lines:>8,} chunks  {size/1e6:.1f} MB", flush=True)
    print(f"  {'TOTAL':30s}  {total:>8,} chunks", flush=True)
    print(f"  Output dir: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--wiki", type=int, default=10000, help="Max Wikipedia articles")
    p.add_argument("--hotpot", type=int, default=5000, help="Max HotpotQA examples")
    p.add_argument("--squad", type=int, default=5000, help="Max SQuAD examples")
    p.add_argument("--hle", type=int, default=500, help="Max HLE examples")
    p.add_argument("--arxiv", type=int, default=10000, help="Max arXiv papers")
    p.add_argument("--only", type=str, default="", help="Only load this dataset")
    args = p.parse_args()

    t0 = time.time()
    if args.only:
        datasets = [args.only]
    else:
        datasets = ["wiki", "hotpot", "squad", "hle", "arxiv"]

    for ds in datasets:
        if ds == "wiki":
            load_simple_wikipedia(args.wiki)
        elif ds == "hotpot":
            load_hotpotqa(args.hotpot)
        elif ds == "squad":
            load_squad(args.squad)
        elif ds == "hle":
            load_hle(args.hle)
        elif ds == "arxiv":
            load_arxiv_abstracts(args.arxiv)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)
    summary()
