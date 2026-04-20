"""
MVP-4B: Generation Fluency Benchmark

Loads real GloVe embeddings, teaches real sentences, queries the system,
and scores fluency of each generation strategy.

This is the empirical test the design doc has been asking for since Round 12.
The question: can this system produce readable text?

Usage:
    python3 benchmarks/generation_benchmark.py [--glove-path PATH]

Output: structured results with fluency scores per strategy.
"""

import sys
import os
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from engine import Engine

# --- Test corpus: real sentences the system should learn ---

TEACH_SENTENCES = [
    # People + achievements
    "shakespeare wrote hamlet",
    "einstein discovered relativity",
    "newton invented calculus",
    "darwin wrote the origin of species",
    "turing invented the computer",
    "galileo discovered the moons of jupiter",
    "beethoven composed the moonlight sonata",
    "curie discovered radium",

    # Capitals
    "paris is the capital of france",
    "london is the capital of england",
    "tokyo is the capital of japan",
    "berlin is the capital of germany",
    "madrid is the capital of spain",
    "rome is the capital of italy",

    # Facts
    "water boils at one hundred degrees",
    "the earth orbits the sun",
    "light travels at three hundred thousand kilometers per second",
    "dna carries genetic information",

    # Descriptions
    "python is a programming language",
    "gold is a precious metal",
    "the pacific is the largest ocean",
    "mount everest is the tallest mountain",
]

# --- Queries to test ---

QUERIES = [
    # Direct recall (in-distribution)
    ("who wrote hamlet", ["shakespeare"]),
    ("who discovered relativity", ["einstein"]),
    ("what is the capital of france", ["paris"]),
    ("what is the capital of japan", ["tokyo"]),
    ("who invented calculus", ["newton"]),
    ("who composed the moonlight sonata", ["beethoven"]),

    # Slight paraphrase
    ("capital of germany", ["berlin"]),
    ("what did curie discover", ["radium", "curie"]),
    ("what did darwin write", ["darwin", "origin", "species"]),

    # Harder (requires multiple concepts)
    ("what is python", ["programming", "language"]),
    ("tell me about gold", ["gold", "precious", "metal"]),
]


def score_answer(answer: str, expected_words: list) -> dict:
    """Score an answer against expected words."""
    answer_lower = answer.lower()
    found = [w for w in expected_words if w in answer_lower]
    recall = len(found) / len(expected_words) if expected_words else 0.0

    # Basic fluency heuristics
    words = answer.split()
    has_structure = len(words) >= 2  # more than one word
    not_just_list = "," not in answer or len(words) > 3  # not just comma-separated
    is_abstain = "don't know" in answer_lower

    fluency = 0.0
    if is_abstain:
        fluency = 0.0
    elif has_structure and not_just_list:
        fluency = 0.8  # template or successor walk
    elif has_structure:
        fluency = 0.4  # concept list but multiple words
    else:
        fluency = 0.2  # single word

    return {
        "recall": recall,
        "fluency": fluency,
        "found": found,
        "missing": [w for w in expected_words if w not in answer_lower],
        "is_abstain": is_abstain,
    }


def run_benchmark(glove_path: str = None):
    """Run the full generation benchmark."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="gen-bench-")

    print("=" * 60)
    print("MVP-4B: Generation Fluency Benchmark")
    print("=" * 60)

    # Init engine
    engine = Engine(data_dir=tmp)

    # Load GloVe
    print(f"\nLoading GloVe embeddings...")
    t0 = time.time()
    if glove_path:
        engine.load_embeddings(glove_path)
    else:
        default_path = os.path.expanduser(
            "~/webmind-research/papers/new-gen-ai/data/glove.6B.300d.txt"
        )
        if os.path.exists(default_path):
            engine.load_embeddings(default_path)
        else:
            print(f"ERROR: GloVe not found at {default_path}")
            print("Usage: python3 benchmarks/generation_benchmark.py --glove-path /path/to/glove.6B.300d.txt")
            return
    t1 = time.time()
    print(f"Loaded {engine.encoder.vocab_size} words in {t1-t0:.1f}s")

    # Teach sentences
    print(f"\nTeaching {len(TEACH_SENTENCES)} sentences...")
    for sentence in TEACH_SENTENCES:
        engine.teach_sentence(sentence)
    stats = engine.stats()
    print(f"KB: {stats['neurons']} neurons, {stats['templates']} templates")

    # Run queries
    print(f"\nRunning {len(QUERIES)} queries...")
    print("-" * 60)

    results_by_strategy = {}
    all_results = []

    for query_text, expected in QUERIES:
        result = engine.query(query_text)
        score = score_answer(result.answer, expected)

        strategy = result.strategy
        if strategy not in results_by_strategy:
            results_by_strategy[strategy] = []
        results_by_strategy[strategy].append(score)
        all_results.append((query_text, result, score))

        status = "MISS" if score["is_abstain"] else "HIT"
        print(f"  [{status}] Q: {query_text}")
        print(f"       A: {result.answer}")
        print(f"       Strategy: {strategy}, Recall: {score['recall']:.0%}, "
              f"Fluency: {score['fluency']:.0%}")
        if score["missing"]:
            print(f"       Missing: {score['missing']}")
        print()

    # Summary
    print("=" * 60)
    print("RESULTS BY STRATEGY")
    print("=" * 60)

    for strategy, scores in sorted(results_by_strategy.items()):
        count = len(scores)
        avg_recall = sum(s["recall"] for s in scores) / count
        avg_fluency = sum(s["fluency"] for s in scores) / count
        abstains = sum(1 for s in scores if s["is_abstain"])
        print(f"\n  {strategy} ({count} queries):")
        print(f"    Avg Recall:  {avg_recall:.0%}")
        print(f"    Avg Fluency: {avg_fluency:.0%}")
        print(f"    Abstains:    {abstains}")

    # Overall
    total = len(all_results)
    hits = sum(1 for _, _, s in all_results if not s["is_abstain"])
    avg_recall = sum(s["recall"] for _, _, s in all_results) / total
    avg_fluency = sum(s["fluency"] for _, _, s in all_results) / total

    print(f"\n{'=' * 60}")
    print(f"OVERALL ({total} queries):")
    print(f"  Hit Rate:    {hits}/{total} ({hits/total:.0%})")
    print(f"  Avg Recall:  {avg_recall:.0%}")
    print(f"  Avg Fluency: {avg_fluency:.0%}")
    print(f"{'=' * 60}")

    # Self-evolution test: correct the misses and re-query
    misses = [(q, exp) for q, r, s in all_results if s["is_abstain"]]
    if misses:
        print(f"\n{'=' * 60}")
        print(f"SELF-EVOLUTION: Correcting {len(misses)} misses and re-querying")
        print(f"{'=' * 60}")

        for query_text, expected in misses:
            # Find the original teach sentence that answers this query
            answer = _find_answer_for(query_text, TEACH_SENTENCES)
            if answer:
                engine.correct(query_text, answer)

        # Re-query the misses
        evolution_hits = 0
        for query_text, expected in misses:
            result = engine.query(query_text)
            score = score_answer(result.answer, expected)
            status = "FIXED" if not score["is_abstain"] else "STILL MISS"
            if not score["is_abstain"]:
                evolution_hits += 1
            print(f"  [{status}] Q: {query_text}")
            print(f"           A: {result.answer}")
            print()

        print(f"  Evolution recovery: {evolution_hits}/{len(misses)} "
              f"({evolution_hits/len(misses):.0%} of misses fixed)")

    engine.close()
    print(f"\nDone. Data at {tmp}")


def _find_answer_for(query: str, sentences: list) -> str:
    """Find the most relevant teach sentence for a query."""
    query_words = set(query.lower().split())
    best = None
    best_overlap = 0
    for s in sentences:
        s_words = set(s.lower().split())
        overlap = len(query_words & s_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best = s
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove-path", type=str, default=None)
    args = parser.parse_args()
    run_benchmark(args.glove_path)
