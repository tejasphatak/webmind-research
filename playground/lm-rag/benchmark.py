"""
Benchmark LM-RAG on NQ Open-Domain (Natural Questions).

Tests:
1. Route accuracy — does the model correctly route to SEARCH?
2. Answer quality — given correct Wikipedia context, does it extract the right answer?
3. End-to-end — full DFS pipeline (slow, run fewer)

Usage:
  python benchmark.py --quick    # 50 questions, route + answer only
  python benchmark.py --full 10  # 10 questions, full DFS (slow)
"""

import os
import sys
import time
import re
import json

def normalize(text):
    """Normalize answer for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\b(the|a|an)\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def exact_match(pred, golds):
    """Check if prediction matches any gold answer."""
    pred_norm = normalize(pred)
    for gold in golds:
        if normalize(gold) == pred_norm:
            return True
        if normalize(gold) in pred_norm or pred_norm in normalize(gold):
            return True
    return False


def run_quick(n=50):
    """Test route + answer skills on NQ questions (no Wikipedia search)."""
    from datasets import load_dataset
    from engine import ModelPool

    pool = ModelPool()
    nq = load_dataset('nq_open', split='validation', streaming=True)

    route_correct = 0
    answer_correct = 0
    total = 0
    total_time = 0

    print(f"NQ Open-Domain Benchmark — {n} questions (route + answer)")
    print("=" * 60)

    for i, ex in enumerate(nq):
        if i >= n:
            break

        q = ex['question']
        golds = ex['answer']
        total += 1

        # Route
        t0 = time.time()
        route = pool.call('route', q, max_len=20)
        dt_route = time.time() - t0

        is_search = route.strip().upper().startswith('SEARCH')
        if is_search:
            route_correct += 1

        # Answer (give a generic context hint — no Wikipedia)
        t0 = time.time()
        answer = pool.call('answer', f"question: {q}", max_len=30)
        dt_answer = time.time() - t0

        total_time += dt_route + dt_answer
        em = exact_match(answer, golds)
        if em:
            answer_correct += 1

        status = '✓' if em else '✗'
        if i < 10 or em:  # Show first 10 + all correct
            print(f"  [{status}] Q: {q[:50]:50s} → {answer[:30]:30s} gold: {golds[0][:20]}")

    print()
    print(f"Route to SEARCH: {route_correct}/{total} ({route_correct*100//total}%)")
    print(f"Answer EM (no context): {answer_correct}/{total} ({answer_correct*100//total}%)")
    print(f"Avg time: {total_time*1000/total:.0f}ms per question")
    print(f"Total time: {total_time:.1f}s")


def run_full(n=5):
    """Full end-to-end with DFS (slow)."""
    from datasets import load_dataset
    from engine import LMRAGEngine

    engine = LMRAGEngine()
    nq = load_dataset('nq_open', split='validation', streaming=True)

    correct = 0
    total = 0

    print(f"NQ Full DFS — {n} questions (end-to-end)")
    print("=" * 60)

    for i, ex in enumerate(nq):
        if i >= n:
            break

        q = ex['question']
        golds = ex['answer']
        total += 1

        t0 = time.time()
        answer = engine.ask(q, verbose=True)
        dt = time.time() - t0

        em = exact_match(answer, golds)
        if em:
            correct += 1

        status = '✓' if em else '✗'
        print(f"  [{status}] ({dt:.1f}s) Q: {q[:40]:40s} → {answer[:30]:30s} gold: {golds[0][:20]}")

    print()
    print(f"EM: {correct}/{total} ({correct*100//total}%)")


if __name__ == '__main__':
    if '--full' in sys.argv:
        n = int(sys.argv[sys.argv.index('--full') + 1]) if len(sys.argv) > sys.argv.index('--full') + 1 else 5
        run_full(n)
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 50
        run_quick(n)
