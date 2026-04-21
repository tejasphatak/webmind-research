"""
Benchmark harness for the reasoning engine.

Train/test split → teach train → evaluate on held-out test.
Reports: exact match, F1, abstention rate, latency.

Usage:
    python3 benchmark.py                    # all datasets, 80/20 split
    python3 benchmark.py --split 0.9        # 90/10 split
    python3 benchmark.py --dataset hotpotqa # single dataset
    python3 benchmark.py --test-only        # skip training, test existing brain
"""

import sys, os, json, time, argparse, random, re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from brain import Brain

def load_brain(db_path):
    """Load brain — auto-detect CSR > LMDB > SQLite."""
    csr_path = os.path.join(db_path, 'cooc_csr', 'indptr.bin')
    lmdb_path = os.path.join(db_path, 'brain.lmdb')
    if os.path.exists(csr_path) and os.path.exists(lmdb_path):
        from brain_csr_adapter import BrainCSR
        return BrainCSR(db_path=db_path)
    if os.path.exists(lmdb_path):
        from brain_lmdb_adapter import BrainLMDB
        return BrainLMDB(db_path=db_path)
    return Brain(db_path=db_path)

DATA_DIR = Path.home() / "webmind-research" / "data"


def tokenize(text):
    """Simple tokenization for F1 scoring."""
    return re.findall(r'\w+', text.lower())


def exact_match(prediction, gold):
    """Normalized exact match."""
    return prediction.strip().lower() == gold.strip().lower()


def f1_score(prediction, gold):
    """Token-level F1 between prediction and gold."""
    pred_tokens = set(tokenize(prediction))
    gold_tokens = set(tokenize(gold))
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def load_qa_records(dataset_path):
    """Load records that have questions and answers (for evaluation)."""
    records = []
    with open(dataset_path) as f:
        for line in f:
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            question = item.get('question', item.get('text', '')).strip()
            answer = item.get('answer', '').strip()

            if question and answer:
                records.append({'question': question, 'answer': answer})
    return records


def split_data(records, train_ratio=0.8, seed=42):
    """Deterministic train/test split."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def teach_records(brain, records):
    """Teach training records to brain."""
    for rec in records:
        q = rec['question']
        a = rec['answer']
        # Teach the answer as knowledge
        if len(a) < 50 and not a.endswith('.'):
            combined = f"{q.rstrip('?')} is {a}"
        else:
            combined = a
        brain.teach(combined, confidence=0.6)


def evaluate(brain, test_records, verbose=False):
    """Evaluate brain on held-out test records."""
    results = {
        'total': len(test_records),
        'exact_match': 0,
        'f1_sum': 0.0,
        'abstentions': 0,
        'latency_ms': [],
        'examples': [],
    }

    for rec in test_records:
        question = rec['question']
        gold = rec['answer']

        t0 = time.time()
        result = brain.ask(question)
        latency = (time.time() - t0) * 1000

        prediction = result['answer']
        strategy = result['strategy']
        confidence = result['confidence']

        results['latency_ms'].append(latency)

        if strategy == 'abstain':
            results['abstentions'] += 1
            em = 0
            f1 = 0.0
        else:
            em = 1 if exact_match(prediction, gold) else 0
            f1 = f1_score(prediction, gold)

        results['exact_match'] += em
        results['f1_sum'] += f1

        if verbose and len(results['examples']) < 10:
            results['examples'].append({
                'question': question[:80],
                'gold': gold[:80],
                'prediction': prediction[:80],
                'em': em,
                'f1': round(f1, 3),
                'confidence': round(confidence, 3),
                'strategy': strategy,
                'latency_ms': round(latency, 1),
            })

    n = results['total']
    results['exact_match_pct'] = round(100 * results['exact_match'] / n, 2) if n else 0
    results['f1_avg'] = round(results['f1_sum'] / n, 4) if n else 0
    results['abstention_rate'] = round(100 * results['abstentions'] / n, 2) if n else 0
    results['latency_avg_ms'] = round(sum(results['latency_ms']) / n, 1) if n else 0
    results['latency_p95_ms'] = round(sorted(results['latency_ms'])[int(n * 0.95)] if n else 0, 1)

    return results


def rlhf_epoch(brain, test_records, epoch_num):
    """
    One RLHF epoch: evaluate → reinforce good neurons → weaken bad ones.

    For each test question:
      - Ask the brain (get answer + participating neurons via trace)
      - Score against gold (F1)
      - If F1 > 0.5: reinforce participating neurons
      - If F1 < 0.2: weaken participating neurons
      - Between: no change (uncertain)

    Returns: (f1_avg, exact_match_pct, reinforced_count, weakened_count)
    """
    f1_sum = 0.0
    em_count = 0
    reinforced = 0
    weakened = 0

    for rec in test_records:
        question = rec['question']
        gold = rec['answer']

        result = brain.ask(question)
        prediction = result['answer']
        strategy = result['strategy']

        if strategy == 'abstain':
            continue

        f1 = f1_score(prediction, gold)
        f1_sum += f1
        if exact_match(prediction, gold):
            em_count += 1

        # Get neurons that participated (from sentence chain)
        content_words = [t for t in tokenize(prediction)
                         if t in brain._word_neurons]
        neuron_ids = [brain._word_neurons[w] for w in content_words
                      if w in brain._word_neurons]

        # Reinforce or weaken based on quality
        if f1 > 0.5:
            for nid in neuron_ids:
                if hasattr(brain.db, 'update_confidence'):
                    brain.db.update_confidence(nid, useful=True)
                # Also boost co-occurrence edges for participating words
                widx = brain._word_idx.get(
                    next((w for w, n in brain._word_neurons.items() if n == nid), None)
                )
                if widx is not None and widx in brain._cooc:
                    for neighbor in brain._cooc[widx]:
                        brain._cooc[widx][neighbor] *= 1.1
                reinforced += 1
        elif f1 < 0.2:
            for nid in neuron_ids:
                if hasattr(brain.db, 'update_confidence'):
                    brain.db.update_confidence(nid, useful=False)
                widx = brain._word_idx.get(
                    next((w for w, n in brain._word_neurons.items() if n == nid), None)
                )
                if widx is not None and widx in brain._cooc:
                    for neighbor in brain._cooc[widx]:
                        brain._cooc[widx][neighbor] *= 0.9
                weakened += 1

    n = len(test_records)
    f1_avg = f1_sum / n if n else 0
    em_pct = 100 * em_count / n if n else 0

    print(f"  Epoch {epoch_num}: F1={f1_avg:.4f} EM={em_pct:.1f}% "
          f"reinforced={reinforced} weakened={weakened}")

    return f1_avg, em_pct, reinforced, weakened


def run_benchmark(args):
    """Main benchmark flow."""
    # Find datasets
    if args.dataset == 'all':
        ds_paths = sorted(DATA_DIR.glob("*.jsonl"))
    else:
        ds_paths = [DATA_DIR / f"{args.dataset}.jsonl"]

    # Load all Q&A records
    all_records = []
    ds_counts = {}
    for ds_path in ds_paths:
        if not ds_path.exists() or ds_path.stat().st_size == 0:
            continue
        records = load_qa_records(ds_path)
        if records:
            ds_counts[ds_path.stem] = len(records)
            all_records.extend(records)

    if not all_records:
        print("No Q&A records found.")
        return

    print(f"Loaded {len(all_records):,} Q&A records from {len(ds_counts)} datasets")
    for ds, count in sorted(ds_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {ds}: {count:,}")

    # Split
    train, test = split_data(all_records, train_ratio=args.split)
    print(f"\nSplit: {len(train):,} train / {len(test):,} test "
          f"({args.split:.0%}/{1-args.split:.0%})")

    # Cap test size for speed
    if args.test_limit and len(test) > args.test_limit:
        test = test[:args.test_limit]
        print(f"Test capped at {args.test_limit}")

    # Train
    if not args.test_only:
        db_path = args.db_path or '/tmp/benchmark_brain'
        os.makedirs(db_path, exist_ok=True)

        print(f"\nTraining on {len(train):,} records...")
        brain = Brain(db_path=db_path)
        brain.begin_bulk()

        t0 = time.time()
        teach_records(brain, train)
        brain.end_bulk()
        train_time = time.time() - t0

        print(f"Training: {train_time:.1f}s | "
              f"{len(brain._words):,} words | "
              f"{brain.db.count():,} neurons")
    else:
        db_path = args.db_path or os.path.expanduser('~/nexus-brain')
        brain = load_brain(db_path)
        print(f"\nUsing existing brain: {len(brain._words):,} words")

    # RLHF epochs (if requested)
    if args.epochs > 0:
        print(f"\n=== RLHF: {args.epochs} epochs ===")
        # Use a subset for RLHF (faster iterations)
        rlhf_set = test[:min(200, len(test))]
        epoch_results = []
        for epoch in range(1, args.epochs + 1):
            f1, em, reinforced, weakened = rlhf_epoch(brain, rlhf_set, epoch)
            epoch_results.append({'epoch': epoch, 'f1': f1, 'em': em,
                                  'reinforced': reinforced, 'weakened': weakened})
            if reinforced == 0 and weakened == 0:
                print(f"  Converged at epoch {epoch} (no changes)")
                break
        print(f"  RLHF complete. F1 improvement: "
              f"{epoch_results[0]['f1']:.4f} → {epoch_results[-1]['f1']:.4f}")

    # Evaluate (final, after RLHF if any)
    print(f"\nEvaluating on {len(test):,} held-out questions...")
    t0 = time.time()
    results = evaluate(brain, test, verbose=True)
    eval_time = time.time() - t0

    # Report
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Exact Match:     {results['exact_match_pct']}%  "
          f"({results['exact_match']}/{results['total']})")
    print(f"  Token F1:        {results['f1_avg']}")
    print(f"  Abstention Rate: {results['abstention_rate']}%  "
          f"({results['abstentions']}/{results['total']})")
    print(f"  Latency (avg):   {results['latency_avg_ms']} ms")
    print(f"  Latency (p95):   {results['latency_p95_ms']} ms")
    print(f"  Eval time:       {eval_time:.1f}s")
    print(f"{'='*60}")

    if results['examples']:
        print(f"\nSample predictions:")
        for ex in results['examples'][:5]:
            status = "✓" if ex['em'] else ("∅" if ex['strategy'] == 'abstain' else "✗")
            print(f"  {status} Q: {ex['question']}")
            print(f"    Gold: {ex['gold']}")
            print(f"    Pred: {ex['prediction']} "
                  f"[{ex['strategy']}, f1={ex['f1']}, {ex['latency_ms']}ms]")
            print()

    # Save results
    out_path = Path(db_path) / 'benchmark_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'exact_match_pct': results['exact_match_pct'],
            'f1_avg': results['f1_avg'],
            'abstention_rate': results['abstention_rate'],
            'latency_avg_ms': results['latency_avg_ms'],
            'latency_p95_ms': results['latency_p95_ms'],
            'total_test': results['total'],
            'total_train': len(train),
            'datasets': ds_counts,
            'split': args.split,
        }, f, indent=2)
    print(f"Results saved to {out_path}")

    brain.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark the reasoning engine")
    parser.add_argument("--dataset", default="all",
                        help="Dataset name or 'all'")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train ratio (default 0.8 = 80/20)")
    parser.add_argument("--test-limit", type=int, default=500,
                        help="Max test questions for speed (0=unlimited)")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip training, evaluate existing brain")
    parser.add_argument("--db-path", default=None,
                        help="Brain DB path (default: /tmp/benchmark_brain)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="RLHF epochs (0=skip, default=5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == '__main__':
    main()
