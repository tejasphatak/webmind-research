#!/usr/bin/env python3
"""
ULI Bulk Training — stream text, build the database.

Feed it text and it learns: words, compounds, definitions, facts, concepts.
No neural network. It reads, parses, detects gaps, and stores knowledge.

Usage:
    # Learn from a file (one sentence per line, or paragraphs)
    python3 scripts/train.py learn --file data.txt

    # Learn from stdin (pipe Wikipedia articles, books, etc.)
    cat article.txt | python3 scripts/train.py learn --pipe

    # Learn from a topic (fetches from web and learns)
    python3 scripts/train.py research "quantum computing"

    # Bulk learn from a JSONL file ({"text": "...", "source": "..."} per line)
    python3 scripts/train.py learn --jsonl corpus.jsonl

    # Learn intents from labeled examples
    # Format: intent_label<TAB>example_query (one per line)
    python3 scripts/train.py intents --file intent_examples.tsv

    # Stats — what did it learn?
    python3 scripts/train.py stats

The database grows with every run. Knowledge compounds.
"""

import argparse
import json
import logging
import os
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('train')


def _get_thinker(auto_search=False):
    """Create a thinker instance for training."""
    from uli.thinker import Thinker
    return Thinker(auto_search=auto_search)


def _split_sentences(text):
    """Split text into sentences. Handles abbreviations, decimals."""
    # Simple sentence splitter — split on .!? followed by space+capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Also split on newlines
    result = []
    for s in sentences:
        for line in s.split('\n'):
            line = line.strip()
            if line and len(line) > 5:
                result.append(line)
    return result


def cmd_learn(args):
    """Learn from text — parse, extract facts, detect gaps, store knowledge."""
    thinker = _get_thinker(auto_search=args.web)

    # Determine input source
    if args.jsonl:
        lines = open(args.jsonl)
        mode = 'jsonl'
    elif args.file:
        text = open(args.file).read()
        lines = _split_sentences(text)
        mode = 'text'
    elif args.pipe:
        text = sys.stdin.read()
        lines = _split_sentences(text)
        mode = 'text'
    else:
        print("Specify --file, --jsonl, or --pipe")
        return

    learned = {'sentences': 0, 'facts': 0, 'gaps': 0, 'gaps_resolved': 0}
    t0 = time.monotonic()

    for line in lines:
        if mode == 'jsonl':
            try:
                obj = json.loads(line)
                text = obj.get('text', '')
                source = obj.get('source', 'corpus')
            except (json.JSONDecodeError, AttributeError):
                continue
            sentences = _split_sentences(text)
        else:
            sentences = [line]
            source = args.file or 'stdin'

        for sentence in sentences:
            if not sentence or len(sentence) < 5:
                continue

            try:
                thought = thinker.think(sentence)
                learned['sentences'] += 1
                learned['facts'] += len(thought.facts)
                learned['gaps'] += len(thought.gaps)

                # Try to resolve gaps if web search is enabled
                if args.web:
                    for gap in thought.gaps:
                        if not gap.answered:
                            if thinker.learn_gap(gap):
                                learned['gaps_resolved'] += 1

                # Progress every 100 sentences
                if learned['sentences'] % 100 == 0:
                    elapsed = time.monotonic() - t0
                    rate = learned['sentences'] / elapsed
                    log.info(
                        "Progress: %d sentences, %d facts, %d gaps (%d resolved) — %.0f sent/s",
                        learned['sentences'], learned['facts'],
                        learned['gaps'], learned['gaps_resolved'], rate
                    )

            except Exception as e:
                log.warning("Failed on: %s — %s", sentence[:50], e)
                continue

    elapsed = time.monotonic() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Sentences processed: {learned['sentences']}")
    print(f"  Facts extracted:     {learned['facts']}")
    print(f"  Gaps detected:       {learned['gaps']}")
    print(f"  Gaps resolved:       {learned['gaps_resolved']}")

    # Show DB stats
    stats = thinker._graph_stats()
    print(f"  Graph nodes:         {stats['nodes']}")
    print(f"  Graph edges:         {stats['edges']}")


def cmd_research(args):
    """Research a topic — fetch from web, learn everything about it."""
    thinker = _get_thinker(auto_search=True)

    topic = args.topic
    log.info("Researching: %s", topic)

    # Generate questions about the topic
    questions = [
        f"What is {topic}?",
        f"How does {topic} work?",
        f"What are the main types of {topic}?",
        f"What is the history of {topic}?",
        f"What are the applications of {topic}?",
        f"Who are the key people in {topic}?",
    ]

    learned = 0
    for q in questions:
        log.info("Asking: %s", q)
        thought = thinker.think(q)
        if thought.response and 'not sure' not in thought.response.lower():
            # Feed the answer back through the thinker to extract facts
            thinker.think(thought.response)
            learned += 1
            log.info("Learned from: %s", q)

            # Resolve any gaps discovered
            for gap in thought.gaps:
                if not gap.answered:
                    if thinker.learn_gap(gap):
                        log.info("Resolved gap: %s", gap.compound)

    stats = thinker._graph_stats()
    print(f"\nResearch complete: {learned}/{len(questions)} questions answered")
    print(f"  Graph nodes: {stats['nodes']}")
    print(f"  Graph edges: {stats['edges']}")


def cmd_intents(args):
    """Learn intents from labeled examples (TSV: intent<TAB>query)."""
    import sqlite3

    db_path = os.path.join(_ROOT, 'data', 'vocab', 'wordnet.db')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if not args.file:
        print("Specify --file with TSV data (intent<TAB>query)")
        return

    # Group words by intent
    intent_words = {}
    count = 0

    for line in open(args.file):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t', 1)
        if len(parts) != 2:
            continue

        intent, query = parts
        intent = intent.strip()
        if not intent.endswith('_intent'):
            intent = f'{intent}_intent'

        # Extract content words from the query
        words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'with', 'this', 'that', 'from', 'they', 'been', 'have',
                     'its', 'will', 'would', 'there', 'their', 'what', 'about',
                     'which', 'when', 'make', 'like', 'time', 'very', 'your',
                     'how', 'each', 'tell', 'does', 'could', 'than', 'them'}
        words -= stopwords

        if intent not in intent_words:
            intent_words[intent] = set()
        intent_words[intent].update(words)
        count += 1

    # Write trigger words to graph
    for intent, words in intent_words.items():
        # Ensure intent node exists
        cur.execute(
            'INSERT OR IGNORE INTO graph_nodes (node_id, label, node_type) VALUES (?,?,?)',
            (intent, intent.replace('_', ' '), 'intent')
        )
        # Add trigger words (chunked into groups of ~10)
        word_list = sorted(words)
        for i in range(0, len(word_list), 10):
            chunk = ' '.join(word_list[i:i+10])
            cur.execute(
                'INSERT OR IGNORE INTO graph_edges (from_id, relation, to_id) VALUES (?,?,?)',
                (intent, 'triggered_by', chunk)
            )

    conn.commit()
    conn.close()

    print(f"Learned {count} examples across {len(intent_words)} intents:")
    for intent, words in sorted(intent_words.items()):
        print(f"  {intent}: {len(words)} trigger words")


def cmd_stats(args):
    """Show what the DB knows."""
    import sqlite3
    db_path = os.path.join(_ROOT, 'data', 'vocab', 'wordnet.db')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute('SELECT COUNT(*) FROM words')
    print(f"Words:          {cur.fetchone()[0]:,}")
    cur.execute('SELECT COUNT(*) FROM compounds')
    print(f"Compounds:      {cur.fetchone()[0]:,}")
    cur.execute('SELECT COUNT(*) FROM senses')
    print(f"Senses:         {cur.fetchone()[0]:,}")
    cur.execute('SELECT COUNT(*) FROM graph_nodes')
    print(f"Graph nodes:    {cur.fetchone()[0]:,}")
    cur.execute('SELECT COUNT(*) FROM graph_edges')
    print(f"Graph edges:    {cur.fetchone()[0]:,}")

    cur.execute("SELECT COUNT(*) FROM graph_edges WHERE source='thinker'")
    print(f"  from thinker: {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM graph_edges WHERE source='inference'")
    print(f"  from inference: {cur.fetchone()[0]:,}")

    # Intent coverage
    cur.execute(
        "SELECT from_id, COUNT(*) FROM graph_edges WHERE relation='triggered_by' "
        "AND from_id LIKE '%_intent' GROUP BY from_id"
    )
    print(f"\nIntent coverage:")
    for intent, count in cur.fetchall():
        cur.execute(
            "SELECT to_id FROM graph_edges WHERE from_id=? AND relation='triggered_by'",
            (intent,)
        )
        all_words = set()
        for row in cur.fetchall():
            all_words.update(row[0].split())
        print(f"  {intent}: {len(all_words)} trigger words ({count} sets)")

    try:
        cur.execute('SELECT COUNT(*) FROM concept_rules')
        print(f"\nLearned rules:  {cur.fetchone()[0]:,}")
    except sqlite3.OperationalError:
        pass

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='ULI Bulk Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest='command')

    p = sub.add_parser('learn', help='Learn from text')
    p.add_argument('--file', help='Text file to learn from')
    p.add_argument('--jsonl', help='JSONL file ({"text": "...", "source": "..."})')
    p.add_argument('--pipe', action='store_true', help='Read from stdin')
    p.add_argument('--web', action='store_true',
                   help='Enable web search to resolve knowledge gaps')

    p = sub.add_parser('research', help='Research a topic from the web')
    p.add_argument('topic', help='Topic to research')

    p = sub.add_parser('intents', help='Learn intents from labeled TSV')
    p.add_argument('--file', required=True, help='TSV file: intent<TAB>query')

    sub.add_parser('stats', help='Show database statistics')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {
        'learn': cmd_learn,
        'research': cmd_research,
        'intents': cmd_intents,
        'stats': cmd_stats,
    }[args.command](args)


if __name__ == '__main__':
    main()
