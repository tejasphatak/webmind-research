#!/usr/bin/env python3
"""
Teach ULI new intents and conversation patterns — no code changes needed.

Usage:
    # Add trigger words for an existing intent
    python3 scripts/teach_intent.py add-triggers coding_intent "docker kubernetes helm terraform"

    # Create a brand new intent type
    python3 scripts/teach_intent.py create-intent translation_intent "translate convert language"

    # Add conversation examples (exact match)
    python3 scripts/teach_intent.py add-example conversation_intent "what can you do"

    # Add not-examples (looks like X but isn't)
    python3 scripts/teach_intent.py add-not-example conversation_intent "can you tell me about"

    # Add conversation patterns (greetings, farewells, introductions, etc.)
    python3 scripts/teach_intent.py add-conv-pattern introductions "i am"
    python3 scripts/teach_intent.py add-conv-phrase introduction_phrases "my name is"

    # List all intents and their triggers
    python3 scripts/teach_intent.py list

    # Test classification on a query
    python3 scripts/teach_intent.py test "Write a Python function to sort a list"

Everything goes into the DB (data/vocab/wordnet.db). The router and thinker
load from DB at startup — no code changes, no redeployment.
"""

import argparse
import json
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB = os.path.join(_ROOT, 'data', 'vocab', 'wordnet.db')


def get_conn():
    return sqlite3.connect(_DB)


def cmd_add_triggers(args):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT OR IGNORE INTO graph_edges (from_id, relation, to_id) VALUES (?,?,?)',
        (args.intent, 'triggered_by', args.words)
    )
    conn.commit()
    print(f'Added triggers to {args.intent}: {args.words}')
    conn.close()


def cmd_create_intent(args):
    conn = get_conn()
    cur = conn.cursor()
    # Create the intent node
    cur.execute(
        'INSERT OR IGNORE INTO graph_nodes (node_id, label, node_type) VALUES (?,?,?)',
        (args.intent, args.intent.replace('_', ' '), 'intent')
    )
    # Add trigger words
    cur.execute(
        'INSERT OR IGNORE INTO graph_edges (from_id, relation, to_id) VALUES (?,?,?)',
        (args.intent, 'triggered_by', args.words)
    )
    conn.commit()
    print(f'Created intent: {args.intent}')
    print(f'Trigger words: {args.words}')
    print(f'\nNote: add Intent.{args.intent.upper().replace("_INTENT","")} to uli/router.py Intent enum')
    conn.close()


def cmd_add_example(args):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT OR IGNORE INTO graph_edges (from_id, relation, to_id) VALUES (?,?,?)',
        (args.intent, 'example', args.text)
    )
    conn.commit()
    print(f'Added example to {args.intent}: "{args.text}"')
    conn.close()


def cmd_add_not_example(args):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT OR IGNORE INTO graph_edges (from_id, relation, to_id) VALUES (?,?,?)',
        (args.intent, 'not_example', args.text)
    )
    conn.commit()
    print(f'Added not-example to {args.intent}: "{args.text}"')
    conn.close()


def cmd_add_conv_pattern(args):
    """Add a word to a conversation pattern set (e.g., introductions, greetings)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM grammar_rules WHERE category='en' AND key='conversation'")
    row = cur.fetchone()
    conv = json.loads(row[0]) if row else {}

    if args.pattern_type not in conv:
        conv[args.pattern_type] = []
    if args.value not in conv[args.pattern_type]:
        conv[args.pattern_type].append(args.value)

    cur.execute(
        'UPDATE grammar_rules SET value=? WHERE category=? AND key=?',
        (json.dumps(conv), 'en', 'conversation')
    )
    conn.commit()
    print(f'Added to {args.pattern_type}: "{args.value}"')
    print(f'Total in {args.pattern_type}: {len(conv[args.pattern_type])}')
    conn.close()


def cmd_list(args):
    conn = get_conn()
    cur = conn.cursor()

    # Intent triggers
    cur.execute(
        "SELECT from_id, to_id FROM graph_edges WHERE relation='triggered_by' "
        "AND from_id LIKE '%_intent' ORDER BY from_id"
    )
    current = None
    for intent, words in cur.fetchall():
        if intent != current:
            print(f'\n{intent}:')
            current = intent
        print(f'  {words}')

    # Conversation examples
    cur.execute(
        "SELECT from_id, to_id FROM graph_edges WHERE relation='example' ORDER BY from_id"
    )
    examples = cur.fetchall()
    if examples:
        print('\nExamples:')
        for intent, text in examples:
            print(f'  {intent}: "{text}"')

    # Conversation patterns
    cur.execute("SELECT value FROM grammar_rules WHERE category='en' AND key='conversation'")
    row = cur.fetchone()
    if row:
        conv = json.loads(row[0])
        print('\nConversation patterns:')
        for k, v in conv.items():
            print(f'  {k}: {v}')

    conn.close()


def cmd_test(args):
    sys.path.insert(0, _ROOT)
    from uli.router import classify
    intent = classify(args.query)
    print(f'Query: "{args.query}"')
    print(f'Intent: {intent.value}')


def main():
    parser = argparse.ArgumentParser(
        description='Teach ULI intents and conversation patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest='command')

    p = sub.add_parser('add-triggers', help='Add trigger words to an intent')
    p.add_argument('intent', help='e.g., coding_intent')
    p.add_argument('words', help='Space-separated trigger words')

    p = sub.add_parser('create-intent', help='Create a new intent type')
    p.add_argument('intent', help='e.g., translation_intent')
    p.add_argument('words', help='Initial trigger words')

    p = sub.add_parser('add-example', help='Add exact-match example')
    p.add_argument('intent', help='e.g., conversation_intent')
    p.add_argument('text', help='The example phrase')

    p = sub.add_parser('add-not-example', help='Add not-example (looks like X but isn\'t)')
    p.add_argument('intent', help='e.g., conversation_intent')
    p.add_argument('text', help='The not-example phrase')

    p = sub.add_parser('add-conv-pattern', help='Add conversation pattern word')
    p.add_argument('pattern_type', help='e.g., introductions, greetings')
    p.add_argument('value', help='The word or phrase')

    p = sub.add_parser('list', help='List all intents and patterns')

    p = sub.add_parser('test', help='Test classification on a query')
    p.add_argument('query', help='The query to classify')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {
        'add-triggers': cmd_add_triggers,
        'create-intent': cmd_create_intent,
        'add-example': cmd_add_example,
        'add-not-example': cmd_add_not_example,
        'add-conv-pattern': cmd_add_conv_pattern,
        'list': cmd_list,
        'test': cmd_test,
    }[args.command](args)


if __name__ == '__main__':
    main()
