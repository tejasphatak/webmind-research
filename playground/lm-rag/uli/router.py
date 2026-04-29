"""
ULI Intent Router — classifies user input into a routing Intent.

classify(text) → Intent

Priority order (first match wins):
  math → coding → creative → factual → research → conversation

The key insight: programming languages are just vocabulary.
A question mentioning "python" + "write" routes to CODING.
A question about recent papers routes to RESEARCH.
Everything else routes to FACTUAL (KG first) or CONVERSATION.

Usage:
    from uli.router import classify, Intent
    intent = classify("Write a Python function to sort a list")
    # → Intent.CODING
"""

import re
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    FACTUAL      = 'factual'       # who/what/where/when → GraphReasoner.try_answer()
    CODING       = 'coding'        # write/explain/debug + PL → CodeAssistant
    RESEARCH     = 'research'      # recent/papers/study → academic MCPs first
    MATH         = 'math'          # numbers, equations → web search (no KG math)
    CREATIVE     = 'creative'      # essay/brainstorm/explain → ContentGenerator
    CONVERSATION = 'conversation'  # greetings, opinions → template response
    OFF_LIMITS   = 'off_limits'    # caught by SafetyGate before routing


# ── Vocabulary sets ───────────────────────────────────────────────────────────

# Programming languages + frameworks — "programming language is just another vocabulary"
_KNOWN_PL: frozenset = frozenset({
    # Languages
    'python', 'javascript', 'typescript', 'rust', 'golang', 'java', 'kotlin',
    'swift', 'ruby', 'scala', 'haskell', 'erlang', 'elixir', 'clojure',
    'c', 'cpp', 'csharp', 'c#', 'c++', 'objectivec', 'objective-c',
    'r', 'matlab', 'julia', 'fortran', 'cobol', 'pascal', 'delphi',
    'lua', 'dart', 'zig', 'nim', 'crystal', 'racket', 'scheme', 'lisp',
    'prolog', 'assembly', 'wasm', 'webassembly',
    'bash', 'shell', 'zsh', 'fish', 'powershell', 'batch',
    'sql', 'plsql', 'tsql', 'nosql',
    'html', 'css', 'scss', 'sass', 'less',
    'json', 'yaml', 'toml', 'xml', 'graphql',
    'solidity', 'vyper',   # blockchain
    'verilog', 'vhdl',     # hardware
    # Frameworks / runtimes (also vocabulary)
    'react', 'vue', 'angular', 'svelte', 'nextjs', 'nuxt', 'remix',
    'django', 'flask', 'fastapi', 'tornado', 'aiohttp',
    'rails', 'sinatra', 'express', 'koa', 'nestjs', 'hapi',
    'spring', 'springboot', 'quarkus', 'micronaut',
    'pytorch', 'tensorflow', 'keras', 'jax', 'sklearn', 'scikit-learn',
    'numpy', 'pandas', 'polars', 'matplotlib', 'seaborn', 'plotly',
    'fastapi', 'pydantic', 'sqlalchemy', 'alembic',
    'docker', 'kubernetes', 'helm', 'terraform', 'ansible',
    'webpack', 'vite', 'esbuild', 'rollup', 'parcel',
    'pytest', 'jest', 'mocha', 'chai', 'vitest', 'playwright', 'cypress',
    'git', 'github', 'gitlab',
    # Databases (also vocabulary)
    'postgresql', 'postgres', 'mysql', 'sqlite', 'mariadb',
    'mongodb', 'redis', 'elasticsearch', 'cassandra', 'dynamodb',
    'clickhouse', 'duckdb', 'bigquery', 'snowflake',
})

_CODING_VERBS: frozenset = frozenset({
    'write', 'code', 'implement', 'build', 'create', 'make',
    'debug', 'fix', 'resolve', 'patch',
    'explain', 'describe', 'show', 'demonstrate', 'example',
    'refactor', 'optimize', 'improve', 'rewrite', 'clean',
    'test', 'review', 'check', 'validate',
    'parse', 'serialize', 'deserialize',
    'compile', 'run', 'execute', 'deploy',
    'install', 'setup', 'configure', 'initialize',
    'what', 'how', 'why', 'when',  # "how does X work in Python" → CODING
})

_RESEARCH_SIGNALS: frozenset = frozenset({
    'paper', 'papers', 'study', 'studies', 'research', 'journal',
    'article', 'publication', 'published', 'arxiv', 'preprint',
    'author', 'authors', 'citation', 'citations', 'cited',
    'recent', 'latest', 'new', 'current', '2024', '2025', '2026',
    'findings', 'results', 'experiment', 'dataset', 'benchmark',
    'according to', 'based on', 'evidence',
})

_CREATIVE_SIGNALS: frozenset = frozenset({
    'write', 'draft', 'compose', 'create',
    'essay', 'article', 'blog', 'post', 'story', 'poem',
    'brainstorm', 'ideas', 'suggest', 'list',
    'explain', 'describe', 'overview', 'summary', 'summarize',
    'timeline', 'history of', 'background on',
})

_MATH_SIGNALS: frozenset = frozenset({
    'calculate', 'compute', 'solve', 'equation', 'formula',
    'integral', 'derivative', 'matrix', 'vector', 'probability',
    'statistics', 'regression', 'eigenvalue',
    'plus', 'minus', 'divide', 'multiply', 'squared', 'cubed',
})

_MATH_PATTERN = re.compile(
    r'[\d\+\-\*\/\=\^\(\)\[\]\{\}]{3,}|'  # expressions: 2+3, x^2
    r'\b\d+[\.,]\d+\b',                    # decimal numbers
)

_CONVERSATION_SIGNALS: frozenset = frozenset({
    'hello', 'hi', 'hey', 'greetings', 'howdy',
    'thanks', 'thank', 'bye', 'goodbye',
    'opinion', 'believe',
    'joke', 'funny', 'laugh',
})

# Multi-word patterns that indicate conversation about the assistant itself
_CONVERSATION_PHRASES: tuple = (
    'what can you do', 'what do you do', 'who are you',
    'what are you', 'how are you', 'help me', 'tell me about yourself',
    'what is your name', "what's your name", 'your name',
    'are you a', 'are you an',
)


# ── Graph-based intent reasoning ─────────────────────────────────────────────
#
# The intent ontology (trigger words, examples, not-examples) is static
# during a process lifetime. Load it once at import, not per-classify call.

import os
import sqlite3

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'vocab', 'wordnet.db'
)

# Cached ontology — populated by _load_intent_ontology() at module load
_INTENT_ONTOLOGY = None


def _load_intent_ontology():
    """Load the full intent ontology from the graph DB into memory.

    Returns a dict with:
      conv_examples: frozenset of conversation example phrases
      not_examples: tuple of not-example prefixes
      triggers: {Intent: frozenset of trigger words}
    Or None if DB unavailable.
    """
    try:
        conn = sqlite3.connect(f'file:{_DB_PATH}?mode=ro', uri=True)
        cur = conn.cursor()

        # Conversation examples (exact match)
        cur.execute(
            "SELECT to_id FROM graph_edges WHERE from_id='conversation_intent' AND relation='example'"
        )
        conv_examples = frozenset(r[0].lower().rstrip('?. ') for r in cur.fetchall())

        # NOT-examples (prefix match)
        cur.execute(
            "SELECT to_id FROM graph_edges WHERE from_id='conversation_intent' AND relation='not_example'"
        )
        not_examples = tuple(r[0].lower().rstrip('?. ') for r in cur.fetchall())

        # Trigger words per intent type
        intent_map = {
            'factual_intent': Intent.FACTUAL,
            'conversation_intent': Intent.CONVERSATION,
            'research_intent': Intent.RESEARCH,
            'coding_intent': Intent.CODING,
            'math_intent': Intent.MATH,
            'creative_intent': Intent.CREATIVE,
        }
        triggers = {}
        for intent_label, intent_enum in intent_map.items():
            cur.execute(
                "SELECT to_id FROM graph_edges WHERE from_id=? AND relation='triggered_by'",
                (intent_label,)
            )
            words = set()
            for row in cur.fetchall():
                words.update(row[0].split())
            triggers[intent_enum] = frozenset(words)

        conn.close()
        return {
            'conv_examples': conv_examples,
            'not_examples': not_examples,
            'triggers': triggers,
        }
    except Exception:
        return None


# Load once at import
_INTENT_ONTOLOGY = _load_intent_ontology()

_POLITE_STARTS = ('can you tell', 'could you tell', 'can you help',
                  'could you help', 'can you show', 'can you find',
                  'can you check', 'can you get', 'would you tell')


def _graph_classify(text: str, words: set, lower: str) -> Optional[Intent]:
    """Reason about intent using the cached ontology.

    No DB queries — everything was loaded at import.
    """
    ontology = _INTENT_ONTOLOGY
    if ontology is None:
        return None

    clean_lower = lower.rstrip('?!. ')

    # 1. Conversation examples (exact match)
    if clean_lower in ontology['conv_examples']:
        return Intent.CONVERSATION

    # 1b. NOT-examples
    is_not_conversation = any(clean_lower.startswith(ne) for ne in ontology['not_examples'])

    # 2. Score each intent by trigger word overlap
    scores = {}
    for intent_enum, trigger_words in ontology['triggers'].items():
        scores[intent_enum] = len(words & trigger_words)

    # 3. Disambiguation
    has_you = bool({'you', 'your', 'yourself'} & words)
    has_content = len(words - {'can', 'you', 'your', 'could', 'would',
                                'tell', 'me', 'help', 'do', 'please',
                                'what', 'the', 'is', 'a', 'an'}) > 0

    if has_you and has_content and is_not_conversation:
        scores[Intent.CONVERSATION] = 0

    if any(clean_lower.startswith(p) for p in _POLITE_STARTS):
        scores[Intent.CONVERSATION] = 0

    # 4. Pick highest-scoring intent (min 1 trigger hit)
    if not scores:
        return None
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    return best


# ── Classifier ────────────────────────────────────────────────────────────────

def classify(text: str) -> Intent:
    """
    Classify user input into an Intent.

    Primary: graph-based reasoning (queries intent ontology in KG).
    Fallback: pattern matching (for speed and when graph is unavailable).
    """
    if not text or not text.strip():
        return Intent.CONVERSATION

    lower = text.lower()
    words = set(re.findall(r"\b[\w#\+\-]+\b", lower))

    # ── Graph reasoning (primary) ────────────────────────────────────────
    graph_intent = _graph_classify(text, words, lower)
    if graph_intent is not None:
        return graph_intent

    # ── Fallback: pattern matching ───────────────────────────────────────
    # Commented out — all intent classification should go through graph
    # reasoning. If the graph doesn't know an intent, teach it by adding
    # triggered_by edges, not by adding code here.
    #
    # # 1. Math — expressions or explicit math vocabulary
    # if _MATH_PATTERN.search(text) and words & _MATH_SIGNALS:
    #     return Intent.MATH
    # if len(words & _MATH_SIGNALS) >= 2:
    #     return Intent.MATH
    #
    # # 2. Coding — any coding verb AND any known PL/framework name
    # has_pl   = bool(words & _KNOWN_PL)
    # has_verb = bool(words & _CODING_VERBS)
    # if has_pl and has_verb:
    #     return Intent.CODING
    # if has_pl and any(w in lower for w in ('what is', 'how does', 'how to', 'explain')):
    #     return Intent.CODING
    #
    # # 3. Creative
    # is_factual_qw = bool(re.match(
    #     r'\b(who|what|where|when|which|how many|how much)\b', lower
    # ))
    # if len(words & _CREATIVE_SIGNALS) >= 1 and not is_factual_qw:
    #     if any(w in lower for w in ('essay', 'brainstorm', 'timeline', 'overview',
    #                                  'summarize', 'summary', 'draft', 'compose',
    #                                  'blog', 'post', 'story', 'poem')):
    #         return Intent.CREATIVE
    #
    # # 4. Research
    # research_hits = len(words & _RESEARCH_SIGNALS)
    # if research_hits >= 2 or any(w in lower for w in (
    #     'arxiv', 'paper', 'published', 'citation', 'journal', 'preprint'
    # )):
    #     return Intent.RESEARCH
    #
    # # 5. Factual
    # if is_factual_qw:
    #     return Intent.FACTUAL
    #
    # # 6. Conversation — phrase-level check
    # if any(lower.startswith(p) for p in _CONVERSATION_PHRASES):
    #     return Intent.CONVERSATION
    # if words & _CONVERSATION_SIGNALS:
    #     return Intent.CONVERSATION

    # Default: if graph had no opinion, treat as factual (will hit web search)
    return Intent.FACTUAL
