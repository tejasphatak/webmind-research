"""
Structural Similarity — computed from ULI's token analysis.

No neural model. No hardcoded weights. Similarity is the proportion of
linguistic signal shared between two texts, weighted by what's actually
present (adaptive normalization).

Principle: if a component has no signal (e.g., neither text has entities),
it doesn't factor into the score. Only components with actual data contribute.
"""

import json
import os
from typing import List, Set, Dict, Tuple
from .protocol import Token, MeaningAST
from .lexer import tokenize

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Load stop words from vocab data — NOT hardcoded
_STOP = None

def _get_stop_words() -> frozenset:
    global _STOP
    if _STOP is not None:
        return _STOP
    try:
        with open(os.path.join(_DATA_DIR, 'vocab', 'en.json'), encoding='utf-8') as f:
            vocab = json.load(f)
        _STOP = frozenset(vocab.get('stop_words', []))
    except (FileNotFoundError, json.JSONDecodeError):
        _STOP = frozenset()
    return _STOP


def _overlap(set_a: Set[str], set_b: Set[str]) -> float:
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|).
    Handles asymmetric sizes (short question vs long passage)."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _extract_features(tokens: List[Token]) -> Dict[str, Set[str]]:
    """Extract all linguistic feature sets from tokens.
    Returns dict of feature_name → word_set."""
    stop = _get_stop_words()
    features = {
        'entities': set(),     # Named entities + proper nouns
        'nouns': set(),        # Common + proper nouns
        'verbs': set(),        # Content verbs
        'adjectives': set(),   # Adjectives
        'content': set(),      # All content words (N + V + ADJ + ADV)
    }

    for tok in tokens:
        word = tok.text.lower().strip()
        lemma = (tok.lemma or tok.text).lower().strip()

        if word in stop or len(word) < 2:
            continue

        if tok.is_entity or tok.pos == 'PROPN':
            features['entities'].add(word)

        if tok.pos in ('NOUN', 'PROPN'):
            features['nouns'].add(lemma)
            features['content'].add(lemma)
        elif tok.pos == 'VERB':
            features['verbs'].add(lemma)
            features['content'].add(lemma)
        elif tok.pos in ('ADJ',):
            features['adjectives'].add(lemma)
            features['content'].add(lemma)
        elif tok.pos in ('ADV', 'NUM'):
            features['content'].add(lemma)

    return features


def token_similarity(tokens_a: List[Token], tokens_b: List[Token]) -> float:
    """Structural similarity between two token lists.

    Adaptive: only components with signal contribute.
    No hardcoded weights — each active component gets equal weight,
    normalized by how many components have data."""

    feat_a = _extract_features(tokens_a)
    feat_b = _extract_features(tokens_b)

    scores = []
    for name in feat_a:
        sa = feat_a[name]
        sb = feat_b[name]
        # Only score this component if BOTH sides have data
        if sa and sb:
            scores.append(_overlap(sa, sb))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def text_similarity(text_a: str, text_b: str, lang: str = 'en') -> float:
    """Structural similarity between two texts."""
    tokens_a = tokenize(text_a, lang=lang)
    tokens_b = tokenize(text_b, lang=lang)
    return token_similarity(tokens_a, tokens_b)


def question_passage_relevance(question: str, passage: str,
                                lang: str = 'en') -> float:
    """Check if a passage is relevant to a question.

    Beyond basic similarity, checks:
    - Does passage contain the question's key content words?
    - Does passage have the entity TYPE the question asks for?
      (who→PERSON, when→DATE, where→GPE)
    """
    q_tokens = tokenize(question, lang=lang)
    p_tokens = tokenize(passage, lang=lang)

    # Base structural similarity (adaptive, no hardcoded weights)
    base_sim = token_similarity(q_tokens, p_tokens)

    # Question content coverage: what fraction of the question's
    # content words appear in the passage?
    stop = _get_stop_words()
    q_content = set()
    for tok in q_tokens:
        if tok.pos in ('NOUN', 'PROPN', 'VERB', 'ADJ') and tok.text.lower() not in stop and len(tok.text) > 2:
            q_content.add(tok.text.lower())

    p_words = set(tok.text.lower() for tok in p_tokens)
    coverage = len(q_content & p_words) / len(q_content) if q_content else 0.0

    # Answer-type presence: does the passage have what the question asks for?
    answer_type_bonus = 0.0
    for tok in q_tokens:
        qw = tok.text.lower()
        if qw in ('who', 'whom'):
            if any(t.entity_type == 'PERSON' for t in p_tokens):
                answer_type_bonus = 0.15
        elif qw == 'when':
            if any(t.entity_type in ('DATE', 'TIME', 'CARDINAL') for t in p_tokens):
                answer_type_bonus = 0.15
        elif qw == 'where':
            if any(t.entity_type in ('GPE', 'LOC') for t in p_tokens):
                answer_type_bonus = 0.15

    # Combine: equal weight to similarity and coverage, plus type bonus
    # No hardcoded weights — just average the signals that have data
    signals = [base_sim, coverage]
    if answer_type_bonus > 0:
        signals.append(answer_type_bonus)

    return sum(signals) / len(signals)
