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


# WordNet synonym cache
_syn_cache = {}

def _get_synonyms(word: str) -> Set[str]:
    """Get synonyms from WordNet. Cached."""
    if word in _syn_cache:
        return _syn_cache[word]
    try:
        from nltk.corpus import wordnet as wn
        syns = set()
        for ss in wn.synsets(word):
            for lemma in ss.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name != word:
                    syns.add(name)
        _syn_cache[word] = syns
        return syns
    except Exception:
        _syn_cache[word] = set()
        return set()


def _overlap_with_synonyms(set_a: Set[str], set_b: Set[str]) -> float:
    """Overlap coefficient with WordNet synonym expansion.
    'plane' matches 'aircraft' because they're synonyms."""
    if not set_a or not set_b:
        return 0.0

    # Direct overlap first
    direct = set_a & set_b
    if direct:
        return len(direct) / min(len(set_a), len(set_b))

    # Synonym expansion: for each word in A, check if any synonym is in B
    syn_matches = 0
    for word in set_a:
        syns = _get_synonyms(word)
        if syns & set_b:
            syn_matches += 1

    if syn_matches > 0:
        return syn_matches / min(len(set_a), len(set_b))

    return 0.0


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
        if sa and sb:
            # Use synonym expansion for content and noun comparisons
            if name in ('content', 'nouns', 'verbs'):
                scores.append(_overlap_with_synonyms(sa, sb))
            else:
                scores.append(_overlap(sa, sb))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def text_similarity(text_a: str, text_b: str, lang: str = 'en') -> float:
    """Structural similarity between two texts."""
    tokens_a, _ = tokenize(text_a, lang=lang)
    tokens_b, _ = tokenize(text_b, lang=lang)
    return token_similarity(tokens_a, tokens_b)


def question_passage_relevance(question: str, passage: str,
                                lang: str = 'en') -> float:
    """Check if a passage is relevant to a question.

    Uses AST comparison: same predicate + same entities + matching roles
    = relevant. Different predicates or no entity overlap = irrelevant.
    """
    from .semantics import tokens_to_ast

    q_tokens, q_spans = tokenize(question, lang=lang)
    p_tokens, p_spans = tokenize(passage, lang=lang)

    q_ast = tokens_to_ast(q_tokens, question, entity_spans=q_spans)
    p_ast = tokens_to_ast(p_tokens, passage, entity_spans=p_spans)

    signals = []

    # 1. Entity overlap with synonym expansion
    #    "plane" matches "aircraft" via WordNet synonyms
    q_ents = set(e.lower() for e in q_ast.entities)
    p_ents = set(e.lower() for e in p_ast.entities)
    if q_ents and p_ents:
        signals.append(_overlap_with_synonyms(q_ents, p_ents))

    # 2. Predicate match — but only for CONTENT predicates
    #    "be/have/do" are copulas that match everything — they carry no meaning
    stop = _get_stop_words()
    if (q_ast.predicate and p_ast.predicate and
        q_ast.predicate not in stop and p_ast.predicate not in stop):
        signals.append(1.0 if q_ast.predicate == p_ast.predicate else 0.0)

    # 3. Agent/subject overlap
    q_agent = q_ast.agent.text.lower() if q_ast.agent and q_ast.agent.text != '?' else ''
    p_agent = p_ast.agent.text.lower() if p_ast.agent else ''
    if q_agent and p_agent:
        # Check if one contains the other (handles "the capital" vs "capital")
        if q_agent in p_agent or p_agent in q_agent or q_agent == p_agent:
            signals.append(1.0)
        else:
            signals.append(0.0)

    # 4. Patient/theme overlap
    q_patient = q_ast.patient.text.lower() if q_ast.patient else ''
    p_patient = p_ast.patient.text.lower() if p_ast.patient else ''
    if q_patient and p_patient:
        if q_patient in p_patient or p_patient in q_patient:
            signals.append(1.0)
        else:
            signals.append(0.0)

    # 5. Answer-type presence — does passage have what question asks for?
    if q_ast.question_word in ('who', 'whom'):
        has_person = any(t.entity_type == 'PERSON' for t in p_tokens)
        signals.append(1.0 if has_person else 0.0)
    elif q_ast.question_word == 'when':
        has_date = any(t.entity_type in ('DATE', 'TIME', 'CARDINAL') for t in p_tokens)
        signals.append(1.0 if has_date else 0.0)
    elif q_ast.question_word == 'where':
        has_place = any(t.entity_type in ('GPE', 'LOC') for t in p_tokens)
        signals.append(1.0 if has_place else 0.0)

    # 6. Non-shared context divergence
    # If texts share a word but NOTHING ELSE overlaps, they're about different things
    # "planet" + gym/fitness vs "planet" + solar/system → different contexts
    q_content = set()
    p_content = set()
    stop = _get_stop_words()
    for t in q_tokens:
        if t.pos in ('NOUN','PROPN','VERB','ADJ') and t.text.lower() not in stop and len(t.text) > 2:
            q_content.add(t.lemma.lower())
    for t in p_tokens:
        if t.pos in ('NOUN','PROPN','VERB','ADJ') and t.text.lower() not in stop and len(t.text) > 2:
            p_content.add(t.lemma.lower())

    shared_content = q_content & p_content
    q_unique = q_content - shared_content
    p_unique = p_content - shared_content

    if q_unique and p_unique:
        # How much do the UNIQUE (non-shared) words overlap?
        # If zero overlap in unique words → probably different topics despite shared word
        context_overlap = _overlap(q_unique, p_unique)
        signals.append(context_overlap)

    # 7. Fallback: token-level content overlap
    base_sim = token_similarity(q_tokens, p_tokens)
    signals.append(base_sim)

    if not signals:
        return 0.0
    return sum(signals) / len(signals)
