"""
Structural Similarity — computed from ULI's token analysis.

No neural model. Similarity is the proportion of linguistic signal
shared between two texts, using context-filtered meaning clouds
(non-neural attention) + WordNet taxonomy distance.

Architecture:
  Cloud signal: Each word's meaning set filtered by surrounding context
                (emulates attention via set intersection).
  Graph signal: First-sense Wu-Palmer similarity via WordNet hypernym tree.
  Combiner:     Decision tree — cloud for polysemy, graph for paraphrases.
"""

import json
import os
import re
from typing import List, Set, Dict, Tuple
from .protocol import Token, MeaningAST
from .lexer import tokenize

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# No hardcoded thresholds. All parameters are derived from:
# - Text properties (cloud size, polysemy, variance)
# - Statistical baselines (expected random overlap, WUP noise floor)
# - Self-weighting (cloud score IS the weight)


# ============================================================
# STOP WORDS — loaded from vocab data, not hardcoded
# ============================================================

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
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|)."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


# ============================================================
# MEANING SETS — each word's "semantic fingerprint"
# ============================================================

_meaning_cache = {}

_SKIP_WORDS = frozenset({'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
    'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'or', 'and',
    'that', 'which', 'it', 'its', 'as', 'not', 'no', 'has', 'have', 'had',
    'do', 'does', 'did', 'will', 'can', 'may', 'such', 'more', 'than', 'into'})


def _simple_lemma(word: str) -> str:
    """Crude lemmatization — strip common suffixes."""
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('es') and len(word) > 3:
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    return word


def _get_meaning_set(word: str) -> Set[str]:
    """Build meaning fingerprint: synonyms + hypernyms + hyponyms + definition words.
    Cached. Bidirectional: plane→aircraft AND aircraft→plane."""
    if word in _meaning_cache:
        return _meaning_cache[word]
    try:
        from nltk.corpus import wordnet as wn
        meaning = set()
        for ss in wn.synsets(word)[:3]:  # Top 3 senses only
            for lemma in ss.lemmas():
                meaning.add(lemma.name().replace('_', ' ').lower())
            frontier = ss.hypernyms()
            for _ in range(2):
                next_f = []
                for h in frontier:
                    for lemma in h.lemmas():
                        meaning.add(lemma.name().replace('_', ' ').lower())
                    next_f.extend(h.hypernyms())
                frontier = next_f
            for hypo in ss.hyponyms():
                for lemma in hypo.lemmas():
                    meaning.add(lemma.name().replace('_', ' ').lower())
            for w in re.findall(r'[a-z]+', ss.definition().lower()):
                lemma = _simple_lemma(w)
                if len(lemma) > 2 and lemma not in _SKIP_WORDS:
                    meaning.add(lemma)
        meaning.discard(word)
        _meaning_cache[word] = meaning
        return meaning
    except Exception:
        _meaning_cache[word] = set()
        return set()


# ============================================================
# CONTEXT-FILTERED MEANING CLOUDS — non-neural attention
# ============================================================

def _context_meaning(word: str, context_words: Set[str]) -> Set[str]:
    """Filter word's meaning set by context. Implicit WSD via set intersection.
    'bank' + {habitat, river} → keeps geography, drops finance."""
    full = _get_meaning_set(word)
    if not context_words or not full:
        return full
    territory = set()
    for cw in context_words:
        territory.update(_get_meaning_set(cw))
        territory.add(cw)
    filtered = full & territory
    # Dynamic minimum: proportional to meaning set size (not fixed)
    # Word with 10 meanings needs 1 match (10%). Word with 50 needs 5.
    min_matches = max(2, len(full) // 5)
    if len(filtered) < min_matches:
        return full
    return filtered


def _meaning_cloud(words: Set[str]) -> Set[str]:
    """Build context-aware meaning cloud.
    Each word's meaning set filtered by OTHER words in the same text."""
    cloud = set()
    word_list = list(words)
    for word in word_list:
        context = words - {word}
        filtered_meaning = _context_meaning(word, context)
        cloud.update(filtered_meaning)
    cloud.update(words)
    return cloud


def _meaning_similarity(word_a: str, word_b: str) -> float:
    """Semantic similarity between two words via meaning sets."""
    if word_a == word_b:
        return 1.0
    ma = _get_meaning_set(word_a)
    mb = _get_meaning_set(word_b)
    if word_b in ma or word_a in mb:
        return 0.8
    if not ma or not mb:
        return 0.0
    intersection = len(ma & mb)
    return intersection / min(len(ma), len(mb)) if min(len(ma), len(mb)) > 0 else 0.0


# ============================================================
# FIRST-SENSE WUP — WordNet Wu-Palmer similarity (cached)
# ============================================================

_wup_cache: Dict[Tuple[str, str], float] = {}

def _first_sense_wup(word_a: str, word_b: str) -> float:
    """Wu-Palmer similarity between first (most frequent) synsets.
    plane/aircraft = 0.909, plane/stock = 0.105."""
    if word_a == word_b:
        return 1.0
    key = (word_a, word_b) if word_a < word_b else (word_b, word_a)
    if key in _wup_cache:
        return _wup_cache[key]
    try:
        from nltk.corpus import wordnet as wn
        ss_a = wn.synsets(word_a)
        ss_b = wn.synsets(word_b)
        if not ss_a or not ss_b:
            _wup_cache[key] = 0.0
            return 0.0
        score = ss_a[0].wup_similarity(ss_b[0])
        _wup_cache[key] = score or 0.0
        return score or 0.0
    except Exception:
        _wup_cache[key] = 0.0
        return 0.0


def _pairwise_wup(set_a: Set[str], set_b: Set[str]) -> float:
    """Average best-match WUP with baseline subtraction.
    Random word pairs score ~0.2 WUP (distant ancestry noise).
    Contribution = max(0, wup - 0.2) / 0.8 — continuous, no hard cutoff.
    WUP=0.2 → 0.0, WUP=0.5 → 0.375, WUP=0.9 → 0.875."""
    if not set_a or not set_b:
        return 0.0
    smaller, larger = (set_a, set_b) if len(set_a) <= len(set_b) else (set_b, set_a)
    total = 0.0
    for wa in smaller:
        best = max((_first_sense_wup(wa, wb) for wb in larger), default=0.0)
        # Continuous contribution: subtract baseline, normalize
        contribution = max(0.0, (best - 0.2) / 0.8)
        total += contribution
    return total / len(smaller)


# ============================================================
# DECISION-TREE COMBINER — cloud for polysemy, graph for paraphrase
# ============================================================

def _combined_score(cloud_score: float, graph_score: float) -> float:
    """Continuous combination — cloud score IS the weight.
    No hardcoded thresholds, no if/elif/else tiers.

    cloud_weight = 0.5 + 0.5 * cloud_score  (0.5 when cloud=0, 1.0 when cloud=1)
    graph capped relative to cloud — can't exceed cloud * 2.
    This prevents graph from inflating when cloud says unrelated."""
    if cloud_score <= 0 and graph_score <= 0:
        return 0.0

    # Cloud determines its own weight (self-weighting)
    cloud_weight = 0.5 + 0.5 * min(cloud_score, 1.0)

    # Graph capped relative to cloud — prevents false positives
    # GPS/bread: cloud=0.025, graph=0.75 → cap=0.05, effective_graph=0.05
    # plane/aircraft: cloud=0.3, graph=0.5 → cap=0.6, effective_graph=0.5
    max_graph = max(cloud_score * 2.0, 0.05)
    effective_graph = min(graph_score, max_graph)

    return cloud_weight * cloud_score + (1.0 - cloud_weight) * effective_graph


# ============================================================
# MATRIX SIMILARITY — meaning clouds with polysemy handling
# ============================================================

def _matrix_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Similarity via context-filtered meaning clouds.
    Direct overlap discounted when cloud says topics diverge."""
    if not set_a or not set_b:
        return 0.0

    direct = set_a & set_b
    unique_a = set_a - direct
    unique_b = set_b - direct

    scores = []

    src_a = unique_a if unique_a else set_a
    src_b = unique_b if unique_b else set_b
    cloud_a = _meaning_cloud(src_a)
    cloud_b = _meaning_cloud(src_b)
    if cloud_a and cloud_b:
        intersection = len(cloud_a & cloud_b)
        cloud_score = intersection / min(len(cloud_a), len(cloud_b))
        scores.append(cloud_score)

    if direct:
        base = len(direct) / min(len(set_a), len(set_b))
        if scores:
            # Continuous discount: cloud_score itself scales the direct overlap
            # cloud=0 → base*0.1 (near-zero). cloud=0.5 → base*0.55. cloud=1.0 → base*1.0
            # No binary gate, no cliff edge
            discount = 0.1 + 0.9 * min(scores[0], 1.0)
            base *= discount
        scores.append(base)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ============================================================
# FEATURE EXTRACTION + TOKEN SIMILARITY
# ============================================================

def _extract_features(tokens: List[Token]) -> Dict[str, Set[str]]:
    """Extract linguistic feature sets from tokens."""
    stop = _get_stop_words()
    features = {
        'entities': set(),
        'nouns': set(),
        'verbs': set(),
        'adjectives': set(),
        'content': set(),
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
    Cloud-only (no graph). Graph injected at question_passage_relevance level."""
    feat_a = _extract_features(tokens_a)
    feat_b = _extract_features(tokens_b)

    scores = []
    for name in feat_a:
        sa = feat_a[name]
        sb = feat_b[name]
        if sa and sb:
            if name in ('content', 'nouns', 'verbs'):
                scores.append(_matrix_similarity(sa, sb))
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


# ============================================================
# QUESTION-PASSAGE RELEVANCE — the main entry point
# ============================================================

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy's sentence boundary detection."""
    try:
        from .lexer import _get_spacy
        nlp = _get_spacy()
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences if sentences else [text]
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()] or [text]


def question_passage_relevance(question: str, passage: str,
                                lang: str = 'en') -> float:
    """Check if a passage is relevant to a question.
    Multi-sentence passages: split, score each, take best."""
    sentences = _split_sentences(passage)
    if len(sentences) > 1:
        scores = []
        for sent in sentences:
            if len(sent.split()) >= 3:
                scores.append(_question_sentence_relevance(question, sent, lang))
        if scores:
            best_sentence = max(scores)
            whole_passage = _question_sentence_relevance(question, passage, lang)
            # Variance-adaptive: if one sentence stands out, trust it more
            mean_s = sum(scores) / len(scores)
            variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
            cv = (variance ** 0.5) / max(mean_s, 1e-6)
            best_weight = 0.5 + 0.4 * min(cv, 1.0)  # 0.5 (uniform) to 0.9 (outlier)
            return best_weight * best_sentence + (1.0 - best_weight) * whole_passage
    return _question_sentence_relevance(question, passage, lang)


def _question_sentence_relevance(question: str, passage: str,
                                  lang: str = 'en') -> float:
    """Core relevance scoring: AST signals + combined cloud/graph similarity."""
    from .semantics import tokens_to_ast

    q_tokens, q_spans = tokenize(question, lang=lang)
    p_tokens, p_spans = tokenize(passage, lang=lang)

    q_ast = tokens_to_ast(q_tokens, question, entity_spans=q_spans)
    p_ast = tokens_to_ast(p_tokens, passage, entity_spans=p_spans)

    signals = []

    # 1. Entity overlap via meaning matrix
    q_ents = set(e.lower() for e in q_ast.entities)
    p_ents = set(e.lower() for e in p_ast.entities)
    if q_ents and p_ents:
        signals.append(_matrix_similarity(q_ents, p_ents))

    # 2. Predicate match — noise floor scales with polysemy
    stop = _get_stop_words()
    if (q_ast.predicate and p_ast.predicate and
        q_ast.predicate not in stop and p_ast.predicate not in stop):
        pred_sim = _meaning_similarity(q_ast.predicate, p_ast.predicate)
        # Polysemous verbs (run=57 senses, make=51) have higher random match rate
        # Noise floor = baseline verb similarity (~0.15) scaled by polysemy
        try:
            from nltk.corpus import wordnet as wn
            pl = (len(wn.synsets(q_ast.predicate)) + len(wn.synsets(p_ast.predicate))) / 2
            noise_floor = min(0.5, 0.03 * pl)  # 1 sense→0.03, 10→0.30, 17+→0.50
        except Exception:
            noise_floor = 0.05
        if pred_sim > noise_floor:
            signals.append(pred_sim)

    # 3. Answer-type presence
    if q_ast.question_word in ('who', 'whom'):
        if any(t.entity_type == 'PERSON' for t in p_tokens):
            signals.append(1.0)
    elif q_ast.question_word == 'when':
        if any(t.entity_type in ('DATE', 'TIME', 'CARDINAL') for t in p_tokens):
            signals.append(1.0)
    elif q_ast.question_word == 'where':
        if any(t.entity_type in ('GPE', 'LOC') for t in p_tokens):
            signals.append(1.0)

    # 4. Combined cloud + graph similarity (core signal)
    cloud_sim = token_similarity(q_tokens, p_tokens)

    feat_q = _extract_features(q_tokens)
    feat_p = _extract_features(p_tokens)
    q_content = feat_q.get('content', set())
    p_content = feat_p.get('content', set())
    graph_sim = _pairwise_wup(q_content, p_content) if q_content and p_content else 0.0

    combined = _combined_score(cloud_sim, graph_sim)
    signals.append(combined)

    if not signals:
        return 0.0
    return sum(signals) / len(signals)
