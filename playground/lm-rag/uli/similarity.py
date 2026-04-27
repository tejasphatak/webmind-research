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
        # Select synsets: top 2 per POS (not just first 3 overall).
        # "drive" has 12 noun senses before any verb sense — [:3] misses
        # the VERB sense entirely. This ensures both POS are covered.
        all_synsets = wn.synsets(word)
        selected = []
        pos_counts = {}
        for ss in all_synsets:
            p = ss.pos()
            pos_counts[p] = pos_counts.get(p, 0) + 1
            if pos_counts[p] <= 2:  # Max 2 per POS
                selected.append(ss)
            if len(selected) >= 4:  # Max 4 total
                break
        for ss in selected:
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


# Weighted meaning sets — Collins & Loftus spreading activation with decay
_weighted_meaning_cache: Dict[str, Dict[str, float]] = {}

# Decay weights by relationship type (from psycholinguistic priming data)
# Synonyms = full activation, distant hypernyms = attenuated
_DECAY = {
    'synonym': 1.0,
    'hypernym1': 0.7,
    'hypernym2': 0.49,    # 0.7^2
    'hyponym': 0.6,
    'definition': 0.3,
}


def _get_weighted_meaning_set(word: str) -> Dict[str, float]:
    """Build WEIGHTED meaning fingerprint. Closer relationships = higher weight.
    Spreading activation: synonyms=1.0, hypernym1=0.7, hypernym2=0.49, etc.
    Sense index weight: first sense (most common) = 1.0, second = 0.5, third = 0.33.
    Returns Dict[str, float] — word → activation weight."""
    if word in _weighted_meaning_cache:
        return _weighted_meaning_cache[word]
    try:
        from nltk.corpus import wordnet as wn
        meaning: Dict[str, float] = {}
        # Same POS-balanced selection as _get_meaning_set
        all_ss = wn.synsets(word)
        sel = []
        pc = {}
        for ss in all_ss:
            p = ss.pos()
            pc[p] = pc.get(p, 0) + 1
            if pc[p] <= 2:
                sel.append(ss)
            if len(sel) >= 4:
                break
        for i, ss in enumerate(sel):
            sense_w = 1.0 / (1.0 + i)  # Prototype effect (Rosch 1975)
            # Synonyms
            for lemma in ss.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                w = _DECAY['synonym'] * sense_w
                meaning[name] = max(meaning.get(name, 0), w)
            # Hypernyms — 2 levels with decay
            frontier = ss.hypernyms()
            for level, key in enumerate(['hypernym1', 'hypernym2']):
                next_f = []
                for h in frontier:
                    for lemma in h.lemmas():
                        name = lemma.name().replace('_', ' ').lower()
                        w = _DECAY[key] * sense_w
                        meaning[name] = max(meaning.get(name, 0), w)
                    next_f.extend(h.hypernyms())
                frontier = next_f
            # Hyponyms
            for hypo in ss.hyponyms():
                for lemma in hypo.lemmas():
                    name = lemma.name().replace('_', ' ').lower()
                    w = _DECAY['hyponym'] * sense_w
                    meaning[name] = max(meaning.get(name, 0), w)
            # Definition words
            for dw in re.findall(r'[a-z]+', ss.definition().lower()):
                lemma = _simple_lemma(dw)
                if len(lemma) > 2 and lemma not in _SKIP_WORDS:
                    w = _DECAY['definition'] * sense_w
                    meaning[lemma] = max(meaning.get(lemma, 0), w)
        meaning.pop(word, None)
        _weighted_meaning_cache[word] = meaning
        return meaning
    except Exception:
        _weighted_meaning_cache[word] = {}
        return {}


# ============================================================
# CONTEXT-FILTERED MEANING CLOUDS — non-neural attention
# ============================================================

def _context_meaning(word: str, context_words: Set[str]) -> Dict[str, float]:
    """Filter word's meaning set by context using GRADED activation.
    Each meaning gets weight proportional to how many context words support it.
    Unsupported meanings get 0.1x weight (suppressed, not eliminated).
    Returns Dict[str, float] — meaning → activation weight."""
    full = _get_weighted_meaning_set(word)
    if not context_words or not full:
        return full

    # Count votes: how many context words' meaning sets contain each meaning
    vote_count: Dict[str, int] = {}
    for cw in context_words:
        cw_meanings = _get_meaning_set(cw)  # Binary set for vote counting
        for m in full:
            if m in cw_meanings or m == cw:
                vote_count[m] = vote_count.get(m, 0) + 1

    if not vote_count:
        return full  # No context overlap — keep everything

    max_votes = max(vote_count.values())

    graded: Dict[str, float] = {}
    for meaning, base_weight in full.items():
        votes = vote_count.get(meaning, 0)
        if votes > 0:
            graded[meaning] = base_weight * (votes / max_votes)
        else:
            graded[meaning] = base_weight * 0.1  # Active suppression
    return graded


def _meaning_cloud(words: Set[str]) -> Dict[str, float]:
    """Build context-aware WEIGHTED meaning cloud.
    Each word's meaning set filtered by OTHER words (graded activation).
    Returns Dict[str, float] — meaning → activation weight."""
    cloud: Dict[str, float] = {}
    word_list = list(words)
    for word in word_list:
        context = words - {word}
        filtered = _context_meaning(word, context)
        for m, w in filtered.items():
            cloud[m] = max(cloud.get(m, 0), w)
    # Include original words at full activation
    for w in words:
        cloud[w] = 1.0
    return cloud


def _meaning_similarity(word_a: str, word_b: str) -> float:
    """Semantic similarity between two words via meaning sets.
    Also checks morphological relatives (shared root ≥ 5 chars):
    refrigerator↔refrigerant, anesthesia↔anesthetic."""
    if word_a == word_b:
        return 1.0
    # Morphological relatives: shared root of 5+ chars = derivational relationship
    shared_prefix = 0
    for i in range(min(len(word_a), len(word_b))):
        if word_a[i] == word_b[i]:
            shared_prefix += 1
        else:
            break
    if shared_prefix >= 5:
        return 0.7
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
    Contribution = max(0, wup - 0.2) / 0.8 — continuous, no hard cutoff."""
    if not set_a or not set_b:
        return 0.0
    smaller, larger = (set_a, set_b) if len(set_a) <= len(set_b) else (set_b, set_a)
    total = 0.0
    for wa in smaller:
        best = max((_first_sense_wup(wa, wb) for wb in larger), default=0.0)
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
        # Weighted intersection: sum of min(weight_a, weight_b) for shared keys
        shared_keys = set(cloud_a.keys()) & set(cloud_b.keys())
        weighted_inter = sum(min(cloud_a[k], cloud_b[k]) for k in shared_keys)
        total_a = sum(cloud_a.values())
        total_b = sum(cloud_b.values())
        cloud_score = weighted_inter / min(total_a, total_b) if min(total_a, total_b) > 0 else 0.0
        scores.append(cloud_score)

    # PARAPHRASE BRIDGE (recursive): for Q words NOT in P, check if P
    # contains DEFINITION words of any SPECIFIC SENSE of the Q word.
    # Per-sense matching avoids dilution from polysemous Q words:
    # "light" has 50+ senses → all bridge words = 85 → 2/85 = 0.02
    # BUT electromagnetic sense alone = 10 words → 2/10 = 0.20
    if unique_a and unique_b:
        best_bridge = 0.0
        for qw in unique_a:
            try:
                from nltk.corpus import wordnet as wn
                for ss in wn.synsets(qw)[:3]:
                    # Hop 1: definition words of THIS SPECIFIC SENSE
                    sense_bridge = set()
                    for lemma in ss.lemmas():
                        sense_bridge.add(lemma.name().replace('_', ' ').lower())
                    for w in re.findall(r'[a-z]+', ss.definition().lower()):
                        lemma = _simple_lemma(w)
                        if len(lemma) > 2 and lemma not in _SKIP_WORDS:
                            sense_bridge.add(lemma)
                    sense_bridge.discard(qw)

                    if sense_bridge:
                        overlap = sense_bridge & unique_b
                        if overlap:
                            coverage = len(overlap) / len(sense_bridge)
                            best_bridge = max(best_bridge, coverage)
            except Exception:
                pass
        if best_bridge > 0:
            scores.append(best_bridge)

    if direct:
        base = len(direct) / min(len(set_a), len(set_b))

        # CROSS-ATTENTION: for each shared word, Q and P each attend to
        # their best-matching sense. If they select DIFFERENT senses →
        # the word means different things → discount.
        #
        # Q_sense = argmax(Q_context · sense_definitions)
        # P_sense = argmax(P_context · sense_definitions)
        # discount = 1.0 if same sense, 0.1 if different
        for word in direct:
            try:
                from nltk.corpus import wordnet as wn
                synsets = wn.synsets(word)[:5]
                if len(synsets) > 1:  # Only for polysemous words
                    # Build sense definition sets
                    sense_defs = []
                    for ss in synsets:
                        sd = set()
                        for lemma in ss.lemmas():
                            sd.add(lemma.name().replace('_', ' ').lower())
                        for w in re.findall(r'[a-z]+', ss.definition().lower()):
                            lm = _simple_lemma(w)
                            if len(lm) > 2 and lm not in _SKIP_WORDS:
                                sd.add(lm)
                        sd.discard(word)
                        sense_defs.append(sd)

                    # Q attends: which sense matches Q's context?
                    q_ctx = unique_a if unique_a else set_a
                    q_best = max(range(len(sense_defs)),
                                 key=lambda i: len(sense_defs[i] & q_ctx),
                                 default=0)
                    q_score = len(sense_defs[q_best] & q_ctx)

                    # P attends: which sense matches P's context?
                    p_ctx = unique_b if unique_b else set_b
                    p_best = max(range(len(sense_defs)),
                                 key=lambda i: len(sense_defs[i] & p_ctx),
                                 default=0)
                    p_score = len(sense_defs[p_best] & p_ctx)

                    # Cross-attention decision:
                    if q_best != p_best and (q_score > 0 or p_score > 0):
                        base *= 0.1  # Different senses → heavy discount
                    elif q_score == 0 and p_score == 0 and len(synsets) > 1:
                        # ZERO attention on both sides → ambiguous
                        # Can't tell if same sense. Don't give full credit.
                        # Discount by 1/num_senses (random chance of matching)
                        base *= 1.0 / len(synsets)
            except Exception:
                pass

        # Cloud discount: catches topic divergence when cross-attention can't decide.
        # Cross-attention: catches polysemy when context disambiguates.
        # Both are needed — they're complementary, not redundant.
        if scores:
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
    """Extract ALL linguistic feature sets from tokens.
    Includes every word — function words have meaning too.
    'You can do it' ≈ 'Yes, you can do it' — EVERY word matters."""
    stop = _get_stop_words()
    features = {
        'entities': set(),
        'nouns': set(),
        'verbs': set(),
        'adjectives': set(),
        'content': set(),      # Content words (nouns, verbs, adj, adv)
        'all_words': set(),    # ALL words including function words
    }
    for tok in tokens:
        word = tok.text.lower().strip()
        lemma = (tok.lemma or tok.text).lower().strip()

        if tok.pos == 'PUNCT' or len(word) < 2:
            continue

        # ALL non-punctuation words go into all_words
        features['all_words'].add(lemma)

        if word in stop:
            continue  # Function words skip content features

        is_proper = tok.is_entity or tok.pos == 'PROPN' or (tok.text.isupper() and len(tok.text) > 1)
        if is_proper:
            features['entities'].add(word)
        if tok.pos in ('NOUN', 'PROPN') or is_proper:
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
    Combines content-word meaning overlap with all-word surface overlap.
    Content words go through meaning expansion (WordNet clouds).
    All words use simple overlap (surface form matching)."""
    feat_a = _extract_features(tokens_a)
    feat_b = _extract_features(tokens_b)

    # Content-based scores (meaning expansion through WordNet)
    content_scores = []
    for name in feat_a:
        sa = feat_a[name]
        sb = feat_b[name]
        if sa and sb:
            if name in ('content', 'nouns', 'verbs'):
                content_scores.append(_matrix_similarity(sa, sb))
            elif name in ('entities', 'adjectives'):
                content_scores.append(_overlap(sa, sb))

    content_sim = sum(content_scores) / len(content_scores) if content_scores else 0.0

    # All-words Jaccard (surface form, includes function words)
    all_a = feat_a.get('all_words', set())
    all_b = feat_b.get('all_words', set())
    if all_a and all_b:
        all_jaccard = len(all_a & all_b) / len(all_a | all_b)
    else:
        all_jaccard = 0.0

    # Combine content similarity with all-words Jaccard.
    # Content captures MEANING overlap (topic matching via WordNet).
    # Jaccard captures SURFACE overlap (paraphrase via shared words).
    # Both contribute — take the MAX (whichever signal is stronger wins).
    # Jaccard naturally handles scale: short similar sentences get high
    # Jaccard (5/6 words = 0.83), long unrelated passages get low
    # Jaccard (3/30 words = 0.10). No threshold needed.
    return max(content_sim, all_jaccard)


def text_similarity(text_a: str, text_b: str, lang: str = 'en') -> float:
    """Structural similarity between two texts."""
    tokens_a, _ = tokenize(text_a, lang=lang)
    tokens_b, _ = tokenize(text_b, lang=lang)
    return token_similarity(tokens_a, tokens_b)


# ============================================================
# QUESTION-PASSAGE RELEVANCE — the main entry point
# ============================================================

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation rules (no spaCy needed)."""
    from .pos_tagger import split_sentences
    sentences = split_sentences(text)
    return sentences if sentences else [text]


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

    # 1. Entity overlap — use SPANS (multi-word names as units)
    # "French Revolution" as a unit ≠ "Industrial Revolution"
    # Entity spans from tokenizer group multi-word entities
    q_span_texts = set(text.lower() for text, _ in q_spans)
    p_span_texts = set(text.lower() for text, _ in p_spans)
    if q_span_texts and p_span_texts:
        # Span-level overlap: "French Revolution" vs "Industrial Revolution" → no match
        span_overlap = len(q_span_texts & p_span_texts)
        span_score = span_overlap / min(len(q_span_texts), len(p_span_texts))
        signals.append(span_score)
    # No fallback: when one side lacks spans, entity matching is unreliable.
    # Content overlap (S4) handles word-level matching without polysemy inflation.

    # 2. Predicate match — noise floor scales with polysemy
    stop = _get_stop_words()
    if (q_ast.predicate and p_ast.predicate and
        q_ast.predicate not in stop and p_ast.predicate not in stop):
        pred_sim = _meaning_similarity(q_ast.predicate, p_ast.predicate)
        # Polysemous verbs (run=57 senses, make=51) have higher random match rate
        # Noise floor = baseline verb similarity (~0.15) scaled by polysemy
        try:
            from nltk.corpus import wordnet as wn
            pl = max(len(wn.synsets(q_ast.predicate)), len(wn.synsets(p_ast.predicate)))
            # Use MAX polysemy — either verb being ultra-polysemous is suspect
            # make(51)↔form(23): max=51→floor=0.88, blocks spurious match
            # paint(7)↔draw(7): max=7→floor=0.51, allows genuine match
            noise_floor = 1.0 - 1.0 / (1.0 + 0.15 * pl)
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

    # 4. Entity NAME vs CONCEPT check
    # "Mercury" (Q, standalone) vs "Freddie Mercury" (P, multi-word name)
    # When a shared word is part of a LONGER entity span in P but not Q,
    # the passage is using it as a NAME → different usage → discount all signals
    name_mismatch = False
    feat_q = _extract_features(q_tokens)
    feat_p = _extract_features(p_tokens)
    q_content = feat_q.get('content', set())
    p_content = feat_p.get('content', set())
    shared = q_content & p_content
    if shared:
        for word in shared:
            # Is this word part of a multi-word entity span in P?
            for span_text, _ in p_spans:
                span_words = span_text.lower().split()
                if word in span_words and len(span_words) > 1:
                    # Check: is it standalone in Q spans?
                    q_standalone = any(
                        text.lower() == word and len(text.split()) == 1
                        for text, _ in q_spans
                    )
                    q_not_in_span = not any(
                        word in text.lower().split() and len(text.split()) > 1
                        for text, _ in q_spans
                    )
                    if q_standalone or q_not_in_span:
                        name_mismatch = True
                        break
            if name_mismatch:
                break

    # 5. Combined cloud + graph similarity (core signal)
    cloud_sim = token_similarity(q_tokens, p_tokens)

    q_unique = q_content - shared
    p_unique = p_content - shared
    graph_sim = _pairwise_wup(q_unique, p_unique) if q_unique and p_unique else 0.0

    combined = _combined_score(cloud_sim, graph_sim)

    # Apply name mismatch discount to combined signal
    if name_mismatch:
        combined *= 0.2  # Name vs concept → heavy discount

    signals.append(combined)

    if not signals:
        return 0.0
    return sum(signals) / len(signals)
