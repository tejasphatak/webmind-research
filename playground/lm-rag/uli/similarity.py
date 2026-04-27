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
import re
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


# WordNet caches
_related_cache = {}
_gloss_cache = {}
_meaning_cache = {}

# ============================================================
# MEANING SETS — each word expands to its "meaning fingerprint"
# ============================================================

_SKIP_WORDS = frozenset({'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
    'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'or', 'and',
    'that', 'which', 'it', 'its', 'as', 'not', 'no', 'has', 'have', 'had',
    'do', 'does', 'did', 'will', 'can', 'may', 'such', 'more', 'than', 'into'})


def _get_meaning_set(word: str) -> Set[str]:
    """Build meaning fingerprint: synonyms + hypernyms + hyponyms + definition words.
    Cached. Bidirectional: plane→aircraft AND aircraft→plane."""
    if word in _meaning_cache:
        return _meaning_cache[word]
    try:
        from nltk.corpus import wordnet as wn
        meaning = set()
        for ss in wn.synsets(word)[:3]:  # Top 3 senses only
            # Lemma names (synonyms)
            for lemma in ss.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                meaning.add(name)
            # 2-level hypernym lemmas (plane → aircraft → vehicle)
            frontier = ss.hypernyms()
            for _ in range(2):
                next_f = []
                for h in frontier:
                    for lemma in h.lemmas():
                        meaning.add(lemma.name().replace('_', ' ').lower())
                    next_f.extend(h.hypernyms())
                frontier = next_f
            # 1-level hyponym lemmas (aircraft → plane, helicopter, ...)
            for hypo in ss.hyponyms():
                for lemma in hypo.lemmas():
                    meaning.add(lemma.name().replace('_', ' ').lower())
            # Definition content words (lemmatized)
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


def _context_meaning(word: str, context_words: Set[str]) -> Set[str]:
    """Filter word's meaning set to only include meanings consistent with context.
    'bank' + {habitat, river} → keeps {water, slope} but drops {financial, institution}.
    This is implicit WSD through set intersection — no Lesk, no tree traversal."""
    full = _get_meaning_set(word)
    if not context_words or not full:
        return full  # No context → keep all senses
    # Build context territory: union of all context words' meanings
    territory = set()
    for cw in context_words:
        territory.update(_get_meaning_set(cw))
        territory.add(cw)  # Include the context words themselves
    # Filter: keep only meaning words that appear in context territory
    filtered = full & territory
    # If too aggressive (< 3 words kept), blend with full set
    if len(filtered) < 3:
        return full
    return filtered


def _meaning_cloud(words: Set[str]) -> Set[str]:
    """Build context-aware meaning cloud.
    Each word's meaning set is filtered by the OTHER words in the same text.
    'bank' in ecology text → only river-bank meanings contribute.
    'bank' in finance text → only financial meanings contribute."""
    cloud = set()
    word_list = list(words)
    for word in word_list:
        context = words - {word}
        filtered_meaning = _context_meaning(word, context)
        cloud.update(filtered_meaning)
    # Include original words IN the cloud — they're valid meaning tokens.
    # Example: "energy" is a passage word AND in photosynthesis's meaning set.
    # Removing it breaks the bridge. Polysemy handled by context filtering above.
    cloud.update(words)
    return cloud


def _meaning_similarity(word_a: str, word_b: str) -> float:
    """Semantic similarity between two words.
    Checks: (1) is word_b a meaning of word_a? (synonym/hypernym)
            (2) is word_a a meaning of word_b?
            (3) do their meaning sets share words? (transitive relation)"""
    if word_a == word_b:
        return 1.0
    ma = _get_meaning_set(word_a)
    mb = _get_meaning_set(word_b)
    # Check if one word IS a meaning of the other (direct relationship)
    if word_b in ma or word_a in mb:
        return 0.8  # Direct synonym/hypernym — strong but not identical
    if not ma or not mb:
        return 0.0
    # Meaning set overlap (transitive relationships)
    intersection = len(ma & mb)
    return intersection / min(len(ma), len(mb)) if min(len(ma), len(mb)) > 0 else 0.0


def _sense_meaning(synset) -> Set[str]:
    """Meaning set for a specific WordNet synset."""
    meaning = set()
    for lemma in synset.lemmas():
        meaning.add(lemma.name().replace('_', ' ').lower())
    for w in re.findall(r'[a-z]+', synset.definition().lower()):
        lemma = _simple_lemma(w)
        if len(lemma) > 2 and lemma not in _SKIP_WORDS:
            meaning.add(lemma)
    for hyp in synset.hypernyms():
        for lemma in hyp.lemmas():
            meaning.add(lemma.name().replace('_', ' ').lower())
    return meaning


def _best_sense_for_context(word: str, context: Set[str]) -> tuple:
    """Pick the WordNet sense whose meaning best matches the context.
    Returns (synset, meaning_set, overlap_score)."""
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(word)[:5]  # Top 5 senses
        if not synsets:
            return None, set(), 0
        best_ss = synsets[0]
        best_meaning = _sense_meaning(best_ss)
        best_overlap = 0
        for ss in synsets:
            m = _sense_meaning(ss)
            overlap = len(m & context)
            if overlap > best_overlap:
                best_overlap = overlap
                best_ss = ss
                best_meaning = m
        best_meaning.discard(word)
        return best_ss, best_meaning, best_overlap
    except Exception:
        return None, set(), 0


def _matrix_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Similarity combining direct overlap + meaning cloud on unique words.
    Direct overlap = baseline. Meaning cloud = semantic depth check.
    For polysemy: when Q has 1 word and it overlaps directly, the score
    depends heavily on whether the unique passage words have semantic
    affinity with the question word's meaning territory."""
    if not set_a or not set_b:
        return 0.0

    direct = set_a & set_b
    unique_a = set_a - direct
    unique_b = set_b - direct

    scores = []

    # Meaning cloud overlap on unique (non-shared) words
    # This determines if the texts are about the same TOPIC
    # even when they share a polysemous word
    src_a = unique_a if unique_a else set_a
    src_b = unique_b if unique_b else set_b
    cloud_a = _meaning_cloud(src_a)
    cloud_b = _meaning_cloud(src_b)
    if cloud_a and cloud_b:
        intersection = len(cloud_a & cloud_b)
        cloud_score = intersection / min(len(cloud_a), len(cloud_b))
        scores.append(cloud_score)

    # Direct word overlap (weighted by cloud context)
    if direct:
        base = len(direct) / min(len(set_a), len(set_b))
        if scores and scores[0] < 0.2:
            # Cloud says topics diverge → heavy discount
            # bank+ecology cloud=0.16 → discount. bank+finance cloud=0.55 → no discount.
            base *= 0.2
        scores.append(base)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _get_related_words(word: str) -> Set[str]:
    """Get synonyms, hypernyms (2 levels), and hyponyms from WordNet. Cached.
    'plane' → {'aircraft', 'airplane', 'aeroplane', ...}
    'aircraft' → {'plane', 'airplane', 'helicopter', ...}"""
    if word in _related_cache:
        return _related_cache[word]
    try:
        from nltk.corpus import wordnet as wn
        related = set()
        for ss in wn.synsets(word):
            # Synonyms (same synset)
            for lemma in ss.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name != word:
                    related.add(name)
            # Hypernyms — 2 levels up (plane → heavier-than-air craft → aircraft)
            frontier = ss.hypernyms()
            for _ in range(2):
                next_frontier = []
                for hyper in frontier:
                    for lemma in hyper.lemmas():
                        name = lemma.name().replace('_', ' ').lower()
                        if name != word:
                            related.add(name)
                    next_frontier.extend(hyper.hypernyms())
                frontier = next_frontier
            # Hyponyms — 1 level down (aircraft → plane)
            for hypo in ss.hyponyms():
                for lemma in hypo.lemmas():
                    name = lemma.name().replace('_', ' ').lower()
                    if name != word:
                        related.add(name)
        _related_cache[word] = related
        return related
    except Exception:
        _related_cache[word] = set()
        return set()


def _simple_lemma(word: str) -> str:
    """Crude lemmatization for gloss words — strip common suffixes.
    'plants'→'plant', 'converts'→'convert', 'chemical'→'chemical'."""
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


def _get_gloss_words(word: str) -> Set[str]:
    """Get content words from WordNet definitions (glosses). Cached + lemmatized.
    Extended Lesk: 'photosynthesis' gloss contains 'convert', 'light', 'energy', 'chemical'.
    This catches paraphrases where no synonym/hypernym path exists."""
    if word in _gloss_cache:
        return _gloss_cache[word]
    try:
        from nltk.corpus import wordnet as wn
        import re
        gloss_words = set()
        skip = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from',
                'or', 'and', 'that', 'which', 'it', 'its', 'as', 'not', 'no',
                'has', 'have', 'had', 'do', 'does', 'did', 'will', 'can',
                'may', 'such', 'more', 'than', 'into', 'used', 'use'}
        for ss in wn.synsets(word):
            # Extract words from definition
            definition = ss.definition()
            tokens = re.findall(r'[a-z]+', definition.lower())
            for t in tokens:
                lemma = _simple_lemma(t)
                if len(lemma) > 2 and lemma not in skip and lemma != word:
                    gloss_words.add(lemma)
                    if lemma != t and len(t) > 2 and t not in skip:
                        gloss_words.add(t)  # Keep both forms
            # Also check example sentences
            for example in ss.examples():
                tokens = re.findall(r'[a-z]+', example.lower())
                for t in tokens:
                    lemma = _simple_lemma(t)
                    if len(lemma) > 2 and lemma not in skip and lemma != word:
                        gloss_words.add(lemma)
        _gloss_cache[word] = gloss_words
        return gloss_words
    except Exception:
        _gloss_cache[word] = set()
        return set()


_specificity_cache = {}

def _word_specificity(word: str) -> float:
    """How specific/unambiguous a word is. 1.0 = one sense, lower = polysemous.
    'photosynthesis' (1 sense) → 1.0, 'bank' (18 senses) → 0.25.
    Used as confidence weight for synonym matches."""
    if word in _specificity_cache:
        return _specificity_cache[word]
    try:
        from nltk.corpus import wordnet as wn
        n_senses = len(wn.synsets(word))
        if n_senses == 0:
            spec = 0.5  # Unknown word — moderate confidence
        elif n_senses == 1:
            spec = 1.0
        else:
            # 2 senses → 0.7, 5 → 0.45, 10 → 0.3, 20 → 0.2
            spec = 1.0 / (1.0 + 0.3 * (n_senses - 1))
        _specificity_cache[word] = spec
        return spec
    except Exception:
        _specificity_cache[word] = 0.5
        return 0.5


# ============================================================
# LESK WSD — Word Sense Disambiguation (Lesk 1986, no neural)
# ============================================================

_lesk_cache = {}

_sig_cache = {}

def _lesk_signature(synset) -> Set[str]:
    """Build signature for a synset: definition + examples + hypernym definitions
    + synset lemma names. Lightweight but effective for WSD."""
    key = synset.name()
    if key in _sig_cache:
        return _sig_cache[key]
    import re
    sig = set()
    skip = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'of', 'in',
            'to', 'for', 'on', 'at', 'by', 'with', 'from', 'or', 'and', 'that',
            'which', 'it', 'its', 'as', 'not', 'no', 'has', 'have', 'do'}
    for text in [synset.definition()] + synset.examples():
        for w in re.findall(r'[a-z]+', text.lower()):
            lemma = _simple_lemma(w)
            if len(lemma) > 2 and lemma not in skip:
                sig.add(lemma)
                sig.add(w)
    # Extended Lesk: include hypernym definitions (one level)
    for hyp in synset.hypernyms():
        for w in re.findall(r'[a-z]+', hyp.definition().lower()):
            lemma = _simple_lemma(w)
            if len(lemma) > 2 and lemma not in skip:
                sig.add(lemma)
        # Include hypernym lemma names (e.g., "financial_institution" → "financial", "institution")
        for lem in hyp.lemmas():
            for part in lem.name().lower().split('_'):
                if len(part) > 2: sig.add(part)
    # Include all lemma names of this synset + hyponyms (one level)
    for lem in synset.lemmas():
        for part in lem.name().lower().split('_'):
            if len(part) > 2: sig.add(part)
    for hypo in synset.hyponyms():
        for lem in hypo.lemmas():
            for part in lem.name().lower().split('_'):
                if len(part) > 2: sig.add(part)
    _sig_cache[key] = sig
    return sig


def _lesk_disambiguate(word: str, context_words: Set[str]):
    """Simplified Lesk: pick the WordNet sense whose signature
    overlaps most with context_words. Returns (synset, overlap_count).
    No context → returns first (most frequent) sense."""
    cache_key = (word, frozenset(context_words))
    if cache_key in _lesk_cache:
        return _lesk_cache[cache_key]
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(word)
        if not synsets:
            _lesk_cache[cache_key] = (None, 0)
            return None, 0
        best_sense = synsets[0]  # Default: most frequent sense
        best_overlap = 0
        for ss in synsets:
            sig = _lesk_signature(ss)
            overlap = len(sig & context_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sense = ss
        _lesk_cache[cache_key] = (best_sense, best_overlap)
        return best_sense, best_overlap
    except Exception:
        _lesk_cache[cache_key] = (None, 0)
        return None, 0


def _same_sense(word: str, context_a: Set[str], context_b: Set[str]) -> float:
    """Check if 'word' is used in the same sense in two contexts.
    Returns 1.0 for same sense, 0.3 for different sense, 0.5 if uncertain."""
    sense_a, overlap_a = _lesk_disambiguate(word, context_a)
    sense_b, overlap_b = _lesk_disambiguate(word, context_b)
    if sense_a is None or sense_b is None:
        return 0.5  # Can't determine
    if sense_a == sense_b:
        return 1.0
    # Different senses: check if they're at least related (same hypernym)
    a_hypers = set(h.name() for h in sense_a.hypernyms())
    b_hypers = set(h.name() for h in sense_b.hypernyms())
    if a_hypers & b_hypers:
        return 0.5  # Related but different
    return 0.2  # Clearly different senses


# ============================================================
# DEFINITION-PASSAGE MATCHING — catches heavy paraphrases
# ============================================================

_def_match_cache = {}

def _definition_passage_score(word: str, passage_content: Set[str]) -> float:
    """Check if the PRIMARY WordNet definition of 'word' overlaps with passage.
    Only checks the top 3 senses (avoids noise from 50+ sense words like 'run').
    Uses synonym expansion on definition words for robustness.
    'photosynthesis' def has 'plant','energy','radiant' → matches passage
    with 'organism','energy','radiation' via synonym expansion."""
    cache_key = (word, frozenset(passage_content))
    if cache_key in _def_match_cache:
        return _def_match_cache[cache_key]
    try:
        from nltk.corpus import wordnet as wn
        import re
        skip = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from',
                'or', 'and', 'that', 'which', 'it', 'its', 'as', 'not', 'no',
                'has', 'have', 'had', 'do', 'does', 'did'}
        best_score = 0.0
        synsets = wn.synsets(word)
        # Only check top 3 senses — polysemous words (50+ senses) create noise
        for ss in synsets[:3]:
            def_words = set()
            for w in re.findall(r'[a-z]+', ss.definition().lower()):
                lemma = _simple_lemma(w)
                if len(lemma) > 2 and lemma not in skip and lemma != word:
                    def_words.add(lemma)
            if not def_words or len(def_words) < 3:
                continue  # Skip short definitions (too noisy)
            # Count matches: direct + synonym-expanded
            matched = 0
            for dw in def_words:
                if dw in passage_content:
                    matched += 1
                else:
                    related = _get_related_words(dw)
                    if related & passage_content:
                        matched += 0.7
            score = matched / len(def_words)
            best_score = max(best_score, score)
        _def_match_cache[cache_key] = best_score
        return best_score
    except Exception:
        _def_match_cache[cache_key] = 0.0
        return 0.0


def _overlap_with_synonyms(set_a: Set[str], set_b: Set[str]) -> float:
    """Overlap coefficient with bidirectional WordNet expansion + gloss overlap.
    Three tiers: direct match > synonym/hypernym match > gloss (definition) match.
    'plane' matches 'aircraft' via hypernym.
    'photosynthesis' matches 'convert sunlight energy' via gloss overlap."""
    if not set_a or not set_b:
        return 0.0

    # Tier 1: Direct overlap
    direct = set_a & set_b
    if direct:
        return len(direct) / min(len(set_a), len(set_b))

    # Tier 2: Bidirectional synonym/hypernym expansion
    # Compute overlap from each direction independently, then average.
    # Weighted by specificity: polysemous words (many senses) get less credit.

    # A→B: what fraction of A words have a synonym in B?
    a_score = 0.0
    for word in set_a:
        related = _get_related_words(word)
        if related & set_b:
            a_score += _word_specificity(word)
    a_overlap = a_score / len(set_a) if set_a else 0.0

    # B→A: what fraction of B words have a synonym in A?
    b_score = 0.0
    for word in set_b:
        related = _get_related_words(word)
        if related & set_a:
            b_score += _word_specificity(word)
    b_overlap = b_score / len(set_b) if set_b else 0.0

    if a_overlap > 0 or b_overlap > 0:
        # Average of both perspectives
        return (a_overlap + b_overlap) / 2.0

    # Tier 3: Gloss (definition) overlap — Extended Lesk
    # For each word in A, check if B words appear in A-word's definition
    gloss_matches = 0
    for word in set_a:
        gloss = _get_gloss_words(word)
        overlap = gloss & set_b
        if len(overlap) >= 2:  # Require 2+ gloss words to match (avoids noise)
            gloss_matches += 1

    # Reverse: for each word in B, check if A words appear in B-word's definition
    for word in set_b:
        gloss = _get_gloss_words(word)
        overlap = gloss & set_a
        if len(overlap) >= 2:
            gloss_matches += 1

    if gloss_matches > 0:
        # Gloss matches are weaker than synonym matches — scale by 0.7
        score = min(gloss_matches / (2 * min(len(set_a), len(set_b))), 1.0) * 0.7
        return score

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
    Uses meaning matrix: each word expands to its semantic fingerprint,
    then similarity = best pairwise match averaged across both directions.
    Adaptive: only components with signal contribute."""

    feat_a = _extract_features(tokens_a)
    feat_b = _extract_features(tokens_b)

    scores = []
    for name in feat_a:
        sa = feat_a[name]
        sb = feat_b[name]
        if sa and sb:
            # Use meaning matrix for content-bearing features
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


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy's sentence boundary detection."""
    try:
        from .lexer import _get_spacy
        nlp = _get_spacy()
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences if sentences else [text]
    except Exception:
        # Fallback: split on sentence-ending punctuation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()] or [text]


def question_passage_relevance(question: str, passage: str,
                                lang: str = 'en') -> float:
    """Check if a passage is relevant to a question.

    For multi-sentence passages: splits into sentences, scores each
    individually, takes the best. Prevents signal dilution from
    irrelevant sentences in a paragraph.

    Uses AST comparison: same predicate + same entities + matching roles
    = relevant. Different predicates or no entity overlap = irrelevant.
    """
    # Split passage into sentences — score each, take best
    sentences = _split_sentences(passage)
    if len(sentences) > 1:
        scores = []
        for sent in sentences:
            if len(sent.split()) >= 3:  # Skip fragments
                scores.append(_question_sentence_relevance(question, sent, lang))
        if scores:
            # Best sentence score dominates, but whole-passage score still contributes
            best_sentence = max(scores)
            whole_passage = _question_sentence_relevance(question, passage, lang)
            # Weighted: 70% best sentence, 30% whole passage
            # (whole passage captures cross-sentence entity references)
            return 0.7 * best_sentence + 0.3 * whole_passage
    return _question_sentence_relevance(question, passage, lang)


def _question_sentence_relevance(question: str, passage: str,
                                  lang: str = 'en') -> float:
    """Core relevance scoring between a question and a single sentence/passage."""
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

    # 2. Predicate match via meaning similarity
    stop = _get_stop_words()
    if (q_ast.predicate and p_ast.predicate and
        q_ast.predicate not in stop and p_ast.predicate not in stop):
        pred_sim = _meaning_similarity(q_ast.predicate, p_ast.predicate)
        if pred_sim > 0.05:
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

    # 4. Token-level meaning matrix similarity (core signal)
    base_sim = token_similarity(q_tokens, p_tokens)
    signals.append(base_sim)

    if not signals:
        return 0.0
    return sum(signals) / len(signals)
