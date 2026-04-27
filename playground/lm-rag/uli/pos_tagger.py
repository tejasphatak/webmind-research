"""
ULI POS Tagger + Rule-Based Dependency Parser

Replaces spaCy entirely. Uses:
- Vocab DB (77K words with POS, senses) for POS tagging
- Grammar JSON (dependency rules, SVO order) for parsing
- Morphological rules for unknown words
- Capitalization + vocab for entity detection

No neural model. No 300MB dependency. All from our data.
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from .protocol import Token

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# ============================================================
# DATA LOADERS — cached
# ============================================================

_vocab_cache: Dict[str, dict] = {}
_grammar_cache: Dict[str, dict] = {}


def _load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _get_vocab(lang: str = 'en') -> dict:
    if lang not in _vocab_cache:
        data = _load_json(os.path.join(DATA_DIR, 'vocab', f'{lang}.json'))
        _vocab_cache[lang] = data.get('words', {})
    return _vocab_cache[lang]


def _get_grammar(lang: str = 'en') -> dict:
    if lang not in _grammar_cache:
        _grammar_cache[lang] = _load_json(os.path.join(DATA_DIR, 'grammar', f'{lang}.json'))
    return _grammar_cache[lang]


# ============================================================
# POS TAGGER — vocab lookup + morphological fallback
# ============================================================

# Function words that don't need vocab lookup
_FUNCTION_WORDS = {
    # Determiners
    'the': 'DET', 'a': 'DET', 'an': 'DET', 'this': 'DET', 'that': 'DET',
    'these': 'DET', 'those': 'DET', 'my': 'DET', 'your': 'DET', 'his': 'DET',
    'her': 'DET', 'its': 'DET', 'our': 'DET', 'their': 'DET', 'some': 'DET',
    'any': 'DET', 'every': 'DET', 'each': 'DET', 'no': 'DET',
    # Prepositions
    'in': 'ADP', 'on': 'ADP', 'at': 'ADP', 'to': 'ADP', 'for': 'ADP',
    'with': 'ADP', 'by': 'ADP', 'from': 'ADP', 'of': 'ADP', 'about': 'ADP',
    'into': 'ADP', 'through': 'ADP', 'during': 'ADP', 'before': 'ADP',
    'after': 'ADP', 'above': 'ADP', 'below': 'ADP', 'between': 'ADP',
    'under': 'ADP', 'near': 'ADP', 'across': 'ADP', 'behind': 'ADP',
    # Conjunctions
    'and': 'CCONJ', 'or': 'CCONJ', 'but': 'CCONJ', 'nor': 'CCONJ',
    'because': 'SCONJ', 'although': 'SCONJ', 'while': 'SCONJ',
    'if': 'SCONJ', 'when': 'SCONJ', 'where': 'SCONJ', 'since': 'SCONJ',
    'until': 'SCONJ', 'unless': 'SCONJ', 'though': 'SCONJ',
    # Auxiliaries
    'is': 'AUX', 'are': 'AUX', 'was': 'AUX', 'were': 'AUX', 'am': 'AUX',
    'be': 'AUX', 'been': 'AUX', 'being': 'AUX',
    'has': 'AUX', 'have': 'AUX', 'had': 'AUX', 'having': 'AUX',
    'do': 'AUX', 'does': 'AUX', 'did': 'AUX',
    'will': 'AUX', 'would': 'AUX', 'shall': 'AUX', 'should': 'AUX',
    'can': 'AUX', 'could': 'AUX', 'may': 'AUX', 'might': 'AUX',
    'must': 'AUX',
    # Pronouns
    'i': 'PRON', 'me': 'PRON', 'we': 'PRON', 'us': 'PRON',
    'you': 'PRON', 'he': 'PRON', 'him': 'PRON', 'she': 'PRON',
    'it': 'PRON', 'they': 'PRON', 'them': 'PRON',
    'who': 'PRON', 'whom': 'PRON', 'what': 'PRON', 'which': 'PRON',
    'how': 'ADV', 'why': 'ADV', 'where': 'ADV', 'when': 'ADV',
    # Adverbs (common)
    'not': 'ADV', 'very': 'ADV', 'also': 'ADV', 'too': 'ADV',
    'then': 'ADV', 'there': 'ADV', 'here': 'ADV', 'now': 'ADV',
    'just': 'ADV', 'only': 'ADV', 'never': 'ADV', 'always': 'ADV',
    # Particles
    'up': 'PART', 'out': 'PART', 'off': 'PART',
}

# Suffix → POS for unknown words
_SUFFIX_POS = [
    ('tion', 'NOUN'), ('sion', 'NOUN'), ('ment', 'NOUN'), ('ness', 'NOUN'),
    ('ity', 'NOUN'), ('ance', 'NOUN'), ('ence', 'NOUN'), ('ism', 'NOUN'),
    ('ist', 'NOUN'), ('ology', 'NOUN'),
    ('ous', 'ADJ'), ('ful', 'ADJ'), ('less', 'ADJ'), ('able', 'ADJ'),
    ('ible', 'ADJ'), ('ive', 'ADJ'), ('ical', 'ADJ'),
    ('ize', 'VERB'), ('ise', 'VERB'), ('ify', 'VERB'), ('ate', 'VERB'),
    ('ly', 'ADV'),
]


def _tag_word(word: str, vocab: dict, is_sentence_start: bool = False,
              prev_pos: str = '') -> str:
    """POS tag a single word using vocab + rules.
    Uses prev_pos for context-aware disambiguation (NOUN vs VERB)."""
    lower = word.lower()

    # Punctuation
    if all(c in '.,;:!?-()[]{}"\'' for c in word):
        return 'PUNCT'

    # Numbers
    if word.replace('.', '').replace(',', '').replace('-', '').isdigit():
        return 'NUM'

    # Function words (closed class — these don't change across domains)
    if lower in _FUNCTION_WORDS:
        return _FUNCTION_WORDS[lower]

    # ALL-CAPS abbreviation (GPS, DNA, RNA, USA)
    if word.isupper() and len(word) > 1:
        entry = vocab.get(word) or vocab.get(lower)
        if entry:
            pos = entry.get('pos', ['NOUN'])
            return pos[0] if pos else 'NOUN'
        return 'PROPN'

    # Capitalized (not sentence start) → proper noun
    if word[0].isupper() and not is_sentence_start and len(word) > 1:
        return 'PROPN'

    # Lemmatize FIRST — catches "painted"→"paint"→VERB, "sat"→"sit"→VERB
    lemma = _lemmatize(lower, 'NOUN', vocab)

    # If word is inflected (-ed, -ing, -s) and lemma is a VERB → tag as VERB
    # "painted" in vocab = ADJ, but lemma "paint" = VERB → use VERB
    if lemma != lower:
        lemma_entry = vocab.get(lemma)
        if lemma_entry and 'VERB' in lemma_entry.get('pos', []):
            if lower.endswith(('ed', 'ing', 's', 'es')):
                return 'VERB'

    # Vocab lookup on lemma (then original)
    entry = vocab.get(lemma) or vocab.get(lower) or vocab.get(word)
    if entry:
        pos_list = entry.get('pos', [])
        if pos_list:
            if len(pos_list) == 1:
                return pos_list[0]
            # Context disambiguation for NOUN/VERB ambiguity
            if 'VERB' in pos_list and 'NOUN' in pos_list:
                # After DET/ADP/ADJ → NOUN ("the bank", "on the bank", "big run")
                if prev_pos in ('DET', 'ADP', 'ADJ'):
                    return 'NOUN'
                # After AUX/PRON/NOUN → VERB ("does GPS work", "I run")
                if prev_pos in ('AUX', 'PRON', 'NOUN', 'PROPN', 'ADV'):
                    return 'VERB'
                return pos_list[0]
            return pos_list[0]

    # Inflected forms → VERB indicators
    if lower.endswith('ed') and len(lower) > 3:
        return 'VERB'
    if lower.endswith('ing') and len(lower) > 4:
        return 'VERB'

    # Morphological fallback
    for suffix, pos in _SUFFIX_POS:
        if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
            return pos

    # Default: noun
    return 'NOUN'


def _lemmatize(word: str, pos: str, vocab: dict) -> str:
    """Get lemma from vocab or suffix stripping."""
    lower = word.lower()
    if lower in vocab:
        return lower

    # Try stripping inflections
    candidates = []
    if lower.endswith('s') and not lower.endswith('ss'):
        candidates.append(lower[:-1])
    if lower.endswith('es'):
        candidates.append(lower[:-2])
    if lower.endswith('ed'):
        candidates.extend([lower[:-2], lower[:-1]])
    if lower.endswith('ing'):
        candidates.extend([lower[:-3], lower[:-3] + 'e'])
    if lower.endswith('ies'):
        candidates.append(lower[:-3] + 'y')
    if lower.endswith('er'):
        candidates.extend([lower[:-2], lower[:-1]])
    if lower.endswith('est'):
        candidates.extend([lower[:-3], lower[:-2]])

    for c in candidates:
        if c in vocab:
            return c
    return lower


# ============================================================
# WORD TOKENIZER — split text into word tokens
# ============================================================

_WORD_PATTERN = re.compile(r"""
    (?:[A-Z]\.)+          |  # Abbreviations like U.S.A.
    \w+(?:[-']\w+)*       |  # Words with hyphens/apostrophes
    [.,;:!?(){}\[\]"']    |  # Punctuation
    \S+                      # Catch-all
""", re.VERBOSE | re.UNICODE)


def _split_words(text: str) -> List[str]:
    """Split text into word tokens."""
    return _WORD_PATTERN.findall(text)


# ============================================================
# DEPENDENCY PARSER — rule-based from grammar JSON
# Uses SVO word order rules to assign basic dependencies.
# ============================================================

def _parse_dependencies(words: List[str], pos_tags: List[str],
                        grammar: dict) -> List[Tuple[str, int]]:
    """Assign dependency relations using grammar rules.
    Returns list of (dep_label, head_index) for each word.

    Strategy (for SVO languages like English):
    1. Find ROOT verb (first content verb, or copula if no content verb)
    2. Nouns BEFORE root → nsubj
    3. Nouns AFTER root → dobj
    4. Adjectives → amod of nearest noun
    5. Adverbs → advmod of nearest verb
    6. Determiners → det of next noun
    7. Prepositions → prep of head, with pobj for following noun
    8. WH-words at start → mark as question
    """
    n = len(words)
    deps = [('', 0)] * n  # (label, head_idx)
    if n == 0:
        return deps

    deps = list(deps)  # Make mutable

    # Find ROOT — the main verb
    root_idx = -1
    # Prefer content verbs over auxiliaries
    for i, (w, p) in enumerate(zip(words, pos_tags)):
        if p == 'VERB':
            root_idx = i
            break
    # Fallback: first AUX as copula
    if root_idx == -1:
        for i, p in enumerate(pos_tags):
            if p == 'AUX':
                root_idx = i
                break
    # Fallback: first content word
    if root_idx == -1:
        for i, p in enumerate(pos_tags):
            if p in ('NOUN', 'PROPN', 'ADJ'):
                root_idx = i
                break
    if root_idx == -1:
        root_idx = 0

    deps[root_idx] = ('ROOT', root_idx)

    # Assign dependencies based on position relative to ROOT
    found_subj = False
    for i, (w, p) in enumerate(zip(words, pos_tags)):
        if i == root_idx:
            continue

        if p in ('NOUN', 'PROPN', 'PRON'):
            if i < root_idx and not found_subj:
                deps[i] = ('nsubj', root_idx)
                found_subj = True
            elif i > root_idx:
                # Check if preceded by preposition
                if i > 0 and pos_tags[i - 1] == 'ADP':
                    deps[i] = ('pobj', i - 1)
                else:
                    deps[i] = ('dobj', root_idx)

        elif p == 'ADJ':
            # Find nearest noun to modify
            nearest_noun = -1
            for j in range(i + 1, min(i + 4, n)):
                if pos_tags[j] in ('NOUN', 'PROPN'):
                    nearest_noun = j
                    break
            if nearest_noun == -1:
                for j in range(i - 1, max(i - 4, -1), -1):
                    if pos_tags[j] in ('NOUN', 'PROPN'):
                        nearest_noun = j
                        break
            deps[i] = ('amod', nearest_noun if nearest_noun >= 0 else root_idx)

        elif p == 'ADV':
            deps[i] = ('advmod', root_idx)

        elif p == 'DET':
            # Find next noun
            for j in range(i + 1, min(i + 4, n)):
                if pos_tags[j] in ('NOUN', 'PROPN'):
                    deps[i] = ('det', j)
                    break
            else:
                deps[i] = ('det', root_idx)

        elif p == 'ADP':
            deps[i] = ('prep', root_idx)

        elif p == 'AUX':
            deps[i] = ('aux', root_idx)

        elif p == 'CCONJ' or p == 'SCONJ':
            deps[i] = ('cc', root_idx)

        elif p == 'NUM':
            # Find nearest noun
            for j in range(i + 1, min(i + 3, n)):
                if pos_tags[j] in ('NOUN', 'PROPN'):
                    deps[i] = ('nummod', j)
                    break
            else:
                deps[i] = ('nummod', root_idx)

        elif p == 'PUNCT':
            deps[i] = ('punct', root_idx)

        else:
            deps[i] = ('dep', root_idx)

    # Handle passive: "was painted by X" → nsubjpass
    for i, (w, p) in enumerate(zip(words, pos_tags)):
        if p == 'AUX' and w.lower() in ('was', 'were', 'been', 'being'):
            if i + 1 < n and pos_tags[i + 1] == 'VERB':
                # Check for passive subject
                for j in range(i):
                    if deps[j][0] == 'nsubj':
                        deps[j] = ('nsubjpass', deps[j][1])
                        break
                # Check for "by" agent
                for j in range(i + 2, n):
                    if words[j].lower() == 'by' and j + 1 < n:
                        if pos_tags[j + 1] in ('NOUN', 'PROPN'):
                            deps[j] = ('agent', deps[j][1])
                        break

    # Handle negation
    for i, (w, p) in enumerate(zip(words, pos_tags)):
        if w.lower() in ('not', "n't", 'never', 'no'):
            deps[i] = ('neg', root_idx)

    return deps


# ============================================================
# ENTITY DETECTION — vocab + capitalization
# ============================================================

def _detect_entities(words: List[str], pos_tags: List[str],
                     vocab: dict) -> List[str]:
    """Detect entity types for each word. Returns list of entity type strings.
    Empty string = not an entity."""
    entities = [''] * len(words)

    for i, (w, p) in enumerate(zip(words, pos_tags)):
        lower = w.lower()
        entry = vocab.get(w) or vocab.get(lower)

        if p == 'PROPN' or (w.isupper() and len(w) > 1):
            # Check vocab senses for entity type
            if entry:
                senses = ' '.join(entry.get('senses', [])).lower()
                if any(k in senses for k in ('country', 'city', 'capital', 'state', 'nation')):
                    entities[i] = 'GPE'
                elif any(k in senses for k in ('person', 'painter', 'scientist', 'writer', 'president')):
                    entities[i] = 'PERSON'
                elif any(k in senses for k in ('organization', 'company', 'institution', 'university')):
                    entities[i] = 'ORG'
                else:
                    entities[i] = 'PROPN'
            else:
                entities[i] = 'PROPN'

        elif p == 'NUM':
            entities[i] = 'CARDINAL'

    return entities


# ============================================================
# SENTENCE SPLITTER
# ============================================================

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation rules."""
    sentences = _SENT_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================
# MAIN TOKENIZER — replaces spaCy
# ============================================================

def tokenize_vocab(text: str, lang: str = 'en') -> Tuple[List[Token], List[Tuple[str, str]]]:
    """Tokenize text using vocab DB + grammar rules. No spaCy.
    Returns (tokens, entity_spans) — same interface as lexer.tokenize()."""
    if not text or not text.strip():
        return [], []

    from .lexer import detect_token_language

    vocab = _get_vocab(lang)
    grammar = _get_grammar(lang)
    words = _split_words(text)

    if not words:
        return [], []

    # POS tag each word (with context from previous POS)
    pos_tags = []
    prev_pos = ''
    for i, w in enumerate(words):
        is_start = (i == 0)
        pos = _tag_word(w, vocab, is_sentence_start=is_start, prev_pos=prev_pos)
        pos_tags.append(pos)
        prev_pos = pos

    # Dependency parse
    deps = _parse_dependencies(words, pos_tags, grammar)

    # Entity detection
    ent_types = _detect_entities(words, pos_tags, vocab)

    # Build Token objects
    tokens = []
    for i, (w, p, (dep, head), etype) in enumerate(zip(words, pos_tags, deps, ent_types)):
        token_lang = detect_token_language(w)
        lemma = _lemmatize(w, p, vocab)
        tokens.append(Token(
            text=w,
            lang=token_lang,
            pos=p,
            dep=dep,
            head_idx=head,
            lemma=lemma,
            is_entity=etype != '',
            entity_type=etype,
        ))

    # Build entity spans (consecutive same-type entities = one span)
    entity_spans = []
    i = 0
    while i < len(tokens):
        if tokens[i].entity_type:
            start = i
            etype = tokens[i].entity_type
            while i < len(tokens) and tokens[i].entity_type == etype:
                i += 1
            span_text = ' '.join(tokens[j].text for j in range(start, i))
            entity_spans.append((span_text, etype))
        else:
            i += 1

    return tokens, entity_spans
