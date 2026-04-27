"""
ULI Lexer — Layer 1: Normalize, detect language, tokenize.
100% rules + database. Zero model calls.
"""

import re
import json
import os
import unicodedata
from typing import List, Dict, Optional, Tuple
from .protocol import Token


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def _load_json(path):
    """Load JSON file, return empty dict if missing."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ============================================================
# Language Detection (script-based + n-gram fallback)
# ============================================================

# Unicode script ranges for fast detection
SCRIPT_RANGES = {
    'devanagari': (0x0900, 0x097F),   # Hindi, Marathi, Sanskrit
    'latin': (0x0041, 0x024F),         # English, French, Spanish, etc.
    'arabic': (0x0600, 0x06FF),
    'cjk': (0x4E00, 0x9FFF),          # Chinese
    'hangul': (0xAC00, 0xD7AF),        # Korean
    'hiragana': (0x3040, 0x309F),      # Japanese
    'katakana': (0x30A0, 0x30FF),
    'cyrillic': (0x0400, 0x04FF),      # Russian
    'tamil': (0x0B80, 0x0BFF),
    'telugu': (0x0C00, 0x0C7F),
    'kannada': (0x0C80, 0x0CFF),
    'gujarati': (0x0A80, 0x0AFF),
    'bengali': (0x0980, 0x09FF),
}

SCRIPT_TO_LANG = {
    'devanagari': 'hi',   # Default to Hindi; Marathi detected by vocab later
    'arabic': 'ar',
    'cjk': 'zh',
    'hangul': 'ko',
    'hiragana': 'ja',
    'katakana': 'ja',
    'cyrillic': 'ru',
    'tamil': 'ta',
    'telugu': 'te',
    'kannada': 'kn',
    'gujarati': 'gu',
    'bengali': 'bn',
}


def detect_script(char: str) -> str:
    """Detect script of a single character."""
    cp = ord(char)
    for script, (lo, hi) in SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return script
    return 'unknown'


def detect_language(text: str) -> str:
    """Detect primary language from text using script analysis.
    Returns ISO 639-1 code. Falls back to langdetect for Latin scripts."""
    scripts = {}
    for ch in text:
        if ch.isalpha():
            s = detect_script(ch)
            scripts[s] = scripts.get(s, 0) + 1

    if not scripts:
        return 'en'  # Default

    primary = max(scripts, key=scripts.get)

    if primary in SCRIPT_TO_LANG:
        return SCRIPT_TO_LANG[primary]

    # Latin script — could be English, French, Spanish, etc.
    # Use langdetect as fallback for Latin scripts
    if primary == 'latin':
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return 'en'

    return 'en'


def detect_token_language(token: str) -> str:
    """Detect language of a single token (for code-switching)."""
    if not token:
        return 'en'
    scripts = set()
    for ch in token:
        if ch.isalpha():
            scripts.add(detect_script(ch))
    if 'devanagari' in scripts:
        return 'mr'  # or 'hi' — differentiate later by vocab
    if 'latin' in scripts:
        return 'en'
    if len(scripts) == 1:
        script = scripts.pop()
        return SCRIPT_TO_LANG.get(script, 'en')
    return 'en'


# ============================================================
# Spell Correction (edit distance, no model)
# ============================================================

def edit_distance_1(word: str) -> set:
    """All words within edit distance 1."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def correct_spelling(word: str, vocab: set, max_dist: int = 2) -> str:
    """Correct spelling using edit distance against vocabulary.
    Returns original if no correction found within max_dist."""
    if word.lower() in vocab:
        return word

    # Edit distance 1
    candidates_1 = edit_distance_1(word.lower()) & vocab
    if candidates_1:
        return max(candidates_1, key=len)  # Prefer longer match

    # Edit distance 2
    if max_dist >= 2:
        candidates_2 = set()
        for w in edit_distance_1(word.lower()):
            candidates_2 |= edit_distance_1(w) & vocab
        if candidates_2:
            return max(candidates_2, key=len)

    return word  # No correction found


# ============================================================
# Normalizer
# ============================================================

class Normalizer:
    """Normalize text using language-specific rules from JSON."""

    def __init__(self, lang: str = 'en'):
        self.lang = lang
        self.rules = _load_json(os.path.join(DATA_DIR, 'normalize', f'{lang}.json'))
        self.abbreviations = self.rules.get('abbreviations', {})
        self.substitutions = self.rules.get('substitutions', {})
        self.contractions = self.rules.get('contractions', {})
        # Build vocab set for spell correction
        vocab_data = _load_json(os.path.join(DATA_DIR, 'vocab', f'{lang}.json'))
        self.vocab = set(vocab_data.get('words', {}).keys())

    def normalize(self, text: str) -> str:
        """Full normalization pipeline.

        NOTE: Spell correction is NOT done here. It was corrupting valid words
        (e.g., "What" → "wheat") because it runs before the parser can identify
        known words. Spell correction should happen post-tokenization, only on
        tokens the parser marks as unknown/OOV. See learner.py for that path.
        """
        text = self._normalize_unicode(text)
        text = self._expand_contractions(text)
        text = self._substitute_characters(text)
        text = self._expand_abbreviations(text)
        # Spell correction intentionally removed from pre-parse pipeline.
        # It belongs post-tokenization where we know which words are real.
        return text.strip()

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode (NFC form)."""
        return unicodedata.normalize('NFC', text)

    def _substitute_characters(self, text: str) -> str:
        """Replace leetspeak and character substitutions."""
        for old, new in self.substitutions.items():
            text = text.replace(old, new)
        return text

    def _expand_contractions(self, text: str) -> str:
        """Expand contractions: dont → don't, im → I'm."""
        words = text.split()
        result = []
        for w in words:
            lower = w.lower().rstrip('.,!?;:')
            if lower in self.contractions:
                result.append(self.contractions[lower])
            else:
                result.append(w)
        return ' '.join(result)

    def _expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations: ppl → people, bc → because."""
        words = text.split()
        result = []
        for w in words:
            lower = w.lower().rstrip('.,!?;:')
            if lower in self.abbreviations:
                result.append(self.abbreviations[lower])
            else:
                result.append(w)
        return ' '.join(result)

    def _correct_spelling(self, text: str) -> str:
        """Fix typos using edit distance against vocab DB."""
        if not self.vocab:
            return text
        words = text.split()
        result = []
        for w in words:
            # Skip short words, numbers, punctuation, URLs
            if len(w) <= 2 or not w.isalpha() or w.startswith('http'):
                result.append(w)
                continue
            corrected = correct_spelling(w, self.vocab)
            result.append(corrected)
        return ' '.join(result)


# ============================================================
# Tokenizer (wraps spaCy for POS/dep, adds language tags)
# ============================================================

_spacy_models = {}

def _get_spacy(lang='en'):
    """Lazy-load spaCy model."""
    if lang not in _spacy_models:
        import spacy
        if lang == 'en':
            _spacy_models[lang] = spacy.load('en_core_web_sm')
        else:
            # For non-English, try to load or fall back to English
            try:
                _spacy_models[lang] = spacy.load(f'{lang}_core_web_sm')
            except OSError:
                _spacy_models[lang] = spacy.load('en_core_web_sm')
    return _spacy_models[lang]


def tokenize(text: str, lang: str = 'en'):
    """Tokenize text using spaCy. Returns (tokens, entity_spans).
    entity_spans are spaCy's grouped multi-word entities."""
    nlp = _get_spacy(lang)
    doc = nlp(text)

    tokens = []
    for tok in doc:
        token_lang = detect_token_language(tok.text)
        tokens.append(Token(
            text=tok.text,
            lang=token_lang,
            pos=tok.pos_,
            dep=tok.dep_,
            head_idx=tok.head.i,
            lemma=tok.lemma_,
            is_entity=tok.ent_type_ != '',
            entity_type=tok.ent_type_,
        ))

    # spaCy's grouped entity spans (multi-word: "Leonardo da Vinci", "World War II")
    entity_spans = [(ent.text, ent.label_) for ent in doc.ents]

    return tokens, entity_spans


# ============================================================
# Emoji handling
# ============================================================

EMOJI_MAP = {
    '🔥': 'excellent',
    '❤️': 'love',
    '😂': 'funny',
    '😢': 'sad',
    '👍': 'agree',
    '👎': 'disagree',
    '🤔': 'thinking',
    '💯': 'perfect',
    '🙏': 'please/thanks',
    '😊': 'happy',
    '😡': 'angry',
    '🎉': 'celebration',
}


def replace_emoji(text: str) -> str:
    """Replace emoji with text equivalents."""
    for emoji, meaning in EMOJI_MAP.items():
        text = text.replace(emoji, f' [{meaning}] ')
    return text


# ============================================================
# URL / mention / hashtag extraction
# ============================================================

URL_PATTERN = re.compile(r'https?://\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
NUMBER_PATTERN = re.compile(r'\$?[\d,]+\.?\d*%?')


def extract_special(text: str) -> Tuple[str, Dict[str, List[str]]]:
    """Extract URLs, mentions, hashtags, numbers from text.
    Returns cleaned text and dict of extracted items."""
    extracted = {
        'urls': URL_PATTERN.findall(text),
        'mentions': MENTION_PATTERN.findall(text),
        'hashtags': HASHTAG_PATTERN.findall(text),
        'numbers': NUMBER_PATTERN.findall(text),
    }
    # Remove URLs from text (keep others)
    text = URL_PATTERN.sub('[URL]', text)
    return text, extracted
